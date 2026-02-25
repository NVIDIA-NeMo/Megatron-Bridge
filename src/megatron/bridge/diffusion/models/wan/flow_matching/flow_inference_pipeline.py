# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import random
import re
import sys
from contextlib import contextmanager
from typing import Tuple

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from megatron.bridge.training.model_load_save import load_megatron_model as _load_megatron_model
from megatron.core import parallel_state
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
    recv_from_prev_pipeline_rank_,
    send_to_next_pipeline_rank,
)
from megatron.core.packed_seq_params import PackedSeqParams
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from megatron.bridge.diffusion.models.wan.utils import grid_sizes_calculation, patchify, unpatchify
from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider


@torch.no_grad()
def _encode_text(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    device: str,
    caption: str,
) -> torch.Tensor:
    caption = caption.strip()
    inputs = tokenizer(
        caption,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
    # Trim to the true (unpadded) sequence length using the attention mask
    true_len = int(inputs["attention_mask"].sum(dim=-1).item())
    outputs = outputs[0, :true_len, :]
    return outputs


class FlowInferencePipeline:
    def __init__(
        self,
        inference_cfg,
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        checkpoint_dir=None,
        checkpoint_step=None,
        t5_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        device_id=0,
        rank=0,
        t5_cpu=False,
        tensor_parallel_size=1,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.float32,
    ):
        r"""
        Initializes the FlowInferencePipeline with the given parameters.

        Args:
            inference_cfg (dict):
                Object containing inference configuration.
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            t5_checkpoint_dir (`str`, *optional*, defaults to None):
                Optional directory containing T5 checkpoint and tokenizer; falls back to `checkpoint_dir` if None.
            vae_checkpoint_dir (`str`, *optional*, defaults to None):
                Optional directory containing VAE checkpoint; falls back to `checkpoint_dir` if None.
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.inference_cfg = inference_cfg
        self.model_id = model_id
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.tensor_parallel_size = tensor_parallel_size
        self.context_parallel_size = context_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.sequence_parallel = sequence_parallel
        self.pipeline_dtype = pipeline_dtype
        self.num_train_timesteps = inference_cfg.num_train_timesteps
        self.param_dtype = inference_cfg.param_dtype
        self.text_len = inference_cfg.text_len

        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=inference_cfg.t5_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
        )

        self.vae_stride = inference_cfg.vae_stride
        self.patch_size = inference_cfg.patch_size
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=inference_cfg.param_dtype,
        )
        self.vae.to(self.device)

        wan_checkpoint_dir = self._select_checkpoint_dir(checkpoint_dir, checkpoint_step)
        self.model = self.setup_model_from_checkpoint(wan_checkpoint_dir)

        # if we use context parallelism, we need to set qkv_format to "thd" for context parallelism
        self.model.config.qkv_format = "thd"  # "sbhd"

        # set self.sp_size=1 for later use, just to respect the original Wan inference code
        self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        self.model.to(self.device)

        self.sample_neg_prompt = inference_cfg.english_sample_neg_prompt

    def setup_model_from_checkpoint(self, checkpoint_dir):
        provider = WanModelProvider()
        provider.tensor_model_parallel_size = self.tensor_parallel_size
        provider.pipeline_model_parallel_size = self.pipeline_parallel_size
        provider.context_parallel_size = self.context_parallel_size
        provider.sequence_parallel = self.sequence_parallel
        provider.pipeline_dtype = self.pipeline_dtype
        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        provider.finalize()
        provider.initialize_model_parallel(seed=0)

        ## Read from megatron checkpoint
        model = _load_megatron_model(
            checkpoint_dir,
            mp_overrides={
                "tensor_model_parallel_size": self.tensor_parallel_size,
                "pipeline_model_parallel_size": self.pipeline_parallel_size,
                "context_parallel_size": self.context_parallel_size,
                "sequence_parallel": self.sequence_parallel,
                "pipeline_dtype": self.pipeline_dtype,
            },
        )
        if isinstance(model, list):
            model = model[0]
        if hasattr(model, "module"):
            model = model.module

        return model

    def _select_checkpoint_dir(self, base_dir: str, checkpoint_step) -> str:
        """
        Resolve checkpoint directory:
        - If checkpoint_step is provided, use base_dir/iter_{step:07d}
        - Otherwise, pick the largest iter_######## subdirectory under base_dir
        """
        if checkpoint_step is not None:
            path = os.path.join(base_dir, f"iter_{int(checkpoint_step):07d}")
            if os.path.isdir(path):
                logging.info(f"Using specified checkpoint: {path}")
                return path
            raise FileNotFoundError(f"Specified checkpoint step {checkpoint_step} not found at {path}")

        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Checkpoint base directory does not exist: {base_dir}")

        pattern = re.compile(r"^iter_(\d+)$")
        try:
            _, latest_path = max(
                (
                    (int(pattern.match(e.name).group(1)), e.path)
                    for e in os.scandir(base_dir)
                    if e.is_dir() and pattern.match(e.name)
                ),
                key=lambda x: x[0],
            )
        except ValueError:
            raise FileNotFoundError(
                f"No checkpoints found under {base_dir}. Expected subdirectories named like 'iter_0001800'."
            )

        logging.info(f"Auto-selected latest checkpoint: {latest_path}")
        return latest_path

    def forward_pp_step(
        self,
        latent_model_input: torch.Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        max_video_seq_len: int,
        timestep: torch.Tensor,
        arg_c: dict,
    ) -> torch.Tensor:
        """
        Forward pass supporting pipeline parallelism.
        """

        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        # PP=1: no pipeline parallelism (avoid touching PP groups which may be uninitialized in unit tests)
        if pp_world_size == 1:
            noise_pred_pp = self.model(latent_model_input, grid_sizes=grid_sizes, t=timestep, **arg_c)
            return noise_pred_pp
        # For PP>1, safe to query stage information
        is_pp_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        is_pp_last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

        # PP>1: pipeline parallelism
        hidden_size = self.model.config.hidden_size
        batch_size = latent_model_input.shape[1]
        # noise prediction shape for communication between first and last pipeline stages
        noise_pred_pp_shape = list(latent_model_input.shape)

        if is_pp_first:
            # First stage: compute multimodal + first PP slice, send activations, then receive sampled token
            hidden_states = self.model(latent_model_input, grid_sizes=grid_sizes, t=timestep, **arg_c)
            send_to_next_pipeline_rank(hidden_states)

            noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=torch.float32)
            return noise_pred_pp

        if is_pp_last:
            # Last stage: recv activations, run final slice + output, sample, broadcast
            recv_buffer = torch.empty(
                (max_video_seq_len, batch_size, hidden_size),
                dtype=next(self.model.parameters()).dtype,
                device=latent_model_input[0].device,
            )
            recv_from_prev_pipeline_rank_(recv_buffer)
            recv_buffer = recv_buffer.to(torch.bfloat16)
            self.model.set_input_tensor(recv_buffer)
            noise_pred_pp = self.model(latent_model_input, grid_sizes=grid_sizes, t=timestep, **arg_c)

            noise_pred_pp = broadcast_from_last_pipeline_stage(
                noise_pred_pp_shape, dtype=noise_pred_pp.dtype, tensor=noise_pred_pp.contiguous()
            )
            return noise_pred_pp

        # Intermediate stages: recv -> run local slice -> send -> receive broadcast token
        recv_buffer = torch.empty(
            (max_video_seq_len, batch_size, hidden_size),
            dtype=next(self.model.parameters()).dtype,
            device=latent_model_input[0].device,
        )
        recv_from_prev_pipeline_rank_(recv_buffer)
        recv_buffer = recv_buffer.to(torch.bfloat16)
        self.model.set_input_tensor(recv_buffer)
        hidden_states = self.model(latent_model_input, grid_sizes=grid_sizes, t=timestep, **arg_c)
        send_to_next_pipeline_rank(hidden_states)

        noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=torch.float32)
        return noise_pred_pp

    def generate(
        self,
        prompts,
        sizes,
        frame_nums,
        shift=5.0,
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            prompts (`list[str]`):
                Text prompt for content generation
            sizes (list[tuple[int, int]]):
                Controls video resolution, (width,height).
            frame_nums (`list[int]`):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """

        # preprocess
        target_shapes = []
        for size, frame_num in zip(sizes, frame_nums):
            target_shapes.append(
                (
                    self.vae.config.z_dim,
                    (frame_num - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2],
                )
            )
        max_video_seq_len = 0
        seq_lens = []
        for target_shape in target_shapes:
            seq_len = (
                math.ceil(
                    (target_shape[2] * target_shape[3])
                    / (self.patch_size[1] * self.patch_size[2])
                    * target_shape[1]
                    / self.sp_size
                )
                * self.sp_size
            )
            seq_lens.append(seq_len)
        max_video_seq_len = max(seq_lens)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        ## process context
        # we implement similar to Wan's diffuser setup
        # (https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/wan/pipeline_wan.py#L157)
        # in which we pad the text to 512, pass through text encoder, and truncate to the actual tokens, then pad with 0s to 512.
        context_max_len = self.text_len
        context_lens = []
        contexts = []
        contexts_null = []
        for prompt in prompts:
            if not self.t5_cpu:
                self.text_encoder.to(self.device)
                context = _encode_text(self.tokenizer, self.text_encoder, self.device, prompt)
                context_null = _encode_text(self.tokenizer, self.text_encoder, self.device, n_prompt)
                if offload_model:
                    self.text_encoder.cpu()
            else:
                context = self.text_encoder([prompt], torch.device("cpu"))[0].to(self.device)
                context_null = self.text_encoder([n_prompt], torch.device("cpu"))[0].to(self.device)
            context_lens.append(context_max_len)  # all samples have the same context_max_len
            contexts.append(context)
            contexts_null.append(context_null)

        # pad to context_max_len tokens, and stack to a tensor of shape [s, b, hidden]
        contexts = [F.pad(context, (0, 0, 0, context_max_len - context.shape[0])) for context in contexts]
        contexts_null = [
            F.pad(context_null, (0, 0, 0, context_max_len - context_null.shape[0])) for context_null in contexts_null
        ]
        contexts = torch.stack(contexts, dim=1)
        contexts_null = torch.stack(contexts_null, dim=1)

        ## setup noise
        noises = []
        for target_shape in target_shapes:
            noises.append(
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g,
                )
            )

        # calculate grid_sizes
        grid_sizes = [
            grid_sizes_calculation(
                input_shape=u.shape[1:],
                patch_size=self.model.patch_size,
            )
            for u in noises
        ]
        grid_sizes = torch.tensor(grid_sizes, dtype=torch.long)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Instantiate per-sample schedulers so each sample maintains its own state
            batch_size_for_schedulers = len(noises)
            schedulers = []
            for _ in range(batch_size_for_schedulers):
                base_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
                s = UniPCMultistepScheduler.from_config(base_sched.config, flow_shift=shift)
                s.set_timesteps(sampling_steps, device=self.device)

                schedulers.append(s)
            timesteps = schedulers[0].timesteps

            # sample videos
            latents = noises

            cu_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)])
            cu_q = cu_q.to(torch.int32).to(self.device)
            cu_kv_self = cu_q
            cu_kv_cross = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(context_lens), dim=0)])
            cu_kv_cross = cu_kv_cross.to(torch.int32).to(self.device)
            packed_seq_params = {
                "self_attention": PackedSeqParams(
                    cu_seqlens_q=cu_q,
                    cu_seqlens_q_padded=cu_q,
                    cu_seqlens_kv=cu_kv_self,
                    cu_seqlens_kv_padded=cu_kv_self,
                    qkv_format=self.model.config.qkv_format,
                ),
                "cross_attention": PackedSeqParams(
                    cu_seqlens_q=cu_q,
                    cu_seqlens_q_padded=cu_q,
                    cu_seqlens_kv=cu_kv_cross,
                    qkv_format=self.model.config.qkv_format,
                ),
            }

            arg_c = {"context": contexts, "max_seq_len": max_video_seq_len, "packed_seq_params": packed_seq_params}
            arg_null = {
                "context": contexts_null,
                "max_seq_len": max_video_seq_len,
                "packed_seq_params": packed_seq_params,
            }

            for _, t in enumerate(tqdm(timesteps)):
                batch_size = len(latents)

                # patchify latents
                unpatchified_latents = latents
                latents = patchify(latents, self.patch_size)
                # pad to have same length
                for i in range(batch_size):
                    latents[i] = F.pad(latents[i], (0, 0, 0, max_video_seq_len - latents[i].shape[0]))
                latents = torch.stack(latents, dim=1)

                latent_model_input = latents
                timestep = [t] * batch_size
                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.forward_pp_step(
                    latent_model_input,
                    grid_sizes=grid_sizes,
                    max_video_seq_len=max_video_seq_len,
                    timestep=timestep,
                    arg_c=arg_c,
                )

                noise_pred_uncond = self.forward_pp_step(
                    latent_model_input,
                    grid_sizes=grid_sizes,
                    max_video_seq_len=max_video_seq_len,
                    timestep=timestep,
                    arg_c=arg_null,
                )

                # run unpatchify
                unpatchified_noise_pred_cond = noise_pred_cond
                unpatchified_noise_pred_cond = unpatchified_noise_pred_cond.transpose(0, 1)  # bring sbhd -> bshd
                # when unpatchifying, the code will truncate the padded videos into the original video shape, based on the grid_sizes.
                unpatchified_noise_pred_cond = unpatchify(
                    unpatchified_noise_pred_cond, grid_sizes, self.vae.config.z_dim, self.patch_size
                )
                unpatchified_noise_pred_uncond = noise_pred_uncond
                unpatchified_noise_pred_uncond = unpatchified_noise_pred_uncond.transpose(0, 1)  # bring sbhd -> bshd
                # when unpatchifying, the code will truncate the padded videos into the original video shape, based on the grid_sizes.
                unpatchified_noise_pred_uncond = unpatchify(
                    unpatchified_noise_pred_uncond, grid_sizes, self.vae.config.z_dim, self.patch_size
                )

                noise_preds = []
                for i in range(batch_size):
                    noise_pred = unpatchified_noise_pred_uncond[i] + guide_scale * (
                        unpatchified_noise_pred_cond[i] - unpatchified_noise_pred_uncond[i]
                    )
                    noise_preds.append(noise_pred)

                # step and update latents
                latents = []
                for i in range(batch_size):
                    temp_x0 = schedulers[i].step(
                        noise_preds[i].unsqueeze(0), t, unpatchified_latents[i].unsqueeze(0), return_dict=False
                    )[0]
                    latents.append(temp_x0.squeeze(0))

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                # Diffusers' VAE decoding
                latents = torch.stack(x0, dim=0)
                latents = latents.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                videos = self.vae.decode(latents).sample
            else:
                videos = None

        del noises, latents
        del schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos if self.rank == 0 else None
