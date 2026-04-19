#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

from easydict import EasyDict


warnings.filterwarnings("ignore")

import random

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.utils import get_batch_on_this_cp_rank
from megatron.bridge.diffusion.models.wan.flow_matching.flow_inference_pipeline import FlowInferencePipeline
from megatron.bridge.diffusion.models.wan.inference import SIZE_CONFIGS, SUPPORTED_SIZES
from megatron.bridge.diffusion.models.wan.inference.utils import cache_video, str2bool
from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider
from megatron.bridge.training.model_load_save import load_megatron_model as _load_megatron_model


class FlowInferencePipelineSBHD(FlowInferencePipeline):
    """Inference pipeline that runs the Wan model in SBHD (batched, no sequence packing) mode."""

    def setup_model_from_checkpoint(self, checkpoint_dir):
        provider = WanModelProvider()
        provider.tensor_model_parallel_size = self.tensor_parallel_size
        provider.pipeline_model_parallel_size = self.pipeline_parallel_size
        provider.context_parallel_size = self.context_parallel_size
        provider.sequence_parallel = self.sequence_parallel
        provider.pipeline_dtype = self.pipeline_dtype
        provider.finalize()
        provider.initialize_model_parallel(seed=0)

        model = _load_megatron_model(
            checkpoint_dir,
            mp_overrides={
                "tensor_model_parallel_size": self.tensor_parallel_size,
                "pipeline_model_parallel_size": self.pipeline_parallel_size,
                "context_parallel_size": self.context_parallel_size,
                "sequence_parallel": self.sequence_parallel,
                "pipeline_dtype": self.pipeline_dtype,
                # Override qkv_format so the model is built with AttnMaskType.no_mask
                # (TE ring attention rejects padding masks with CP, and SBHD uses loss_mask
                # for gradient correctness instead of attention masking).
                "qkv_format": "sbhd",
            },
        )
        if isinstance(model, list):
            model = model[0]
        if hasattr(model, "module"):
            model = model.module
        return model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parent hardcodes config.qkv_format = "thd" after setup; restore it.
        self.model.config.qkv_format = "sbhd"

    # NOTE: _stack_contexts is intentionally NOT overridden here.
    # The model was trained with context always padded to 512 tokens, so the
    # cross-attention softmax is calibrated to 512 KV positions. Padding to a
    # shorter per-batch max shifts the attention distribution and degrades quality.

    def forward_pp_step(self, latent_model_input, grid_sizes, max_video_seq_len, timestep, arg_c):
        # SBHD mode: no packed_seq_params — model uses the batched (no_mask) path.
        arg_c = {**arg_c, "packed_seq_params": None}

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1:
            S = latent_model_input.shape[0]
            multiple = 2 * cp_size
            assert S % multiple == 0, (
                f"SBHD inference: video seq_len={S} must be divisible by "
                f"2*context_parallel_size={multiple}"
            )
            # Split latent across CP ranks using the same ZigZag interleaving as training.
            # get_batch_on_this_cp_rank operates on BSHD [B, S, D], so transpose first.
            lmi_bshd = get_batch_on_this_cp_rank({"x": latent_model_input.transpose(0, 1).contiguous()})["x"]
            latent_model_input = lmi_bshd.transpose(0, 1)  # [S/CP, B, D]
            max_video_seq_len = S // cp_size

        noise_pred = super().forward_pp_step(latent_model_input, grid_sizes, max_video_seq_len, timestep, arg_c)

        if cp_size > 1:
            # All-gather noise_pred from every CP rank and reconstruct the full sequence.
            # get_batch_on_this_cp_rank splits with ZigZag: each rank k holds
            #   [k*S/(2CP)..(k+1)*S/(2CP)-1]  (front half)
            # + [(2CP-1-k)*S/(2CP)..(2CP-k)*S/(2CP)-1]  (back half)
            # Inverse: gather fronts in rank order + backs in reverse rank order.
            noise_pred = noise_pred.contiguous()  # must be contiguous: torch.empty_like on a non-contiguous tensor preserves its strides, causing all_gather to write seq-major data into batch-major memory and scramble batch items
            s_per_rank = noise_pred.shape[0]
            half = s_per_rank // 2
            gathered = [torch.empty_like(noise_pred) for _ in range(cp_size)]
            dist.all_gather(gathered, noise_pred, group=parallel_state.get_context_parallel_group())
            fronts = [g[:half] for g in gathered]
            backs = [g[half:] for g in gathered]
            noise_pred = torch.cat(fronts + backs[::-1], dim=0)  # [S_padded, B, D]
            noise_pred = noise_pred[:S]  # trim CP-alignment padding

        return noise_pred


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    assert args.task in SUPPORTED_SIZES, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check: only validate provided --sizes; default handled later
    if args.sizes is not None and len(args.sizes) > 0:
        for s in args.sizes:
            assert s in SUPPORTED_SIZES[args.task], (
                f"Unsupport size {s} for task {args.task}, supported sizes are: "
                f"{', '.join(SUPPORTED_SIZES[args.task])}"
            )
        # SBHD batch mode requires all samples to have the same shape so that
        # no attention mask is needed (AttnMaskType.no_mask).
        assert len(set(args.sizes)) == 1, (
            f"SBHD batch mode requires all --sizes to be identical, got: {args.sizes}"
        )
    if args.frame_nums is not None and len(args.frame_nums) > 1:
        assert len(set(args.frame_nums)) == 1, (
            f"SBHD batch mode requires all --frame_nums to be identical, got: {args.frame_nums}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument(
        "--task", type=str, default="t2v-14B", choices=list(SUPPORTED_SIZES.keys()), help="The task to run."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=None,
        choices=list(SIZE_CONFIGS.keys()),
        help=(
            "Video size (WIDTH*HEIGHT). Provide one value to broadcast to all prompts, "
            "or one per prompt (all must be identical in SBHD batch mode). "
            "Example: --sizes 1280*720"
        ),
    )
    parser.add_argument(
        "--frame_nums",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Number of frames (4n+1). Provide one value to broadcast to all prompts, "
            "or one per prompt (all must be identical in SBHD batch mode)."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="The path to the main WAN checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help=(
            "Optional training step to load, e.g. 1800 -> iter_0001800. "
            "If not provided, the latest (largest) step in --checkpoint_dir is used.",
        ),
    )
    parser.add_argument(
        "--t5_checkpoint_dir", type=str, default=None, help="Optional directory containing T5 checkpoint/tokenizer"
    )
    parser.add_argument(
        "--vae_checkpoint_dir", type=str, default=None, help="Optional directory containing VAE checkpoint"
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--save_file", type=str, default=None, help="The file to save the generated image or video to."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="A list of prompts to generate multiple images or videos. Example: --prompts 'a cat' 'a dog'",
    )
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the image or video.")
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--sequence_parallel", type=str2bool, default=False, help="Sequence parallel.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):  # noqa: D103
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    videos = []

    if args.offload_model is None:
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    inference_cfg = EasyDict(
        {
            # t5
            "t5_dtype": torch.bfloat16,
            "text_len": 512,
            # vae
            "vae_stride": (4, 8, 8),
            # transformer
            "param_dtype": torch.bfloat16,
            "patch_size": (1, 2, 2),
            # others
            "num_train_timesteps": 1000,
            "sample_fps": 16,
            "chinese_sample_neg_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "english_sample_neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        }
    )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {inference_cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task:
        # Resolve prompts list (default to example prompt)
        if args.prompts is not None and len(args.prompts) > 0:
            prompts = args.prompts
        else:
            prompts = [EXAMPLE_PROMPT[args.task]["prompt"]]

        # Resolve size: single value broadcasts to all prompts (SBHD requires uniform shape).
        if args.sizes is not None and len(args.sizes) > 0:
            size_keys = args.sizes[:1] * len(prompts)
        else:
            size_keys = [SUPPORTED_SIZES[args.task][0]] * len(prompts)

        # Resolve frame count: single value broadcasts to all prompts.
        if args.frame_nums is not None and len(args.frame_nums) > 0:
            frame_nums = args.frame_nums[:1] * len(prompts)
        else:
            frame_nums = [81] * len(prompts)

        logging.info(
            f"SBHD batch mode: {len(prompts)} prompt(s), size={size_keys[0]}, frame_num={frame_nums[0]}"
        )

        logging.info("Creating flow inference pipeline (SBHD mode).")
        pipeline = FlowInferencePipelineSBHD(
            inference_cfg=inference_cfg,
            checkpoint_dir=args.checkpoint_dir,
            model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            checkpoint_step=args.checkpoint_step,
            t5_checkpoint_dir=args.t5_checkpoint_dir,
            vae_checkpoint_dir=args.vae_checkpoint_dir,
            device_id=device,
            rank=rank,
            t5_cpu=args.t5_cpu,
            tensor_parallel_size=args.tensor_parallel_size,
            context_parallel_size=args.context_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            sequence_parallel=args.sequence_parallel,
            pipeline_dtype=torch.float32,
        )

        rank = dist.get_rank() if dist.is_initialized() else rank
        if rank == 0:
            print("Running inference with tensor_parallel_size:", args.tensor_parallel_size)
            print("Running inference with context_parallel_size:", args.context_parallel_size)
            print("Running inference with pipeline_parallel_size:", args.pipeline_parallel_size)
            print("Running inference with sequence_parallel:", args.sequence_parallel)
            print("\n\n\n")

        logging.info("Generating videos ...")
        videos = pipeline.generate(
            prompts=prompts,
            sizes=[SIZE_CONFIGS[size] for size in size_keys],
            frame_nums=frame_nums,
            shift=args.sample_shift,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

        if rank == 0:
            for i, video in enumerate(videos):
                formatted_experiment_name = (args.save_file) if args.save_file is not None else "DefaultExp"
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:50]
                suffix = ".mp4"
                formatted_save_file = (
                    f"{args.task}_{formatted_experiment_name}_videoindex{int(i)}_size{size_keys[i].replace('*', 'x') if sys.platform == 'win32' else size_keys[i]}_{formatted_prompt}_{formatted_time}"
                    + suffix
                )

                logging.info(f"Saving generated video to {formatted_save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=formatted_save_file,
                    fps=inference_cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
