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

import logging
import os
from functools import partial
from typing import Iterable

import torch
import wandb
from einops import rearrange
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_model_config

from megatron.bridge.diffusion.common.utils.save_video import save_video
from megatron.bridge.diffusion.models.dit.dit_data_process import dit_data_step
from megatron.bridge.diffusion.models.dit.edm.edm_pipeline import EDMPipeline


logger = logging.getLogger(__name__)


class DITForwardStep:
    def __init__(self):
        self.diffusion_pipeline = EDMPipeline(sigma_data=0.5)
        self.valid = False
        self.train = True

    def on_validation_start(self, state, batch, model):
        C, T, H, W = batch["latent_shape"][0]
        latent = self.diffusion_pipeline.generate_samples_from_batch(
            model,
            batch,
            guidance=model.config.val_generation_guidance,
            state_shape=batch["video"].shape,
            num_steps=model.config.val_generation_num_steps,
            is_negative_prompt=True if "neg_context_embeddings" in batch else False,
        )
        caption = batch["video_metadata"][0]["caption"] if "caption" in batch["video_metadata"][0] else "no caption"
        latent = latent[0, None, : batch["seq_len_q"][0]]
        latent = rearrange(
            latent,
            "b (T H W) (ph pw pt c) -> b c (T pt) (H ph) (W pw)",
            ph=model.config.patch_spatial,
            pw=model.config.patch_spatial,
            pt=model.config.patch_temporal,
            c=C,
            T=T // model.config.patch_temporal,
            H=H // model.config.patch_spatial,
            W=W // model.config.patch_spatial,
        )

        vae = model.config.configure_vae().to("cuda")

        decoded_video = (1.0 + vae.decode(latent / self.diffusion_pipeline.sigma_data)).clamp(0, 2) / 2
        decoded_video = (decoded_video * 255).to(torch.uint8).permute(0, 2, 3, 4, 1).cpu().numpy()
        rank = torch.distributed.get_rank()

        image_folder = os.path.join(state.cfg.checkpoint.save, "validation_generation")
        os.makedirs(image_folder, exist_ok=True)
        save_video(
            grid=decoded_video[0],
            fps=decoded_video.shape[1],
            H=decoded_video.shape[2],
            W=decoded_video.shape[3],
            video_save_quality=5,
            video_save_path=f"{image_folder}/step={state.train_state.step}_rank={rank}.mp4",
            caption=caption,
        )

        # Log the video to Weights & Biases
        is_last_dp_rank = parallel_state.get_data_parallel_rank() == (
            parallel_state.get_data_parallel_world_size() - 1
        )

        last_dp_local_rank = parallel_state.get_data_parallel_world_size() - 1
        dp_group = parallel_state.get_data_parallel_group()
        dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
        wandb_rank = dp_ranks[last_dp_local_rank]

        if is_last_dp_rank:
            gather_list = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        else:
            gather_list = None

        torch.distributed.gather_object(
            obj=(decoded_video[0], caption),
            object_gather_list=gather_list,
            dst=wandb_rank,
            group=parallel_state.get_data_parallel_group(),
        )

        if is_last_dp_rank and state.wandb_logger is not None:
            if gather_list is not None:
                videos = []
                for video_data, video_caption in gather_list:
                    video_data_transposed = video_data.transpose(0, 3, 1, 2)
                    videos.append(wandb.Video(video_data_transposed, fps=24, format="mp4", caption=video_caption))

                state.wandb_logger.log({"prediction": videos})

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, partial]:
        """Forward training step.

        Args:
            state: Global state for the run
            data_iterator: Input data iterator
            model: The GPT Model
            return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

        Returns:
            tuple containing the output tensor and the loss function
        """
        batch = self.data_process(state, data_iterator, model, return_schedule_plan)
        if model.training and self.valid:
            self.train = True
            self.valid = False
        elif (not model.training) and self.train:
            self.train = False
            self.valid = True
            self.on_validation_start(state, batch, model)
        return self.forward_step(state, batch, model, return_schedule_plan)

    def data_process(
        self, state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, partial]:
        timers = state.timers
        straggler_timer = state.straggler_timer

        config = get_model_config(model)

        timers("batch-generator", log_level=2).start()
        qkv_format = getattr(config, "qkv_format", "sbhd")
        with straggler_timer(bdata=True):
            batch = dit_data_step(qkv_format, data_iterator)
        return batch

    def forward_step(self, state, batch, model, return_schedule_plan: bool = False):
        timers = state.timers
        timers("batch-generator").stop()

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
        straggler_timer = state.straggler_timer
        with straggler_timer:
            if parallel_state.is_pipeline_last_stage():
                # Self.diffusion_pipeline should not know anything about the model
                # TODO: we need to sepearte the noise ingection process from the pipeline itself
                output_batch, loss = self.diffusion_pipeline.training_step(model, batch, 0)
                output_tensor = torch.mean(loss, dim=-1)
            else:
                output_tensor = self.diffusion_pipeline.training_step(model, batch, 0)

        loss = output_tensor
        if "loss_mask" not in batch or batch["loss_mask"] is None:
            loss_mask = torch.ones_like(loss)
        else:
            loss_mask = batch["loss_mask"]
        loss_function = self._create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
        return output_tensor, loss_function

    def _create_loss_function(
        self, loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
    ) -> partial:
        """Create a partial loss function with the specified configuration.

        Args:
            loss_mask: Used to mask out some portions of the loss
            check_for_nan_in_loss: Whether to check for NaN values in the loss
            check_for_spiky_loss: Whether to check for spiky loss values

        Returns:
            A partial function that can be called with output_tensor to compute the loss
        """
        return partial(
            masked_next_token_loss,
            loss_mask,
            check_for_nan_in_loss=check_for_nan_in_loss,
            check_for_spiky_loss=check_for_spiky_loss,
        )
