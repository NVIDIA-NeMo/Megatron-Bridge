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
from functools import partial
from typing import Iterable

import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config

from megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan import (
    WanAdapter,
    WanFlowMatchingPipeline,
)
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)


def wan_data_step(qkv_format, dataloader_iter):  # noqa: D103
    batch = next(dataloader_iter)
    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    # Construct packed sequence parameters
    if ("seq_len_q" in batch) and ("seq_len_kv" in batch):
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")

        cu_seqlens = batch["seq_len_q"].cumsum(dim=0).to(torch.int32)
        cu_seqlens = torch.cat((zero, cu_seqlens))

        cu_seqlens_padded = batch["seq_len_q_padded"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_padded = torch.cat((zero, cu_seqlens_padded))

        cu_seqlens_kv = batch["seq_len_kv"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        cu_seqlens_kv_padded = batch["seq_len_kv_padded"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv_padded = torch.cat((zero, cu_seqlens_kv_padded))

        batch["packed_seq_params"] = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                qkv_format=qkv_format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                qkv_format=qkv_format,
            ),
        }

    # tranpose from "sbhd" to "bshd" to be compatible with flow matching pipeline
    batch["video_latents"] = batch["video_latents"].transpose(0, 1)

    return batch


class WanForwardStep:  # noqa: D101
    def __init__(
        self,
        use_sigma_noise: bool = True,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 3.0,
        mix_uniform_ratio: float = 0.1,
        sigma_min: float = 0.0,  # Default: no clamping (pretrain)
        sigma_max: float = 1.0,  # Default: no clamping (pretrain)
    ):
        self.diffusion_pipeline = WanFlowMatchingPipeline(
            model_adapter=WanAdapter(),
            timestep_sampling=timestep_sampling,
            logit_mean=logit_mean,
            logit_std=logit_std,
            flow_shift=flow_shift,
            mix_uniform_ratio=mix_uniform_ratio,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        self.use_sigma_noise = use_sigma_noise
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.flow_shift = flow_shift
        self.mix_uniform_ratio = mix_uniform_ratio
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: VisionModule
    ) -> tuple[torch.Tensor, partial]:
        """
        Forward training step.
        """
        timers = state.timers
        straggler_timer = state.straggler_timer

        config = get_model_config(model)

        timers("batch-generator", log_level=2).start()

        qkv_format = getattr(config, "qkv_format", "sbhd")
        with straggler_timer(bdata=True):
            batch = wan_data_step(qkv_format, data_iterator)
        timers("batch-generator").stop()

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss

        # run diffusion training step
        with straggler_timer:
            weighted_loss, average_weighted_loss, loss_mask, metrics = self.diffusion_pipeline.step(
                model,
                batch,
            )
            output_tensor = torch.mean(weighted_loss, dim=-1)
            batch["loss_mask"] = loss_mask

        # TODO: do we need to gather output with sequence or context parallelism here
        #       especially when we have pipeline parallelism

        loss = output_tensor
        if "loss_mask" not in batch or batch["loss_mask"] is None:
            loss_mask = torch.ones_like(loss)
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
