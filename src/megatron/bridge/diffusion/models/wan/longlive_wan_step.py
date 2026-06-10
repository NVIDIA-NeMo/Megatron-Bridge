# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import inspect
import logging
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.utils import get_model_config

from megatron.bridge.diffusion.common.flow_matching.adapters.base import FlowMatchingContext
from megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan import (
    WanAdapter,
    WanFlowMatchingPipeline,
)
from megatron.bridge.diffusion.models.wan.longlive_wan_utils import (
    ChunkSelectionStrategy,
    apply_longlive_noising,
    build_longlive_loss_mask,
    build_teacher_forcing_self_attention_mask,
    select_longlive_chunks,
    split_self_attention_mask_rows,
)
from megatron.bridge.diffusion.models.wan.utils import thd_partition_indices
from megatron.bridge.diffusion.models.wan.wan_step import _WAN_MODE_DEFAULTS, WanForwardStep, wan_data_step
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)


def _callable_accepts_kwarg(fn: Any, kwarg_name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return kwarg_name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def _model_accepts_explicit_self_attention_mask(model: nn.Module) -> bool:
    module = getattr(model, "module", model)
    decoder = getattr(module, "decoder", None)
    forward = getattr(decoder, "forward", None)
    return forward is not None and _callable_accepts_kwarg(forward, "self_attention_mask")


class LongLiveWanAdapter(WanAdapter):
    """WAN adapter that forwards an optional LongLive self-attention mask to the model."""

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        inputs = super().prepare_inputs(context)
        self_attention_mask = context.batch.get("self_attention_mask")
        if self_attention_mask is not None and parallel_state.get_context_parallel_world_size() > 1:
            packed_seq_params = context.batch["packed_seq_params"]
            row_indices = thd_partition_indices(
                cu_seqlens_q_padded=packed_seq_params["self_attention"].cu_seqlens_q_padded,
                total_s=self_attention_mask.size(-1),
                cp_group=parallel_state.get_context_parallel_group(),
                device=self_attention_mask.device,
            )
            self_attention_mask = split_self_attention_mask_rows(self_attention_mask, row_indices)
        inputs["self_attention_mask"] = self_attention_mask
        return inputs

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        model_pred = model(
            x=inputs["noisy_latents"],
            grid_sizes=inputs["grid_sizes"],
            t=inputs["timesteps"],
            context=inputs["context_embeddings"],
            packed_seq_params=inputs["packed_seq_params"],
            self_attention_mask=inputs.get("self_attention_mask"),
        )
        return self.post_process_prediction(model_pred)


class LongLiveWanFlowMatchingPipeline(WanFlowMatchingPipeline):
    """WAN flow-matching pipeline with clean history and a noisy target temporal chunk."""

    def __init__(
        self,
        *args,
        target_chunk_frames: int = 1,
        chunk_selection_strategy: ChunkSelectionStrategy = "random",
        fallback_to_standard_wan: bool = True,
        teacher_forcing_mask_max_tokens: int | None = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_chunk_frames = target_chunk_frames
        self.chunk_selection_strategy = chunk_selection_strategy
        self.fallback_to_standard_wan = fallback_to_standard_wan
        self.teacher_forcing_mask_max_tokens = teacher_forcing_mask_max_tokens

    def should_build_explicit_self_attention_mask(self, seq_len: int, model: nn.Module) -> bool:
        """Use dense teacher-forcing masks only when explicitly requested and supported."""
        model_config = get_model_config(model)
        has_window = getattr(model_config, "window_size", None) is not None
        max_tokens = self.teacher_forcing_mask_max_tokens
        dense_mask_requested = max_tokens is None or (max_tokens > 0 and seq_len <= max_tokens)
        if dense_mask_requested:
            if _model_accepts_explicit_self_attention_mask(model):
                return True
            if has_window:
                return False
            raise ValueError(
                "LongLiveWan explicit dense teacher-forcing masks require the model decoder to accept "
                "self_attention_mask. Configure model.window_size to use windowed attention instead."
            )
        if not has_window:
            raise ValueError(
                "LongLiveWan training requires model.window_size when explicit dense teacher-forcing masks "
                f"are disabled for seq_len={seq_len}. Increase teacher_forcing_mask_max_tokens only with "
                "Megatron-Core support for self_attention_mask."
            )
        return False

    def step(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if "video_latents" not in batch:
            raise KeyError("LongLiveWan requires 'video_latents' in the batch")

        latents = batch["video_latents"].to(device, dtype=dtype)
        batch_size = latents.shape[0]
        data_type = batch.get("data_type", "video")
        task_type = self.determine_task_type(data_type)

        chunks_or_none = select_longlive_chunks(
            grid_sizes=batch["grid_sizes"],
            seq_len_q=batch["seq_len_q"],
            seq_len_q_padded=batch["seq_len_q_padded"],
            target_chunk_frames=self.target_chunk_frames,
            strategy=self.chunk_selection_strategy,
        )
        if any(chunk is None for chunk in chunks_or_none):
            if not self.fallback_to_standard_wan:
                raise ValueError("At least one WAN sample is too short for LongLive clean history + target chunk")
            logger.info(
                "LongLiveWan fallback to standard WAN flow matching: at least one sample is too short "
                "for clean history plus target chunk"
            )
            weighted_loss, average_weighted_loss, loss_mask, metrics = super().step(
                model=model,
                batch=batch,
                device=device,
                dtype=dtype,
                global_step=global_step,
            )
            metrics["longlive_fallback"] = 1
            metrics["longlive_target_tokens"] = 0
            return weighted_loss, average_weighted_loss, loss_mask, metrics

        chunks = [chunk for chunk in chunks_or_none if chunk is not None]
        sigma, timesteps, sampling_method = self.sample_timesteps(batch_size, device)
        noise = torch.randn_like(latents, dtype=torch.float32)
        mixed_latents = apply_longlive_noising(latents, noise, sigma, chunks, self.noise_schedule).to(dtype)

        base_loss_mask = batch["loss_mask"]
        batch["loss_mask"] = build_longlive_loss_mask(base_loss_mask, chunks)
        global_target_tokens = int(batch["loss_mask"].float().sum().item())
        use_explicit_mask = self.should_build_explicit_self_attention_mask(latents.size(1), model)
        if use_explicit_mask:
            batch["self_attention_mask"] = build_teacher_forcing_self_attention_mask(
                total_seq_len=latents.size(1),
                chunks=chunks,
                device=latents.device,
            )
        else:
            batch.pop("self_attention_mask", None)

        context = FlowMatchingContext(
            noisy_latents=mixed_latents,
            latents=latents,
            timesteps=timesteps,
            sigma=sigma,
            task_type=task_type,
            data_type=data_type,
            device=device,
            dtype=dtype,
            cfg_dropout_prob=self.cfg_dropout_prob,
            batch=batch,
        )
        inputs = self.model_adapter.prepare_inputs(context)
        model_pred = self.model_adapter.forward(model, inputs)
        target = noise - latents.float()

        weighted_loss, average_weighted_loss, _unweighted_loss, average_unweighted_loss, loss_weight, loss_mask = (
            self.compute_loss(model_pred, target, sigma, batch)
        )

        if torch.isnan(average_weighted_loss) or average_weighted_loss > 100:
            logger.error("[ERROR] LongLiveWan loss explosion! Loss=%.3f", average_weighted_loss.item())
            raise ValueError(f"Loss exploded: {average_weighted_loss.item()}")

        metrics = {
            "loss": average_weighted_loss.item(),
            "unweighted_loss": average_unweighted_loss.item(),
            "sigma_min": sigma.min().item(),
            "sigma_max": sigma.max().item(),
            "sigma_mean": sigma.mean().item(),
            "weight_min": loss_weight.min().item(),
            "weight_max": loss_weight.max().item(),
            "timestep_min": timesteps.min().item(),
            "timestep_max": timesteps.max().item(),
            "sampling_method": sampling_method,
            "task_type": task_type,
            "data_type": data_type,
            "longlive_fallback": 0,
            "longlive_target_tokens": global_target_tokens,
            "longlive_target_chunk_frames": self.target_chunk_frames,
            "longlive_explicit_self_attention_mask": int(use_explicit_mask),
        }
        return weighted_loss, average_weighted_loss, loss_mask, metrics


class LongLiveWanForwardStep(WanForwardStep):
    """Forward training step for LongLiveWan MVP."""

    def __init__(
        self,
        mode: str = "pretrain",
        use_sigma_noise: bool = True,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 3.0,
        mix_uniform_ratio: float = 0.1,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        target_chunk_frames: int = 1,
        chunk_selection_strategy: ChunkSelectionStrategy = "random",
        teacher_forcing_mask_max_tokens: int | None = 0,
    ):
        if mode is not None:
            if mode not in _WAN_MODE_DEFAULTS:
                raise ValueError(f"Unknown WAN mode '{mode}'. Choose from: {list(_WAN_MODE_DEFAULTS)}")
            defaults = _WAN_MODE_DEFAULTS[mode]
            timestep_sampling = defaults.get("timestep_sampling", timestep_sampling)
            logit_std = defaults.get("logit_std", logit_std)
            flow_shift = defaults.get("flow_shift", flow_shift)
            mix_uniform_ratio = defaults.get("mix_uniform_ratio", mix_uniform_ratio)
        self.diffusion_pipeline = LongLiveWanFlowMatchingPipeline(
            model_adapter=LongLiveWanAdapter(),
            timestep_sampling=timestep_sampling,
            logit_mean=logit_mean,
            logit_std=logit_std,
            flow_shift=flow_shift,
            mix_uniform_ratio=mix_uniform_ratio,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            target_chunk_frames=target_chunk_frames,
            chunk_selection_strategy=chunk_selection_strategy,
            teacher_forcing_mask_max_tokens=teacher_forcing_mask_max_tokens,
        )
        self.use_sigma_noise = use_sigma_noise
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.flow_shift = flow_shift
        self.mix_uniform_ratio = mix_uniform_ratio
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.target_chunk_frames = target_chunk_frames
        self.chunk_selection_strategy = chunk_selection_strategy
        self.teacher_forcing_mask_max_tokens = teacher_forcing_mask_max_tokens

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: VisionModule
    ) -> tuple[torch.Tensor, partial]:
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

        with straggler_timer:
            weighted_loss, _average_weighted_loss, loss_mask, _metrics = self.diffusion_pipeline.step(model, batch)
            output_tensor = torch.mean(weighted_loss, dim=-1)
            batch["loss_mask"] = loss_mask

        loss_mask = batch["loss_mask"] if batch.get("loss_mask") is not None else torch.ones_like(output_tensor)
        loss_function = self._create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
        return output_tensor, loss_function
