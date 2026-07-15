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

import logging
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config

from megatron.bridge.diffusion.common.flow_matching.adapters.base import FlowMatchingContext
from megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan import (
    WanAdapter,
    WanFlowMatchingPipeline,
)
from megatron.bridge.diffusion.models.wan.longlive_wan_utils import (
    ChunkSelectionStrategy,
    build_longlive_paired_latents_and_masks,
    build_longlive_paired_sequence_metadata,
    build_paired_teacher_forcing_self_attention_mask,
    duplicate_context_for_longlive_paired_chunks,
    select_longlive_paired_chunks,
    split_self_attention_mask_rows,
)
from megatron.bridge.diffusion.models.wan.utils import thd_partition_indices
from megatron.bridge.diffusion.models.wan.wan_step import _WAN_MODE_DEFAULTS, WanForwardStep, wan_data_step
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)


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
        inputs["grid_frame_offsets"] = context.batch.get("grid_frame_offsets")
        return inputs

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        model_pred = model(
            x=inputs["noisy_latents"],
            grid_sizes=inputs["grid_sizes"],
            t=inputs["timesteps"],
            context=inputs["context_embeddings"],
            packed_seq_params=inputs["packed_seq_params"],
            self_attention_mask=inputs.get("self_attention_mask"),
            grid_frame_offsets=inputs.get("grid_frame_offsets"),
        )
        return self.post_process_prediction(model_pred)


def _build_packed_seq_params(
    qkv_format: str,
    seq_len_q: torch.Tensor,
    seq_len_q_padded: torch.Tensor,
    seq_len_kv: torch.Tensor,
    seq_len_kv_padded: torch.Tensor,
) -> dict[str, PackedSeqParams]:
    zero = torch.zeros(1, dtype=torch.int32, device=seq_len_q.device)
    cu_seqlens_q = torch.cat((zero, seq_len_q.cumsum(dim=0).to(torch.int32)))
    cu_seqlens_q_padded = torch.cat((zero, seq_len_q_padded.cumsum(dim=0).to(torch.int32)))
    cu_seqlens_kv = torch.cat((zero, seq_len_kv.cumsum(dim=0).to(torch.int32)))
    cu_seqlens_kv_padded = torch.cat((zero, seq_len_kv_padded.cumsum(dim=0).to(torch.int32)))
    return {
        "self_attention": PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv=cu_seqlens_q,
            cu_seqlens_kv_padded=cu_seqlens_q_padded,
            qkv_format=qkv_format,
        ),
        "cross_attention": PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            qkv_format=qkv_format,
        ),
    }


class LongLiveWanFlowMatchingPipeline(WanFlowMatchingPipeline):
    """WAN flow-matching pipeline with LongLive paired clean/noisy teacher forcing."""

    def __init__(
        self,
        *args,
        target_chunk_frames: int = 1,
        chunk_selection_strategy: ChunkSelectionStrategy = "random",
        fallback_to_standard_wan: bool = True,
        teacher_forcing_mask_max_tokens: int | None = 8192,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_chunk_frames = target_chunk_frames
        self.chunk_selection_strategy = chunk_selection_strategy
        self.fallback_to_standard_wan = fallback_to_standard_wan
        self.teacher_forcing_mask_max_tokens = teacher_forcing_mask_max_tokens

    def should_build_explicit_self_attention_mask(self, seq_len: int, model: nn.Module) -> bool:
        """Use dense teacher-forcing masks when the paired sequence is small enough."""
        max_tokens = self.teacher_forcing_mask_max_tokens
        dense_mask_requested = max_tokens is None or (max_tokens > 0 and seq_len <= max_tokens)
        if dense_mask_requested:
            return True
        raise ValueError(
            "LongLiveWan paired clean/noisy teacher forcing requires an explicit AR self-attention mask. "
            f"The paired sequence has {seq_len} tokens, exceeding teacher_forcing_mask_max_tokens={max_tokens}. "
            "A TE sliding window is not equivalent because it cannot prevent noisy chunks from attending to "
            "previous noisy chunks. Lower the validation resolution/sequence length or add block-sparse mask support."
        )

    def validate_qkv_format(self, qkv_format: str) -> None:
        """Require SBHD until LongLive self-attention boundaries are sample-level."""
        if qkv_format != "sbhd":
            raise ValueError(
                "LongLiveWan paired clean/noisy teacher forcing currently requires qkv_format='sbhd'. "
                f"Got qkv_format={qkv_format!r}. THD treats each clean/noisy chunk as a hard packed-sequence boundary, "
                "which prevents noisy chunks from attending to previous clean chunks. Context parallelism "
                "requires THD and is not supported by this dense-mask LongLive path yet."
            )

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

        chunks = select_longlive_paired_chunks(
            grid_sizes=batch["grid_sizes"],
            seq_len_q=batch["seq_len_q"],
            seq_len_q_padded=batch["seq_len_q_padded"],
            target_chunk_frames=self.target_chunk_frames,
        )
        if not chunks:
            if not self.fallback_to_standard_wan:
                raise ValueError("At least one WAN sample is required for LongLive paired teacher forcing")
            logger.info("LongLiveWan fallback to standard WAN flow matching: no valid temporal chunks found")
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

        sigma, timesteps, sampling_method = self.sample_timesteps(batch_size, device)
        noise = torch.randn_like(latents, dtype=torch.float32)
        paired_latents, paired_target, paired_loss_mask = build_longlive_paired_latents_and_masks(
            clean_latents=latents,
            noise=noise,
            sigma=sigma,
            base_loss_mask=batch["loss_mask"],
            chunks=chunks,
            noise_schedule=self.noise_schedule,
        )
        paired_latents = paired_latents.to(dtype)

        grid_sizes, grid_frame_offsets, seq_len_q, seq_len_q_padded = build_longlive_paired_sequence_metadata(
            chunks=chunks,
            device=latents.device,
        )
        context_embeddings, seq_len_kv, seq_len_kv_padded = duplicate_context_for_longlive_paired_chunks(
            context_embeddings=batch["context_embeddings"],
            seq_len_kv=batch["seq_len_kv"],
            seq_len_kv_padded=batch["seq_len_kv_padded"],
            chunks=chunks,
        )
        qkv_format = batch["packed_seq_params"]["self_attention"].qkv_format
        self.validate_qkv_format(qkv_format)
        batch["video_latents"] = paired_latents
        batch["grid_sizes"] = grid_sizes
        batch["grid_frame_offsets"] = grid_frame_offsets
        batch["seq_len_q"] = seq_len_q
        batch["seq_len_q_padded"] = seq_len_q_padded
        batch["seq_len_kv"] = seq_len_kv
        batch["seq_len_kv_padded"] = seq_len_kv_padded
        batch["context_embeddings"] = context_embeddings
        batch["loss_mask"] = paired_loss_mask.to(device=latents.device)
        batch["packed_seq_params"] = _build_packed_seq_params(
            qkv_format=qkv_format,
            seq_len_q=seq_len_q,
            seq_len_q_padded=seq_len_q_padded,
            seq_len_kv=seq_len_kv,
            seq_len_kv_padded=seq_len_kv_padded,
        )
        global_target_tokens = int(batch["loss_mask"].float().sum().item())
        use_explicit_mask = self.should_build_explicit_self_attention_mask(paired_latents.size(1), model)
        if use_explicit_mask:
            batch["self_attention_mask"] = build_paired_teacher_forcing_self_attention_mask(
                total_seq_len=paired_latents.size(1),
                chunks=chunks,
                device=latents.device,
            )
        else:
            batch.pop("self_attention_mask", None)

        context = FlowMatchingContext(
            noisy_latents=paired_latents,
            latents=paired_latents,
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
        target = paired_target

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
            "longlive_paired_chunks": len(chunks),
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
        teacher_forcing_mask_max_tokens: int | None = 8192,
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
