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

"""Qwen3-Omni thinker training step helpers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config, get_pg_size

from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.utils.flop_utils import (
    accumulate_flops_metadata,
    get_model_chunk_vp_stage,
    vision_patch_stats_from_grid_thw,
)
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.bridge.training.utils.padding_utils import (
    get_padded_sequence_length,
    pad_batch_sequence_tensors,
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.state import GlobalState


_MULTIMODAL_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "video_second_per_grid",
    "visual_inputs",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lengths",
)
_PACKED_SEQ_DEVICE_KEYS = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
_PACKED_SEQ_HOST_KEYS = ("max_seqlen_q", "max_seqlen_kv")
_PACKED_SEQ_PARAM_KEYS = (*_PACKED_SEQ_DEVICE_KEYS, *_PACKED_SEQ_HOST_KEYS, "total_tokens")


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, Any]:
    """Get a thinker-training batch from the iterator."""

    del use_mtp, is_first_pp_stage
    batch = next(data_iterator)

    required_device_keys = set(_MULTIMODAL_KEYS)
    required_host_keys = set()
    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))
    if "cu_seqlens_q" in batch:
        required_device_keys.update(key for key in _PACKED_SEQ_DEVICE_KEYS if key in batch)
        required_host_keys.update(key for key in _PACKED_SEQ_HOST_KEYS if key in batch)
        if batch.get("padding_mask") is not None:
            required_device_keys.add("padding_mask")

    batch_required_keys: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "attention_mask" and skip_getting_attention_mask_from_dataset:
            continue
        if key in required_device_keys:
            if key == "visual_inputs":
                if value is None:
                    batch_required_keys[key] = None
                else:
                    batch_required_keys[key] = value
                    for visual_key, visual_value in value.__dict__.items():
                        value.__dict__[visual_key] = (
                            visual_value.cuda(non_blocking=True) if visual_value is not None else None
                        )
            else:
                batch_required_keys[key] = value.cuda(non_blocking=True) if value is not None else None
        elif key in required_host_keys:
            batch_required_keys[key] = value.cpu() if value is not None else None
        elif key == "total_tokens" and "cu_seqlens_q" in batch:
            batch_required_keys[key] = int(value) if value is not None else None
        else:
            batch_required_keys[key] = value

    return batch_required_keys


def _normalize_multimodal_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize multimodal batch tensors for Qwen3-Omni model forward."""

    normalized: dict[str, torch.Tensor] = {}
    visual_inputs = batch.get("visual_inputs")
    if visual_inputs is not None:
        normalized.update(visual_inputs.normalized_for_model())
    for key in _MULTIMODAL_KEYS:
        if key == "visual_inputs":
            continue
        value = batch.get(key)
        if value is None:
            continue
        if key in ("pixel_values", "pixel_values_videos") and value.dim() == 5:
            bsz, nitems, channels, height, width = value.shape
            normalized[key] = value.view(bsz * nitems, channels, height, width)
        elif key in ("image_grid_thw", "video_grid_thw") and value.dim() == 3:
            normalized[key] = value.view(-1, value.size(-1))
        elif key == "video_second_per_grid" and value.dim() > 1:
            normalized[key] = value.reshape(-1)
        else:
            normalized[key] = value
    return normalized


def get_batch(data_iterator: Iterable, cfg: "ConfigContainer", use_mtp: bool = False, *, pg_collection) -> tuple[...]:
    """Generate a minimal thinker-training batch."""

    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)
    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )

    packed_seq_metadata = (
        {key: batch[key] for key in _PACKED_SEQ_PARAM_KEYS if batch.get(key) is not None}
        if batch.get("cu_seqlens_q") is not None
        else None
    )
    if batch.get("padding_mask") is not None and packed_seq_metadata is not None:
        packed_seq_metadata["padding_mask"] = batch["padding_mask"]

    if getattr(cfg.model, "pipeline_model_parallel_size", 1) > 1 and packed_seq_metadata is None:
        seq_len = cfg.model.seq_length
        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
        tokens_or_input = pad_or_truncate_2d_to_len(tokens_or_input, seq_len, seq_len, pad_value=0)
        if batch.get("tokens") is not None:
            batch["tokens"] = tokens_or_input
        else:
            batch["input_ids"] = tokens_or_input
        batch["labels"] = pad_or_truncate_2d_to_len(batch.get("labels"), seq_len, seq_len, pad_value=-100)
        batch["loss_mask"] = pad_or_truncate_2d_to_len(batch.get("loss_mask"), seq_len, seq_len, pad_value=0)
        batch["position_ids"] = pad_or_truncate_pos_to_len(batch.get("position_ids"), seq_len, seq_len)
        if batch.get("attention_mask") is not None:
            batch["attention_mask"] = pad_or_truncate_attn_to_len(batch.get("attention_mask"), seq_len, seq_len)

    multimodal_inputs = _normalize_multimodal_inputs(batch)
    return (
        (batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        multimodal_inputs,
        packed_seq_metadata,
    )


def pad_batch_sequences_for_context_parallel(
    tokens: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    pg_collection,
    *,
    force_to_seq_length: bool = False,
    seq_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Pad dense sequence tensors before Megatron's CP zigzag split.

    Dense CP partitions each sequence into ``2 * cp_size`` chunks.  Padding here
    keeps the step-level tensors compatible with Megatron's CP slicing while the
    full ``input_ids`` tensor remains available for model-internal mRoPE.
    """

    tp_size = get_pg_size(getattr(pg_collection, "tp", None))
    cp_size = get_pg_size(getattr(pg_collection, "cp", None))
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    target_len = get_padded_sequence_length(
        tokens.size(1),
        divisible_by,
        force_to_seq_length=force_to_seq_length,
        seq_length=seq_length,
        validate_forced_seq_length=True,
        error_context="dense context parallelism",
    )

    return pad_batch_sequence_tensors(tokens, labels, loss_mask, attention_mask, position_ids, target_len)


def _get_dense_batch_on_this_cp_rank(batch: dict[str, Any], cp_group) -> dict[str, Any]:
    """Slice dense CP tensors, including 2D attention masks from VLM datasets."""

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None and attention_mask.dim() == 2:
        batch = dict(batch)
        batch["_attention_mask_2d"] = batch.pop("attention_mask")
        batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=cp_group)
        batch["attention_mask"] = batch.pop("_attention_mask_2d")
        return batch

    return get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=cp_group)


def forward_step(
    state: "GlobalState",
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """Forward training step for Qwen3-Omni thinker."""

    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    pg_collection = get_pg_collection(model)
    pp_size = get_pg_size(getattr(pg_collection, "pp", None))
    cp_size = get_pg_size(getattr(pg_collection, "cp", None))
    ep_size = get_pg_size(getattr(pg_collection, "ep", None))
    force_to_seq_length = pp_size > 1 or ep_size > 1
    with straggler_timer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, multimodal_inputs, packed_seq_metadata = get_batch(
            data_iterator, state.cfg, use_mtp, pg_collection=pg_collection
        )
    timers("batch-generator").stop()

    enable_in_batch_packing = getattr(state.cfg.dataset, "enable_in_batch_packing", False)
    if enable_in_batch_packing and getattr(state.cfg.dataset, "defer_in_batch_packing_to_step", False):
        raise ValueError(
            "qwen3_omni_step requires collate-time in-batch packing; set defer_in_batch_packing_to_step=False"
        )
    if enable_in_batch_packing and packed_seq_metadata is None:
        raise ValueError(
            "Qwen3-Omni in-batch packing expects collator-owned cu_seqlens_q metadata. "
            "Ensure the collator receives enable_in_batch_packing=True."
        )
    packed_seq_params = get_packed_seq_params(packed_seq_metadata) if packed_seq_metadata is not None else None

    if packed_seq_params is None and cp_size > 1:
        tokens, labels, loss_mask, attention_mask, position_ids = pad_batch_sequences_for_context_parallel(
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            pg_collection,
            force_to_seq_length=force_to_seq_length,
            seq_length=getattr(config, "seq_length", getattr(state.cfg.model, "seq_length", None)),
        )

    # Accumulate FLOPS metadata across micro-batches. Packed batches provide
    # cu_seqlens so the helper can use the THD attention term; dense batches
    # fall back to BSHD math. Vision-patch tracking still applies.
    # Qwen vision attention isolates temporal frames. Preserve additive patch
    # statistics for every media/frame boundary instead of collapsing them into
    # one quadratic attention sequence.
    model_cfg = getattr(state.cfg, "model", config)
    vision_config = getattr(model_cfg, "vision_config", None)
    if vision_config is None:
        vision_config = getattr(getattr(model_cfg, "thinker_config", None), "vision_config", None)
    # Every DP rank using this VLM step must enter the same reduction, including
    # a rank whose current sample happens to contain no media.
    vision_patch_stats = (0, 0, 0)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
    if isinstance(multimodal_inputs, dict):
        for grid in (multimodal_inputs.get("image_grid_thw"), multimodal_inputs.get("video_grid_thw")):
            if grid is not None and grid.numel() > 0:
                grid_stats = vision_patch_stats_from_grid_thw(grid, spatial_merge_size=spatial_merge_size)
                vision_patch_stats = tuple(
                    total + delta for total, delta in zip(vision_patch_stats, grid_stats, strict=True)
                )
    cu_seqlens = None
    cu_seqlens_unpadded = None
    if packed_seq_params is not None and cp_size == 1:
        cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        if cu_seqlens is None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        else:
            cu_seqlens_unpadded = packed_seq_params.cu_seqlens_q
    accumulate_flops_metadata(
        state,
        tokens,
        vp_stage=get_model_chunk_vp_stage(model),
        cu_seqlens=cu_seqlens,
        cu_seqlens_unpadded=cu_seqlens_unpadded,
        vision_patch_stats=vision_patch_stats,
    )

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }

    if packed_seq_params is not None:
        forward_args["packed_seq_params"] = packed_seq_params
        if packed_seq_metadata is not None and packed_seq_metadata.get("padding_mask") is not None:
            forward_args["padding_mask"] = packed_seq_metadata["padding_mask"]
    elif cp_size > 1:
        original_tokens = tokens.clone()
        forward_args = _get_dense_batch_on_this_cp_rank(forward_args, cp_group=pg_collection.cp)
        forward_args["input_ids"] = original_tokens
        forward_args["packed_seq_params"] = None
    else:
        forward_args["packed_seq_params"] = None

    forward_args.update(multimodal_inputs)

    # The Omni thinker computes multimodal mRoPE internally from full input_ids.
    forward_args["position_ids"] = None
    loss_mask = forward_args["loss_mask"]

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                forward_args["input_ids"],
                forward_args["position_ids"],
                forward_args["attention_mask"],
                labels=forward_args["labels"],
                loss_mask=loss_mask,
                packed_seq_params=forward_args.get("packed_seq_params"),
                padding_mask=forward_args.get("padding_mask"),
            )
            loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
            return schedule_plan, loss_function
        model_output = model(**forward_args)
        if isinstance(model_output, tuple):
            output_tensor, model_loss_mask = model_output
            if model_loss_mask is not None:
                loss_mask = model_loss_mask
        else:
            output_tensor = model_output

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
    return output_tensor, loss_function
