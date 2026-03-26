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
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.utils.padding_utils import (
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
    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))

    batch_required_keys: dict[str, Any] = {}
    for key, value in batch.items():
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

    if getattr(cfg.model, "pipeline_model_parallel_size", 1) > 1:
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
    )


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
    with straggler_timer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, multimodal_inputs = get_batch(
            data_iterator, state.cfg, use_mtp, pg_collection=pg_collection
        )
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }
    forward_args.update(multimodal_inputs)

    # The Omni thinker computes multimodal mRoPE internally from full input_ids.
    forward_args = get_batch_on_this_cp_rank(forward_args, cp_group=pg_collection.cp)
    forward_args["input_ids"] = tokens
    forward_args["position_ids"] = None

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
            return schedule_plan, loss_function
        output_tensor = model(**forward_args)

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
    return output_tensor, loss_function
