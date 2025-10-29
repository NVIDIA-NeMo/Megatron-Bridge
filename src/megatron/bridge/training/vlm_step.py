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
from typing import Any, Iterable

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.bridge.training.utils.padding_utils import (
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)


logger = logging.getLogger(__name__)


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
) -> dict[str, Any]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.
        use_mtp: Whether Multi-Token Prediction layers are enabled.
        skip_getting_attention_mask_from_dataset: If set, the dataset will pass a None attention mask.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    # Instead of raw tensors, expect a single 'visual_inputs' object in batch
    required_device_keys.add("visual_inputs")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            if key == "visual_inputs":
                if val is None:
                    _batch_required_keys[key] = None
                else:
                    _batch_required_keys[key] = val
                    # Move all visual inputs contained tensors to CUDA
                    for k, v in val.__dict__.items():
                        _batch_required_keys[key].__dict__[k] = v.cuda(non_blocking=True) if v is not None else None
            else:
                _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Any,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, max_seqlen, visual_inputs (container of optional modalities)
    """
    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
    )

    # Slice only text tensors for context parallelism
    cp_keys = ("tokens", "input_ids", "labels", "loss_mask", "attention_mask", "position_ids")
    cp_slice = {k: batch.get(k) for k in cp_keys if k in batch}
    cp_slice = get_batch_on_this_cp_rank(cp_slice)
    for k, v in cp_slice.items():
        batch[k] = v

    # When using pipeline parallelism, ensure fixed shapes equal to cfg.model.seq_length
    if getattr(cfg.model, "pipeline_model_parallel_size", 1) > 1:
        seq_len = cfg.model.seq_length

        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
        tokens_or_input = pad_or_truncate_2d_to_len(tokens_or_input, seq_len, seq_len, pad_value=0)
        if batch.get("tokens") is not None:
            batch["tokens"] = tokens_or_input  # type: ignore[assignment]
        else:
            batch["input_ids"] = tokens_or_input  # type: ignore[assignment]
        batch["labels"] = pad_or_truncate_2d_to_len(batch.get("labels"), seq_len, seq_len, pad_value=-100)  # type: ignore[assignment]
        batch["loss_mask"] = pad_or_truncate_2d_to_len(batch.get("loss_mask"), seq_len, seq_len, pad_value=0)  # type: ignore[assignment]
        batch["position_ids"] = pad_or_truncate_pos_to_len(batch.get("position_ids"), seq_len, seq_len)  # type: ignore[assignment]
        if batch.get("attention_mask") is not None:
            batch["attention_mask"] = pad_or_truncate_attn_to_len(batch.get("attention_mask"), seq_len, seq_len)  # type: ignore[assignment]
    else:
        # No PP: pad sequence length to nearest multiple of 128 for efficiency (capped at model seq_length)
        seq_cap = cfg.model.seq_length

        def _ceil_to_mult(n: int, mult: int) -> int:
            return ((n + mult - 1) // mult) * mult

        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
        if tokens_or_input is not None:
            cur_len = tokens_or_input.size(1)
            target_len = min(seq_cap, _ceil_to_mult(cur_len, 128))

            # tokens/input_ids
            padded_tokens = pad_or_truncate_2d_to_len(tokens_or_input, target_len, seq_cap, pad_value=0)
            if batch.get("tokens") is not None:
                batch["tokens"] = padded_tokens  # type: ignore[assignment]
            else:
                batch["input_ids"] = padded_tokens  # type: ignore[assignment]

            # labels and loss mask
            batch["labels"] = pad_or_truncate_2d_to_len(batch.get("labels"), target_len, seq_cap, pad_value=-100)  # type: ignore[assignment]
            batch["loss_mask"] = pad_or_truncate_2d_to_len(batch.get("loss_mask"), target_len, seq_cap, pad_value=0)  # type: ignore[assignment]

            # position_ids: extend with increasing positions
            pos = batch.get("position_ids")
            pos = pad_or_truncate_pos_to_len(pos, target_len, seq_cap)
            if pos is not None:
                batch["position_ids"] = pos  # type: ignore[assignment]

            # attention_mask if present
            attn = batch.get("attention_mask")
        if attn is not None:
            attn = pad_or_truncate_attn_to_len(attn, target_len, seq_cap)
            batch["attention_mask"] = attn  # type: ignore[assignment]

    visual_inputs = batch.get("visual_inputs")

    return (
        (batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")),
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
        visual_inputs,
    )


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
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
    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            cu_seqlens_argmin,
            max_seqlen,
            visual_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp)
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if visual_inputs is not None:
        forward_args.update(visual_inputs.normalized_for_model())

    # Add packed sequence support
    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
        }
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

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
        else:
            output_tensor = model(**forward_args)

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
