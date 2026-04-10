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

"""
Audio-language model training step, independent of vlm_step.py.
"""

import logging
import math
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_model_config

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
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.vlm_step import pack_batch_sequences


logger = logging.getLogger(__name__)


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, Any]:
    """Get a batch of data from the iterator for audio-language models.

    Uses the ``audio_inputs`` batch key instead of ``visual_inputs``.
    """
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    # Audio path: expect 'audio_inputs' object in batch
    required_device_keys.add("audio_inputs")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            if key == "audio_inputs":
                if val is None:
                    _batch_required_keys[key] = None
                else:
                    _batch_required_keys[key] = val
                    # Move all audio inputs contained tensors to CUDA
                    for k, v in val.__dict__.items():
                        _batch_required_keys[key].__dict__[k] = v.cuda(non_blocking=True) if v is not None else None
            else:
                _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    # Preserve collator's 2D padding mask for sequence packing length detection.
    raw_attn = batch.get("attention_mask")
    if isinstance(raw_attn, torch.Tensor) and raw_attn.dim() == 2:
        _batch_required_keys["_padding_mask"] = raw_attn.cuda(non_blocking=True)

    return _batch_required_keys


def get_batch(data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection) -> tuple[...]:
    """Generate a batch for audio-language models.

    Adapted from vlm_step.get_batch but uses ``audio_inputs`` key.
    """
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)

    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )
    enable_packing = getattr(cfg.dataset, "pack_sequences_in_batch", False)

    if not enable_packing:
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
                batch["loss_mask"] = pad_or_truncate_2d_to_len(
                    batch.get("loss_mask"), target_len, seq_cap, pad_value=0
                )  # type: ignore[assignment]

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

    audio_inputs = batch.get("audio_inputs")
    cp_size = pg_collection.cp.size() if pg_collection is not None and pg_collection.cp is not None else 1
    tp_size = pg_collection.tp.size() if pg_collection is not None and pg_collection.tp is not None else 1
    has_sp = getattr(cfg.model, "sequence_parallel", False)

    if enable_packing:
        # Pack sequences
        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")

        cp_multiple = 2 * cp_size if cp_size > 1 else 1
        sp_multiple = cp_size * tp_size if has_sp and tp_size > 1 else 1
        pad_multiple = math.lcm(cp_multiple, sp_multiple)

        (
            packed_tokens,
            packed_labels,
            packed_loss_mask,
            packed_attention_mask,
            packed_position_ids,
            cu_seqlens,
            max_seqlen,
        ) = pack_batch_sequences(
            tokens=tokens_or_input,
            labels=batch.get("labels"),
            loss_mask=batch.get("loss_mask"),
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            pad_token_id=0,
            pad_to_multiple_of=pad_multiple,
            padding_mask=batch.get("_padding_mask"),
        )

        # Update batch dict with packed tensors
        if batch.get("tokens") is not None:
            batch["tokens"] = packed_tokens
        else:
            batch["input_ids"] = packed_tokens
        batch["labels"] = packed_labels
        batch["loss_mask"] = packed_loss_mask
        batch["attention_mask"] = packed_attention_mask
        batch["position_ids"] = packed_position_ids

        logger.debug(f"Packed batch: cu_seqlens={cu_seqlens.tolist()}, max_seqlen={max_seqlen}")
    else:
        cu_seqlens = None
        max_seqlen = None

    return (
        (batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        cu_seqlens,
        max_seqlen,
        audio_inputs,
    )


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step for audio-language models.

    Uses a local get_batch that extracts audio_inputs instead of visual_inputs.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The audio-language model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    pg_collection = get_pg_collection(model)
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            max_seqlen,
            audio_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }

    if audio_inputs is not None:
        forward_args.update(audio_inputs.normalized_for_model())

    # Add packed sequence support
    if cu_seqlens is not None:
        cu_seqlens_argmin = torch.tensor(len(cu_seqlens))  # no padding in cu_seqlens since packing is done in-batch
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "cu_seqlens_argmin": cu_seqlens_argmin,
        }
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        model_output = model(**forward_args)
        if isinstance(model_output, tuple):
            output_tensor, loss_mask = model_output
        else:
            output_tensor = model_output

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
