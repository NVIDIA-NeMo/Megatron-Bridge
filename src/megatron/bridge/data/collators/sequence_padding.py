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

"""Ordinary collate-time sequence padding and truncation helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from megatron.bridge.data.datasets.utils import IGNORE_INDEX


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


@contextmanager
def use_processor_right_padding(processor: Any) -> Iterator[None]:
    """Temporarily force a processor tokenizer to use right padding."""
    tokenizer = getattr(processor, "tokenizer", processor)
    should_restore = tokenizer is not None and hasattr(tokenizer, "padding_side")
    previous_padding_side = getattr(tokenizer, "padding_side", None)
    if should_restore:
        tokenizer.padding_side = "right"
    try:
        yield
    finally:
        if should_restore:
            tokenizer.padding_side = previous_padding_side


def _token_key(batch: MutableMapping[str, Any]) -> str:
    if isinstance(batch.get("tokens"), torch.Tensor):
        return "tokens"
    if isinstance(batch.get("input_ids"), torch.Tensor):
        return "input_ids"
    raise ValueError("Sequence batch must contain a 2D 'input_ids' or 'tokens' tensor.")


def _set_tokens(batch: MutableMapping[str, Any], token_key: str, value: torch.Tensor) -> None:
    batch[token_key] = value
    if token_key == "input_ids" and "tokens" in batch:
        batch["tokens"] = value
    elif token_key == "tokens" and "input_ids" in batch:
        batch["input_ids"] = value


def _pad_or_truncate_2d(
    value: torch.Tensor | None,
    target_length: int,
    pad_value: int | float,
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(value.shape)}.")
    current_length = value.size(1)
    if current_length < target_length:
        return F.pad(value, (0, target_length - current_length), value=pad_value)
    if current_length > target_length:
        return value[:, :target_length].contiguous()
    return value.contiguous()


def _pad_or_truncate_position_ids(
    position_ids: torch.Tensor | None,
    target_length: int,
) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if position_ids.dim() != 2:
        raise ValueError(f"Expected 2D position_ids, got shape {tuple(position_ids.shape)}.")
    current_length = position_ids.size(1)
    if current_length < target_length:
        addition = (
            torch.arange(current_length, target_length, device=position_ids.device, dtype=position_ids.dtype)
            .unsqueeze(0)
            .expand(position_ids.size(0), -1)
        )
        return torch.cat([position_ids, addition], dim=1).contiguous()
    if current_length > target_length:
        return position_ids[:, :target_length].contiguous()
    return position_ids.contiguous()


def _pad_or_truncate_attention_mask(
    attention_mask: torch.Tensor | None,
    target_length: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    pad_value = False if attention_mask.dtype == torch.bool else 0
    if attention_mask.dim() == 2:
        current_length = attention_mask.size(1)
        if current_length < target_length:
            return F.pad(attention_mask, (0, target_length - current_length), value=pad_value)
        if current_length > target_length:
            return attention_mask[:, :target_length].contiguous()
        return attention_mask.contiguous()
    if attention_mask.dim() == 4:
        attention_mask = attention_mask[:, :, :target_length, :target_length]
        _, _, query_length, key_length = attention_mask.shape
        if query_length < target_length or key_length < target_length:
            return F.pad(
                attention_mask,
                (0, target_length - key_length, 0, target_length - query_length),
                value=pad_value,
            )
        return attention_mask.contiguous()
    raise ValueError(f"attention_mask must be 2D or 4D, got shape {tuple(attention_mask.shape)}.")


def pad_or_truncate_sequence_batch(
    batch: MutableMapping[str, Any],
    *,
    sequence_length: int | None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
    sequence_tensor_pad_values: Mapping[str, int | float] | None = None,
) -> None:
    """Pad or truncate sequence-aligned tensors without applying packing."""
    token_key = _token_key(batch)
    tokens = batch[token_key]
    if not isinstance(tokens, torch.Tensor) or tokens.dim() != 2:
        raise ValueError("Sequence batch preparation expects a 2D token tensor.")
    if sequence_length is None:
        return
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    target_length = (
        sequence_length
        if pad_to_max_length
        else min(sequence_length, _ceil_to_multiple(tokens.size(1), pad_to_multiple_of))
    )
    _set_tokens(batch, token_key, _pad_or_truncate_2d(tokens, target_length, pad_token_id))
    batch["labels"] = _pad_or_truncate_2d(batch.get("labels"), target_length, ignore_index)
    batch["loss_mask"] = _pad_or_truncate_2d(batch.get("loss_mask"), target_length, 0)
    batch["position_ids"] = _pad_or_truncate_position_ids(batch.get("position_ids"), target_length)
    batch["attention_mask"] = _pad_or_truncate_attention_mask(batch.get("attention_mask"), target_length)
    for key, tensor_pad_value in (sequence_tensor_pad_values or {}).items():
        if batch.get(key) is not None:
            batch[key] = _pad_or_truncate_2d(batch[key], target_length, tensor_pad_value)
