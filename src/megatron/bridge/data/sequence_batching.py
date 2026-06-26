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

"""Collate-time sequence batch padding, truncation, and packing helpers."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch
import torch.nn.functional as F

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.sequence_packing import _pack_padded_sequence


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


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


def _pad_or_truncate_2d(x: torch.Tensor | None, target_len: int, pad_value: int | float) -> torch.Tensor | None:
    if x is None:
        return None
    if x.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(x.shape)}.")
    current_len = x.size(1)
    if current_len < target_len:
        return F.pad(x, (0, target_len - current_len), value=pad_value)
    if current_len > target_len:
        return x[:, :target_len].contiguous()
    return x.contiguous()


def _pad_or_truncate_position_ids(position_ids: torch.Tensor | None, target_len: int) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if position_ids.dim() != 2:
        raise ValueError(f"Expected 2D position_ids, got shape {tuple(position_ids.shape)}.")
    current_len = position_ids.size(1)
    if current_len < target_len:
        addition = (
            torch.arange(current_len, target_len, device=position_ids.device, dtype=position_ids.dtype)
            .unsqueeze(0)
            .expand(position_ids.size(0), -1)
        )
        return torch.cat([position_ids, addition], dim=1).contiguous()
    if current_len > target_len:
        return position_ids[:, :target_len].contiguous()
    return position_ids.contiguous()


def _pad_or_truncate_attention_mask(attention_mask: torch.Tensor | None, target_len: int) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    pad_value = False if attention_mask.dtype == torch.bool else 0
    if attention_mask.dim() == 2:
        current_len = attention_mask.size(1)
        if current_len < target_len:
            return F.pad(attention_mask, (0, target_len - current_len), value=pad_value)
        if current_len > target_len:
            return attention_mask[:, :target_len].contiguous()
        return attention_mask.contiguous()
    if attention_mask.dim() == 4:
        attention_mask = attention_mask[:, :, :target_len, :target_len]
        _, _, query_len, key_len = attention_mask.shape
        if query_len < target_len or key_len < target_len:
            return F.pad(attention_mask, (0, target_len - key_len, 0, target_len - query_len), value=pad_value)
        return attention_mask.contiguous()
    raise ValueError(f"attention_mask must be 2D or 4D, got shape {tuple(attention_mask.shape)}.")


def pad_or_pack_sequence(
    batch: MutableMapping[str, Any],
    *,
    sequence_length: int | None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Pad, truncate, or pack sequence tensors for the training step.

    This is the collate-time policy helper for sequence tensors. When packing
    is enabled it still uses an internal pad-then-pack helper, because the
    current model collates first produce padded tensors. Longer term, packing
    collates should build flattened packed tensors directly.

    Args:
        batch: Mutable collate batch with ``input_ids`` or ``tokens`` plus
            ``labels``, ``loss_mask``, ``position_ids``, and optional
            ``attention_mask``.
        sequence_length: Model sequence cap. If unset, non-packed batches are
            left at the processor's batch-max length.
        pad_to_max_length: If true, pad/truncate non-packed batches directly to
            ``sequence_length``. This preserves the former PP/EP fixed-shape path.
        pad_to_multiple_of: Efficient non-packed length multiple used when
            ``pad_to_max_length`` is false.
        enable_in_batch_packing: If true, flatten the microbatch and emit packed-sequence
            metadata instead of returning a padded attention mask.
        in_batch_packing_pad_to_multiple_of: Per-sequence packed length multiple
            for CP/SP constraints.
        pad_token_id: Token value for inserted padding.
        ignore_index: Label value for inserted padding.
    """
    token_key = _token_key(batch)
    tokens = batch[token_key]
    if not isinstance(tokens, torch.Tensor) or tokens.dim() != 2:
        raise ValueError("Sequence batch preparation expects a 2D token tensor.")

    if enable_in_batch_packing:
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None and (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.dim() != 2
            or attention_mask.shape != tokens.shape
        ):
            batch["attention_mask"] = None
        _pack_padded_sequence(
            batch,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            tokens_key=token_key,
        )
        # Legacy VLM packing always carried both padded and unpadded metadata,
        # even when no extra per-sequence padding was inserted. Keep that
        # contract so PackedSeqParams takes the same path as the former
        # training-step packer.
        if batch.get("cu_seqlens_unpadded") is None:
            batch["cu_seqlens_unpadded"] = batch["cu_seqlens"].clone()
            batch["cu_seqlens_unpadded_argmin"] = batch["cu_seqlens_argmin"].clone()
        return

    if sequence_length is None:
        return

    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    current_len = tokens.size(1)
    if pad_to_max_length:
        target_len = sequence_length
    else:
        target_len = min(sequence_length, _ceil_to_multiple(current_len, pad_to_multiple_of))

    _set_tokens(batch, token_key, _pad_or_truncate_2d(tokens, target_len, pad_token_id))
    batch["labels"] = _pad_or_truncate_2d(batch.get("labels"), target_len, ignore_index)
    batch["loss_mask"] = _pad_or_truncate_2d(batch.get("loss_mask"), target_len, 0)
    batch["position_ids"] = _pad_or_truncate_position_ids(batch.get("position_ids"), target_len)
    batch["attention_mask"] = _pad_or_truncate_attention_mask(batch.get("attention_mask"), target_len)
