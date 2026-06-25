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


def _sequence_lengths(tokens: torch.Tensor, *, pad_token_id: int, padding_mask: torch.Tensor | None) -> list[int]:
    lengths = []
    batch_size, seq_len = tokens.shape
    for idx in range(batch_size):
        if padding_mask is not None:
            length = int(padding_mask[idx].sum().item())
        else:
            non_pad_mask = tokens[idx] != pad_token_id
            if non_pad_mask.all():
                length = seq_len
            elif non_pad_mask.any():
                length = int(non_pad_mask.nonzero(as_tuple=True)[0][-1].item()) + 1
            else:
                length = 0
        lengths.append(length)
    return lengths


def _validate_sequence_tensor(
    batch: MutableMapping[str, Any],
    key: str,
    *,
    tokens: torch.Tensor,
) -> torch.Tensor | None:
    tensor = batch.get(key)
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor) or tensor.shape != tokens.shape:
        raise ValueError(f"'{key}' must match token shape for direct sequence packing.")
    return tensor


def pack_sequence_batch_to_mcore_thd(
    batch: MutableMapping[str, Any],
    *,
    token_key: str | None = None,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
    pad_to_multiple_of: int = 1,
) -> None:
    """Pack a collated sequence batch directly into MCore THD layout.

    This helper emits the current ``PackedSeqParams`` field names consumed by
    Megatron-Core and Transformer Engine: ``cu_seqlens_q``, ``cu_seqlens_kv``,
    optional padded cu-seqlens, and ``max_seqlen_q`` / ``max_seqlen_kv``.

    Args:
        batch: Mutable collate batch with 2D token, position, and optional
            label/loss-mask tensors.
        token_key: Token key to use. If unset, detected from ``tokens`` or
            ``input_ids``.
        pad_token_id: Token value for padding inserted by ``pad_to_multiple_of``.
        ignore_index: Label value for inserted padding.
        pad_to_multiple_of: Optional per-sequence packed length multiple.

    Raises:
        ValueError: If required tensors are missing or the batch contains no
            non-padding tokens.
    """
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be >= 1.")

    token_key = token_key if token_key is not None else _token_key(batch)
    tokens = batch[token_key]
    if not isinstance(tokens, torch.Tensor) or tokens.dim() != 2:
        raise ValueError("Direct sequence packing expects a 2D token tensor.")

    position_ids = batch.get("position_ids")
    if not isinstance(position_ids, torch.Tensor) or position_ids.dim() != 2 or position_ids.shape != tokens.shape:
        raise ValueError("Direct sequence packing expects 2D 'position_ids' matching token shape.")

    labels = _validate_sequence_tensor(batch, "labels", tokens=tokens)
    loss_mask = _validate_sequence_tensor(batch, "loss_mask", tokens=tokens)

    attention_mask = batch.get("attention_mask")
    padding_mask = None
    if attention_mask is not None:
        if (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.dim() != 2
            or attention_mask.shape != tokens.shape
        ):
            raise ValueError("'attention_mask' must match token shape for direct sequence packing.")
        padding_mask = attention_mask.to(device=tokens.device)

    lengths = _sequence_lengths(tokens, pad_token_id=pad_token_id, padding_mask=padding_mask)
    valid_indices = [idx for idx, length in enumerate(lengths) if length > 0]
    if not valid_indices:
        raise ValueError("Cannot pack a batch with no non-padding tokens.")

    unpadded_lengths = [lengths[idx] for idx in valid_indices]
    padded_lengths = [_ceil_to_multiple(length, pad_to_multiple_of) for length in unpadded_lengths]

    cu_seqlens = [0]
    cu_seqlens_padded = [0]
    for length, padded_length in zip(unpadded_lengths, padded_lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_length)

    total_len = cu_seqlens_padded[-1]
    device = tokens.device
    packed_tokens = torch.full((1, total_len), pad_token_id, dtype=tokens.dtype, device=device)
    packed_position_ids = torch.zeros((1, total_len), dtype=position_ids.dtype, device=position_ids.device)
    packed_labels = (
        torch.full((1, total_len), ignore_index, dtype=labels.dtype, device=labels.device)
        if labels is not None
        else None
    )
    packed_loss_mask = (
        torch.zeros((1, total_len), dtype=loss_mask.dtype, device=loss_mask.device) if loss_mask is not None else None
    )

    offset = 0
    for batch_idx, length, padded_length in zip(valid_indices, unpadded_lengths, padded_lengths):
        packed_tokens[0, offset : offset + length] = tokens[batch_idx, :length]
        packed_position_ids[0, offset : offset + length] = position_ids[batch_idx, :length]
        if packed_labels is not None and labels is not None:
            packed_labels[0, offset : offset + length] = labels[batch_idx, :length]
        if packed_loss_mask is not None and loss_mask is not None:
            packed_loss_mask[0, offset : offset + length] = loss_mask[batch_idx, :length]

        pad_len = padded_length - length
        if pad_len > 0:
            start_pos = position_ids[batch_idx, length - 1] + 1
            packed_position_ids[0, offset + length : offset + padded_length] = torch.arange(
                start_pos,
                start_pos + pad_len,
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
        offset += padded_length

    _set_tokens(batch, token_key, packed_tokens)
    if packed_labels is not None:
        batch["labels"] = packed_labels
    if packed_loss_mask is not None:
        batch["loss_mask"] = packed_loss_mask
    batch["position_ids"] = packed_position_ids
    batch["attention_mask"] = None

    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    batch["cu_seqlens_q"] = cu_seqlens_t
    batch["cu_seqlens_kv"] = cu_seqlens_t
    if cu_seqlens_padded != cu_seqlens:
        cu_seqlens_padded_t = torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=device)
        batch["cu_seqlens_q_padded"] = cu_seqlens_padded_t
        batch["cu_seqlens_kv_padded"] = cu_seqlens_padded_t
    batch["max_seqlen_q"] = torch.tensor(max(padded_lengths), dtype=torch.int32)
    batch["max_seqlen_kv"] = torch.tensor(max(padded_lengths), dtype=torch.int32)


def prepare_padded_or_packed_sequence_batch(
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

    This is the collate-time policy helper for sequence tensors. Non-packed
    batches are padded/truncated to the requested shape. Packed batches are
    emitted directly in MCore THD layout with current packed-sequence metadata.

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
        pack_sequence_batch_to_mcore_thd(
            batch,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            token_key=token_key,
        )
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
