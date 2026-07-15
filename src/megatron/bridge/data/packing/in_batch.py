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

"""Collate-time construction of Megatron-Core THD packed sequence batches."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch

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


def _right_padded_sequence_lengths(
    tokens: torch.Tensor, *, pad_token_id: int, padding_mask: torch.Tensor | None
) -> list[int]:
    if padding_mask is not None:
        active_mask = padding_mask.to(device=tokens.device, dtype=torch.bool)
    else:
        active_mask = tokens != pad_token_id

    lengths_t = active_mask.sum(dim=1)
    positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
    expected_mask = positions < lengths_t.unsqueeze(1)
    if not torch.equal(active_mask, expected_mask):
        raise ValueError("Direct sequence packing requires right-padded input rows.")

    return [int(length.item()) for length in lengths_t]


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


def _validate_sequence_row_tensor(row: Mapping[str, Any], key: str, *, length: int) -> torch.Tensor | None:
    tensor = row.get(key)
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 1 or tensor.numel() != length:
        raise ValueError(f"'{key}' must be a 1D tensor matching its token row for direct sequence packing.")
    return tensor


def build_mcore_thd_sequence_batch_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    token_key: str = "input_ids",
    sequence_length: int | None = None,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
    pad_to_multiple_of: int = 1,
    sequence_tensor_pad_values: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    """Build an MCore THD batch directly from unpadded sequence rows.

    Args:
        rows: Per-example mappings containing 1D sequence tensors.
        token_key: Token tensor key present in each row.
        sequence_length: Optional maximum length for each unpadded row.
        pad_token_id: Token value for per-sequence alignment padding.
        ignore_index: Label value for per-sequence alignment padding.
        pad_to_multiple_of: Per-sequence alignment multiple for CP/SP.
        sequence_tensor_pad_values: Additional sequence-aligned tensor keys and
            the value used for alignment padding.

    Returns:
        A single-row THD batch with current MCore packed-sequence metadata.

    Raises:
        ValueError: If rows are empty, inconsistent, padded, or overlength.
    """
    if not rows:
        raise ValueError("Cannot pack an empty sequence row list.")
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be >= 1.")
    if sequence_length is not None and sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    extra_pad_values = dict(sequence_tensor_pad_values or {})
    reserved_keys = {token_key, "position_ids", "labels", "loss_mask", "attention_mask"}
    if reserved_keys.intersection(extra_pad_values):
        raise ValueError("Additional sequence tensor keys must not replace standard sequence tensors.")

    normalized_rows: list[dict[str, torch.Tensor]] = []
    for row in rows:
        tokens = row.get(token_key)
        if not isinstance(tokens, torch.Tensor) or tokens.dim() != 1:
            raise ValueError(f"'{token_key}' must be a 1D tensor for direct sequence packing.")
        row_length = tokens.numel()
        if row_length == 0:
            raise ValueError("Cannot pack an empty sequence row.")
        if sequence_length is not None and row_length > sequence_length:
            raise ValueError(
                f"Packed sequence row length {row_length} exceeds configured sequence_length {sequence_length}."
            )

        normalized_row = {token_key: tokens}
        for key in ("position_ids", "labels", "loss_mask", *extra_pad_values):
            tensor = _validate_sequence_row_tensor(row, key, length=row_length)
            if key == "position_ids" and tensor is None:
                raise ValueError("Direct sequence packing requires a position_ids row.")
            if tensor is not None:
                normalized_row[key] = tensor

        attention_mask = _validate_sequence_row_tensor(row, "attention_mask", length=row_length)
        if attention_mask is not None and not attention_mask.to(dtype=torch.bool).all():
            raise ValueError("Direct sequence packing requires unpadded input rows.")
        normalized_rows.append(normalized_row)

    for key in ("labels", "loss_mask", *extra_pad_values):
        present = [key in row for row in normalized_rows]
        if any(present) and not all(present):
            raise ValueError(f"'{key}' must be present for every sequence row or omitted from all rows.")

    unpadded_lengths = [row[token_key].numel() for row in normalized_rows]
    padded_lengths = [_ceil_to_multiple(length, pad_to_multiple_of) for length in unpadded_lengths]
    cu_seqlens = [0]
    cu_seqlens_padded = [0]
    for length, padded_length in zip(unpadded_lengths, padded_lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_length)

    first_row = normalized_rows[0]
    first_tokens = first_row[token_key]
    total_length = cu_seqlens_padded[-1]
    packed: dict[str, Any] = {
        token_key: torch.full((1, total_length), pad_token_id, dtype=first_tokens.dtype, device=first_tokens.device),
        "position_ids": torch.zeros(
            (1, total_length),
            dtype=first_row["position_ids"].dtype,
            device=first_row["position_ids"].device,
        ),
        "attention_mask": None,
    }

    output_pad_values: dict[str, int | float] = {"labels": ignore_index, "loss_mask": 0, **extra_pad_values}
    for key, pad_value in output_pad_values.items():
        if key in first_row:
            tensor = first_row[key]
            packed[key] = torch.full((1, total_length), pad_value, dtype=tensor.dtype, device=tensor.device)

    offset = 0
    for row, length, padded_length in zip(normalized_rows, unpadded_lengths, padded_lengths):
        packed[token_key][0, offset : offset + length] = row[token_key]
        packed["position_ids"][0, offset : offset + length] = row["position_ids"]
        for key in output_pad_values:
            if key in packed:
                packed[key][0, offset : offset + length] = row[key]

        pad_length = padded_length - length
        if pad_length > 0:
            start_position = row["position_ids"][-1] + 1
            packed["position_ids"][0, offset + length : offset + padded_length] = torch.arange(
                start_position,
                start_position + pad_length,
                dtype=row["position_ids"].dtype,
                device=row["position_ids"].device,
            )
        offset += padded_length

    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=first_tokens.device)
    packed["cu_seqlens_q"] = cu_seqlens_t
    packed["cu_seqlens_kv"] = cu_seqlens_t
    if pad_to_multiple_of > 1:
        cu_seqlens_padded_t = torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=first_tokens.device)
        packed["cu_seqlens_q_padded"] = cu_seqlens_padded_t
        packed["cu_seqlens_kv_padded"] = cu_seqlens_padded_t
    packed["max_seqlen_q"] = torch.tensor(max(padded_lengths), dtype=torch.int32)
    packed["max_seqlen_kv"] = torch.tensor(max(padded_lengths), dtype=torch.int32)
    # MCore uses total_tokens together with the physical (padded) boundaries
    # to construct seq_idx, which resets Mamba/SSM state between packed rows.
    packed["total_tokens"] = total_length
    return packed


def pack_right_padded_sequence_batch_to_mcore_thd(
    batch: MutableMapping[str, Any],
    *,
    token_key: str | None = None,
    sequence_length: int | None = None,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
    pad_to_multiple_of: int = 1,
    sequence_tensor_pad_values: Mapping[str, int | float] | None = None,
) -> None:
    """Pack a right-padded sequence batch into MCore THD layout.

    This helper emits the current ``PackedSeqParams`` field names consumed by
    Megatron-Core and Transformer Engine: ``cu_seqlens_q``, ``cu_seqlens_kv``,
    optional padded cu-seqlens, and ``max_seqlen_q`` / ``max_seqlen_kv``.

    Args:
        batch: Mutable collate batch with 2D token, position, and optional
            label/loss-mask tensors.
        token_key: Token key to use. If unset, detected from ``tokens`` or
            ``input_ids``.
        sequence_length: Optional maximum length for each unpadded row.
        pad_token_id: Token value for padding inserted by ``pad_to_multiple_of``.
        ignore_index: Label value for inserted padding.
        pad_to_multiple_of: Optional per-sequence packed length multiple.
        sequence_tensor_pad_values: Additional sequence-aligned tensor keys and
            their alignment padding values.

    Raises:
        ValueError: If required tensors are missing or the batch contains no
            non-padding tokens.
    """
    token_key = token_key if token_key is not None else _token_key(batch)
    tokens = batch[token_key]
    if not isinstance(tokens, torch.Tensor) or tokens.dim() != 2:
        raise ValueError("Direct sequence packing expects a 2D token tensor.")

    position_ids = batch.get("position_ids")
    if not isinstance(position_ids, torch.Tensor) or position_ids.dim() != 2 or position_ids.shape != tokens.shape:
        raise ValueError("Direct sequence packing expects 2D 'position_ids' matching token shape.")

    labels = _validate_sequence_tensor(batch, "labels", tokens=tokens)
    loss_mask = _validate_sequence_tensor(batch, "loss_mask", tokens=tokens)
    extra_tensors = {
        key: _validate_sequence_tensor(batch, key, tokens=tokens) for key in (sequence_tensor_pad_values or {})
    }

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

    lengths = _right_padded_sequence_lengths(tokens, pad_token_id=pad_token_id, padding_mask=padding_mask)
    if any(length == 0 for length in lengths):
        raise ValueError("Cannot pack a batch containing an empty sequence row.")

    rows = []
    for batch_idx in range(len(lengths)):
        length = lengths[batch_idx]
        row = {
            token_key: tokens[batch_idx, :length],
            "position_ids": position_ids[batch_idx, :length],
        }
        if labels is not None:
            row["labels"] = labels[batch_idx, :length]
        if loss_mask is not None:
            row["loss_mask"] = loss_mask[batch_idx, :length]
        for key, tensor in extra_tensors.items():
            if tensor is not None:
                row[key] = tensor[batch_idx, :length]
        rows.append(row)

    packed = build_mcore_thd_sequence_batch_from_rows(
        rows,
        token_key=token_key,
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        pad_to_multiple_of=pad_to_multiple_of,
        sequence_tensor_pad_values=sequence_tensor_pad_values,
    )
    _set_tokens(batch, token_key, packed.pop(token_key))
    for key in (
        "labels",
        "loss_mask",
        "position_ids",
        "attention_mask",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "max_seqlen_q",
        "max_seqlen_kv",
        "total_tokens",
        *(sequence_tensor_pad_values or {}),
    ):
        if key in packed:
            batch[key] = packed[key]
        elif key in {"cu_seqlens_q_padded", "cu_seqlens_kv_padded"}:
            batch.pop(key, None)
