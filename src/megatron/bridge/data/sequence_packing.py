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

"""Internal helpers for collate-time in-batch sequence packing."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch


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


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _pack_padded_sequence(
    batch: MutableMapping[str, Any],
    *,
    pad_token_id: int = 0,
    ignore_index: int = -100,
    pad_to_multiple_of: int = 1,
    tokens_key: str = "tokens",
    input_ids_key: str = "input_ids",
    labels_key: str = "labels",
    loss_mask_key: str = "loss_mask",
    position_ids_key: str = "position_ids",
    attention_mask_key: str = "attention_mask",
) -> None:
    """Convert a padded microbatch to packed sequence layout in place.

    This is a transitional pad-then-pack helper for callers that already rely
    on tokenizer/processor padding. New collators should prefer constructing
    the packed layout directly instead of padding first and stripping padding
    here.

    The helper mutates ``batch`` in place. It converts text-like tensors from
    ``[B, S]`` to ``[1, sum(L_i)]`` and emits metadata consumed by
    ``megatron.bridge.training.gpt_step.get_packed_seq_params``.

    Args:
        batch: Batch dictionary containing at least tokens/input_ids and position_ids.
        pad_token_id: Token value to write for padding inserted by ``pad_to_multiple_of``.
        ignore_index: Label value to write for inserted padding.
        pad_to_multiple_of: Optional per-sample packed length multiple.
        tokens_key: Preferred token key.
        input_ids_key: Optional alias key for tokens.
        labels_key: Key containing labels to pack when present.
        loss_mask_key: Key containing loss mask to pack when present.
        position_ids_key: Key containing position ids to pack.
        attention_mask_key: Key containing 1/0 padding mask. Set to ``None`` after packing.
    """
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be >= 1.")

    tokens = batch.get(tokens_key)
    if tokens is None:
        tokens = batch.get(input_ids_key)
    if tokens is None:
        raise ValueError(f"Batch must contain '{tokens_key}' or '{input_ids_key}' for sequence packing.")
    if not isinstance(tokens, torch.Tensor) or tokens.dim() != 2:
        raise ValueError("Sequence packing expects a 2D token tensor.")

    position_ids = batch.get(position_ids_key)
    if not isinstance(position_ids, torch.Tensor) or position_ids.dim() != 2:
        raise ValueError(f"Sequence packing expects a 2D '{position_ids_key}' tensor.")

    padding_mask = batch.get(attention_mask_key)
    if padding_mask is not None:
        if not isinstance(padding_mask, torch.Tensor) or padding_mask.shape != tokens.shape:
            raise ValueError(f"'{attention_mask_key}' must match token shape for sequence packing.")
        padding_mask = padding_mask.to(device=tokens.device)

    lengths = _sequence_lengths(tokens, pad_token_id=pad_token_id, padding_mask=padding_mask)
    valid_indices = [idx for idx, length in enumerate(lengths) if length > 0]
    if not valid_indices:
        raise ValueError("Cannot pack a batch with no non-padding tokens.")

    unpadded_lengths = [lengths[idx] for idx in valid_indices]
    padded_lengths = [_ceil_to_multiple(length, pad_to_multiple_of) for length in unpadded_lengths]

    cu_seqlens = [0]
    cu_seqlens_unpadded = [0]
    for padded_len, unpadded_len in zip(padded_lengths, unpadded_lengths):
        cu_seqlens.append(cu_seqlens[-1] + padded_len)
        cu_seqlens_unpadded.append(cu_seqlens_unpadded[-1] + unpadded_len)

    total_len = cu_seqlens[-1]
    device = tokens.device
    packed_tokens = torch.full((1, total_len), pad_token_id, dtype=tokens.dtype, device=device)
    packed_position_ids = torch.zeros((1, total_len), dtype=position_ids.dtype, device=position_ids.device)

    labels = batch.get(labels_key)
    packed_labels = None
    if labels is not None:
        if not isinstance(labels, torch.Tensor) or labels.shape != tokens.shape:
            raise ValueError(f"'{labels_key}' must match token shape for sequence packing.")
        packed_labels = torch.full((1, total_len), ignore_index, dtype=labels.dtype, device=labels.device)

    loss_mask = batch.get(loss_mask_key)
    packed_loss_mask = None
    if loss_mask is not None:
        if not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != tokens.shape:
            raise ValueError(f"'{loss_mask_key}' must match token shape for sequence packing.")
        packed_loss_mask = torch.zeros((1, total_len), dtype=loss_mask.dtype, device=loss_mask.device)

    offset = 0
    for batch_idx, length, padded_len in zip(valid_indices, unpadded_lengths, padded_lengths):
        packed_tokens[0, offset : offset + length] = tokens[batch_idx, :length]
        packed_position_ids[0, offset : offset + length] = position_ids[batch_idx, :length]
        if packed_labels is not None:
            packed_labels[0, offset : offset + length] = labels[batch_idx, :length]
        if packed_loss_mask is not None:
            packed_loss_mask[0, offset : offset + length] = loss_mask[batch_idx, :length]

        pad_len = padded_len - length
        if pad_len > 0:
            start_pos = position_ids[batch_idx, length - 1] + 1
            packed_position_ids[0, offset + length : offset + padded_len] = torch.arange(
                start_pos,
                start_pos + pad_len,
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
        offset += padded_len

    batch[tokens_key] = packed_tokens
    if input_ids_key in batch:
        batch[input_ids_key] = packed_tokens
    if packed_labels is not None:
        batch[labels_key] = packed_labels
    if packed_loss_mask is not None:
        batch[loss_mask_key] = packed_loss_mask
    batch[position_ids_key] = packed_position_ids
    batch[attention_mask_key] = None
    batch["cu_seqlens"] = torch.tensor([cu_seqlens], dtype=torch.int32, device=device)
    batch["cu_seqlens_argmin"] = torch.tensor([[len(cu_seqlens)]], dtype=torch.int32)
    batch["max_seqlen"] = torch.tensor([[max(padded_lengths)]], dtype=torch.int32)
    if cu_seqlens_unpadded != cu_seqlens:
        batch["cu_seqlens_unpadded"] = torch.tensor([cu_seqlens_unpadded], dtype=torch.int32, device=device)
        batch["cu_seqlens_unpadded_argmin"] = torch.tensor([[len(cu_seqlens_unpadded)]], dtype=torch.int32)


def _pack_padded_sequence_as_legacy_tuple(
    tokens: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int = 0,
    pad_to_multiple_of: int = 1,
    padding_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert padded sequence tensors and return the legacy tuple form.

    Internal compatibility wrapper for older step-time callers. It preserves
    the legacy tuple return contract while reusing the transitional
    pad-then-pack helper above.
    """
    batch: dict[str, Any] = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": padding_mask,
        "position_ids": position_ids,
    }
    _pack_padded_sequence(
        batch,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    return (
        batch["input_ids"],
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch["position_ids"],
        batch["cu_seqlens"].squeeze(),
        batch["max_seqlen"].squeeze().to(device=tokens.device),
    )
