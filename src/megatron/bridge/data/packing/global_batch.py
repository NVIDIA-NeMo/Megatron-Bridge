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

"""Data-side contract for global-batch online packing.

Global-batch packing is the fourth packing mode in Bridge. Unlike offline,
in-batch, and Energon packing, the bins are formed once per training step by
Megatron-Core's sequence-packing scheduler after a data-parallel all-gather of
sample lengths, so the candidate pool spans the whole global batch across
DP x CP ranks. That is what makes dynamic context parallelism possible.

The scheduler consumes *unpacked* per-sample dicts, one sequence per sample,
delivered through an identity collate. This module holds that sample contract
and the helpers datasets use to produce it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch


REQUIRED_SAMPLE_KEYS: tuple[str, ...] = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "original_seq_len",
    "padded_seq_len",
)
"""Keys every unpacked sample must carry for the Megatron-Core scheduler.

``tokens``/``labels``/``position_ids`` are ``int64 [L]``, ``loss_mask`` is
``float32 [L]``, and the two lengths are ``int32 [1]`` tensors (``L`` is the
padded length).
"""


def identity_collate(samples: list[Any]) -> list[Any]:
    """Return the samples as a list instead of stacking them.

    The scheduler pulls ``micro_batch_size`` samples per ``next()`` and packs
    them itself; the default collate would stack the variable-length tensors
    and destroy the per-sample lengths it needs.
    """
    return list(samples)


def fold_alignment_padding(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Report the padded length as the sequence length so bins carry no padding between sequences.

    Alignment padding stays loss-masked and causally inert; only the boundaries
    Megatron-Core reports to the attention backend change. Use this for models
    whose head dimension Transformer Engine only trains with FlashAttention
    (head_dim >= 256), because FlashAttention rejects THD bins flagged as padded.
    """
    sample["original_seq_len"] = sample["padded_seq_len"].clone()
    return sample


def make_unpacked_collate(*, fold_padding: bool) -> Callable[[list[Any]], list[Any]]:
    """Return the collate for datasets that yield unpacked samples."""
    if not fold_padding:
        return identity_collate

    def _collate(samples: list[Any]) -> list[Any]:
        return [fold_alignment_padding(sample) for sample in samples]

    return _collate


def global_batch_packing_padding_multiple(
    *, dynamic_cp: bool, dp_size: int, cp_size: int, tp_size: int = 1, sequence_parallel: bool = False
) -> int:
    """Return the length multiple every unpacked sample must be padded to.

    Context-parallel THD slicing cuts each sequence into ``2 * group`` zigzag
    chunks. Under dynamic CP a sequence may run on any group of the DP x CP pool,
    so the multiple is ``2 * dp * cp`` (the pool size must be a power of two, which
    ``ConfigContainer.validate`` enforces); static CP uses ``2 * cp``. Sequence
    parallelism additionally needs TP-divisible chunks. This is the same rule
    Megatron's ``SFTDataset._calculate_padding_divisor`` applies.
    """
    if dynamic_cp:
        multiple = 2 * dp_size * cp_size
    else:
        multiple = 2 * cp_size if cp_size > 1 else 1
    if sequence_parallel and tp_size > 1:
        multiple *= tp_size
    return multiple


def build_unpacked_sequence_sample(
    tokens: Sequence[int] | torch.Tensor,
    labels: Sequence[int] | torch.Tensor,
    loss_mask: Sequence[int] | Sequence[float] | torch.Tensor,
    *,
    pad_to_multiple_of: int,
    pad_token_id: int,
    ignore_label_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build one scheduler sample from an unpadded sequence.

    Args:
        tokens: Input token ids for one sequence.
        labels: Next-token targets aligned with ``tokens``.
        loss_mask: 1 where ``labels`` are supervised, 0 elsewhere.
        pad_to_multiple_of: Alignment multiple for CP THD slicing.
        pad_token_id: Token written into alignment padding.
        ignore_label_id: Label written into alignment padding (defaults to ``pad_token_id``).

    Returns:
        A dict with :data:`REQUIRED_SAMPLE_KEYS`; padding positions have ``loss_mask == 0``.
    """
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be >= 1.")
    tokens_t = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
    labels_t = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
    loss_mask_t = torch.as_tensor(loss_mask, dtype=torch.float32).reshape(-1)
    length = tokens_t.numel()
    if length == 0:
        raise ValueError("Cannot build a scheduler sample from an empty sequence.")
    if labels_t.numel() != length or loss_mask_t.numel() != length:
        raise ValueError("tokens, labels, and loss_mask must have the same length.")

    padded = math.ceil(length / pad_to_multiple_of) * pad_to_multiple_of
    pad = padded - length
    if pad:
        label_pad = pad_token_id if ignore_label_id is None else ignore_label_id
        tokens_t = torch.cat([tokens_t, torch.full((pad,), pad_token_id, dtype=torch.int64)])
        labels_t = torch.cat([labels_t, torch.full((pad,), label_pad, dtype=torch.int64)])
        loss_mask_t = torch.cat([loss_mask_t, torch.zeros(pad, dtype=torch.float32)])
    return {
        "tokens": tokens_t,
        "labels": labels_t,
        "loss_mask": loss_mask_t,
        "position_ids": torch.arange(padded, dtype=torch.int64),
        "original_seq_len": torch.tensor([length], dtype=torch.int32),
        "padded_seq_len": torch.tensor([padded], dtype=torch.int32),
    }
