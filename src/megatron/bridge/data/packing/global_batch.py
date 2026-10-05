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

Offline, in-batch, and Energon packing form bins from a local candidate pool
(the dataset, one microbatch, or one worker's buffer). Global-batch packing forms
them once per training step inside Megatron-Core's sequence-packing scheduler,
after a data-parallel all-gather of sample lengths, so the candidate pool is the
whole global batch across DP x CP ranks.

The scheduler consumes *unpacked* per-sample dicts, one sequence per sample,
delivered as a list rather than stacked into a batch. GPT-SFT datasets produce
them when ``enable_global_batch_packing`` is set; this module holds that sample
contract and the helper that builds one sample.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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
