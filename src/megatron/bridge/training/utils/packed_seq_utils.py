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

from __future__ import annotations

import torch
from megatron.core.packed_seq_params import PackedSeqParams


def _trim_cu_seqlens(cu_seqlens: torch.Tensor, cu_seqlens_argmin: torch.Tensor | None) -> torch.Tensor:
    """Trim padded cu_seqlens tail safely.

    If argmin metadata is available from dataloader, trust it. Otherwise, detect
    the first zero in the tail region after index 0 (the leading 0 is valid).
    """
    if cu_seqlens_argmin is not None:
        return cu_seqlens[: int(cu_seqlens_argmin.item())]

    padded_tail = torch.nonzero(cu_seqlens[1:] == 0, as_tuple=False)
    first_tail = int(padded_tail[0].item() + 1) if padded_tail.numel() else cu_seqlens.numel()
    return cu_seqlens[:first_tail]


def get_packed_seq_params(batch: dict[str, torch.Tensor]) -> PackedSeqParams:
    """Build packed sequence parameters from a batch dictionary.

    The function trims optional sentinel tails via argmin metadata and returns
    a `PackedSeqParams` instance suitable for packed sequence kernels.

    Args:
        batch: A dictionary containing packed-sequence metadata. Expected keys:
            `cu_seqlens`, optional `cu_seqlens_unpadded`, optional argmins, and
            optional `max_seqlen`.

    Returns:
        PackedSeqParams for THD (`qkv_format="thd"`). If unpadded boundaries
        are provided, they are used for q/kv while padded boundaries remain in
        `*_padded` fields for kernel compatibility.
    """

    cu_seqlens_padded = batch["cu_seqlens"].squeeze()
    cu_seqlens_unpadded = batch.get("cu_seqlens_unpadded")
    if cu_seqlens_unpadded is not None:
        cu_seqlens_unpadded = cu_seqlens_unpadded.squeeze()

    cu_seqlens_argmin = batch.get("cu_seqlens_argmin")
    cu_seqlens_unpadded_argmin = batch.get("cu_seqlens_unpadded_argmin")

    # Note: if argmin metadata is absent, fallback tail detection can still incur
    # device-to-host synchronization.
    cu_seqlens_padded = _trim_cu_seqlens(cu_seqlens_padded, cu_seqlens_argmin)

    if cu_seqlens_unpadded is not None:
        cu_seqlens_unpadded = _trim_cu_seqlens(cu_seqlens_unpadded, cu_seqlens_unpadded_argmin)

    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    # When cu_seqlens_unpadded is present (pad_seq_to_mult > 1), pass both unpadded and padded
    # for proper THD CP support. Otherwise, just use cu_seqlens_padded to avoid slower TE kernel.
    if cu_seqlens_unpadded is not None:
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_unpadded,
            cu_seqlens_kv=cu_seqlens_unpadded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
    else:
        # Follow Megatron-LM data_schedule.py convention: set all four
        # cu_seqlens fields to padded values so packed kernels consume a
        # single consistent boundary set.
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
