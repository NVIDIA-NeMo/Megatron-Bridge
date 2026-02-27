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


def get_packed_seq_params(batch: dict[str, torch.Tensor]) -> PackedSeqParams:
    """Build packed sequence parameters from a batch dictionary.

    The function squeezes possible batch dimensions and removes any padding
    marked by -1 values. It returns a `PackedSeqParams` instance suitable for
    packed sequence attention kernels.

    Args:
        batch: A dictionary containing packed-sequence metadata. Expected keys:
            `cu_seqlens`, optional `cu_seqlens_unpadded`, optional argmins, and
            optional `max_seqlen`.

    Returns:
        PackedSeqParams with identical q/kv parameters and `qkv_format` set to
        "thd".
    """

    cu_seqlens_padded = batch["cu_seqlens"].squeeze()
    cu_seqlens_unpadded = batch.get("cu_seqlens_unpadded")
    if cu_seqlens_unpadded is not None:
        cu_seqlens_unpadded = cu_seqlens_unpadded.squeeze()

    cu_seqlens_argmin = batch.get("cu_seqlens_argmin")
    cu_seqlens_unpadded_argmin = batch.get("cu_seqlens_unpadded_argmin")

    # note: if argmin is not pre-computed in the dataloader, torch.argmin here will incur a
    # device-to-host synchronization, which can slow down training
    if cu_seqlens_argmin is not None:
        cu_seqlens_padded = cu_seqlens_padded[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens_padded = cu_seqlens_padded[: torch.argmin(cu_seqlens_padded)]

    if cu_seqlens_unpadded is not None:
        if cu_seqlens_unpadded_argmin is not None:
            cu_seqlens_unpadded = cu_seqlens_unpadded[: cu_seqlens_unpadded_argmin.item()]
        else:
            cu_seqlens_unpadded = cu_seqlens_unpadded[: torch.argmin(cu_seqlens_unpadded)]

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
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
