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
        batch: A dictionary possibly containing `cu_seqlens`, optional
            `cu_seqlens_argmin`, and optional `max_seqlen` tensors.

    Returns:
        PackedSeqParams with identical q/kv parameters and `qkv_format` set to
        "thd".
    """

    cu_seqlens = batch["cu_seqlens"].squeeze()

    cu_seqlens_argmin = batch.get("cu_seqlens_argmin", None)
    if cu_seqlens_argmin is not None:
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )
