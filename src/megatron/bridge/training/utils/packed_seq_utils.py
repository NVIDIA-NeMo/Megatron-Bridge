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

from typing import Optional, Tuple

import torch
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_hybrid_cp_rank,
    get_thd_batch_on_this_cp_rank,
)


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


def ensure_cu_seqlens_padded_for_cp(batch: dict[str, torch.Tensor], cp_group=None) -> dict[str, torch.Tensor]:
    """Ensure `cu_seqlens_padded` matches CP/SP padding rules when packing is used.

    Padding rule: when CP is enabled, pad each sequence length to a multiple of 2*cp_size.
    If sequence parallel is also enabled, pad to 2*cp_size*tp_size.
    """
    if "cu_seqlens" not in batch or batch.get("cu_seqlens_padded") is not None:
        return batch

    # Resolve parallel sizes
    if cp_group is not None:
        cp_size = cp_group.size()
    else:
        cp_size = parallel_state.get_context_parallel_world_size()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    has_sp = True

    # Flatten in case upstream provided an extra batch dim (e.g., [1, B+1]).
    cu_seqlens = batch["cu_seqlens"].reshape(-1).to(dtype=torch.int32)
    if cp_size <= 1:
        batch["cu_seqlens"] = cu_seqlens
        batch["cu_seqlens_padded"] = cu_seqlens
        return batch

    pad_factor = 2 * cp_size * (tp_size if has_sp else 1)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    padded_lengths = ((lengths + pad_factor - 1) // pad_factor) * pad_factor
    cu_seqlens_padded = torch.cat(
        [
            torch.zeros(1, device=cu_seqlens.device, dtype=cu_seqlens.dtype),
            padded_lengths.cumsum(0),
        ]
    )

    batch["cu_seqlens"] = cu_seqlens
    batch["cu_seqlens_padded"] = cu_seqlens_padded.to(dtype=torch.int32)
    batch["max_seqlen"] = torch.tensor(
        [int(padded_lengths.max().item())],
        device=cu_seqlens.device,
        dtype=torch.int32,
    )

    return batch


def prepare_packed_seq_params_and_slice_cp(
    batch: dict[str, torch.Tensor], cp_group=None
) -> Tuple[dict[str, torch.Tensor], Optional[PackedSeqParams]]:
    """Slice a batch for Context Parallelism and build PackedSeqParams when packing is used.

    This mirrors the Megatron-LM training path so packed THD batches get the CP-aware
    metadata (`cu_seqlens_*_padded`, `cp_group`, `local_cp_size`) before being passed
    into TransformerEngine attention kernels.

    Args:
        batch: Dictionary containing text tensors plus optional packing metadata:
            `cu_seqlens`, `cu_seqlens_padded`, `max_seqlen`, and `local_cp_size`.
        cp_group: Optional context-parallel process group to use for slicing and for
            attaching to the resulting PackedSeqParams.

    Returns:
        A tuple of (possibly sliced) batch dict and an optional PackedSeqParams.
    """

    cu_seqlens = batch.pop("cu_seqlens", None)
    cu_seqlens_padded = batch.pop("cu_seqlens_padded", None)
    max_seqlen = batch.pop("max_seqlen", None)
    local_cp_size = batch.pop("local_cp_size", None)
    if local_cp_size is not None and torch.is_tensor(local_cp_size):
        local_cp_size = int(local_cp_size.item())

    # Resolve CP group/size even if one was not passed explicitly.
    resolved_cp_group = cp_group
    if resolved_cp_group is None:
        try:
            resolved_cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        except Exception:
            resolved_cp_group = None
    cp_size = resolved_cp_group.size() if resolved_cp_group is not None else 1

    packed_seq_params: Optional[PackedSeqParams] = None
    if cu_seqlens is None and local_cp_size is None:
        # No packing metadata; standard CP slicing.
        batch = get_batch_on_this_cp_rank(batch, resolved_cp_group)
    elif local_cp_size is None:
        # Packed THD format
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)
        if cu_seqlens_padded is not None and cu_seqlens_padded.dtype != torch.int32:
            cu_seqlens_padded = cu_seqlens_padded.to(torch.int32)
        # Ensure all tensors are on the same device as cu_seqlens before TE indexing
        if cu_seqlens is not None:
            target_device = cu_seqlens.device
            for key, data in batch.items():
                if data is not None and torch.is_tensor(data) and data.device != target_device:
                    batch[key] = data.to(target_device, non_blocking=True)
        if cu_seqlens_padded is None:
            if cp_size > 1:
                raise RuntimeError("Context Parallel + packed sequences require `cu_seqlens_padded` in the batch.")
            cu_seqlens_padded = cu_seqlens
        if max_seqlen is None:
            raise RuntimeError("Packed sequences require `max_seqlen` in the batch.")
        # Slice text inputs and targets for THD CP so logits/labels stay aligned.
        thd_keys = ("tokens", "input_ids", "position_ids", "labels", "loss_mask")
        thd_batch = {k: v for k, v in batch.items() if k in thd_keys and v is not None}
        thd_batch, packed_seq_params = get_thd_batch_on_this_cp_rank(
            thd_batch, cu_seqlens, cu_seqlens_padded, max_seqlen, resolved_cp_group
        )
        # Merge sliced tensors back.
        for k, v in thd_batch.items():
            batch[k] = v
        if packed_seq_params.cp_group is None:
            packed_seq_params.cp_group = resolved_cp_group
    else:
        # Hybrid CP format
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(batch, local_cp_size, resolved_cp_group)
        if packed_seq_params.cp_group is None:
            packed_seq_params.cp_group = resolved_cp_group
    return batch, packed_seq_params
