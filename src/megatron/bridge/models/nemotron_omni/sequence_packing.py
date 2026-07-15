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

"""Sequence packing for Nemotron Omni's model-owned media merge."""

from typing import Optional

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection


def pack_sequences_from_attention_mask(
    tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pre_process: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """Pack padded sequences and take exactly one zigzag CP shard.

    ``tensor`` may contain token IDs, embeddings, or other sequence-aligned
    values. Its first two dimensions must be ``[batch, sequence]``. Each
    sequence is padded to the active TP/CP alignment before CP sharding so the
    returned ``PackedSeqParams`` describes the full, pre-shard THD layout.
    """

    batch_size = tensor.shape[0]
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()

    sequence_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    if pg_collection is not None:
        tp_size = pg_collection.tp.size()
        cp_size = pg_collection.cp.size()
        cp_rank = pg_collection.cp.rank()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

    alignment = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    padding = (alignment - sequence_lengths % alignment) % alignment
    padded_lengths = sequence_lengths + padding

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=tensor.device)
    cu_seqlens[1:] = torch.cumsum(sequence_lengths, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=tensor.device)
    cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)

    lengths_cpu = sequence_lengths.tolist()
    padded_lengths_cpu = padded_lengths.tolist()
    cu_padded_cpu = cu_seqlens_padded.tolist()
    max_seqlen = max(padded_lengths_cpu)

    if pre_process:
        shape = list(tensor.shape[1:])
        shape[0] = sum(padded_lengths_cpu) // cp_size
        packed = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

        for batch_index in range(batch_size):
            valid = tensor[batch_index, attention_mask[batch_index]]
            if cp_size == 1:
                start = cu_padded_cpu[batch_index]
                packed[start : start + lengths_cpu[batch_index]] = valid
                continue

            padded_length = padded_lengths_cpu[batch_index]
            local_length = padded_length // cp_size
            half_local_length = local_length // 2
            local_start = cu_padded_cpu[batch_index] // cp_size

            first_start = half_local_length * cp_rank
            first_end = half_local_length * (cp_rank + 1)
            packed[local_start : local_start + half_local_length] = valid[first_start:first_end]

            second_start = padded_length - half_local_length * (cp_rank + 1)
            second_end = min(padded_length - half_local_length * cp_rank, valid.shape[0])
            second_length = second_end - second_start
            if second_length > 0:
                packed[local_start + half_local_length : local_start + half_local_length + second_length] = valid[
                    second_start:second_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        # Mamba runs on the reconstructed full sequence after its CP
        # all-to-all, so seq_idx must describe global padded boundaries rather
        # than the local CP shard returned above.
        total_tokens=sum(padded_lengths_cpu),
    )

    if pre_process:
        return packed.unsqueeze(0), packed_seq_params
    return tensor, packed_seq_params
