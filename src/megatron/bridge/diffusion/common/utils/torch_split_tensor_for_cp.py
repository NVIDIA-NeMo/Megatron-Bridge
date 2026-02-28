# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import torch
import transformer_engine_torch as tex
from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, get_rank, get_world_size


def cat_outputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup, thd_cu_seqlens: Optional[Tensor] = None) -> Tensor:
    """
    Concatenates tensors from multiple processes along a specified dimension.

    This function gathers tensors from all processes in the given process group
    and concatenates them along the specified dimension.

    Args:
        x (Tensor): The input tensor to be gathered and concatenated.
        seq_dim (int): The dimension along which to concatenate the gathered tensors.
        cp_group (ProcessGroup): The process group containing all the processes involved in the gathering.
        thd_cu_seqlens (Tensor, optional): THD cumulative sequence lengths used during partitioning. Provide
            this to restore the original token order after gathering.

    Returns:
        Tensor: A tensor resulting from the concatenation of tensors from all processes. If `thd_cu_seqlens`
        is provided, the tensor is reordered to match the original (pre-partition) sequence order.

    Raises:
        RuntimeError: If the gathering of tensors fails.
    """
    # Number of processes in the group
    world_size = get_world_size(cp_group)
    # List to hold tensors from each rank
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    # Attempt to gather tensors from all ranks
    all_gather(gathered_tensors, x, group=cp_group)

    # Concatenate tensors along the specified dimension
    gathered = torch.cat(gathered_tensors, dim=seq_dim)
    total_seq_len = int(thd_cu_seqlens[-1].item())
    # Rebuild the global index ordering used during THD partitioning.
    cp_rank = get_rank(cp_group)
    local_indices = tex.thd_get_partitioned_indices(thd_cu_seqlens, total_seq_len, world_size, cp_rank).to(
        device=x.device, dtype=torch.long
    )

    # Gather indices from all ranks to compute the inverse permutation.
    gathered_indices = [torch.empty_like(local_indices) for _ in range(world_size)]
    all_gather(gathered_indices, local_indices, group=cp_group)
    global_indices = torch.cat(gathered_indices, dim=0)

    if global_indices.numel() != gathered.size(seq_dim):
        raise RuntimeError("Gathered indices size does not match gathered tensor along sequence dimension.")

    restore_order = torch.argsort(global_indices, dim=0)
    gathered = gathered.index_select(seq_dim, restore_order.to(device=gathered.device))
    return gathered.contiguous()
