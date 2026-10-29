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

"""Stage-boundary packing for pipeline-parallel Kimi K3."""

import torch


def bank_num_rows(layer_idx: int, block_size: int) -> int:
    """Return the number of AttnRes snapshots present before ``layer_idx``."""
    if layer_idx <= 0:
        raise ValueError("layer_idx must be positive")
    return (layer_idx + block_size - 1) // block_size


def pack_stage_boundary(prefix_sum: torch.Tensor, block_residual: torch.Tensor) -> torch.Tensor:
    """Pack the AttnRes prefix and snapshot bank into one pipeline tensor."""
    if block_residual.shape[-2] <= 0:
        raise ValueError("A pipeline boundary must follow the first AttnRes snapshot")
    return torch.cat((prefix_sum, block_residual.flatten(-2)), dim=-1)


def unpack_stage_boundary(
    packed: torch.Tensor,
    hidden_size: int,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the prefix and snapshot bank received from a pipeline stage."""
    expected = (1 + num_rows) * hidden_size
    if packed.shape[-1] != expected:
        raise ValueError(f"stage-boundary payload width {packed.shape[-1]} != (1 + {num_rows}) * {hidden_size}")
    prefix_sum = packed[..., :hidden_size].contiguous()
    block_residual = packed[..., hidden_size:].unflatten(-1, (num_rows, hidden_size)).contiguous()
    return prefix_sum, block_residual
