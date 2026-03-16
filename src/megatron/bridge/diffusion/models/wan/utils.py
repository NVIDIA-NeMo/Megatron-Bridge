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

import math
from typing import Tuple

import torch
import torch.distributed as dist
import transformer_engine_torch as tex


def grid_sizes_calculation(
    input_shape: Tuple[int, int, int],  # (F_latents, H_latents, W_latents)
    patch_size: Tuple[int, int, int],  # (pF, pH, pW)
) -> Tuple[int, int, int]:
    """
    Compute the (f,h,w) output spatial/temporal dimensions of a Conv3d patch embedder.
    """

    F_latents, H_latents, W_latents = input_shape
    pF, pH, pW = patch_size
    F_patches = F_latents // pF
    H_patches = H_latents // pH
    W_patches = W_latents // pW

    return [F_patches, H_patches, W_patches]


def patchify(x, patch_size):
    """
    Convert a list of reconstructed video tensor into patch embeddings.
    This method is the inverse of `unpatchify`.

    Args:
        x (list[torch.Tensor]): list of tensors, each with shape [c, F_patches * pF, H_patches * pH, W_patches * pW]
        patch_size (tuple): (pF, pH, pW)

    Returns:
        torch.Tensor: shape [ (F_patches * H_patches * W_patches), (c * pF * pH * pW)],
    """
    out = []
    for u in x:
        c, F_pF, H_pH, W_pW = u.shape
        pF, pH, pW = patch_size
        assert F_pF % pF == 0 and H_pH % pH == 0 and W_pW % pW == 0, (
            "Spatial dimensions must be divisible by patch size."
        )

        F_patches, H_patches, W_patches = F_pF // pF, H_pH // pH, W_pW // pW

        # split spatial dims into (grid, patch) and reorder to match original patch layout:
        # start: (c, F_patches * pF, H_patches * pH, W_patches * pW)
        # reshape -> (c, F_patches, pF, H_patches, pH, W_patches, pW)
        # permute -> (F_patches, H_patches, W_patches, pF, pH, pW, c)
        t = u.reshape(c, F_patches, pF, H_patches, pH, W_patches, pW)
        t = t.permute(1, 3, 5, 2, 4, 6, 0)

        num_patches = F_patches * H_patches * W_patches
        out.append(t.reshape(num_patches, c * (pF * pH * pW)))
    return out


def unpatchify(
    x: list[torch.Tensor], grid_sizes: list[Tuple[int, int, int]], out_dim: int, patch_size: Tuple[int, int, int]
) -> list[torch.Tensor]:
    """
    Reconstruct video tensors from patch embeddings into a list of videotensors.

    Args:
        x (list[torch.Tensor]):
            list of tensors, each with shape [seq_len, c * pF * pH * pW]
        grid_sizes (list[Tuple[int, int, int]]):
            list of tensors, each with original spatial-temporal grid dimensions before patching,
                (3 dimensions correspond to F_patches, H_patches, W_patches)

    Returns:
        list[torch.Tensor]: list of tensors, each with shape [c, F_latents, H_latents, W_latents]
    """

    c = out_dim
    out = []
    for u, v in zip(x, grid_sizes):
        u = u[: math.prod(v)].view(*v, *patch_size, c)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        u = u.reshape(c, *[i * j for i, j in zip(v, patch_size)])
        out.append(u)
    return out


def thd_split_inputs_cp(
    x: torch.Tensor, cu_seqlens_q_padded: torch.Tensor, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """
    Split a THD-packed tensor across CP ranks for inputs shaped [S, B, ...].

    Args:
        x: [S, B, ...] tensor (sequence first).
        cu_seqlens_q_padded: 1D int32 THD cu_seqlens (padded) used for packing.
        cp_group: context-parallel process group.

    Returns:
        x_local: [S_local, B, ...] shard for this CP rank.
    """
    # Move to [B, S, ...] to use THD partitioning along S
    x_bs = x.transpose(0, 1).contiguous()  # [B, S, ...]

    total_S = x_bs.size(1)
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    # Compute this rank's THD partition indices (same API as during gather)
    idx = tex.thd_get_partitioned_indices(
        cu_seqlens_q_padded,  # int32 offsets
        total_S,
        cp_size,
        cp_rank,
    ).to(device=x_bs.device, dtype=torch.long)  # [S_local]

    # Take the shard along sequence dim
    x_local_bs = x_bs.index_select(dim=1, index=idx).contiguous()  # [B, S_local, ...]

    # Return to [S, B, ...]
    x_local = x_local_bs.transpose(0, 1).contiguous()  # [S_local, B, ...]
    return x_local
