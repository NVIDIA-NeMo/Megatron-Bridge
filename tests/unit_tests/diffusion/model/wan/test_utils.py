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

from unittest.mock import MagicMock, patch

import torch

from megatron.bridge.diffusion.models.wan.utils import grid_sizes_calculation, patchify, thd_split_inputs_cp, unpatchify


def test_grid_sizes_calculation_basic():
    input_shape = (4, 8, 6)
    patch_size = (1, 2, 3)
    f, h, w = grid_sizes_calculation(input_shape, patch_size)
    assert (f, h, w) == (4, 4, 2)


def test_patchify_unpatchify_roundtrip():
    # Video latent: [c, F_patches * pF, H_patches * pH, W_patches * pW]
    c = 3
    F_patches, H_patches, W_patches = 2, 2, 3
    patch_size = (1, 2, 2)
    F_latents = F_patches * patch_size[0]
    H_latents = H_patches * patch_size[1]
    W_latents = W_patches * patch_size[2]

    x = [torch.randn(c, F_latents, H_latents, W_latents)]

    patches = patchify(x, patch_size)
    assert isinstance(patches, list) and len(patches) == 1
    seq_len, dim = patches[0].shape
    assert seq_len == F_patches * H_patches * W_patches
    assert dim == c * (patch_size[0] * patch_size[1] * patch_size[2])

    # Unpatchify and compare
    y = unpatchify(patches, [[F_patches, H_patches, W_patches]], out_dim=c, patch_size=patch_size)
    assert isinstance(y, list) and len(y) == 1
    assert y[0].shape == x[0].shape
    torch.testing.assert_close(y[0], x[0], rtol=1e-5, atol=1e-5)


@patch("megatron.bridge.diffusion.models.wan.utils.tex")
@patch("megatron.bridge.diffusion.models.wan.utils.dist")
def test_thd_split_inputs_cp_selects_correct_indices(mock_dist, mock_tex):
    S, B, D = 8, 2, 4
    cp_size, cp_rank = 2, 0

    mock_dist.get_world_size.return_value = cp_size
    mock_dist.get_rank.return_value = cp_rank
    indices = torch.tensor([0, 2, 4, 6])
    mock_tex.thd_get_partitioned_indices.return_value = indices

    x = torch.arange(S * B * D, dtype=torch.float32).reshape(S, B, D)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    cp_group = MagicMock()

    result = thd_split_inputs_cp(x, cu_seqlens, cp_group)

    assert result.shape == (len(indices), B, D)
    x_bs = x.transpose(0, 1)
    expected = x_bs.index_select(dim=1, index=indices).transpose(0, 1)
    torch.testing.assert_close(result, expected)
    mock_tex.thd_get_partitioned_indices.assert_called_once_with(cu_seqlens, S, cp_size, cp_rank)


@patch("megatron.bridge.diffusion.models.wan.utils.tex")
@patch("megatron.bridge.diffusion.models.wan.utils.dist")
def test_thd_split_inputs_cp_rank1(mock_dist, mock_tex):
    S, B, D = 8, 2, 4
    cp_size, cp_rank = 2, 1

    mock_dist.get_world_size.return_value = cp_size
    mock_dist.get_rank.return_value = cp_rank
    indices = torch.tensor([1, 3, 5, 7])
    mock_tex.thd_get_partitioned_indices.return_value = indices

    x = torch.arange(S * B * D, dtype=torch.float32).reshape(S, B, D)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    cp_group = MagicMock()

    result = thd_split_inputs_cp(x, cu_seqlens, cp_group)

    assert result.shape == (4, B, D)
    x_bs = x.transpose(0, 1)
    expected = x_bs.index_select(dim=1, index=indices).transpose(0, 1)
    torch.testing.assert_close(result, expected)


@patch("megatron.bridge.diffusion.models.wan.utils.tex")
@patch("megatron.bridge.diffusion.models.wan.utils.dist")
def test_thd_split_inputs_cp_preserves_extra_dims(mock_dist, mock_tex):
    """Verify the function works with higher-dimensional tensors [S, B, H, W]."""
    S, B, H, W = 6, 2, 3, 5

    mock_dist.get_world_size.return_value = 3
    mock_dist.get_rank.return_value = 2
    indices = torch.tensor([2, 5])
    mock_tex.thd_get_partitioned_indices.return_value = indices

    x = torch.randn(S, B, H, W)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    cp_group = MagicMock()

    result = thd_split_inputs_cp(x, cu_seqlens, cp_group)

    assert result.shape == (2, B, H, W)
    x_bs = x.transpose(0, 1)
    expected = x_bs.index_select(dim=1, index=indices).transpose(0, 1)
    torch.testing.assert_close(result, expected)


@patch("megatron.bridge.diffusion.models.wan.utils.tex")
@patch("megatron.bridge.diffusion.models.wan.utils.dist")
def test_thd_split_inputs_cp_single_rank(mock_dist, mock_tex):
    """When cp_size=1, the entire tensor should be returned."""
    S, B, D = 4, 1, 2

    mock_dist.get_world_size.return_value = 1
    mock_dist.get_rank.return_value = 0
    indices = torch.arange(S)
    mock_tex.thd_get_partitioned_indices.return_value = indices

    x = torch.randn(S, B, D)
    cu_seqlens = torch.tensor([0, S], dtype=torch.int32)
    cp_group = MagicMock()

    result = thd_split_inputs_cp(x, cu_seqlens, cp_group)

    assert result.shape == x.shape
    torch.testing.assert_close(result, x)
