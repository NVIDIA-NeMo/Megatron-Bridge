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

import torch

from megatron.bridge.diffusion.models.wan.utils import grid_sizes_calculation, patchify, unpatchify


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
