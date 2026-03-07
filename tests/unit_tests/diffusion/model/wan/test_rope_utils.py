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

import pytest
import torch

from megatron.bridge.diffusion.models.wan.rope_utils import Wan3DRopeEmbeddings


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_wan3d_rope_embeddings_shapes_and_padding():
    # Small, CPU-friendly config
    n_head = 2
    dim_head = 8  # must be divisible with the internal splits
    max_position_len = 16
    rope = Wan3DRopeEmbeddings(dim_head=dim_head, max_position_len=max_position_len)

    # Two samples with different (f, h, w)
    grid_sizes = torch.tensor([[2, 3, 2], [4, 1, 1]], dtype=torch.int32)
    seq_lens = [(2 * 3 * 2), (4 * 1 * 1)]
    padded_lens = [seq_lens[0] + 2, seq_lens[1]]  # pad first sample

    cu_seqlens_q_padded = torch.tensor([0, padded_lens[0], padded_lens[0] + padded_lens[1]], dtype=torch.int32)

    out = rope(
        n_head=n_head,
        dim_head=dim_head,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        grid_sizes=grid_sizes,
        device=torch.device("cpu"),
    )

    # Total concatenated length equals sum of padded lens
    assert out.shape == (sum(padded_lens), 1, 1, dim_head)

    # Check that padding region for the first sample is zero
    first_seq_len = seq_lens[0]
    first_padded_len = padded_lens[0]
    tail = out[first_seq_len:first_padded_len]
    assert torch.all(tail == 0), "Padded region should be zeros"
