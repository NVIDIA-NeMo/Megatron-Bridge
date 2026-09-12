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

import pytest
import torch

from megatron.bridge.diffusion.common.cp_utils import (
    _reorder_zigzag_chunks,
    local_zigzag_mask,
    zigzag_slice,
)


# ---------------------------------------------------------------------------
# zigzag_slice
# ---------------------------------------------------------------------------
class TestZigzagSlice:
    def test_cp1_is_identity(self):
        x = torch.arange(12).view(1, 12, 1)
        assert torch.equal(zigzag_slice(x, cp_rank=0, cp_size=1, seq_dim=1), x)

    def test_cp2_load_balanced_slices(self):
        # seq=8, cp=2 -> 4 chunks of 2: [0,1][2,3][4,5][6,7].
        # rank r owns chunks [r] and [2*cp-1-r]: rank0 -> 0 & 3 -> {0,1,6,7};
        # rank1 -> 1 & 2 -> {2,3,4,5}.
        x = torch.arange(8).view(1, 8)
        r0 = zigzag_slice(x, cp_rank=0, cp_size=2, seq_dim=1)
        r1 = zigzag_slice(x, cp_rank=1, cp_size=2, seq_dim=1)
        assert r0.flatten().tolist() == [0, 1, 6, 7]
        assert r1.flatten().tolist() == [2, 3, 4, 5]
        # each rank holds seq/cp tokens
        assert r0.shape[1] == 4 and r1.shape[1] == 4

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_shards_reassemble_to_original(self, cp_size):
        seq = 8 * cp_size
        x = torch.arange(seq * 3).view(1, seq, 3)
        shards = [zigzag_slice(x, r, cp_size, seq_dim=1) for r in range(cp_size)]
        # undo the zigzag: each rank contributes two chunks (idx r and 2cp-1-r)
        chunks = {}
        for r, s in enumerate(shards):
            half = s.shape[1] // 2
            chunks[r] = s[:, :half]
            chunks[2 * cp_size - 1 - r] = s[:, half:]
        recon = torch.cat([chunks[i] for i in range(2 * cp_size)], dim=1)
        assert torch.equal(recon, x)

    def test_requires_divisible_by_2cp(self):
        x = torch.arange(6).view(1, 6)  # 6 not divisible by 2*cp=8
        with pytest.raises(AssertionError):
            zigzag_slice(x, cp_rank=0, cp_size=4, seq_dim=1)


# ---------------------------------------------------------------------------
# local_zigzag_mask
# ---------------------------------------------------------------------------
class TestLocalZigzagMask:
    def test_cp1_all_true(self):
        m = local_zigzag_mask(10, cp_rank=0, cp_size=1, device="cpu")
        assert m.dtype == torch.bool and m.shape == (10,) and bool(m.all())

    def test_cp2_owned_positions(self):
        # seq=8, cp=2: rank0 owns {0,1,6,7}, rank1 owns {2,3,4,5}.
        m0 = local_zigzag_mask(8, cp_rank=0, cp_size=2, device="cpu")
        m1 = local_zigzag_mask(8, cp_rank=1, cp_size=2, device="cpu")
        assert m0.nonzero().flatten().tolist() == [0, 1, 6, 7]
        assert m1.nonzero().flatten().tolist() == [2, 3, 4, 5]

    def test_masks_partition_the_sequence(self):
        # across all ranks every position is owned exactly once
        cp_size, seq = 4, 32
        total = torch.zeros(seq, dtype=torch.int)
        for r in range(cp_size):
            total += local_zigzag_mask(seq, r, cp_size, device="cpu").int()
        assert torch.equal(total, torch.ones(seq, dtype=torch.int))

    def test_requires_divisible_by_2cp(self):
        with pytest.raises(AssertionError):
            local_zigzag_mask(6, cp_rank=0, cp_size=4, device="cpu")


# ---------------------------------------------------------------------------
# _reorder_zigzag_chunks (the gather/scatter reassembly used in autograd
# backward; exercised here on CPU without a process group)
# ---------------------------------------------------------------------------
class TestReorderZigzagChunks:
    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_reassembles_original_sequence(self, cp_size):
        # `gathered` mirrors an all_gather of per-rank zigzag shards: rank r holds
        # chunks [r] and [2*cp-1-r] concatenated -- exactly what zigzag_slice yields.
        seq = 8 * cp_size
        x = torch.arange(seq * 2).view(1, seq, 2)
        gathered = [zigzag_slice(x, r, cp_size, seq_dim=1) for r in range(cp_size)]
        recon = _reorder_zigzag_chunks(gathered, cp_size, seq_dim=1)
        assert torch.equal(recon, x)

    def test_result_is_contiguous(self):
        cp_size, seq = 2, 16
        x = torch.arange(seq).view(1, seq)
        gathered = [zigzag_slice(x, r, cp_size, seq_dim=1) for r in range(cp_size)]
        recon = _reorder_zigzag_chunks(gathered, cp_size, seq_dim=1)
        assert recon.is_contiguous()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
