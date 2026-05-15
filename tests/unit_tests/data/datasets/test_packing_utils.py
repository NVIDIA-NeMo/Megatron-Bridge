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

from typing import List

import numpy as np
import pytest

from megatron.bridge.data.datasets.packing_utils import first_fit, first_fit_decreasing


def _first_fit_linear(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """Reference: original O(N²) linear-scan first_fit before segment tree."""
    res = []
    res_sums = []
    for s in seqlens:
        first_bin = -1
        for i, cur_sum in enumerate(res_sums):
            if cur_sum + s <= pack_size:
                first_bin = i
                break
        if first_bin == -1:
            res.append([s])
            res_sums.append(s)
        else:
            res[first_bin].append(s)
            res_sums[first_bin] += s
    return res


class TestFirstFitPacking:
    """Test cases for first_fit bin-packing algorithm."""

    def test_first_fit_decreasing_sorted_order(self):
        """Test first_fit_decreasing sorts sequences before packing."""
        seqlens = [1111, 8192, 4096, 1000]
        pack_size = 2048

        result = first_fit_decreasing(seqlens, pack_size)
        assert result == [[8192], [4096], [1111], [1000]]

    def test_bin_capacity_not_exceeded(self):
        """Test no bin exceeds the pack_size limit."""
        np.random.seed(7)
        seqlens = list(np.random.randint(1, 2048, size=10000))
        pack_size = 2048

        result = first_fit(seqlens, pack_size)
        for bin_contents in result:
            assert sum(bin_contents) <= pack_size


class TestSegmentTreeMatchesLinearScan:
    """Verify segment-tree first_fit produces identical results to the original linear-scan."""

    def test_matches_on_small_input(self):
        """Test a small, hand-crafted example."""
        seqlens = [500, 600, 500, 400, 700]
        pack_size = 1200
        assert first_fit(seqlens, pack_size) == _first_fit_linear(seqlens, pack_size)

    def test_matches_on_oversized_sequences(self):
        """Test sequences that individually exceed pack_size are still placed in their own bins."""
        seqlens = [4096, 3000, 5000]
        pack_size = 2048
        assert first_fit(seqlens, pack_size) == _first_fit_linear(seqlens, pack_size)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_on_random_input(self, seed):
        """Test random inputs of varying sizes."""
        np.random.seed(seed)
        seqlens = list(np.random.randint(1, 2048, size=5000))
        pack_size = 2048
        assert first_fit(seqlens, pack_size) == _first_fit_linear(seqlens, pack_size)
