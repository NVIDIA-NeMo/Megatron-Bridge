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

from megatron.bridge.data.datasets.packing_utils import (
    balanced_index_order,
    first_fit,
    first_fit_decreasing,
)


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


@pytest.mark.unit
def test_dead_cost_balanced_pack_removed():
    # F15: cost_balanced_pack is dead (production reorders via balanced_index_order) -> deleted.
    import megatron.bridge.data.datasets.packing_utils as packing_utils

    assert not hasattr(packing_utils, "cost_balanced_pack")


@pytest.mark.unit
def test_dead_maybe_reorder_global_batch_removed():
    # F15: maybe_reorder_global_batch deleted; production uses compute_reorder_perm + reorder_and_slice.
    import megatron.bridge.data.megatron_mimo.dp_utils as dp_utils

    assert not hasattr(dp_utils, "maybe_reorder_global_batch")


@pytest.mark.unit
class TestBalancedIndexOrder:
    def test_valid_permutation_and_equal_groups(self):
        costs = [5.0, 1.0, 4.0, 2.0, 3.0, 6.0]
        order = balanced_index_order(costs, n_groups=2)
        assert sorted(order) == list(range(6))  # permutation
        # contiguous equal halves
        g0, g1 = order[:3], order[3:]
        assert len(g0) == 3 and len(g1) == 3

    def test_balances_contiguous_shards(self):
        # grouped heavies; naive contiguous split would imbalance
        costs = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
        order = balanced_index_order(costs, n_groups=3)  # 2 per group
        groups = [order[0:2], order[2:4], order[4:6]]
        totals = [sum(costs[i] for i in g) for g in groups]
        assert max(totals) - min(totals) == 0.0  # each group one heavy + one light

    def test_determinism(self):
        costs = [3.0, 1.0, 9.0, 0.5, 4.0, 2.0]
        assert balanced_index_order(costs, 2) == balanced_index_order(costs, 2)

    def test_errors(self):
        with pytest.raises(ValueError):
            balanced_index_order([1.0, 2.0, 3.0], 2)  # 3 not divisible by 2
        with pytest.raises(ValueError):
            balanced_index_order([1.0], 0)
