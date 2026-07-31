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

import random
from typing import Any, List

import pytest

from megatron.bridge.diffusion.data.common.sequence_packing_utils import (
    find_first_bin_that_fits,
    first_fit,
    first_fit_decreasing,
    packing_length,
)


def _linear_scan_first_fit(seqlens: List[Any], pack_size: int) -> List[List[Any]]:
    """Reference: the original O(N^2) linear-scan first_fit, before the segment tree."""
    res = []
    for s in seqlens:
        first_bin = find_first_bin_that_fits(res, s, pack_size)
        if first_bin == -1:
            res.append([s])
        else:
            res[first_bin].append(s)
    return res


class _LengthKeyedSample:
    """Stand-in for DiffusionSample with the same __add__/__radd__/__lt__ contract.

    first_fit is used by the diffusion task encoder to pack sample objects, not raw
    integers, so the packing core must key on length while returning the objects
    themselves. This mirrors that contract without pulling in torch or energon.
    """

    def __init__(self, seq_len: int, uid: int):
        self.seq_len = seq_len
        self.uid = uid

    def __add__(self, other: Any) -> int:
        if isinstance(other, _LengthKeyedSample):
            return self.seq_len + other.seq_len
        if isinstance(other, int):
            return self.seq_len + other
        raise NotImplementedError

    def __radd__(self, other: Any) -> int:
        if isinstance(other, int):
            return self.seq_len + other
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, _LengthKeyedSample):
            return self.seq_len < other.seq_len
        if isinstance(other, int):
            return self.seq_len < other
        raise NotImplementedError


def test_find_first_bin_that_fits():
    """Test find_first_bin_that_fits function."""
    # Test case: Find a bin that fits
    bins = [[5, 3], [10], [2, 2, 2]]
    s = 2
    bin_size = 10
    result = find_first_bin_that_fits(bins, s, bin_size)
    assert result == 0, "Should return index 0 as first bin (5+3+2=10) fits"

    # Test case: No bin fits
    bins = [[8, 2], [9, 1], [10]]
    s = 5
    bin_size = 10
    result = find_first_bin_that_fits(bins, s, bin_size)
    assert result == -1, "Should return -1 as no bin can accommodate size 5"

    # Test case: Empty bins list
    bins = []
    s = 5
    bin_size = 10
    result = find_first_bin_that_fits(bins, s, bin_size)
    assert result == -1, "Should return -1 for empty bins list"

    # Test case: First bin doesn't fit, but second does
    bins = [[9], [5], [3]]
    s = 4
    bin_size = 10
    result = find_first_bin_that_fits(bins, s, bin_size)
    assert result == 1, "Should return index 1 as second bin (5+4=9) fits"


def test_first_fit():
    """Test first_fit bin packing algorithm."""
    # Test case: Simple packing scenario
    seqlens = [5, 3, 2, 7, 4]
    pack_size = 10
    result = first_fit(seqlens, pack_size)

    # Verify all sequences are packed
    all_items = [item for bin in result for item in bin]
    assert sum(all_items) == sum(seqlens), "Sum of all packed items should equal sum of input"

    # Verify no bin exceeds pack_size
    for bin in result:
        assert sum(bin) <= pack_size, f"Bin {bin} exceeds pack_size {pack_size}"

    # Verify expected packing: [5, 3, 2], [7], [4] (first-fit order)
    assert len(result) == 3, "Should create 3 bins"
    assert result[0] == [5, 3, 2], "First bin should contain [5, 3, 2]"
    assert result[1] == [7], "Second bin should contain [7]"
    assert result[2] == [4], "Third bin should contain [4]"


def test_first_fit_decreasing():
    """Test first_fit_decreasing bin packing algorithm."""
    # Test case: Same sequences as first_fit but sorted in decreasing order
    seqlens = [5, 3, 2, 7, 4]
    pack_size = 10
    result = first_fit_decreasing(seqlens, pack_size)

    # Verify all sequences are packed
    all_items = [item for bin in result for item in bin]
    assert sum(all_items) == sum(seqlens), "Sum of all packed items should equal sum of input"

    # Verify no bin exceeds pack_size
    for bin in result:
        assert sum(bin) <= pack_size, f"Bin {bin} exceeds pack_size {pack_size}"

    # Verify expected packing: sorted [7, 5, 4, 3, 2] -> [7, 3], [5, 4, 2] (more efficient)
    assert len(result) <= 3, "Should create at most 3 bins"
    # First-fit-decreasing should pack: [7, 3], [5, 4], [2]
    assert result[0] == [7, 3], "First bin should contain [7, 3]"
    assert result[1] == [5, 4], "Second bin should contain [5, 4]"
    assert result[2] == [2], "Third bin should contain [2]"


class TestSegmentTreeMatchesLinearScan:
    """The segment-tree core must reproduce the original linear scan exactly.

    first_fit is order- and identity-sensitive: the diffusion task encoder packs
    sample objects and relies on which bin each one lands in, so "same bin count"
    is not a strong enough guarantee. These compare full bin assignments.
    """

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_on_random_int_input(self, seed):
        """Random integer lengths, including zero-length and oversized entries."""
        rng = random.Random(seed)
        pack_size = rng.choice([16, 128, 2048, 8192])
        # Range exceeds pack_size so oversized items (own bin) are covered too.
        seqlens = [rng.randint(0, pack_size + 200) for _ in range(rng.choice([1, 5, 50, 400]))]

        assert first_fit(seqlens, pack_size) == _linear_scan_first_fit(seqlens, pack_size)
        assert first_fit_decreasing(seqlens, pack_size) == _linear_scan_first_fit(
            sorted(seqlens, reverse=True), pack_size
        )

    @pytest.mark.parametrize(
        "seqlens,pack_size",
        [
            ([], 100),
            ([0] * 10, 100),
            ([0, 5, 0, 95, 0, 1], 100),
            ([500], 100),
            ([500, 600, 700], 100),
            ([50, 50, 50, 50], 100),
        ],
        ids=["empty", "all-zero", "zero-mixed", "single-oversize", "all-oversize", "exact-fit"],
    )
    def test_matches_on_edge_cases(self, seqlens, pack_size):
        """Zero-length items are the tricky case: an unopened bin also reads as 0 capacity."""
        assert first_fit(seqlens, pack_size) == _linear_scan_first_fit(seqlens, pack_size)


class TestPacksSampleObjects:
    """first_fit must pack length-keyed objects, not just ints (diffusion task encoder path)."""

    def _samples(self, lengths):
        return [_LengthKeyedSample(length, uid) for uid, length in enumerate(lengths)]

    def test_returns_original_objects(self):
        samples = self._samples([5, 3, 2, 7, 4])
        result = first_fit(samples, 10)

        assert [[s.uid for s in b] for b in result] == [[0, 1, 2], [3], [4]]
        assert all(isinstance(s, _LengthKeyedSample) for b in result for s in b)

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_linear_scan_on_objects(self, seed):
        rng = random.Random(1000 + seed)
        pack_size = rng.choice([256, 4096, 8192])
        samples = self._samples([rng.randint(0, pack_size // 2 + 50) for _ in range(rng.choice([1, 40, 300]))])

        expected = _linear_scan_first_fit(samples, pack_size)
        assert [[s.uid for s in b] for b in first_fit(samples, pack_size)] == [[s.uid for s in b] for b in expected]

    def test_every_sample_placed_exactly_once(self):
        rng = random.Random(7)
        samples = self._samples([rng.randint(1, 2000) for _ in range(200)])

        placed = sorted(s.uid for b in first_fit(samples, 4096) for s in b)
        assert placed == sorted(s.uid for s in samples)

    def test_bins_respect_capacity(self):
        rng = random.Random(11)
        pack_size = 4096
        samples = self._samples([rng.randint(1, pack_size) for _ in range(200)])

        for abin in first_fit(samples, pack_size):
            assert sum(s.seq_len for s in abin) <= pack_size


def test_packing_length_handles_ints_and_samples():
    """packing_length is what keeps the module decoupled from DiffusionSample."""
    assert packing_length(42) == 42
    assert packing_length(_LengthKeyedSample(128, uid=0)) == 128
