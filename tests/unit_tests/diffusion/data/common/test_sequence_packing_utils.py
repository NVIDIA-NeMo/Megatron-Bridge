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

from megatron.bridge.diffusion.data.common.sequence_packing_utils import (
    find_first_bin_that_fits,
    first_fit,
    first_fit_decreasing,
)


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
