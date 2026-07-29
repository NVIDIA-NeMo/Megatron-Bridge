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


from typing import Any, List

from megatron.bridge.data.packing.algorithms import first_fit as _shared_first_fit


def packing_length(item: Any) -> int:
    """
    Returns the bin-packing length of an item.

    Items are either plain sequence lengths or objects that report their length by
    adding with an int -- `DiffusionSample` does this, returning its padded query
    sequence length when one is set and its unpadded length otherwise. Going through
    `0 + item` keeps this module free of a hard dependency on the sample type, and
    matches the length the previous `sum(bin) + s` capacity check used.

    Args:
      item: A sequence length, or an object implementing `__radd__` against an int.

    Returns:
      The integer length used for bin-packing capacity accounting.
    """
    return 0 + item


def find_first_bin_that_fits(bins: List[List[int]], s: int, bin_size: int) -> int:
    """
    Finds the first bin in a list of bins that has enough space to fit a sequence of size 's'.

    Args:
      bins: A list of lists, where each inner list represents a bin and contains the current elements in that bin.
      s: The size of the sequence to be placed in a bin.
      bin_size: The maximum capacity of each bin.

    Returns:
      The index of the first bin that can fit the sequence 's', or -1 if no such bin exists.
    """
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit algorithm.

    Delegates to the shared segment-tree implementation in
    `megatron.bridge.data.packing.algorithms`, which places each item in O(log N)
    rather than rescanning and re-summing every open bin. Bin assignments are
    identical to the previous linear-scan implementation.

    Args:
      seqlens: The sequences to pack. Entries are either integer lengths or objects
        that report their length when added to an int (see `packing_length`); the
        original entries are what end up in the returned bins.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, where each inner list represents a bin and contains the
        entries assigned to that bin.
    """
    return _shared_first_fit(seqlens, pack_size, item_lengths=[packing_length(item) for item in seqlens])


def first_fit_decreasing(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit Decreasing algorithm.

    This is a variation of the First-Fit algorithm where the sequences are sorted by decreasing length before packing.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    sorted_seqlens = sorted(seqlens, reverse=True)
    return first_fit(sorted_seqlens, pack_size)
