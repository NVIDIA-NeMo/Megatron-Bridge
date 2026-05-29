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

from megatron.bridge.data.vlm_datasets.step37_flickr8k.packing import pack


pytestmark = pytest.mark.unit


def _ranges(result):
    return result.packed_sample_ranges


def test_exact_fill_flushes_pack():
    # 2 + 2 == max_len -> a single full pack of both samples.
    result = pack([2, 2], max_len=4, oversize_policy="drop")
    assert _ranges(result) == [(0, 2)]
    assert result.num_packed_samples == 1
    assert result.num_droped == 0


def test_greedy_fill_starts_new_pack_on_overflow():
    # 3 then 2: 3 fits, +2 overflows max_len=4 -> putback -> new pack with 2.
    result = pack([3, 2], max_len=4, oversize_policy="drop")
    assert _ranges(result) == [(0, 1), (1, 1)]
    assert result.num_packed_samples == 2
    assert result.num_droped == 0


def test_trailing_partial_pack_is_flushed():
    result = pack([2, 1], max_len=4, oversize_policy="drop")
    # 2 + 1 = 3 < 4, never hits exact-fill, flushed at the end as one pack.
    assert _ranges(result) == [(0, 2)]
    assert result.num_packed_samples == 1


def test_oversize_drop_policy_skips_sample():
    # Middle sample (5) exceeds max_len=4 and is dropped; neighbours pack.
    result = pack([2, 5, 2], max_len=4, oversize_policy="drop")
    assert result.num_droped == 1
    # The two size-2 samples land in separate packs (idx 0 and idx 2).
    assert _ranges(result) == [(0, 1), (2, 1)]


def test_oversize_extend_policy_keeps_sample_in_own_pack():
    result = pack([2, 5, 2], max_len=4, oversize_policy="extend")
    assert result.num_droped == 0
    # 2 flushed, 5 forced into its own pack, 2 flushed -> three packs.
    assert _ranges(result) == [(0, 1), (1, 1), (2, 1)]


def test_empty_input():
    result = pack([], max_len=4, oversize_policy="drop")
    assert _ranges(result) == []
    assert result.num_packed_samples == 0
    assert result.num_droped == 0


def test_packed_ranges_cover_all_kept_samples():
    sizes = [1, 1, 1, 1, 1]
    result = pack(sizes, max_len=2, oversize_policy="drop")
    # Each pair fills exactly, last single is flushed.
    assert _ranges(result) == [(0, 2), (2, 2), (4, 1)]
    total_packed = sum(n for _, n in _ranges(result))
    assert total_packed + result.num_droped == len(sizes)
