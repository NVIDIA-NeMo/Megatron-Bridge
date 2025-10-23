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

"""Tests for iterator utilities."""

import pytest

from megatron.bridge.data.iterator_utils import make_data_iterator_list


def test_make_data_iterator_list_single_chunk():
    """Test that single chunk returns iterator as-is."""
    data = iter([1, 2, 3, 4, 5])
    model = ["single_chunk"]

    result = make_data_iterator_list(model, data)

    # Should return the same iterator
    assert result is data
    assert next(result) == 1
    assert next(result) == 2


def test_make_data_iterator_list_multiple_chunks():
    """Test that multiple chunks get caching iterators."""
    data = iter([{"batch": 1}, {"batch": 2}, {"batch": 3}])
    model = ["chunk1", "chunk2", "chunk3"]  # 3 virtual pipeline stages

    result = make_data_iterator_list(model, data)

    # Should return a list of iterators
    assert isinstance(result, list)
    assert len(result) == 3

    # All iterators should yield the same data (cached)
    batch1_chunk0 = next(result[0])  # Main iterator fetches and caches
    batch1_chunk1 = next(result[1])  # Proxy reads from cache
    batch1_chunk2 = next(result[2])  # Proxy reads from cache

    assert batch1_chunk0 == {"batch": 1}
    assert batch1_chunk1 == {"batch": 1}
    assert batch1_chunk2 == {"batch": 1}

    # Next batch
    batch2_chunk0 = next(result[0])
    batch2_chunk1 = next(result[1])
    batch2_chunk2 = next(result[2])

    assert batch2_chunk0 == {"batch": 2}
    assert batch2_chunk1 == {"batch": 2}
    assert batch2_chunk2 == {"batch": 2}


def test_make_data_iterator_list_two_chunks():
    """Test with 2 virtual pipeline stages."""
    data = iter(range(10))
    model = ["chunk1", "chunk2"]

    result = make_data_iterator_list(model, data)

    assert isinstance(result, list)
    assert len(result) == 2

    # Iterate through several batches
    for expected_val in range(5):
        val0 = next(result[0])
        val1 = next(result[1])
        assert val0 == expected_val
        assert val1 == expected_val


def test_make_data_iterator_list_empty_model_list():
    """Test with empty model list."""
    data = iter([1, 2, 3])
    model = []

    result = make_data_iterator_list(model, data)

    # Empty list should return the iterator as-is
    assert result is data


def test_make_data_iterator_list_exhaust_iterator():
    """Test that StopIteration propagates correctly."""
    data = iter([1, 2])
    model = ["chunk1", "chunk2"]

    result = make_data_iterator_list(model, data)

    # Consume all data
    assert next(result[0]) == 1
    assert next(result[1]) == 1
    assert next(result[0]) == 2
    assert next(result[1]) == 2

    # Should raise StopIteration
    with pytest.raises(StopIteration):
        next(result[0])
