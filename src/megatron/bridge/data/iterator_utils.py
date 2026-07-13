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

"""Iterator utilities for handling virtual pipeline parallelism."""

from typing import Iterator, TypeVar, Union


DataT = TypeVar("DataT")


def make_data_iterator_list(
    model: list, data_iterator: Iterator[DataT]
) -> Union[Iterator[DataT], list[Iterator[DataT]]]:
    """Convert data iterator into form expected by Megatron with virtual pipeline parallelism.

    With interleaved/virtual pipeline parallelism, Megatron expects a list of one data
    iterator per model chunk. Each model chunk independently gets data from its data
    iterator, so we need to interact with the data iterator multiple times for each
    microbatch step. Instead of incorporating this logic into the data loader, we lazily
    cache the source iterator by microbatch index and expose one cache view per model chunk.

    Args:
        model: List of model chunks (when virtual PP is used) or single-element list
        data_iterator: Iterator yielding microbatch data

    Returns:
        If model has only 1 chunk: returns the iterator as-is
        If model has multiple chunks: returns a list of iterators with caching behavior
            - Each iterator can advance independently
            - All iterators see the same microbatch sequence regardless of scheduler order

    Example:
        >>> # With virtual PP size = 2 (model has 2 chunks)
        >>> iters = make_data_iterator_list(model=[chunk1, chunk2], data_iterator=iter(microbatches))
        >>> # len(iters) == 2
        >>> # Both iters[0] and iters[1] will yield the same microbatch data
        >>> batch_from_chunk0 = next(iters[0])  # Fetches from data_iterator, caches
        >>> batch_from_chunk1 = next(iters[1])  # Reads from cache, same data
    """
    # Single model chunk - no caching needed
    if not isinstance(model, list) or len(model) <= 1:
        return data_iterator

    class SharedCache:
        """Shared lazy cache over an iterator.

        Any model chunk may be scheduled first. Fetching by index fills the cache
        up to that position, so all chunk iterators see the same microbatch
        sequence without assuming chunk 0 advances first.
        """

        def __init__(self, iterator: Iterator[DataT]):
            self.iterator = iterator
            self.cache: list[DataT] = []

        def get(self, index: int) -> DataT:
            """Return the cached value at index, reading the source iterator if needed."""
            while len(self.cache) <= index:
                self.cache.append(next(self.iterator))
            return self.cache[index]

    class CacheView:
        """Per-model-chunk iterator view into a shared cache."""

        def __init__(self, cache: SharedCache):
            self.cache = cache
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            """Return the next value for this chunk from the shared cache."""
            val = self.cache.get(self.index)
            self.index += 1
            return val

    shared_cache = SharedCache(data_iterator)
    return [CacheView(shared_cache) for _ in model]
