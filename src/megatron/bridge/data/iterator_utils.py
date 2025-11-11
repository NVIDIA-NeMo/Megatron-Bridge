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

import queue
from typing import Iterator, TypeVar, Union


DataT = TypeVar("DataT")


def make_data_iterator_list(
    model: list, data_iterator: Iterator[DataT]
) -> Union[Iterator[DataT], list[Iterator[DataT]]]:
    """Convert data iterator into form expected by Megatron with virtual pipeline parallelism.

    With interleaved/virtual pipeline parallelism, Megatron expects a list of one data
    iterator per model chunk. Each model chunk independently gets data from its data
    iterator, so we need to interact with the data iterator multiple times for each
    microbatch step. Instead of incorporating this logic into the data loader, we cache
    the iterator's output to the first model chunk and reuse it in the other model chunks.

    Args:
        model: List of model chunks (when virtual PP is used) or single-element list
        data_iterator: Iterator yielding microbatch data

    Returns:
        If model has only 1 chunk: returns the iterator as-is
        If model has multiple chunks: returns a list of iterators with caching behavior
            - First iterator in list consumes from data_iterator and caches values
            - Remaining iterators are proxies that read from the cache

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

    class CachingIterator:
        """Iterator wrapper that caches values for proxy iterators.

        When the main iterator is advanced, it caches the value and distributes
        it to all registered proxy iterators.
        """

        class Proxy:
            """Proxy iterator that reads from the cache.

            Assumed to never advance past the caching iterator.
            """

            def __init__(self):
                self.cache = queue.Queue()

            def __iter__(self):
                return self

            def __next__(self):
                return self.cache.get_nowait()

        def __init__(self, iterator: Iterator[DataT]):
            self.iterator = iterator
            self.proxies = []

        def make_proxy(self):
            """Create a new proxy iterator that reads from this cache."""
            self.proxies.append(CachingIterator.Proxy())
            return self.proxies[-1]

        def __iter__(self):
            return self

        def __next__(self):
            """Advance the main iterator and cache the value for all proxies."""
            val = next(self.iterator)
            for proxy in self.proxies:
                proxy.cache.put(val)
            return val

    # Create list of iterator wrappers - one per model chunk
    # First iterator is the main caching iterator
    # Remaining iterators are proxies that read from the cache
    iters = [CachingIterator(data_iterator)]
    while len(iters) < len(model):
        iters.append(iters[0].make_proxy())

    return iters
