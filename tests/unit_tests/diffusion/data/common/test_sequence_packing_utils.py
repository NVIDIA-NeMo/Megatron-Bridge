# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

import pytest

from megatron.bridge.data.packing.algorithms import first_fit_decreasing
from megatron.bridge.diffusion.data.common.sequence_packing_utils import packing_length


def _linear_scan_first_fit_decreasing(lengths, pack_size):
    """Reference: the original O(N^2) linear-scan first-fit-decreasing the diffusion packer used."""
    res = []
    for s in sorted(lengths, reverse=True):
        placed = False
        for abin in res:
            if sum(abin) + s <= pack_size:
                abin.append(s)
                placed = True
                break
        if not placed:
            res.append([s])
    return res


class _Sample:
    """Stand-in for DiffusionSample's bin-packing contract.

    The diffusion task encoder packs live sample objects, keying capacity on the padded
    query sequence length when one is set and the unpadded length otherwise (this mirrors
    DiffusionSample.__radd__). This reproduces just that contract, without pulling in
    torch or energon.
    """

    def __init__(self, seq_len_q: int, *, seq_len_q_padded: int | None = None, uid: int = 0):
        self.seq_len_q = seq_len_q
        self.seq_len_q_padded = seq_len_q_padded
        self.uid = uid

    def __radd__(self, other: Any) -> int:
        if isinstance(other, int):
            return self.length + other
        raise NotImplementedError

    @property
    def length(self) -> int:
        return self.seq_len_q_padded if self.seq_len_q_padded is not None else self.seq_len_q


def test_packing_length_handles_plain_ints():
    assert packing_length(42) == 42
    assert packing_length(0) == 0


def test_packing_length_prefers_padded_query_length():
    """packing_length must match DiffusionSample: the padded length wins when it is set."""
    assert packing_length(_Sample(100)) == 100
    assert packing_length(_Sample(100, seq_len_q_padded=128)) == 128


class TestDiffusionEncoderPacking:
    """The diffusion task encoder call pattern: pack sample objects through the shared packer.

    select_samples_to_pack() calls the shared first_fit_decreasing with item_lengths derived
    from packing_length, so it is order- and identity-sensitive: it relies on which bin each
    sample lands in. These assert full bin assignments and object identity, not just counts.
    """

    def _pack(self, samples, pack_size):
        return first_fit_decreasing(samples, pack_size, item_lengths=[packing_length(s) for s in samples])

    def test_returns_original_sample_objects(self):
        samples = [_Sample(length, uid=i) for i, length in enumerate([5, 3, 2, 7, 4])]
        result = self._pack(samples, 10)

        assert all(isinstance(s, _Sample) for b in result for s in b)
        # Sorted desc -> 7,5,4,3,2 (uids 3,0,4,1,2); first-fit-decreasing packs
        # [7,3],[5,4],[2] -> uids [[3,1],[0,4],[2]].
        assert [[s.uid for s in b] for b in result] == [[3, 1], [0, 4], [2]]

    def test_uses_padded_length_for_capacity(self):
        """Padded length drives capacity: unpadded would pack two per bin, padded does not."""
        samples = [_Sample(100, seq_len_q_padded=128, uid=i) for i in range(3)]
        result = self._pack(samples, 200)
        assert all(len(b) == 1 for b in result)

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_linear_scan_reference(self, seed):
        rng = random.Random(seed)
        pack_size = rng.choice([256, 4096, 8192])
        lengths = [rng.randint(0, pack_size // 2 + 50) for _ in range(rng.choice([1, 40, 300]))]
        samples = [_Sample(length, uid=i) for i, length in enumerate(lengths)]

        packed_lengths = [[s.length for s in b] for b in self._pack(samples, pack_size)]
        assert packed_lengths == _linear_scan_first_fit_decreasing(lengths, pack_size)

    def test_every_sample_placed_exactly_once(self):
        rng = random.Random(7)
        samples = [_Sample(rng.randint(1, 2000), uid=i) for i in range(200)]

        placed = sorted(s.uid for b in self._pack(samples, 4096) for s in b)
        assert placed == list(range(200))

    def test_bins_respect_capacity(self):
        rng = random.Random(11)
        pack_size = 4096
        samples = [_Sample(rng.randint(1, pack_size), uid=i) for i in range(200)]

        for abin in self._pack(samples, pack_size):
            assert sum(s.length for s in abin) <= pack_size
