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

from megatron.bridge.data.vlm_datasets.step37_flickr8k.samplers import (
    LoopedSequentialSampler,
    LoopedShuffleSampler,
    WeightedRandomSampler,
)


pytestmark = pytest.mark.unit


def _take(sampler, n):
    return [next(sampler) for _ in range(n)]


# ─── LoopedSequentialSampler ────────────────────────────────────────────────


def test_sequential_loops_in_order():
    sampler = LoopedSequentialSampler(3)
    assert _take(sampler, 7) == [0, 1, 2, 0, 1, 2, 0]


# ─── LoopedShuffleSampler ───────────────────────────────────────────────────


def test_shuffle_epoch_is_a_permutation():
    sampler = LoopedShuffleSampler(size=4, base_seed=7)
    epoch = _take(sampler, 4)
    assert sorted(epoch) == [0, 1, 2, 3]


def test_shuffle_is_deterministic_for_same_seed():
    a = LoopedShuffleSampler(size=5, base_seed=99)
    b = LoopedShuffleSampler(size=5, base_seed=99)
    assert _take(a, 10) == _take(b, 10)


def test_shuffle_same_order_for_each_epoch():
    sampler = LoopedShuffleSampler(size=4, base_seed=7, same_order_for_each_epoch=True)
    first = _take(sampler, 4)
    second = _take(sampler, 4)
    assert first == second


def test_shuffle_state_dict_roundtrip():
    sampler = LoopedShuffleSampler(size=6, base_seed=1234)
    _take(sampler, 4)  # advance mid-epoch
    state = sampler.state_dict()

    restored = LoopedShuffleSampler(size=6, base_seed=1234)
    restored.load_state_dict(state)
    assert _take(restored, 8) == _take(sampler, 8)


# ─── WeightedRandomSampler ──────────────────────────────────────────────────


def test_weighted_uniform_round_robins():
    sampler = WeightedRandomSampler(size=4)  # uniform weights
    assert _take(sampler, 8) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_weighted_respects_weight_ratio():
    sampler = WeightedRandomSampler(size=2, weights=[1.0, 3.0])
    drawn = _take(sampler, 8)
    assert drawn.count(0) == 2
    assert drawn.count(1) == 6


def test_weighted_rejects_bad_weights_length():
    with pytest.raises(ValueError):
        WeightedRandomSampler(size=3, weights=[1.0, 1.0])


def test_weighted_rejects_nonpositive_weights():
    with pytest.raises(ValueError):
        WeightedRandomSampler(size=2, weights=[1.0, 0.0])


def test_weighted_state_dict_roundtrip():
    sampler = WeightedRandomSampler(size=2, weights=[1.0, 3.0])
    _take(sampler, 3)
    state = sampler.state_dict()
    assert state["counts"] == pytest.approx([1.0, 2.0])

    restored = WeightedRandomSampler(size=2, weights=[1.0, 3.0])
    restored.load_state_dict(state)
    assert _take(restored, 5) == _take(sampler, 5)
