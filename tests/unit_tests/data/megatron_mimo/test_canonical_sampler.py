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

"""Unit tests for canonical-grid batch sampling (MegatronMIMO scalable reads)."""

import pytest

from megatron.bridge.data.megatron_mimo.canonical_sampler import (
    build_canonical_group_batch_sampler,
    canonical_grid_size,
    covered_canonical_groups,
)


class _FakeDataset:
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


def _windows(sampler, count: int):
    it = iter(sampler)
    return [next(it) for _ in range(count)]


class TestGridHelpers:
    def test_grid_size_is_lcm(self):
        assert canonical_grid_size([1, 2]) == 2
        assert canonical_grid_size([2, 4]) == 4
        assert canonical_grid_size([2, 3]) == 6

    def test_grid_size_rejects_invalid(self):
        with pytest.raises(ValueError, match="positive"):
            canonical_grid_size([2, 0])

    def test_covered_groups_are_contiguous(self):
        assert covered_canonical_groups(0, 1, 2) == [0, 1]
        assert covered_canonical_groups(1, 2, 2) == [1]
        assert covered_canonical_groups(1, 2, 4) == [2, 3]

    def test_covered_groups_rejects_non_divisor(self):
        with pytest.raises(ValueError, match="not divisible"):
            covered_canonical_groups(0, 3, 4)


class TestFactoryValidation:
    def test_rejects_drop_last_false(self):
        with pytest.raises(ValueError, match="drop_last"):
            build_canonical_group_batch_sampler(
                dataloader_type="single",
                dataset=_FakeDataset(16),
                consumed_samples=0,
                micro_batch_size=4,
                grid_size=2,
                groups=[0],
                data_sharding=True,
                drop_last=False,
            )

    def test_rejects_indivisible_micro_batch(self):
        with pytest.raises(ValueError, match="divisible"):
            build_canonical_group_batch_sampler(
                dataloader_type="single",
                dataset=_FakeDataset(16),
                consumed_samples=0,
                micro_batch_size=5,
                grid_size=2,
                groups=[0],
                data_sharding=True,
            )

    def test_rejects_unsupported_dataloader_type(self):
        with pytest.raises(ValueError, match="single.*cyclic"):
            build_canonical_group_batch_sampler(
                dataloader_type="batch",
                dataset=_FakeDataset(16),
                consumed_samples=0,
                micro_batch_size=4,
                grid_size=2,
                groups=[0],
                data_sharding=True,
            )


def _build(groups, *, dataloader_type, total=48, consumed=0, mbs=4, grid=2, data_sharding=True):
    return build_canonical_group_batch_sampler(
        dataloader_type=dataloader_type,
        dataset=_FakeDataset(total),
        consumed_samples=consumed,
        micro_batch_size=mbs,
        grid_size=grid,
        groups=groups,
        data_sharding=data_sharding,
    )


class TestCrossGeometryAlignment:
    """The core invariant: a rank covering groups {0, 1} reads, per window, exactly the
    concatenation of what single-group ranks read — for shuffling samplers included."""

    @pytest.mark.parametrize("data_sharding", [True, False])
    def test_cyclic_merged_equals_concat_of_group_streams(self, data_sharding):
        merged = _build([0, 1], dataloader_type="cyclic", data_sharding=data_sharding)
        group0 = _build([0], dataloader_type="cyclic", data_sharding=data_sharding)
        group1 = _build([1], dataloader_type="cyclic", data_sharding=data_sharding)

        merged_windows = _windows(merged, 6)
        g0_windows = _windows(group0, 6)
        g1_windows = _windows(group1, 6)

        for m, g0, g1 in zip(merged_windows, g0_windows, g1_windows):
            assert m == g0 + g1
            assert len(m) == 4

    def test_cyclic_group_streams_are_disjoint(self):
        group0 = _build([0], dataloader_type="cyclic")
        group1 = _build([1], dataloader_type="cyclic")
        seen0 = {i for w in _windows(group0, 6) for i in w}
        seen1 = {i for w in _windows(group1, 6) for i in w}
        assert seen0.isdisjoint(seen1)

    def test_single_merged_reproduces_contiguous_windows(self):
        merged = _build([0, 1], dataloader_type="single")
        assert _windows(merged, 3) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    def test_single_group_streams_match_slice_semantics(self):
        # Group g's stream equals the [g*2, (g+1)*2) slice of each contiguous window,
        # i.e. exactly what slice_batch_for_megatron_mimo would keep for dp rank g.
        group1 = _build([1], dataloader_type="single")
        assert _windows(group1, 3) == [[2, 3], [6, 7], [10, 11]]

    def test_heterogeneous_grid_four_groups(self):
        # language dp=4 (one group per rank) with vision dp=2 (two groups per rank).
        vision_rank0 = _build([0, 1], dataloader_type="cyclic", grid=4, mbs=8)
        lang_rank0 = _build([0], dataloader_type="cyclic", grid=4, mbs=8)
        lang_rank1 = _build([1], dataloader_type="cyclic", grid=4, mbs=8)
        for m, g0, g1 in zip(_windows(vision_rank0, 4), _windows(lang_rank0, 4), _windows(lang_rank1, 4)):
            assert m == g0 + g1


class TestResume:
    @pytest.mark.parametrize("dataloader_type", ["single", "cyclic"])
    def test_resume_continues_the_stream(self, dataloader_type):
        full = _build([0, 1], dataloader_type=dataloader_type)
        full_windows = _windows(full, 6)

        # Two windows consumed -> global consumed = 2 * micro_batch_size.
        resumed = _build([0, 1], dataloader_type=dataloader_type, consumed=8)
        resumed_windows = _windows(resumed, 4)

        assert resumed_windows == full_windows[2:]

    def test_resume_across_epoch_boundary(self):
        # total=8, window=4 -> 2 windows per epoch. consumed=12 resumes at the
        # second window of epoch 1.
        merged = _build([0, 1], dataloader_type="cyclic", total=8)
        epoch0 = list(iter(merged))
        epoch1 = list(iter(merged))
        full_seq = epoch0 + epoch1

        resumed = _build([0, 1], dataloader_type="cyclic", total=8, consumed=12)
        assert next(iter(resumed)) == full_seq[3]


class TestEpochBoundary:
    def test_cyclic_reiterates_with_new_epoch(self):
        # total=8, window=4 -> 2 windows per epoch; sub-samplers advance their own
        # consumed_samples, so re-iterating the wrapper enters the next epoch.
        merged = _build([0, 1], dataloader_type="cyclic", total=8)
        epoch0 = list(iter(merged))
        assert len(epoch0) == 2
        epoch1 = list(iter(merged))
        assert len(epoch1) == 2
        assert {i for w in epoch0 for i in w} == set(range(8))
        assert {i for w in epoch1 for i in w} == set(range(8))
        # The sub-samplers must have advanced (a replayed epoch 0 would pass the
        # set checks above): consumed grew and the epoch-1 order is a new shuffle.
        assert merged.samplers[0].consumed_samples == 16
        assert epoch1 != epoch0


class TestNonMultipleDatasetSize:
    @pytest.mark.parametrize("data_sharding", [True, False])
    def test_groups_stay_aligned_when_total_is_not_a_multiple(self, data_sharding):
        # 54 % 8 != 0: without truncation the cyclic data_sharding=False stride gives
        # groups unequal window counts ([7, 7, 6, 6]) and desyncs the modules.
        groups = [
            _build([g], dataloader_type="cyclic", total=54, mbs=8, grid=4, data_sharding=data_sharding)
            for g in range(4)
        ]
        counts = [len(list(iter(g))) for g in groups]
        assert counts == [6, 6, 6, 6]

        merged = _build([0, 1, 2, 3], dataloader_type="cyclic", total=54, mbs=8, grid=4, data_sharding=data_sharding)
        windows = list(iter(merged))
        assert len(windows) == 6
        assert all(len(w) == 8 for w in windows)
