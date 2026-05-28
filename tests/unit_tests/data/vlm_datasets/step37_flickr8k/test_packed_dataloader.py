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

from megatron.bridge.data.vlm_datasets.step37_flickr8k.packed_dataloader import (
    MixedPackedDataloader,
)


pytestmark = pytest.mark.unit


class _Item:
    """A fake sample: ``len(item)`` is its packed-NTP length."""

    def __init__(self, ntp_len: int, tag: int):
        self._n = ntp_len
        self.tag = tag

    def __len__(self) -> int:
        return self._n


class _FakeDataset:
    def __init__(self, sizes):
        self._items = [_Item(n, i) for i, n in enumerate(sizes)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)


def _tags_transform(items):
    return [it.tag for it in items]


def test_sequential_schedule_is_deterministic():
    ds = _FakeDataset([2, 2, 2, 2])
    loader = MixedPackedDataloader(
        datasets=[ds],
        epochs=[1.0],
        max_length=4,
        oversize_policy="extend",
        transform=_tags_transform,
        dataset_sampling="sequential",
    )

    # 4 samples of size 2, packed to max_length 4 -> two packs of two samples.
    assert len(loader) == 2
    assert loader[0] == [0, 1]
    assert loader[1] == [2, 3]
    # Map-style: same index returns the same pack on every call.
    assert loader[0] == loader[0]


def test_empty_datasets_raise():
    with pytest.raises(ValueError):
        MixedPackedDataloader(datasets=[], epochs=[], max_length=4)


def test_datasets_epochs_length_mismatch_raises():
    with pytest.raises(ValueError):
        MixedPackedDataloader(datasets=[_FakeDataset([2])], epochs=[1.0, 1.0], max_length=4)


def test_nonpositive_epoch_raises():
    with pytest.raises(ValueError):
        MixedPackedDataloader(datasets=[_FakeDataset([2, 2])], epochs=[0.0], max_length=4)


def test_all_samples_dropped_raises_empty_pack():
    # Single sample longer than max_length with drop policy -> nothing packed.
    with pytest.raises(ValueError):
        MixedPackedDataloader(
            datasets=[_FakeDataset([5])],
            epochs=[1.0],
            max_length=4,
            oversize_policy="drop",
            dataset_sampling="sequential",
        )


def test_normalize_dataset_sampling_validates_strategy():
    with pytest.raises(ValueError):
        MixedPackedDataloader._normalize_dataset_sampling("bogus", 1)
    assert MixedPackedDataloader._normalize_dataset_sampling("random", 2) == [
        "random",
        "random",
    ]
