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

"""Parity tests: ragged packing pipeline must match the legacy object-array pipeline."""

import numpy as np
import pytest

from megatron.bridge.data.packing.algorithms import (
    create_hist,
    create_hist_from_lengths,
    create_packing_strategy,
    fill_packing_strategy,
    fill_packing_strategy_ragged,
)
from megatron.bridge.data.packing.offline import _materialize_dataset_items


SEED = 1234
MAX_SEQ_LENGTH = 32
PACK_SIZE = 24
EOS_ID = 7


def _make_items(num_items: int, mask_style: str = "bool"):
    """Build tokenized-looking samples with a spread of lengths.

    mask_style: "bool" -> loss_mask on every item, "mixed" -> some items carry
    answer_start_idx instead of loss_mask (exercises the fallback path),
    "int" -> integer 0/1 loss masks.
    """
    rng = np.random.default_rng(SEED)
    items = []
    for i in range(num_items):
        # Cycle through lengths including 1 (runtime len 0) and over-length items.
        runtime_length = [3, 7, 11, 15, 40, 5, 9, 21][i % 8]
        input_ids = rng.integers(10, 100, size=runtime_length + 1).tolist()
        item = {"input_ids": input_ids}
        if mask_style == "bool":
            item["loss_mask"] = [bool(x) for x in rng.integers(0, 2, size=runtime_length + 1)]
        elif mask_style == "int":
            item["loss_mask"] = [int(x) for x in rng.integers(0, 2, size=runtime_length + 1)]
        elif mask_style == "mixed" and runtime_length in (11, 15):
            item["answer_start_idx"] = int(rng.integers(2, runtime_length + 1))
        else:
            item["loss_mask"] = [bool(x) for x in rng.integers(0, 2, size=runtime_length + 1)]
        items.append(item)
    return items


class _ListDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _legacy_packed_output(items, seed):
    dataset = np.array([dict(item) for item in items], dtype=object)
    sequences, histogram = create_hist(dataset, MAX_SEQ_LENGTH)
    np.random.seed(seed)
    assignments, _ = create_packing_strategy(histogram, PACK_SIZE, "first_fit_shuffle")
    return fill_packing_strategy(assignments, sequences, PACK_SIZE, EOS_ID)


def _ragged_packed_output(items, seed, num_workers=1):
    store = _materialize_dataset_items(_ListDataset(items), num_workers)
    try:
        groups, histogram = create_hist_from_lengths(store.input_ids_lengths() - 1, MAX_SEQ_LENGTH)
        np.random.seed(seed)
        assignments, _ = create_packing_strategy(histogram, PACK_SIZE, "first_fit_shuffle")
        return fill_packing_strategy_ragged(assignments, store, groups, PACK_SIZE, EOS_ID)
    finally:
        store.close()


def _packed_npy_bytes(output_data):
    import io

    buffer = io.BytesIO()
    np.save(buffer, output_data, allow_pickle=True)
    return buffer.getvalue()


@pytest.mark.parametrize("mask_style", ["bool", "int", "mixed"])
def test_ragged_pipeline_matches_legacy_output(mask_style):
    """Same input + seed must produce identical packed rows in both pipelines."""
    items = _make_items(200, mask_style=mask_style)

    legacy = _legacy_packed_output(items, SEED)
    ragged = _ragged_packed_output(items, SEED)

    assert len(ragged) == len(legacy) > 0
    for legacy_bin, ragged_bin in zip(legacy, ragged):
        assert list(legacy_bin.keys()) == list(ragged_bin.keys())
        assert legacy_bin["input_ids"] == ragged_bin["input_ids"]
        assert legacy_bin["loss_mask"] == ragged_bin["loss_mask"]
        assert legacy_bin["seq_start_id"] == ragged_bin["seq_start_id"]


def test_ragged_pipeline_matches_legacy_npy_bytes():
    """The serialized packed artifact must be byte-identical between pipelines."""
    items = _make_items(200, mask_style="bool")

    legacy = _legacy_packed_output(items, SEED)
    ragged = _ragged_packed_output(items, SEED)

    assert _packed_npy_bytes(legacy) == _packed_npy_bytes(ragged)


@pytest.mark.parametrize("num_workers", [2, 4])
def test_ragged_pipeline_matches_legacy_output_parallel(num_workers):
    """The fork-based parallel materialization must feed the same packing output."""
    items = _make_items(200, mask_style="bool")

    legacy = _legacy_packed_output(items, SEED)
    ragged = _ragged_packed_output(items, SEED, num_workers=num_workers)

    assert len(ragged) == len(legacy) > 0
    for legacy_bin, ragged_bin in zip(legacy, ragged):
        assert legacy_bin["input_ids"] == ragged_bin["input_ids"]
        assert legacy_bin["loss_mask"] == ragged_bin["loss_mask"]
        assert legacy_bin["seq_start_id"] == ragged_bin["seq_start_id"]


@pytest.mark.parametrize("num_workers", [1, 3])
def test_ragged_pipeline_matches_legacy_output_multi_chunk(monkeypatch, num_workers):
    """Chunk boundaries (multiple pools and store segments) must not change the output."""
    from megatron.bridge.data.packing import offline as offline_module

    monkeypatch.setattr(offline_module, "_MATERIALIZE_CHUNK_SIZE", 7)
    items = _make_items(50, mask_style="bool")

    legacy = _legacy_packed_output(items, SEED)
    ragged = _ragged_packed_output(items, SEED, num_workers=num_workers)

    assert len(ragged) == len(legacy) > 0
    for legacy_bin, ragged_bin in zip(legacy, ragged):
        assert legacy_bin["input_ids"] == ragged_bin["input_ids"]
        assert legacy_bin["loss_mask"] == ragged_bin["loss_mask"]
        assert legacy_bin["seq_start_id"] == ragged_bin["seq_start_id"]


def test_ragged_pipeline_raises_when_neither_mask_nor_answer_start():
    """A length group with neither loss_mask nor answer_start_idx must raise like the legacy path."""

    class TinyDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"input_ids": [1, 2, 3]}

    store = _materialize_dataset_items(TinyDataset(), 1)
    try:
        groups, histogram = create_hist_from_lengths(store.input_ids_lengths() - 1, MAX_SEQ_LENGTH)
        np.random.seed(SEED)
        assignments, _ = create_packing_strategy(histogram, PACK_SIZE, "first_fit_shuffle")
        with pytest.raises(ValueError, match="loss_mask and answer_start_idx missing"):
            fill_packing_strategy_ragged(assignments, store, groups, PACK_SIZE, EOS_ID)
    finally:
        store.close()
