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

import numpy as np
import pytest
import torch

from megatron.bridge.data.packing.ragged_store import RaggedStore, RaggedStoreWriter


SCHEMA = ("input_ids", "loss_mask", "answer_start_idx")


def _make_items(n: int, with_loss_mask: bool = True, with_answer_start: bool = False):
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        item = {"input_ids": rng.integers(0, 1000, size=rng.integers(1, 20)).tolist()}
        if with_loss_mask:
            item["loss_mask"] = [bool(x) for x in rng.integers(0, 2, size=len(item["input_ids"]))]
        if with_answer_start:
            item["answer_start_idx"] = int(rng.integers(0, len(item["input_ids"])))
        items.append(item)
    return items


def test_ragged_store_roundtrip_variable_lengths(tmp_path):
    items = _make_items(37)
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=10)
    writer.append_many(items)
    store = writer.finalize()

    assert store.num_items == 37
    for i, item in enumerate(items):
        np.testing.assert_array_equal(store.column("input_ids").get(i), item["input_ids"])
        np.testing.assert_array_equal(store.column("loss_mask").get(i), item["loss_mask"])


def test_ragged_store_persists_across_reopen(tmp_path):
    items = _make_items(12)
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=5)
    writer.append_many(items)
    writer.finalize()

    reopened = RaggedStore.open(tmp_path / "store")
    assert reopened.num_items == 12
    for i, item in enumerate(items):
        np.testing.assert_array_equal(reopened.column("input_ids").get(i), item["input_ids"])


def test_ragged_store_missing_keys_are_zero_length(tmp_path):
    items = _make_items(6, with_loss_mask=False, with_answer_start=True)
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=4)
    writer.append_many(items)
    store = writer.finalize()

    lengths = store.column("loss_mask").lengths
    assert np.all(lengths == 0)
    asidx = store.column("answer_start_idx").lengths
    assert np.all(asidx == 1)
    np.testing.assert_array_equal(
        store.column("answer_start_idx").gather(np.arange(6)).reshape(-1), [item["answer_start_idx"] for item in items]
    )


def test_ragged_store_empty(tmp_path):
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=8)
    store = writer.finalize()

    assert store.num_items == 0
    assert len(store.column("input_ids").lengths) == 0


def test_ragged_store_gather_equal_lengths(tmp_path):
    items = [{"input_ids": [i, i + 1, i + 2], "loss_mask": [True, False, True]} for i in range(5)]
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=2)
    writer.append_many(items)
    store = writer.finalize()

    gathered = store.column("input_ids").gather(np.array([4, 0, 2]))
    np.testing.assert_array_equal(gathered, [[4, 5, 6], [0, 1, 2], [2, 3, 4]])


def test_ragged_store_gather_unequal_lengths_raises(tmp_path):
    items = _make_items(4)
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=2)
    writer.append_many(items)
    store = writer.finalize()

    with pytest.raises(ValueError, match="equal length"):
        store.column("input_ids").gather(np.arange(4))


def test_ragged_store_accepts_torch_tensors(tmp_path):
    items = [
        {
            "input_ids": torch.LongTensor([5, 6, 7]),
            "loss_mask": torch.BoolTensor([False, True, True]),
        }
    ]
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=2)
    writer.append_many(items)
    store = writer.finalize()

    np.testing.assert_array_equal(store.column("input_ids").get(0), [5, 6, 7])
    np.testing.assert_array_equal(store.column("loss_mask").get(0), [False, True, True])
    assert store.column("loss_mask").data.dtype == np.bool_


def test_ragged_store_preserves_bool_dtype(tmp_path):
    items = [{"input_ids": [1, 2], "loss_mask": [True, False]}]
    writer = RaggedStoreWriter(tmp_path / "store", SCHEMA, chunk_size=2)
    writer.append_many(items)
    store = writer.finalize()

    assert store.column("loss_mask").data.dtype == np.bool_
    assert store.column("loss_mask").get(0).tolist() == [True, False]
