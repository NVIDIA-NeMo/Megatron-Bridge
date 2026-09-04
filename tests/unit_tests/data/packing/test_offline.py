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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.bridge.data.packing.offline import (
    _materialize_dataset_items,
    _pre_pad_data_point,
    prepare_gpt_sft_packed_data,
    tokenize_dataset,
)


PAD_ID = 0


def test_configured_seed_controls_offline_packing(monkeypatch):
    """Identical input and configured seed must produce identical packed rows."""

    class TinyDataset:
        tokenizer = SimpleNamespace(eod=PAD_ID)
        pad_seq_length_to_mult = 1

        def __init__(self):
            runtime_lengths = (6, 6, 4, 4)
            self.items = [
                {
                    "input_ids": [sample_id * 10 + token_id for token_id in range(runtime_length + 1)],
                    "loss_mask": [True] * (runtime_length + 1),
                }
                for sample_id, runtime_length in enumerate(runtime_lengths)
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def prepare_with_ambient_seed(ambient_seed):
        packed_outputs = []
        np.random.seed(ambient_seed)
        real_save = np.save

        def fake_save(file, arr, *args, **kwargs):
            # Let chunk saves inside _materialize_dataset_items write to disk,
            # but capture the final output save from prepare_gpt_sft_packed_data.
            if "chunk_" in str(file):
                real_save(file, arr, *args, **kwargs)
            else:
                packed_outputs.append(arr)

        monkeypatch.setattr(np, "save", fake_save)
        prepare_gpt_sft_packed_data(
            input_path=Path("unused.jsonl"),
            output_path=Path("unused.npy"),
            output_metadata_path=None,
            packed_sequence_size=10,
            tokenizer=SimpleNamespace(eos_id=PAD_ID),
            max_seq_length=10,
            seed=777,
            packing_algorithm="first_fit_shuffle",
            num_tokenizer_workers=1,
            dataset_builder=lambda *args, **kwargs: TinyDataset(),
        )
        return packed_outputs[0]

    assert prepare_with_ambient_seed(0) == prepare_with_ambient_seed(4)


def test_pre_pad_data_point_chat_tensors_do_not_raise():
    """Chat path returns torch tensors; padding must not raise TypeError (see issue #2610)."""
    data = {
        "input_ids": torch.LongTensor([5, 6, 7]),
        "loss_mask": torch.BoolTensor([False, True, True]),
        "context_ids": torch.LongTensor([5, 6]),
    }
    # stored max_stored_length_to_pad=9 -> input_ids padded to length 9
    _pre_pad_data_point(data, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert isinstance(data["input_ids"], list)
    assert isinstance(data["loss_mask"], list)
    # loss_mask must end up the same length as input_ids, otherwise fill_packing_strategy's
    # np.array([...loss_mask...]) raises an inhomogeneous-shape error when samples are grouped.
    assert len(data["loss_mask"]) == len(data["input_ids"])
    # padded loss_mask positions carry 0 (no loss on pad tokens)
    assert data["loss_mask"][3:] == [0] * (len(data["loss_mask"]) - 3)
    assert data["input_ids"][3:] == [PAD_ID] * (len(data["input_ids"]) - 3)


def test_pre_pad_data_point_equalizes_loss_mask_lengths():
    """Two samples that round to the same padded input length must get equal-length loss_masks."""
    a = {"input_ids": torch.LongTensor([1, 2, 3]), "loss_mask": torch.BoolTensor([False, True, True])}
    b = {
        "input_ids": torch.LongTensor([1, 2, 3, 4, 5]),
        "loss_mask": torch.BoolTensor([False, False, True, True, True]),
    }
    # both round up to the same multiple-of-8 target
    _pre_pad_data_point(a, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)
    _pre_pad_data_point(b, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert len(a["input_ids"]) == len(b["input_ids"])
    assert len(a["loss_mask"]) == len(b["loss_mask"]) == len(a["input_ids"])


def test_pre_pad_data_point_non_chat_lists_still_work():
    """Non-chat (GPTSFTDataset) path returns plain lists without loss_mask; must be unaffected."""
    data = {"input_ids": [9, 9, 9], "context_ids": [9, 9]}
    _pre_pad_data_point(data, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert data["input_ids"] == [9, 9, 9] + [PAD_ID] * 6
    assert "loss_mask" not in data


def test_pre_pad_data_point_truncates_overlong_to_target_plus_one():
    """Overlong sequences retain a CP-divisible runtime length after truncation."""
    data = {"input_ids": list(range(20)), "loss_mask": [1] * 20}
    _pre_pad_data_point(data, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert len(data["input_ids"]) == 9
    assert len(data["loss_mask"]) == len(data["input_ids"])
    assert (len(data["input_ids"]) - 1) % 8 == 0
    assert data["input_ids"][-1] == 8


def test_pre_pad_data_point_near_pack_size_trims_to_target_plus_one():
    """Near-pack-size sequences retain a CP-divisible runtime length without overflowing."""
    data = {"input_ids": list(range(12)), "loss_mask": [1] * 12}
    _pre_pad_data_point(data, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert len(data["input_ids"]) == 9
    assert len(data["loss_mask"]) == len(data["input_ids"])
    assert (len(data["input_ids"]) - 1) % 8 == 0
    assert data["input_ids"][-1] == 8


def test_pre_pad_data_point_keeps_already_divisible_stored_length():
    """A stored length of runtime multiple + 1 must not add another padding bucket."""
    data = {"input_ids": list(range(17)), "loss_mask": [1] * 17}
    _pre_pad_data_point(data, max_seq_length=40, max_stored_length_to_pad=17, pad_id=PAD_ID)

    assert data["input_ids"] == list(range(17))
    assert len(data["loss_mask"]) == 17
    assert (len(data["input_ids"]) - 1) % 8 == 0


def test_tokenize_dataset_caps_runtime_padding_target_to_pack_size():
    """CP padding should let runtime length reach the divisible pack-size cap."""
    factory_kwargs = {}

    class TinyDataset:
        tokenizer = SimpleNamespace(eod=PAD_ID)
        pad_seq_length_to_mult = 8
        max_seq_length = 17

        def __init__(self):
            self.items = [{"input_ids": list(range(length))} for length in (16, 17, 20, 21)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def fake_build_sft_split(*args, **kwargs):
        factory_kwargs.update(kwargs)
        return TinyDataset()

    dataset = tokenize_dataset(
        Path("unused.jsonl"),
        tokenizer=object(),
        max_seq_length=16,
        seed=123,
        pad_seq_to_mult=8,
        num_tokenizer_workers=1,
        dataset_builder=fake_build_sft_split,
    )

    assert [len(item["input_ids"]) for item in dataset] == [17, 17, 17, 17]
    assert all((len(item["input_ids"]) - 1) % 8 == 0 for item in dataset)
    assert max(len(item["input_ids"]) - 1 for item in dataset) == 16
    assert factory_kwargs["seq_length"] == 17


def test_tokenize_dataset_ceil_uses_runtime_length_not_stored_length():
    """Stored length runtime multiple + 1 should not be rounded up by stored length."""

    class TinyDataset:
        tokenizer = SimpleNamespace(eod=PAD_ID)
        pad_seq_length_to_mult = 8
        max_seq_length = 40

        def __init__(self):
            self.items = [{"input_ids": list(range(length))} for length in (17, 18)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = tokenize_dataset(
        Path("unused.jsonl"),
        tokenizer=object(),
        max_seq_length=40,
        seed=123,
        pad_seq_to_mult=8,
        num_tokenizer_workers=1,
        dataset_builder=lambda *args, **kwargs: TinyDataset(),
    )

    assert [len(item["input_ids"]) for item in dataset] == [17, 25]
    assert all((len(item["input_ids"]) - 1) % 8 == 0 for item in dataset)


def test_tokenize_dataset_rejects_padding_multiple_without_positive_target():
    """Padding must not silently reduce every sample to a zero-token runtime segment."""

    class TinyDataset:
        tokenizer = SimpleNamespace(eod=PAD_ID)
        pad_seq_length_to_mult = 8
        max_seq_length = 7

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise AssertionError("invalid padding should fail before materializing samples")

    with pytest.raises(ValueError, match="must be at least the effective padding multiple"):
        tokenize_dataset(
            Path("unused.jsonl"),
            tokenizer=object(),
            max_seq_length=7,
            seed=123,
            pad_seq_to_mult=8,
            num_tokenizer_workers=1,
            dataset_builder=lambda *args, **kwargs: TinyDataset(),
        )


def test_materialize_dataset_items_uses_serial_path_for_non_positive_workers(monkeypatch):
    """Non-positive worker counts should not create a multiprocessing pool."""

    class TinyDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index + 10

    import multiprocessing as mp

    def fail_pool(*args, **kwargs):
        raise AssertionError("Pool should not be constructed for non-positive worker counts")

    monkeypatch.setattr(mp, "Pool", fail_pool)

    assert _materialize_dataset_items(TinyDataset(), -1).tolist() == [10, 11, 12]
    assert _materialize_dataset_items(TinyDataset(), 0).tolist() == [10, 11, 12]


def test_materialize_dataset_items_empty_dataset():
    """An empty dataset must return an empty array, not raise from concatenate."""

    class TinyDataset:
        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise AssertionError("no items should be fetched from an empty dataset")

    result = _materialize_dataset_items(TinyDataset(), -1)
    assert result.shape == (0,)


def test_materialize_dataset_items_parallel_returns_correct_items():
    """The parallel (fork) path should return all items correctly."""

    class TinyDataset:
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return index + 100

    result = _materialize_dataset_items(TinyDataset(), 2)
    assert sorted(result.tolist()) == [100, 101, 102, 103, 104]


def test_materialize_dataset_items_preserves_dict_items():
    """Dict items (like tokenized samples) must survive chunked serialization."""

    class TinyDataset:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"input_ids": np.array([index, index + 1]), "loss_mask": np.array([1, 0])}

    # Serial
    result_serial = _materialize_dataset_items(TinyDataset(), num_workers=None)
    assert len(result_serial) == 4
    assert np.array_equal(result_serial[0]["input_ids"], [0, 1])

    # Parallel
    result_parallel = _materialize_dataset_items(TinyDataset(), 2)
    assert len(result_parallel) == 4
    # Items may be reordered in parallel mode; check by lookup
    ids = {tuple(r["input_ids"].tolist()) for r in result_parallel}
    assert ids == {(0, 1), (1, 2), (2, 3), (3, 4)}
