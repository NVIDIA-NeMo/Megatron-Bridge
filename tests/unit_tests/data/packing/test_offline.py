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

import megatron.bridge.data.packing.offline as offline_module
from megatron.bridge.data.packing.offline import (
    _get_shared_dataset_item,
    _init_shared_dataset_worker,
    _materialize_dataset_items,
    _pre_pad_data_point,
    prepare_gpt_sft_packed_data,
    tokenize_dataset,
)


PAD_ID = 0


def test_worker_error_identifies_dataset_index():
    class FailingDataset:
        def __getitem__(self, index):
            raise ValueError(f"invalid row {index}")

    _init_shared_dataset_worker(FailingDataset())

    with pytest.raises(ValueError, match="invalid row 17") as error:
        _get_shared_dataset_item(17)

    assert error.value.__notes__ == ["Failed to prepare packed-SFT dataset index 17."]


def test_configured_seed_controls_offline_packing(monkeypatch, tmp_path):
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

        def save_staged(path, rows):
            packed_outputs.append(rows)
            Path(path).write_bytes(b"staged numpy")

        monkeypatch.setattr(np, "save", save_staged)
        prepare_gpt_sft_packed_data(
            input_path=Path("unused.jsonl"),
            output_path=tmp_path / "unused.npy",
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


def test_streaming_packing_rejects_legacy_numpy_output():
    """The bounded writer is specific to Parquet output."""
    with pytest.raises(ValueError, match="requires a Parquet output path"):
        prepare_gpt_sft_packed_data(
            input_path=Path("unused.jsonl"),
            output_path=Path("unused.npy"),
            output_metadata_path=None,
            packed_sequence_size=8,
            tokenizer=SimpleNamespace(eos_id=PAD_ID),
            max_seq_length=8,
            stream_packed_parquet=True,
            dataset_builder=lambda *args, **kwargs: pytest.fail("validation must precede dataset construction"),
        )


def test_streaming_output_is_not_published_when_metadata_write_fails(tmp_path, monkeypatch):
    output_path = tmp_path / "packed.idx.parquet"
    metadata_path = tmp_path / "missing" / "packed.metadata.json"
    monkeypatch.setattr("megatron.bridge.data.packing.offline.tokenize_dataset", lambda *args, **kwargs: [])
    monkeypatch.setattr("megatron.bridge.data.packing.offline.create_hist", lambda *args: ([], {}))
    monkeypatch.setattr(
        "megatron.bridge.data.packing.offline.create_packing_strategy",
        lambda *args: ([], {"packing_efficiency": 100.0}),
    )
    monkeypatch.setattr("megatron.bridge.data.packing.offline.iter_packing_strategy", lambda *args: iter(()))

    def write_staged_output(rows, path, *, _publish_on_completion):
        assert list(rows) == []
        assert _publish_on_completion is False
        Path(path).write_bytes(b"staged parquet")

    monkeypatch.setattr(
        "megatron.bridge.data.packing.parquet.write_packed_parquet_streaming",
        write_staged_output,
    )

    with pytest.raises(FileNotFoundError):
        prepare_gpt_sft_packed_data(
            input_path=Path("unused.jsonl"),
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=8,
            tokenizer=SimpleNamespace(eos_id=PAD_ID),
            max_seq_length=8,
            stream_packed_parquet=True,
            dataset_builder=lambda *args, **kwargs: pytest.fail("tokenization is stubbed"),
        )

    assert not output_path.exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == []


@pytest.mark.parametrize("output_suffix", [".idx.parquet", ".npy"])
def test_materialized_output_is_not_published_when_metadata_write_fails(
    tmp_path,
    monkeypatch,
    output_suffix,
):
    output_path = tmp_path / f"packed{output_suffix}"
    metadata_path = tmp_path / "missing" / "packed.metadata.json"
    monkeypatch.setattr("megatron.bridge.data.packing.offline.tokenize_dataset", lambda *args, **kwargs: [])
    monkeypatch.setattr("megatron.bridge.data.packing.offline.create_hist", lambda *args: ([], {}))
    monkeypatch.setattr(
        "megatron.bridge.data.packing.offline.create_packing_strategy",
        lambda *args: ([], {"packing_efficiency": 100.0}),
    )
    monkeypatch.setattr("megatron.bridge.data.packing.offline.fill_packing_strategy", lambda *args: [])

    if output_suffix == ".idx.parquet":
        monkeypatch.setattr(
            "megatron.bridge.data.packing.parquet.write_packed_parquet",
            lambda rows, path: Path(path).write_bytes(b"staged parquet"),
        )
    else:
        monkeypatch.setattr(
            "megatron.bridge.data.packing.offline.np.save",
            lambda path, rows: Path(path).write_bytes(b"staged numpy"),
        )

    with pytest.raises(FileNotFoundError):
        prepare_gpt_sft_packed_data(
            input_path=Path("unused.jsonl"),
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=8,
            tokenizer=SimpleNamespace(eos_id=PAD_ID),
            max_seq_length=8,
            dataset_builder=lambda *args, **kwargs: pytest.fail("tokenization is stubbed"),
        )

    assert not output_path.exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == []


@pytest.mark.parametrize("stream_packed_parquet", [False, True])
@pytest.mark.parametrize("existing_metadata", [False, True])
def test_metadata_is_rolled_back_when_output_publication_fails(
    tmp_path,
    monkeypatch,
    stream_packed_parquet,
    existing_metadata,
):
    output_path = tmp_path / "packed.idx.parquet"
    metadata_path = tmp_path / "packed.metadata.json"
    original_metadata = b'[{"existing": true}]'
    if existing_metadata:
        metadata_path.write_bytes(original_metadata)

    monkeypatch.setattr("megatron.bridge.data.packing.offline.tokenize_dataset", lambda *args, **kwargs: [])
    monkeypatch.setattr("megatron.bridge.data.packing.offline.create_hist", lambda *args: ([], {}))
    monkeypatch.setattr(
        "megatron.bridge.data.packing.offline.create_packing_strategy",
        lambda *args: ([], {"packing_efficiency": 100.0}),
    )

    if stream_packed_parquet:
        monkeypatch.setattr("megatron.bridge.data.packing.offline.iter_packing_strategy", lambda *args: iter(()))

        def write_staged_output(rows, path, *, _publish_on_completion):
            assert list(rows) == []
            assert _publish_on_completion is False
            Path(path).write_bytes(b"staged parquet")

        monkeypatch.setattr(
            "megatron.bridge.data.packing.parquet.write_packed_parquet_streaming",
            write_staged_output,
        )
    else:
        monkeypatch.setattr("megatron.bridge.data.packing.offline.fill_packing_strategy", lambda *args: [])
        monkeypatch.setattr(
            "megatron.bridge.data.packing.parquet.write_packed_parquet",
            lambda rows, path: Path(path).write_bytes(b"staged parquet"),
        )

    publish_staged_path = offline_module._publish_staged_path

    def fail_output_publication(staged_path, destination_path):
        if Path(destination_path) == output_path:
            raise OSError("synthetic output publication failure")
        publish_staged_path(staged_path, destination_path)

    monkeypatch.setattr(offline_module, "_publish_staged_path", fail_output_publication)

    with pytest.raises(OSError, match="synthetic output publication failure"):
        prepare_gpt_sft_packed_data(
            input_path=Path("unused.jsonl"),
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=8,
            tokenizer=SimpleNamespace(eos_id=PAD_ID),
            max_seq_length=8,
            stream_packed_parquet=stream_packed_parquet,
            dataset_builder=lambda *args, **kwargs: pytest.fail("tokenization is stubbed"),
        )

    assert not output_path.exists()
    if existing_metadata:
        assert metadata_path.read_bytes() == original_metadata
        assert sorted(item.name for item in tmp_path.iterdir()) == [metadata_path.name]
    else:
        assert not metadata_path.exists()
        assert sorted(item.name for item in tmp_path.iterdir()) == []


def test_pre_pad_data_point_chat_tensors_do_not_raise():
    """Chat tensors should retain compact storage while being padded (see issue #2610)."""
    data = {
        "input_ids": torch.LongTensor([5, 6, 7]),
        "loss_mask": torch.BoolTensor([False, True, True]),
        "context_ids": torch.LongTensor([5, 6]),
    }
    # stored max_stored_length_to_pad=9 -> input_ids padded to length 9
    _pre_pad_data_point(data, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert isinstance(data["input_ids"], torch.Tensor)
    assert isinstance(data["loss_mask"], torch.Tensor)
    # loss_mask must end up the same length as input_ids, otherwise fill_packing_strategy's
    # np.array([...loss_mask...]) raises an inhomogeneous-shape error when samples are grouped.
    assert len(data["loss_mask"]) == len(data["input_ids"])
    # padded loss_mask positions carry 0 (no loss on pad tokens)
    assert data["loss_mask"][3:].tolist() == [False] * (len(data["loss_mask"]) - 3)
    assert data["input_ids"][3:].tolist() == [PAD_ID] * (len(data["input_ids"]) - 3)


def test_pre_pad_data_point_numpy_arrays_retain_compact_storage():
    data = {
        "input_ids": np.asarray([5, 6, 7], dtype=np.int64),
        "loss_mask": np.asarray([False, True, True], dtype=np.bool_),
    }

    _pre_pad_data_point(data, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert isinstance(data["input_ids"], np.ndarray)
    assert isinstance(data["loss_mask"], np.ndarray)
    assert data["input_ids"].tolist() == [5, 6, 7] + [PAD_ID] * 6
    assert data["loss_mask"].tolist() == [False, True, True] + [False] * 6


def test_pre_pad_data_point_truncated_tensor_releases_original_storage():
    data = {"input_ids": torch.arange(20), "loss_mask": torch.ones(20, dtype=torch.bool)}

    _pre_pad_data_point(data, max_seq_length=16, max_stored_length_to_pad=9, pad_id=PAD_ID)

    assert data["input_ids"].tolist() == list(range(9))
    assert data["input_ids"].untyped_storage().nbytes() == 9 * data["input_ids"].element_size()
    assert data["loss_mask"].untyped_storage().nbytes() == 9 * data["loss_mask"].element_size()


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

    def fail_pool(*args, **kwargs):
        raise AssertionError("Pool should not be constructed for non-positive worker counts")

    monkeypatch.setattr("megatron.bridge.data.packing.offline.Pool", fail_pool)

    assert _materialize_dataset_items(TinyDataset(), -1).tolist() == [10, 11, 12]
    assert _materialize_dataset_items(TinyDataset(), 0).tolist() == [10, 11, 12]


def test_materialize_dataset_items_discards_fields_unused_by_packing():
    """The packing boundary keeps only compact fields consumed by packing."""

    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "input_ids": [10, 11, 12],
                "loss_mask": [False, True, True],
                "context_ids": [10],
                "answer_ids": [11, 12],
                "metadata": {"tools": [{"large": "unused"}]},
            }

    item = _materialize_dataset_items(TinyDataset(), 1)[0]
    assert set(item) == {"input_ids", "loss_mask"}
    assert isinstance(item["input_ids"], np.ndarray)
    assert isinstance(item["loss_mask"], np.ndarray)
    assert item["input_ids"].tolist() == [10, 11, 12]
    assert item["loss_mask"].tolist() == [False, True, True]


def test_materialize_dataset_items_converts_chat_tensors_before_worker_return():
    """Compact NumPy values avoid retaining one shared-memory tensor file per field and sample."""

    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "input_ids": torch.tensor([10, 11, 12]),
                "loss_mask": torch.tensor([False, True, True]),
            }

    item = _materialize_dataset_items(TinyDataset(), 1)[0]
    assert isinstance(item["input_ids"], np.ndarray)
    assert isinstance(item["loss_mask"], np.ndarray)
    assert item["input_ids"].dtype == np.int64
    assert item["loss_mask"].dtype == np.bool_


@pytest.mark.parametrize("pool_fails", [False, True])
def test_materialize_dataset_items_configures_and_restores_worker_resources(monkeypatch, pool_fails):
    """The multiprocessing path should use file-backed tensor sharing only while its pool runs."""

    class TinyDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index + 20

    pool_calls = []

    class FakePool:
        def __init__(self, num_workers, *, initializer, initargs):
            pool_calls.append((num_workers, initializer, initargs))
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def imap(self, function, indexes):
            if pool_fails:
                raise RuntimeError("worker failure")
            return map(function, indexes)

    sharing_strategy_calls = []
    nofile_limit_calls = []
    monkeypatch.setattr("megatron.bridge.data.packing.offline.Pool", FakePool)
    monkeypatch.setattr(torch.multiprocessing, "get_sharing_strategy", lambda: "file_descriptor")
    monkeypatch.setattr(torch.multiprocessing, "set_sharing_strategy", sharing_strategy_calls.append)
    monkeypatch.setattr("megatron.bridge.data.packing.offline.resource.getrlimit", lambda _: (256, 4096))
    monkeypatch.setattr(
        "megatron.bridge.data.packing.offline.resource.setrlimit",
        lambda _, limits: nofile_limit_calls.append(limits),
    )

    dataset = TinyDataset()
    if pool_fails:
        with pytest.raises(RuntimeError, match="worker failure"):
            _materialize_dataset_items(dataset, 2)
    else:
        assert _materialize_dataset_items(dataset, 2).tolist() == [20, 21]
    assert sharing_strategy_calls == ["file_system", "file_descriptor"]
    assert nofile_limit_calls == [(4096, 4096), (256, 4096)]
    assert pool_calls == [(2, _init_shared_dataset_worker, (dataset,))]
