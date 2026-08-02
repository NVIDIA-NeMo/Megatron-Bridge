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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from megatron.bridge.data.builders.gpt_sft import build_gpt_sft_split
from megatron.bridge.data.packing import indexed as indexed_module
from megatron.bridge.data.packing.indexed import (
    GPTSFTPackedIndexedDataset,
    PackedSFTIndexedDataset,
    decode_packed_row,
    encode_packed_row,
    is_packed_indexed_dataset,
    resolve_packed_indexed_prefixes,
    resolve_packed_indexed_prefixes_with_retry,
    write_packed_indexed_dataset,
)


pytestmark = pytest.mark.unit


def _make_row(offset: int = 0) -> dict[str, list[int]]:
    return {
        "input_ids": [10 + offset, 11 + offset, 12 + offset, 20 + offset, 21 + offset, 22 + offset],
        "loss_mask": [0, 1, 0, 1, 1, 0],
        "seq_start_id": [0, 3],
    }


def _make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.eos_id = 999
    tokenizer.eod = 999
    tokenizer.pad_id = 999
    return tokenizer


def _make_training_dataset(path: str, *, max_num_samples: int | None = None) -> GPTSFTPackedIndexedDataset:
    return GPTSFTPackedIndexedDataset(
        file_path=path,
        tokenizer=_make_tokenizer(),
        max_seq_length=16,
        max_num_samples=max_num_samples,
        pad_to_max_length=False,
        pad_seq_length_to_mult=1,
        add_bos=False,
        add_eos=False,
        add_sep=False,
        seed=42,
        label_key="output",
        answer_only_loss=True,
        truncation_field="input",
        prompt_template="{input} {output}",
        return_cu_seqlen=True,
    )


def test_encode_decode_round_trip_preserves_packed_row() -> None:
    row = _make_row()
    encoded = encode_packed_row(row)
    decoded = decode_packed_row(encoded, row_index=0)

    assert encoded.dtype == np.int32
    assert np.any(encoded < 0), "enabled loss-mask bits should set the int32 sign bit"
    np.testing.assert_array_equal(decoded["input_ids"], row["input_ids"])
    np.testing.assert_array_equal(decoded["loss_mask"], row["loss_mask"])
    assert decoded["seq_start_id"] == row["seq_start_id"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("input_ids", [], "must not be empty"),
        ("input_ids", [-1, 11, 12, 20, 21, 22], "range"),
        ("loss_mask", [0, 2, 0, 1, 1, 0], "binary"),
        ("loss_mask", [0, 1], "length"),
        ("seq_start_id", [1, 3], "start with 0"),
        ("seq_start_id", [0, 0], "strictly increasing"),
    ],
)
def test_encode_rejects_invalid_rows(field: str, value: list[int], error: str) -> None:
    row = _make_row()
    row[field] = value
    with pytest.raises(ValueError, match=error):
        encode_packed_row(row)


def test_writer_reader_and_path_resolution(tmp_path) -> None:
    prefix = tmp_path / "train.sft"
    rows = [_make_row(), _make_row(offset=100)]
    assert write_packed_indexed_dataset(rows, prefix) == str(prefix)
    assert (tmp_path / "train.sft.bin").is_file()
    assert (tmp_path / "train.sft.idx").is_file()
    assert is_packed_indexed_dataset(prefix)
    assert is_packed_indexed_dataset(tmp_path / "train.sft.bin")
    assert resolve_packed_indexed_prefixes(tmp_path) == [str(prefix)]
    assert resolve_packed_indexed_prefixes(tmp_path / "*.sft") == [str(prefix)]

    dataset = PackedSFTIndexedDataset(prefix)
    assert len(dataset) == len(rows)
    for index, expected in enumerate(rows):
        actual = dataset[index]
        np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])
        np.testing.assert_array_equal(actual["loss_mask"], expected["loss_mask"])
        assert actual["seq_start_id"] == expected["seq_start_id"]


def test_invalid_overwrite_preserves_existing_pair(tmp_path) -> None:
    prefix = tmp_path / "train.sft"
    original_rows = [_make_row(), _make_row(offset=100)]
    write_packed_indexed_dataset(original_rows, prefix)
    original_bin = (tmp_path / "train.sft.bin").read_bytes()
    original_idx = (tmp_path / "train.sft.idx").read_bytes()
    invalid_row = _make_row(offset=200)
    invalid_row["loss_mask"][0] = 2

    with pytest.raises(ValueError, match="binary"):
        write_packed_indexed_dataset([_make_row(offset=300), invalid_row], prefix)

    assert (tmp_path / "train.sft.bin").read_bytes() == original_bin
    assert (tmp_path / "train.sft.idx").read_bytes() == original_idx
    assert not list(tmp_path.glob("train.sft.tmp-*"))


def test_finalize_failure_preserves_existing_pair_and_cleans_temporary_files(tmp_path, monkeypatch) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)
    original_bin = (tmp_path / "train.sft.bin").read_bytes()
    original_idx = (tmp_path / "train.sft.idx").read_bytes()

    def fail_finalize(self, idx_path):
        self.data_file.close()
        raise OSError("injected finalize failure")

    monkeypatch.setattr(indexed_module.IndexedDatasetBuilder, "finalize", fail_finalize)

    with pytest.raises(OSError, match="injected"):
        write_packed_indexed_dataset([_make_row(offset=100)], prefix)

    assert (tmp_path / "train.sft.bin").read_bytes() == original_bin
    assert (tmp_path / "train.sft.idx").read_bytes() == original_idx
    assert not list(tmp_path.glob("train.sft.tmp-*"))


def test_temporary_cleanup_failure_does_not_mask_primary_error(tmp_path, monkeypatch) -> None:
    def fail_finalize(self, idx_path):
        self.data_file.close()
        raise OSError("injected finalize failure")

    original_unlink = indexed_module._unlink_path

    def fail_temporary_cleanup(path):
        if ".tmp-" in path:
            raise OSError("injected cleanup failure")
        original_unlink(path)

    monkeypatch.setattr(indexed_module.IndexedDatasetBuilder, "finalize", fail_finalize)
    monkeypatch.setattr(indexed_module, "_unlink_path", fail_temporary_cleanup)

    with pytest.raises(OSError, match="injected finalize failure"):
        write_packed_indexed_dataset([_make_row()], tmp_path / "train.sft")


@pytest.mark.parametrize("failing_suffix", [".bin", ".idx"])
def test_publication_failure_restores_existing_pair(tmp_path, monkeypatch, failing_suffix) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)
    original_bin = (tmp_path / "train.sft.bin").read_bytes()
    original_idx = (tmp_path / "train.sft.idx").read_bytes()
    original_replace = indexed_module._replace_path
    failure_injected = False

    def fail_temporary_publish(source, destination):
        nonlocal failure_injected
        if not failure_injected and ".tmp-" in source and source.endswith(failing_suffix):
            failure_injected = True
            raise OSError(f"injected {failing_suffix} publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(indexed_module, "_replace_path", fail_temporary_publish)

    with pytest.raises(OSError, match="publication failure"):
        write_packed_indexed_dataset([_make_row(offset=100)], prefix)

    assert (tmp_path / "train.sft.bin").read_bytes() == original_bin
    assert (tmp_path / "train.sft.idx").read_bytes() == original_idx
    assert not list(tmp_path.glob("train.sft.tmp-*"))
    assert not list(tmp_path.glob("train.sft.backup-*"))


@pytest.mark.parametrize("failing_suffix", [".bin", ".idx"])
def test_backup_failure_leaves_existing_pair_untouched(tmp_path, monkeypatch, failing_suffix) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)
    original_bin = (tmp_path / "train.sft.bin").read_bytes()
    original_idx = (tmp_path / "train.sft.idx").read_bytes()
    original_replace = indexed_module._replace_path

    def fail_backup(source, destination):
        if source == f"{prefix}{failing_suffix}" and ".backup-" in destination:
            raise OSError(f"injected {failing_suffix} backup failure")
        original_replace(source, destination)

    monkeypatch.setattr(indexed_module, "_replace_path", fail_backup)

    with pytest.raises(OSError, match="backup failure"):
        write_packed_indexed_dataset([_make_row(offset=100)], prefix)

    assert (tmp_path / "train.sft.bin").read_bytes() == original_bin
    assert (tmp_path / "train.sft.idx").read_bytes() == original_idx
    assert not list(tmp_path.glob("train.sft.tmp-*"))
    assert not list(tmp_path.glob("train.sft.backup-*"))


def test_writer_rejects_direct_object_storage_output() -> None:
    with pytest.raises(ValueError, match="write a local pair"):
        write_packed_indexed_dataset([_make_row()], "msc://profile/train.sft")


def test_multi_shard_reader_validates_every_shard(tmp_path) -> None:
    write_packed_indexed_dataset([_make_row()], tmp_path / "shard_000.sft")
    invalid_prefix = tmp_path / "shard_001.sft"
    builder = indexed_module.IndexedDatasetBuilder(f"{invalid_prefix}.bin", dtype=np.int32)
    builder.add_item(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))
    builder.end_document()
    builder.finalize(f"{invalid_prefix}.idx")

    with pytest.raises(ValueError, match="shard_001.*invalid magic"):
        PackedSFTIndexedDataset(tmp_path)


def test_indexed_resolution_retries_after_stale_visibility(monkeypatch) -> None:
    calls = []

    def resolve(_spec):
        calls.append(None)
        if len(calls) < 3:
            raise ValueError("stale")
        return ["train.sft"]

    monkeypatch.setattr(indexed_module, "resolve_packed_indexed_prefixes", resolve)
    monkeypatch.setattr(indexed_module, "_refresh_directory_metadata", lambda _spec: None)

    assert resolve_packed_indexed_prefixes_with_retry("train.sft", max_attempts=3, backoff_s=0) == ["train.sft"]
    assert len(calls) == 3


def test_remote_reader_disables_mmap_and_builds_object_storage_config(monkeypatch) -> None:
    encoded = encode_packed_row(_make_row())
    observed = []

    class FakeIndexedDataset:
        def __init__(self, prefix, *, mmap, object_storage_config):
            observed.append((prefix, mmap, object_storage_config))

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return encoded

    config = SimpleNamespace(path_to_idx_cache="/cache", bin_chunk_nbytes=123)
    monkeypatch.setattr(indexed_module, "resolve_packed_indexed_prefixes", lambda _spec: ["msc://p/train.sft"])
    monkeypatch.setattr(indexed_module, "make_object_storage_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(indexed_module, "IndexedDataset", FakeIndexedDataset)

    dataset = PackedSFTIndexedDataset(
        "msc://p/train.sft",
        object_storage_cache_path="/cache",
        object_storage_bin_chunk_nbytes=123,
    )

    assert len(dataset) == 1
    assert observed == [("msc://p/train.sft", False, config)]


def test_packed_sequence_specs_accepts_complete_indexed_pair(tmp_path) -> None:
    from megatron.bridge.data.packing import PackedSequenceSpecs

    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)

    specs = PackedSequenceSpecs(packed_sequence_size=16, packed_train_data_path=prefix)

    assert specs.packed_train_data_path == str(prefix)


def test_training_dataset_preserves_per_sequence_shift_and_mask(tmp_path) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)
    dataset = _make_training_dataset(str(prefix))
    sample = dataset[0]
    batch = dataset.collate_fn([sample])

    assert sample["seq_boundaries"] == [0, 3, 6]
    assert batch["tokens"].tolist() == [[10, 11, 20, 21]]
    assert batch["labels"].tolist() == [[11, 12, 21, 22]]
    assert batch["loss_mask"].tolist() == [[0, 1, 1, 1]]
    assert batch["position_ids"].tolist() == [[0, 1, 0, 1]]
    assert batch["cu_seqlens"].tolist() == [[0, 2, 4, -1]]


def test_builder_routes_indexed_dataset(tmp_path) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)
    dataset = build_gpt_sft_split(
        prefix,
        tokenizer=_make_tokenizer(),
        seq_length=16,
        memmap_workers=1,
        seed=42,
        packed_sequence_size=16,
        pad_seq_to_mult=1,
        dataset_kwargs={"pad_seq_length_to_mult": 1},
    )
    assert isinstance(dataset, GPTSFTPackedIndexedDataset)


def test_negative_index_zeroes_loss_mask(tmp_path) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)
    dataset = _make_training_dataset(str(prefix))
    assert dataset[-1]["loss_mask"].tolist() == [0] * 6


def test_samples_mapping_uses_mapped_row_index(tmp_path) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row(), _make_row(offset=100)], prefix)
    dataset = _make_training_dataset(str(prefix))
    dataset.samples_mapping = np.asarray([[1, 0, 1]], dtype=np.int64)

    assert dataset[0]["input_ids"].tolist() == _make_row(offset=100)["input_ids"]


def test_single_row_single_sample_mapping_remains_one_dimensional(tmp_path) -> None:
    prefix = tmp_path / "train.sft"
    write_packed_indexed_dataset([_make_row()], prefix)
    dataset = _make_training_dataset(str(prefix), max_num_samples=1)

    assert dataset.samples_mapping.shape == (1,)
    assert dataset[0]["input_ids"].tolist() == _make_row()["input_ids"]


def test_parquet_and_indexed_produce_identical_samples_and_batches(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    from megatron.bridge.data.packing.parquet import GPTSFTPackedParquetDataset, write_packed_parquet

    rows = [_make_row(), _make_row(offset=100)]
    indexed_prefix = tmp_path / "train.sft"
    parquet_path = tmp_path / "train.idx.parquet"
    write_packed_indexed_dataset(rows, indexed_prefix)
    write_packed_parquet(rows, parquet_path)
    indexed = _make_training_dataset(str(indexed_prefix))
    parquet = GPTSFTPackedParquetDataset(
        file_path=str(parquet_path),
        tokenizer=_make_tokenizer(),
        max_seq_length=16,
        max_num_samples=None,
        pad_to_max_length=False,
        pad_seq_length_to_mult=1,
        add_bos=False,
        add_eos=False,
        add_sep=False,
        seed=42,
        label_key="output",
        answer_only_loss=True,
        truncation_field="input",
        prompt_template="{input} {output}",
        return_cu_seqlen=True,
    )

    for row_index in range(len(rows)):
        indexed_sample = indexed[row_index]
        parquet_sample = parquet[row_index]
        np.testing.assert_array_equal(indexed_sample["input_ids"], parquet_sample["input_ids"])
        np.testing.assert_array_equal(indexed_sample["loss_mask"], parquet_sample["loss_mask"])
        assert indexed_sample["seq_boundaries"] == parquet_sample["seq_boundaries"]
        indexed_batch = indexed.collate_fn([indexed_sample])
        parquet_batch = parquet.collate_fn([parquet_sample])
        assert indexed_batch.keys() == parquet_batch.keys()
        for key in indexed_batch:
            if isinstance(indexed_batch[key], torch.Tensor):
                torch.testing.assert_close(indexed_batch[key], parquet_batch[key])
            else:
                assert indexed_batch[key] == parquet_batch[key]
