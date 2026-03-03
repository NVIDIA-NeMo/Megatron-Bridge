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

"""Integration tests for GPTSFTPackedParquetDataset.

These tests create real Parquet files and exercise the dataset end-to-end,
covering _locate_row, row-group caching, multi-file support, schema validation,
and the validate_row helper.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Optional dependency — skip the entire module when pyarrow is missing.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from megatron.bridge.data.datasets.packed_parquet import (
    GPTSFTPackedParquetDataset,
    _resolve_parquet_paths,
    is_packed_parquet_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer_mock():
    """Return a minimal tokenizer mock sufficient for GPTSFTPackedDataset."""
    tok = MagicMock()
    tok.eos_id = 0
    tok.eod = 0
    tok.pad_id = 0
    return tok


def _make_packed_row(n_tokens: int = 64, n_seqs: int = 2):
    """Build a single packed-row dict.

    Returns input_ids, loss_mask, and seq_start_id with valid invariants.
    """
    assert n_seqs >= 1
    tokens_per_seq = n_tokens // n_seqs
    seq_start_id = [i * tokens_per_seq for i in range(n_seqs)]
    return {
        "input_ids": list(range(1, n_tokens + 1)),
        "loss_mask": [1] * n_tokens,
        "seq_start_id": seq_start_id,
    }


def _write_parquet(path: str | Path, rows: list[dict], row_group_size: int = 500):
    """Write a list of packed-row dicts to a Parquet file."""
    table = pa.table(
        {
            "input_ids": [row["input_ids"] for row in rows],
            "loss_mask": [row["loss_mask"] for row in rows],
            "seq_start_id": [row["seq_start_id"] for row in rows],
        }
    )
    pq.write_table(table, str(path), row_group_size=row_group_size)


# ---------------------------------------------------------------------------
# Tests: is_packed_parquet_file
# ---------------------------------------------------------------------------


class TestIsPackedParquetFile:
    def test_direct_idx_parquet(self):
        assert is_packed_parquet_file("data.idx.parquet") is True

    def test_direct_idx_pq(self):
        assert is_packed_parquet_file("data.idx.pq") is True

    def test_glob_pattern(self):
        assert is_packed_parquet_file("shard_*.idx.parquet") is True

    def test_regular_parquet_rejected(self):
        assert is_packed_parquet_file("data.parquet") is False

    def test_case_insensitive(self):
        assert is_packed_parquet_file("DATA.IDX.PARQUET") is True


# ---------------------------------------------------------------------------
# Tests: _resolve_parquet_paths
# ---------------------------------------------------------------------------


class TestResolveParquetPaths:
    def test_single_file(self, tmp_path):
        f = tmp_path / "data.idx.parquet"
        _write_parquet(f, [_make_packed_row()])
        paths = _resolve_parquet_paths(str(f))
        assert paths == [str(f)]

    def test_glob_pattern(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"shard_{i:03d}.idx.parquet"
            _write_parquet(f, [_make_packed_row()])
        paths = _resolve_parquet_paths(str(tmp_path / "shard_*.idx.parquet"))
        assert len(paths) == 3
        assert paths == sorted(paths)

    def test_directory(self, tmp_path):
        for i in range(2):
            _write_parquet(tmp_path / f"s{i}.idx.parquet", [_make_packed_row()])
        paths = _resolve_parquet_paths(str(tmp_path))
        assert len(paths) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _resolve_parquet_paths(str(tmp_path / "nonexistent.idx.parquet"))

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No packed Parquet files found"):
            _resolve_parquet_paths(str(tmp_path))


# ---------------------------------------------------------------------------
# Tests: GPTSFTPackedParquetDataset
# ---------------------------------------------------------------------------


def _make_dataset(file_path, max_num_samples=None, **kwargs):
    """Construct a GPTSFTPackedParquetDataset with minimal config."""
    return GPTSFTPackedParquetDataset(
        file_path=str(file_path),
        tokenizer=_make_tokenizer_mock(),
        max_seq_length=4096,
        max_num_samples=max_num_samples,
        pad_to_max_length=False,
        add_bos=False,
        add_eos=False,
        add_sep=False,
        seed=42,
        answer_only_loss=True,
        truncation_field="input",
        prompt_template="{input} {output}",
        return_cu_seqlen=True,
        **kwargs,
    )


class TestPackedParquetDatasetSingleFile:
    """Tests using a single Parquet file."""

    @pytest.fixture()
    def parquet_file(self, tmp_path):
        rows = [_make_packed_row(n_tokens=64, n_seqs=2) for _ in range(10)]
        path = tmp_path / "data.idx.parquet"
        _write_parquet(path, rows, row_group_size=5)
        return path

    def test_len(self, parquet_file):
        ds = _make_dataset(parquet_file)
        assert len(ds) == 10

    def test_getitem_returns_expected_keys(self, parquet_file):
        ds = _make_dataset(parquet_file)
        sample = ds[0]
        assert "input_ids" in sample
        assert "loss_mask" in sample
        assert "seq_boundaries" in sample

    def test_getitem_seq_boundaries(self, parquet_file):
        ds = _make_dataset(parquet_file)
        sample = ds[0]
        # seq_start_id was [0, 32], so boundaries should be [0, 32, 64]
        assert sample["seq_boundaries"] == [0, 32, 64]

    def test_getitem_all_rows(self, parquet_file):
        ds = _make_dataset(parquet_file)
        for i in range(len(ds)):
            sample = ds[i]
            assert len(sample["input_ids"]) == 64
            assert len(sample["loss_mask"]) == 64

    def test_negative_index_zeroes_loss_mask(self, parquet_file):
        ds = _make_dataset(parquet_file)
        sample = ds[-1]
        assert all(m == 0 for m in sample["loss_mask"])

    def test_max_num_samples(self, parquet_file):
        ds = _make_dataset(parquet_file, max_num_samples=5)
        assert len(ds) == 5

    def test_oversampling(self, parquet_file):
        ds = _make_dataset(parquet_file, max_num_samples=25)
        assert len(ds) == 25
        # Should be able to iterate all
        for i in range(len(ds)):
            ds[i]


class TestPackedParquetDatasetMultiFile:
    """Tests using multiple Parquet files."""

    @pytest.fixture()
    def multi_parquet(self, tmp_path):
        # 3 shards with 5, 10, 7 rows respectively
        counts = [5, 10, 7]
        for i, n in enumerate(counts):
            rows = [_make_packed_row(n_tokens=32, n_seqs=1) for _ in range(n)]
            _write_parquet(tmp_path / f"shard_{i:03d}.idx.parquet", rows, row_group_size=4)
        return tmp_path

    def test_total_len(self, multi_parquet):
        ds = _make_dataset(str(multi_parquet / "shard_*.idx.parquet"))
        assert len(ds) == 22

    def test_getitem_across_files(self, multi_parquet):
        ds = _make_dataset(str(multi_parquet / "shard_*.idx.parquet"))
        for i in range(len(ds)):
            sample = ds[i]
            assert len(sample["input_ids"]) == 32

    def test_locate_row_boundaries(self, multi_parquet):
        ds = _make_dataset(str(multi_parquet / "shard_*.idx.parquet"))
        # Row 0 -> file 0, row 4 -> file 0, row 5 -> file 1, row 15 -> file 2
        file_idx_0, _, _ = ds._locate_row(0)
        file_idx_4, _, _ = ds._locate_row(4)
        file_idx_5, _, _ = ds._locate_row(5)
        file_idx_15, _, _ = ds._locate_row(15)
        assert file_idx_0 == 0
        assert file_idx_4 == 0
        assert file_idx_5 == 1
        assert file_idx_15 == 2


class TestPackedParquetDatasetRowGroupCache:
    """Tests for row-group caching behavior."""

    @pytest.fixture()
    def parquet_file(self, tmp_path):
        rows = [_make_packed_row(n_tokens=16, n_seqs=1) for _ in range(20)]
        path = tmp_path / "data.idx.parquet"
        _write_parquet(path, rows, row_group_size=5)
        return path

    def test_cache_reuse_within_row_group(self, parquet_file):
        ds = _make_dataset(parquet_file)
        ds[0]  # Load row group 0
        cached_table = ds._cached_row_group_table
        ds[1]  # Same row group
        assert ds._cached_row_group_table is cached_table

    def test_cache_eviction_on_new_row_group(self, parquet_file):
        ds = _make_dataset(parquet_file)
        ds[0]  # Row group 0
        old_table = ds._cached_row_group_table
        ds[5]  # Row group 1
        assert ds._cached_row_group_table is not old_table


class TestPackedParquetSchemaValidation:
    """Tests for schema validation during _load_dataset."""

    def test_missing_column_raises(self, tmp_path):
        # Write a parquet file missing 'loss_mask'
        table = pa.table(
            {
                "input_ids": [[1, 2, 3]],
                "seq_start_id": [[0]],
            }
        )
        path = tmp_path / "bad.idx.parquet"
        pq.write_table(table, str(path))

        with pytest.raises(ValueError, match="missing required columns"):
            _make_dataset(path)

    def test_empty_file_raises(self, tmp_path):
        table = pa.table(
            {
                "input_ids": pa.array([], type=pa.list_(pa.int32())),
                "loss_mask": pa.array([], type=pa.list_(pa.int8())),
                "seq_start_id": pa.array([], type=pa.list_(pa.int32())),
            }
        )
        path = tmp_path / "empty.idx.parquet"
        pq.write_table(table, str(path))

        with pytest.raises(ValueError, match="empty"):
            _make_dataset(path)


class TestValidateRow:
    """Tests for the static validate_row method."""

    def test_valid_row(self):
        GPTSFTPackedParquetDataset.validate_row(
            idx=0,
            input_ids=[1, 2, 3, 4],
            loss_mask=[1, 1, 0, 1],
            seq_start_id=[0, 2],
        )

    def test_loss_mask_length_mismatch(self):
        with pytest.raises(ValueError, match="loss_mask length"):
            GPTSFTPackedParquetDataset.validate_row(
                idx=0,
                input_ids=[1, 2, 3],
                loss_mask=[1, 1],
                seq_start_id=[0],
            )

    def test_seq_start_id_not_starting_with_zero(self):
        with pytest.raises(ValueError, match="must start with 0"):
            GPTSFTPackedParquetDataset.validate_row(
                idx=0,
                input_ids=[1, 2, 3],
                loss_mask=[1, 1, 1],
                seq_start_id=[1],
            )

    def test_seq_start_id_empty(self):
        with pytest.raises(ValueError, match="must start with 0"):
            GPTSFTPackedParquetDataset.validate_row(
                idx=0,
                input_ids=[1, 2, 3],
                loss_mask=[1, 1, 1],
                seq_start_id=[],
            )

    def test_seq_start_id_out_of_bounds(self):
        with pytest.raises(ValueError, match=">="):
            GPTSFTPackedParquetDataset.validate_row(
                idx=0,
                input_ids=[1, 2, 3],
                loss_mask=[1, 1, 1],
                seq_start_id=[0, 5],
            )

    def test_seq_start_id_not_non_decreasing(self):
        with pytest.raises(ValueError, match="not non-decreasing"):
            GPTSFTPackedParquetDataset.validate_row(
                idx=0,
                input_ids=[1, 2, 3, 4],
                loss_mask=[1, 1, 1, 1],
                seq_start_id=[0, 3, 1],
            )


class TestPackedParquetClose:
    """Tests for resource cleanup."""

    def test_close_clears_state(self, tmp_path):
        path = tmp_path / "data.idx.parquet"
        _write_parquet(path, [_make_packed_row()])
        ds = _make_dataset(path)
        ds[0]  # Force reader open
        assert len(ds._parquet_files) > 0

        ds.close()
        assert len(ds._parquet_files) == 0
        assert ds._cached_row_group_table is None

    def test_double_close_safe(self, tmp_path):
        path = tmp_path / "data.idx.parquet"
        _write_parquet(path, [_make_packed_row()])
        ds = _make_dataset(path)
        ds.close()
        ds.close()  # Should not raise
