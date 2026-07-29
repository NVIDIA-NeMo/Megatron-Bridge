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

from typing import List

import numpy as np
import pytest

from megatron.bridge.data.datasets.packing_utils import (
    calculate_avg_seqlen,
    first_fit,
    first_fit_decreasing,
)


def _first_fit_linear(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """Reference: original O(N²) linear-scan first_fit before segment tree."""
    res = []
    res_sums = []
    for s in seqlens:
        first_bin = -1
        for i, cur_sum in enumerate(res_sums):
            if cur_sum + s <= pack_size:
                first_bin = i
                break
        if first_bin == -1:
            res.append([s])
            res_sums.append(s)
        else:
            res[first_bin].append(s)
            res_sums[first_bin] += s
    return res


class TestFirstFitPacking:
    """Test cases for first_fit bin-packing algorithm."""

    def test_first_fit_decreasing_sorted_order(self):
        """Test first_fit_decreasing sorts sequences before packing."""
        seqlens = [1111, 8192, 4096, 1000]
        pack_size = 2048

        result = first_fit_decreasing(seqlens, pack_size)
        assert result == [[8192], [4096], [1111], [1000]]

    def test_bin_capacity_not_exceeded(self):
        """Test no bin exceeds the pack_size limit."""
        np.random.seed(7)
        seqlens = list(np.random.randint(1, 2048, size=10000))
        pack_size = 2048

        result = first_fit(seqlens, pack_size)
        for bin_contents in result:
            assert sum(bin_contents) <= pack_size


class TestSegmentTreeMatchesLinearScan:
    """Verify segment-tree first_fit produces identical results to the original linear-scan."""

    def test_matches_on_small_input(self):
        """Test a small, hand-crafted example."""
        seqlens = [500, 600, 500, 400, 700]
        pack_size = 1200
        assert first_fit(seqlens, pack_size) == _first_fit_linear(seqlens, pack_size)

    def test_matches_on_oversized_sequences(self):
        """Test sequences that individually exceed pack_size are still placed in their own bins."""
        seqlens = [4096, 3000, 5000]
        pack_size = 2048
        assert first_fit(seqlens, pack_size) == _first_fit_linear(seqlens, pack_size)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_on_random_input(self, seed):
        """Test random inputs of varying sizes."""
        np.random.seed(seed)
        seqlens = list(np.random.randint(1, 2048, size=5000))
        pack_size = 2048
        assert first_fit(seqlens, pack_size) == _first_fit_linear(seqlens, pack_size)


# ---------------------------------------------------------------------------
# calculate_avg_seqlen: format-aware loader (npy + parquet single/glob/dir)
# ---------------------------------------------------------------------------
#
# Two hand-crafted packed rows shared by every fixture below so all formats
# must return the same stats:
#   row A: seq_start_id=[0, 4], input_ids len 8 -> boundaries [0,4,8]
#          per-seq token counts = [3, 3]  (each minus 1 EOS)
#   row B: seq_start_id=[0],    input_ids len 5 -> boundaries [0,5]
#          per-seq token counts = [4]
# Aggregated over both rows (gbs=1, drop_remainder=True -> count=2):
#   seq_count_accum   = 2 + 1              = 3
#   total_len_accum   = (3+3) + 4          = 10
#   seqlen_sq_accum   = (9+9) + 16         = 34
# -> (count, total, sq_individual, sq_per_row) = (3/2, 10/2, 34/3, 34/2)
_AVG_SEQLEN_ROWS = [
    {"input_ids": list(range(8)), "seq_start_id": [0, 4]},
    {"input_ids": list(range(5)), "seq_start_id": [0]},
]
_AVG_SEQLEN_EXPECTED = (3 / 2, 10 / 2, 34 / 3, 34 / 2)


def _write_avg_seqlen_parquet(path, rows) -> None:
    """Write the two columns calculate_avg_seqlen reads to a parquet shard."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "input_ids": [r["input_ids"] for r in rows],
            "seq_start_id": [r["seq_start_id"] for r in rows],
        }
    )
    pq.write_table(table, str(path))


class TestCalculateAvgSeqlen:
    """calculate_avg_seqlen must load npy and parquet (single/glob/dir) identically."""

    def _assert_expected(self, stats):
        assert stats == pytest.approx(_AVG_SEQLEN_EXPECTED)

    def test_npy_legacy(self, tmp_path):
        path = tmp_path / "training_4096.npy"
        np.save(path, np.array(_AVG_SEQLEN_ROWS, dtype=object))
        stats = calculate_avg_seqlen(str(path), gbs=1, max_seq_len=8, drop_remainder=True)
        self._assert_expected(stats)

    def test_parquet_single_file(self, tmp_path):
        path = tmp_path / "training_4096.idx.parquet"
        _write_avg_seqlen_parquet(path, _AVG_SEQLEN_ROWS)
        stats = calculate_avg_seqlen(str(path), gbs=1, max_seq_len=8, drop_remainder=True)
        self._assert_expected(stats)

    def test_parquet_glob_spec_matches_single_file(self, tmp_path):
        """A glob spec (sharded parquet) must resolve+read, not crash -- guards the
        _packed_data_exists vs calculate_avg_seqlen input-set consistency."""
        _write_avg_seqlen_parquet(tmp_path / "shard_000.idx.parquet", _AVG_SEQLEN_ROWS[:1])
        _write_avg_seqlen_parquet(tmp_path / "shard_001.idx.parquet", _AVG_SEQLEN_ROWS[1:])
        glob_spec = str(tmp_path / "shard_*.idx.parquet")
        stats = calculate_avg_seqlen(glob_spec, gbs=1, max_seq_len=8, drop_remainder=True)
        self._assert_expected(stats)

    def test_parquet_directory_spec(self, tmp_path):
        """A directory spec is also accepted (globs *.parquet under the dir)."""
        _write_avg_seqlen_parquet(tmp_path / "shard_000.idx.parquet", _AVG_SEQLEN_ROWS[:1])
        _write_avg_seqlen_parquet(tmp_path / "shard_001.idx.parquet", _AVG_SEQLEN_ROWS[1:])
        stats = calculate_avg_seqlen(str(tmp_path), gbs=1, max_seq_len=8, drop_remainder=True)
        self._assert_expected(stats)
