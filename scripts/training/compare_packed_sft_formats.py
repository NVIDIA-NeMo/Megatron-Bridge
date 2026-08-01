#!/usr/bin/env python3
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

"""Validate semantic parity and read performance of packed SFT formats."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megatron.bridge.data.packing.indexed import PackedSFTIndexedDataset, resolve_packed_indexed_prefixes
from megatron.bridge.data.packing.paths import resolve_packed_parquet_paths


logger = logging.getLogger(__name__)
_FIELDS = ("input_ids", "loss_mask", "seq_start_id")


@dataclass(frozen=True)
class ReadStats:
    """Sequential-read measurements for one packed dataset format."""

    rows: int
    tokens: int
    seconds: float
    bytes_on_disk: int

    @property
    def rows_per_second(self) -> float:
        """Return sequential rows decoded per second."""
        return self.rows / self.seconds

    @property
    def tokens_per_second(self) -> float:
        """Return sequential tokens decoded per second."""
        return self.tokens / self.seconds


def _import_parquet():
    """Import PyArrow lazily so indexed-only training does not require it."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Parquet comparison requires the optional 'parquet' dependencies") from error
    return pq


def _iter_parquet_rows(path_spec: str) -> Iterator[dict[str, list[int]]]:
    """Yield packed rows from all resolved Parquet shards."""
    pq = _import_parquet()
    for path in resolve_packed_parquet_paths(path_spec):
        parquet_file = pq.ParquetFile(path)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index, columns=list(_FIELDS))
            for row_index in range(table.num_rows):
                yield {field: table.column(field)[row_index].as_py() for field in _FIELDS}


def _parquet_row_count(path_spec: str) -> int:
    """Return the row count across all Parquet shards."""
    pq = _import_parquet()
    return sum(pq.read_metadata(path).num_rows for path in resolve_packed_parquet_paths(path_spec))


def _limited_count(total_rows: int, max_rows: int) -> int:
    return total_rows if max_rows <= 0 else min(total_rows, max_rows)


def _local_file_size(paths: list[str]) -> int:
    total = 0
    for path in paths:
        try:
            total += Path(path).stat().st_size
        except OSError:
            return 0
    return total


def validate_parity(parquet_path: str, indexed_path: str, *, max_rows: int) -> int:
    """Validate row-level equality between Parquet and indexed datasets."""
    indexed = PackedSFTIndexedDataset(indexed_path)
    parquet_rows = _parquet_row_count(parquet_path)
    if max_rows <= 0 and parquet_rows != len(indexed):
        raise ValueError(f"Row count mismatch: parquet={parquet_rows}, indexed={len(indexed)}")
    rows_to_compare = min(_limited_count(parquet_rows, max_rows), _limited_count(len(indexed), max_rows))
    parquet_iterator = _iter_parquet_rows(parquet_path)
    for row_index in range(rows_to_compare):
        parquet_row = next(parquet_iterator)
        indexed_row = indexed[row_index]
        for field in _FIELDS:
            if not np.array_equal(np.asarray(parquet_row[field]), np.asarray(indexed_row[field])):
                raise ValueError(f"Parity mismatch at row {row_index}, field '{field}'")
    return rows_to_compare


def benchmark_indexed(path_spec: str, *, max_rows: int) -> ReadStats:
    """Measure sequential indexed decoding throughput."""
    dataset = PackedSFTIndexedDataset(path_spec)
    rows = _limited_count(len(dataset), max_rows)
    start = time.perf_counter()
    token_count = sum(len(dataset[row_index]["input_ids"]) for row_index in range(rows))
    seconds = max(time.perf_counter() - start, 1e-12)
    prefixes = resolve_packed_indexed_prefixes(path_spec)
    paths = [path for prefix in prefixes for path in (f"{prefix}.bin", f"{prefix}.idx")]
    return ReadStats(rows, token_count, seconds, _local_file_size(paths))


def benchmark_parquet(path_spec: str, *, max_rows: int) -> ReadStats:
    """Measure sequential Parquet decoding throughput."""
    rows = _limited_count(_parquet_row_count(path_spec), max_rows)
    start = time.perf_counter()
    token_count = 0
    for row_index, row in enumerate(_iter_parquet_rows(path_spec)):
        if row_index >= rows:
            break
        token_count += len(row["input_ids"])
    seconds = max(time.perf_counter() - start, 1e-12)
    paths = resolve_packed_parquet_paths(path_spec)
    return ReadStats(rows, token_count, seconds, _local_file_size(paths))


def _log_stats(name: str, stats: ReadStats) -> None:
    logger.info(
        "%s: rows=%d tokens=%d time=%.4fs rows/s=%.1f tokens/s=%.1f size=%.2f MiB",
        name,
        stats.rows,
        stats.tokens,
        stats.seconds,
        stats.rows_per_second,
        stats.tokens_per_second,
        stats.bytes_on_disk / (1024 * 1024),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, help="Packed Parquet file, directory, or glob.")
    parser.add_argument("--indexed", required=True, help="Packed SFT .bin/.idx prefix, directory, or glob.")
    parser.add_argument("--max-rows", type=int, default=0, help="Rows to compare and benchmark; 0 means all rows.")
    return parser.parse_args()


def main() -> None:
    """Run parity validation followed by sequential-read measurements."""
    args = parse_args()
    compared_rows = validate_parity(args.parquet, args.indexed, max_rows=args.max_rows)
    logger.info("Parity validation passed for %d rows", compared_rows)
    _log_stats("Parquet", benchmark_parquet(args.parquet, max_rows=args.max_rows))
    _log_stats("bin/idx", benchmark_indexed(args.indexed, max_rows=args.max_rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    main()
