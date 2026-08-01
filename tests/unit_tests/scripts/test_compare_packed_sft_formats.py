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

import importlib.util
import sys
from pathlib import Path

import pytest

from megatron.bridge.data.packing.indexed import write_packed_indexed_dataset
from megatron.bridge.data.packing.parquet import write_packed_parquet


pytest.importorskip("pyarrow")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "training" / "compare_packed_sft_formats.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_packed_sft_formats_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _rows() -> list[dict[str, list[int]]]:
    return [
        {
            "input_ids": [10 + offset, 11 + offset, 12 + offset, 20 + offset],
            "loss_mask": [0, 1, 0, 0],
            "seq_start_id": [0, 3],
        }
        for offset in range(4)
    ]


def test_compare_reports_parity_and_read_stats(tmp_path) -> None:
    module = _load_module()
    rows = _rows()
    parquet_path = tmp_path / "train.idx.parquet"
    indexed_prefix = tmp_path / "train.sft"
    write_packed_parquet(rows, parquet_path)
    write_packed_indexed_dataset(rows, indexed_prefix)

    assert module.validate_parity(str(parquet_path), str(indexed_prefix), max_rows=0) == len(rows)
    parquet_stats = module.benchmark_parquet(str(parquet_path), max_rows=0)
    indexed_stats = module.benchmark_indexed(str(indexed_prefix), max_rows=0)
    assert parquet_stats.rows == indexed_stats.rows == len(rows)
    assert parquet_stats.tokens == indexed_stats.tokens == 16
    assert parquet_stats.bytes_on_disk > 0
    assert indexed_stats.bytes_on_disk > 0


def test_compare_detects_field_mismatch(tmp_path) -> None:
    module = _load_module()
    parquet_rows = _rows()
    indexed_rows = _rows()
    indexed_rows[2]["loss_mask"][1] = 0
    parquet_path = tmp_path / "train.idx.parquet"
    indexed_prefix = tmp_path / "train.sft"
    write_packed_parquet(parquet_rows, parquet_path)
    write_packed_indexed_dataset(indexed_rows, indexed_prefix)

    with pytest.raises(ValueError, match="row 2, field 'loss_mask'"):
        module.validate_parity(str(parquet_path), str(indexed_prefix), max_rows=0)
