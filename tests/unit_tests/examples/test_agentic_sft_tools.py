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

"""Tests for the run comparison and item-token tools of ``examples/agentic_sft``."""

import json
import logging
import runpy
from pathlib import Path

import pytest
import torch

from megatron.bridge.data.packing.global_batch import build_unpacked_sequence_sample


pytestmark = pytest.mark.unit

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "agentic_sft"
_COMPARE = runpy.run_path(str(_EXAMPLE_DIR / "compare_runs.py"))
_ITEM_TOKENS = runpy.run_path(str(_EXAMPLE_DIR / "item_tokens.py"))


def _log(step_seconds: list[float], global_batch_size: int) -> str:
    return "\n".join(
        f" iteration {i:8d}/{len(step_seconds):8d} | consumed samples: {i * global_batch_size:12d} | "
        f"elapsed time per iteration (ms): {1000 * seconds:.1f} | lm loss: 1.500000E+00 |"
        for i, seconds in enumerate(step_seconds, start=1)
    )


class _RowDataset(torch.utils.data.Dataset):
    """Items of several sequences, collated like global-batch packing's unpacked samples."""

    def __init__(self, rows: list[list[int]], pad_to_multiple_of: int) -> None:
        self.rows = rows
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> list[int]:
        return self.rows[index]

    def collate_fn(self, batch: list[list[int]]) -> list[dict[str, torch.Tensor]]:
        return [
            build_unpacked_sequence_sample(
                [1] * length,
                [1] * length,
                [1.0] * length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_token_id=0,
            )
            for row in batch
            for length in row
        ]


def test_item_sequence_tokens_reads_items_in_order_without_alignment_padding():
    dataset = _RowDataset([[5, 3], [16], [7, 1, 2]], pad_to_multiple_of=8)

    assert _ITEM_TOKENS["item_sequence_tokens"](dataset) == [[5, 3], [16], [7, 1, 2]]
    assert _ITEM_TOKENS["item_sequence_tokens"](dataset, num_items=2) == [[5, 3], [16]]


def test_capacity_shares_group_sequences_by_the_smallest_cp_group_that_fits():
    shares = _ITEM_TOKENS["capacity_shares"]([10, 16, 17, 32, 100], tokens_per_rank=16)

    assert list(shares) == [1, 2, 8]
    assert shares[1] == pytest.approx((2 / 5, 26 / 175))
    assert shares[2] == pytest.approx((2 / 5, 49 / 175))
    assert shares[8] == pytest.approx((1 / 5, 100 / 175))


def test_read_item_tokens_sums_the_sequences_of_each_item(tmp_path: Path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([[5, 3], [16], 9]))

    assert _COMPARE["read_item_tokens"](path) == [8, 16, 9]


def test_tokens_per_step_rejects_token_counts_that_do_not_cover_the_run():
    tokens_per_step = _COMPARE["tokens_per_step"]

    assert tokens_per_step([1, 2, 3, 4], 2, [1, 2]) == {1: 3, 2: 7}
    with pytest.raises(ValueError, match="only 4 token counts"):
        tokens_per_step([1, 2, 3, 4], 2, [1, 2, 3])


def test_item_tokens_compare_runs_that_read_different_items(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    # Online: two samples per step; offline: one packed row per step. Both take one second per step.
    (tmp_path / "online.log").write_text(_log([9.0, 1.0, 1.0], 2))
    (tmp_path / "offline.log").write_text(_log([9.0, 1.0, 1.0], 1))
    (tmp_path / "online.json").write_text(json.dumps([[100], [100], [100], [200], [300], [100]]))
    (tmp_path / "offline.json").write_text(json.dumps([[100, 100], [300, 300], [300, 300]]))

    with caplog.at_level(logging.INFO):
        _COMPARE["main"](
            [
                "--baseline",
                str(tmp_path / "online.log"),
                "--candidate",
                str(tmp_path / "offline.log"),
                "--warmup",
                "1",
                "--baseline-item-tokens",
                str(tmp_path / "online.json"),
                "--baseline-global-batch-size",
                "2",
                "--candidate-item-tokens",
                str(tmp_path / "offline.json"),
                "--candidate-global-batch-size",
                "1",
            ]
        )

    # Steps 2-3: 700 online tokens and 1,200 offline tokens in two seconds each.
    assert "350 runtime tokens/s" in caplog.text
    assert "600 runtime tokens/s" in caplog.text
    assert "+71.43% throughput (runtime tokens/s)" in caplog.text


def test_item_tokens_need_both_runs_and_exclude_mock_lengths(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(_log([1.0, 1.0], 1))
    items = tmp_path / "items.json"
    items.write_text(json.dumps([[1], [1]]))
    common = ["--baseline", str(log), "--candidate", str(log)]

    with pytest.raises(SystemExit):
        _COMPARE["main"]([*common, "--baseline-item-tokens", str(items), "--global-batch-size", "1"])
    with pytest.raises(SystemExit):
        _COMPARE["main"]([*common, "--item-tokens", str(items)])
    with pytest.raises(SystemExit):
        _COMPARE["main"](
            [*common, "--item-tokens", str(items), "--global-batch-size", "1", "--mock-seq-length", "64"]
        )
