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

"""Length distribution and file contracts of the agentic SFT mock dataset."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.unit

_SOURCE = Path(__file__).resolve().parents[3] / "examples/agentic_sft/mock_agentic_data.py"
_SPEC = importlib.util.spec_from_file_location("agentic_sft_mock_data_under_test", _SOURCE)
mock = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mock
_SPEC.loader.exec_module(mock)

SEQ = 65536


def _bucket(lengths: np.ndarray, index: int) -> np.ndarray:
    low, high = mock.bucket_bounds(SEQ)[index]
    return lengths[(lengths >= low) & (lengths <= high)]


def test_bucket_bounds_follow_minimum_cp_capacity():
    assert mock.bucket_bounds(SEQ) == [(1, 16384), (16385, 32768), (32769, 65536)]
    for bad in (0, 31, 65535):
        with pytest.raises(ValueError, match="multiple of 32"):
            mock.bucket_bounds(bad)


def test_bucket_counts_are_exact_largest_remainder_splits():
    assert mock.bucket_counts(5120) == [2560, 1536, 1024]
    assert mock.bucket_counts(256) == [128, 77, 51]
    assert mock.bucket_counts(1600) == [800, 480, 320]
    assert sum(mock.bucket_counts(7)) == 7
    for count, ratios in ((-1, mock.BUCKET_RATIOS), (3, (0.5, 0.2, 0.2)), (3, (1, 0))):
        with pytest.raises(ValueError):
            mock.bucket_counts(count, ratios)


def test_lengths_reproduce_the_mixed_distribution_counts_and_token_shares():
    lengths = mock.runtime_lengths(5120, SEQ, seed=1234)

    assert [len(_bucket(lengths, i)) for i in range(3)] == [2560, 1536, 1024]
    shares = [_bucket(lengths, i).sum() / lengths.sum() for i in range(3)]
    # The original example reports 19.16% / 36.01% / 44.83% token shares for its mixed selection.
    assert shares == pytest.approx([0.1916, 0.3601, 0.4483], abs=0.02)
    assert lengths.min() >= 1 and lengths.max() <= SEQ
    assert not np.array_equal(lengths, np.sort(lengths)), "buckets must be interleaved, not grouped"


def test_lengths_are_deterministic_in_the_seed():
    first = mock.runtime_lengths(200, SEQ, seed=1)
    assert np.array_equal(first, mock.runtime_lengths(200, SEQ, seed=1))
    assert not np.array_equal(first, mock.runtime_lengths(200, SEQ, seed=2))


def test_rows_hold_exactly_the_intended_tokens_with_a_supervised_completion(tmp_path):
    manifest = mock.write_mock_dataset(
        tmp_path, seq_length=4096, num_train=24, num_validation=0, vocab_size=1000, seed=5
    )

    lengths = manifest["splits"]["training"]["runtime_lengths"]
    rows = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert len(rows) == len(lengths) == 24
    for row, length in zip(rows, lengths):
        prompt, completion = row["input"].split(" "), row["output"].split(" ")
        tokens = [int(token) for token in prompt + completion]
        assert len(tokens) == length + 1, "the single next-token shift leaves `length` positions"
        assert min(tokens) >= 1 and max(tokens) <= 996, "ID 0 and the last three special-token IDs stay reserved"
        assert 1 <= len(completion) < len(tokens)
    assert not (tmp_path / "validation.jsonl").exists()


def test_validation_split_is_written_on_request(tmp_path):
    manifest = mock.write_mock_dataset(
        tmp_path, seq_length=4096, num_train=8, num_validation=5, vocab_size=1000, seed=5
    )

    assert len(manifest["splits"]["validation"]["runtime_lengths"]) == 5
    assert len((tmp_path / "validation.jsonl").read_text().splitlines()) == 5


def test_an_identical_request_reuses_files_and_a_different_one_regenerates(tmp_path):
    arguments = {"seq_length": 4096, "num_train": 8, "num_validation": 0, "vocab_size": 1000, "seed": 5}
    first = mock.write_mock_dataset(tmp_path, **arguments)
    stamp = (tmp_path / "training.jsonl").stat().st_mtime_ns

    assert mock.write_mock_dataset(tmp_path, **arguments) == first
    assert (tmp_path / "training.jsonl").stat().st_mtime_ns == stamp

    changed = mock.write_mock_dataset(tmp_path, **{**arguments, "num_train": 9})
    assert len(changed["splits"]["training"]["runtime_lengths"]) == 9


def test_every_caller_with_the_same_arguments_gets_byte_identical_rows(tmp_path):
    arguments = {"seq_length": 4096, "num_train": 6, "num_validation": 0, "vocab_size": 1000, "seed": 9}
    mock.write_mock_dataset(tmp_path / "a", **arguments)
    mock.write_mock_dataset(tmp_path / "b", **arguments)

    assert (tmp_path / "a/training.jsonl").read_bytes() == (tmp_path / "b/training.jsonl").read_bytes()


def test_ranks_other_than_local_zero_wait_for_the_manifest(tmp_path, monkeypatch):
    arguments = {"seq_length": 4096, "num_train": 4, "num_validation": 0, "vocab_size": 1000, "seed": 3}
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(TimeoutError, match="was not written"):
        mock.ensure_mock_dataset(tmp_path, timeout_s=0.05, **arguments)

    monkeypatch.setenv("LOCAL_RANK", "0")
    written = mock.ensure_mock_dataset(tmp_path, **arguments)
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert mock.ensure_mock_dataset(tmp_path, timeout_s=5, **arguments) == written


def test_invalid_counts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="num_train must be positive"):
        mock.write_mock_dataset(tmp_path, seq_length=4096, num_train=0, num_validation=0, vocab_size=1000, seed=1)


def test_local_tokenizer_reads_rows_back_as_exactly_the_intended_ids(tmp_path):
    from transformers import AutoTokenizer

    manifest = mock.write_mock_dataset(
        tmp_path, seq_length=4096, num_train=4, num_validation=0, vocab_size=1000, seed=11
    )
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / manifest["tokenizer"], local_files_only=True)

    assert len(tokenizer) == 1000
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (997, 998, 999)
    for line in (tmp_path / "training.jsonl").read_text().splitlines():
        row = json.loads(line)
        for column in ("input", "output"):
            ids = tokenizer(row[column], add_special_tokens=False)["input_ids"]
            assert ids == [int(token) for token in row[column].split(" ")]


def test_tokenizer_needs_room_for_its_special_tokens(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        mock.write_mock_tokenizer(tmp_path / "tokenizer", vocab_size=4)
