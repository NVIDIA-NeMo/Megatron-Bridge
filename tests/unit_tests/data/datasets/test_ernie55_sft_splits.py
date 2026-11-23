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

"""The held-out splits of ERNIE 5.5's SFT dataset: ``valid_path`` and ``test_path``.

These tests read JSONL files and build no model, so they run on Megatron-Core main as well as dev. They are here
and not in ``tests/unit_tests/models/ernie`` because that directory's ``conftest.py`` keeps ``test_ernie55_*.py``
out of collection.
"""

import json
from pathlib import Path

import pytest

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.models.ernie.ernie55_data import Ernie55SFTDatasetConfig


pytestmark = pytest.mark.unit


def _write_jsonl(path: Path, samples: list[tuple[list[int], list[int]]]) -> str:
    path.write_text("".join(json.dumps({"input_ids": ids, "labels": labels}) + "\n" for ids, labels in samples))
    return str(path)


def _build_all_splits(config: Ernie55SFTDatasetConfig):
    config.finalize()
    return config.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=1, test_samples=1))


def test_validation_and_test_splits_read_their_own_files(tmp_path):
    train_path = _write_jsonl(tmp_path / "train.jsonl", [([1, 2, 3, 4], [1, 2, 3, 4])])
    valid_path = _write_jsonl(tmp_path / "valid.jsonl", [([5, 6, 7], [5, 6, 7])])
    test_path = _write_jsonl(tmp_path / "test.jsonl", [([8, 9], [8, 9])])
    config = Ernie55SFTDatasetConfig(
        seq_length=16, pad_token_id=0, path=train_path, valid_path=valid_path, test_path=test_path
    )
    train, valid, test = _build_all_splits(config)
    assert train[0] == [([1, 2, 3, 4], [2, 3, 4, -100])]
    assert valid[0] == [([5, 6, 7], [6, 7, -100])]
    assert test[0] == [([8, 9], [9, -100])]
    # The split configurations are copies that only differ in the file they read.
    assert valid.config.path == valid_path and valid.config.seq_length == 16
    assert config.path == train_path and config.valid_path == valid_path


def test_splits_default_to_the_training_file(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", [([1, 2, 3, 4], [1, 2, 3, 4])])
    config = Ernie55SFTDatasetConfig(seq_length=16, pad_token_id=0, path=path)
    train, valid, test = _build_all_splits(config)
    assert valid.config is config and test.config is config
    assert valid[0] == train[0] == test[0]


def test_the_two_held_out_files_are_independent(tmp_path):
    train_path = _write_jsonl(tmp_path / "train.jsonl", [([1, 2, 3, 4], [1, 2, 3, 4])])
    test_path = _write_jsonl(tmp_path / "test.jsonl", [([8, 9], [8, 9])])
    config = Ernie55SFTDatasetConfig(seq_length=16, pad_token_id=0, path=train_path, test_path=test_path)
    train, valid, test = _build_all_splits(config)
    # Only test_path is set: the validation split still reads the training file, the test split its own file.
    assert valid.config is config and valid[0] == train[0]
    assert test.config is not config and test[0] == [([8, 9], [9, -100])]


@pytest.mark.parametrize("split", ["valid_path", "test_path"])
def test_held_out_files_need_a_training_file(tmp_path, split):
    held_out = _write_jsonl(tmp_path / "held_out.jsonl", [([5, 6, 7], [5, 6, 7])])
    config = Ernie55SFTDatasetConfig(seq_length=16, pad_token_id=0, mock_document_lengths=(4,), **{split: held_out})
    with pytest.raises(ValueError, match="set dataset.path too"):
        config.finalize()
