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

"""Unit tests for HFDatasetBuilder._load_dataset split normalization.

When load_dataset is called with a specific split, it returns a Dataset (not
DatasetDict).  The builder wraps it in a DatasetDict keyed by a normalized
split name.  These tests verify the normalization logic without network access
by mocking load_dataset.
"""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict

from megatron.bridge.data.builders.hf_dataset import HFDatasetBuilder


def _noop_process_fn(example, _tokenizer=None):
    return {"input": "", "output": "", "original_answers": []}


def _make_builder(split: str | None) -> HFDatasetBuilder:
    """Create a minimal HFDatasetBuilder with the given split value."""
    return HFDatasetBuilder(
        dataset_name="dummy/dataset",
        tokenizer=MagicMock(),
        process_example_fn=_noop_process_fn,
        split=split,
        seq_length=512,
        val_proportion=0.1,
    )


@pytest.mark.unit
class TestLoadDatasetSplitNormalization:
    """Verify that _load_dataset wraps a single Dataset with the correct split key."""

    @pytest.mark.parametrize(
        "split,expected_key",
        [
            ("train", "train"),
            ("train_1M", "train"),
            ("train[:10%]", "train"),
            ("validation", "validation"),
            ("valid", "validation"),
            ("val", "validation"),
            ("eval", "validation"),
            ("test", "test"),
            ("test[:100]", "test"),
        ],
    )
    @patch("megatron.bridge.data.builders.hf_dataset.load_dataset")
    def test_single_dataset_wrapped_with_correct_key(self, mock_load, split, expected_key):
        mock_dataset = MagicMock(spec=Dataset)
        mock_load.return_value = mock_dataset

        builder = _make_builder(split=split)
        result = builder._load_dataset()

        assert isinstance(result, DatasetDict)
        assert list(result.keys()) == [expected_key]
        assert result[expected_key] is mock_dataset

    @patch("megatron.bridge.data.builders.hf_dataset.load_dataset")
    def test_unknown_split_defaults_to_train(self, mock_load):
        mock_dataset = MagicMock(spec=Dataset)
        mock_load.return_value = mock_dataset

        builder = _make_builder(split="custom_split")
        result = builder._load_dataset()

        assert isinstance(result, DatasetDict)
        assert list(result.keys()) == ["train"]

    @patch("megatron.bridge.data.builders.hf_dataset.load_dataset")
    def test_none_split_returns_datasetdict_unchanged(self, mock_load):
        """When split=None, load_dataset returns a DatasetDict — no wrapping needed."""
        mock_ddict = MagicMock(spec=DatasetDict)
        mock_load.return_value = mock_ddict

        builder = _make_builder(split=None)
        result = builder._load_dataset()

        assert result is mock_ddict

    @patch("megatron.bridge.data.builders.hf_dataset.load_dataset")
    def test_datasetdict_returned_as_is(self, mock_load):
        """If load_dataset happens to return a DatasetDict even with a split, pass it through."""
        mock_ddict = DatasetDict({"train": MagicMock(spec=Dataset)})
        mock_load.return_value = mock_ddict

        builder = _make_builder(split="train")
        result = builder._load_dataset()

        assert result is mock_ddict
