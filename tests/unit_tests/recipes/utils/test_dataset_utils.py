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

from unittest.mock import patch

import pytest

from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths


@pytest.mark.unit
class TestGetBlendFieldsFromDataPaths:
    """Test cases for the get_blend_fields_from_data_paths function."""

    def test_mock_mode_explicit(self):
        """Test function with explicit mock=True."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(mock=True)

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_mock_mode_no_data_config(self):
        """Test function with no data configuration (should default to mock)."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths()

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_mock_mode_with_data_paths_but_mock_true(self):
        """Test function with data paths but mock=True (should ignore data paths)."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["/path/to/data1", "/path/to/data2"], mock=True
        )

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_data_paths(self):
        """Test function with data_paths and blend weights returned."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["/path/to/data1", "/path/to/data2"]
        )

        assert blend == (["/path/to/data1", "/path/to/data2"], None)
        assert blend_per_split is None
        assert split == "9999,8,2"

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["0.6", "/path/to/data1", "0.4", "/path/to/data2"]
        )

        assert blend == (["/path/to/data1", "/path/to/data2"], [0.6, 0.4])
        assert blend_per_split is None
        assert split == "9999,8,2"

    def test_data_args_path_with_blend_weights(self):
        """Test function with data_args_path and blend weights returned."""

        import tempfile

        content = "0.6\n/path/to/data1\n0.4\n/path/to/data2\n"
        with tempfile.NamedTemporaryFile(prefix="datasrc_") as data_args_file:
            data_args_file.write(str.encode(content))
            data_args_file.seek(0)

            blend, blend_per_split, split = get_blend_fields_from_data_paths(data_args_path=data_args_file.name)

            assert blend == (["/path/to/data1", "/path/to/data2"], [0.6, 0.4])
            assert blend_per_split is None
            assert split == "9999,8,2"

    def test_per_split_paths_with_blend_per_split_weights(self):
        """Test function with train/valid/test paths and blend_per_split weights."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            train_data_path=["/path/to/train1", "/path/to/train2"],
            valid_data_path=["/path/to/valid1"],
            test_data_path=["/path/to/test1", "/path/to/test2"],
        )

        assert blend is None
        assert blend_per_split == [
            (["/path/to/train1", "/path/to/train2"], None),
            (["/path/to/valid1"], None),
            (["/path/to/test1", "/path/to/test2"], None),
        ]
        assert split is None

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            train_data_path=["0.8", "/path/to/train1", "0.2", "/path/to/train2"],
            valid_data_path=["0.7", "/path/to/valid1", "0.3", "/path/to/valid2"],
            test_data_path=["0.6", "/path/to/test1", "0.4", "/path/to/test2"],
        )

        assert blend is None
        assert blend_per_split == [
            (["/path/to/train1", "/path/to/train2"], [0.8, 0.2]),
            (["/path/to/valid1", "/path/to/valid2"], [0.7, 0.3]),
            (["/path/to/test1", "/path/to/test2"], [0.6, 0.4]),
        ]
        assert split is None

    def test_per_split_data_args_path_with_blend_per_split_weights(self):
        """Test function with per_split_data_args_path and blend_per_split weights."""

        import json
        import tempfile

        content = {
            "train": ["0.8", "/path/to/train1", "0.2", "/path/to/train2"],
            "valid": ["0.7", "/path/to/valid1", "0.3", "/path/to/valid2"],
            "test": ["0.6", "/path/to/test1", "0.4", "/path/to/test2"],
        }
        with tempfile.NamedTemporaryFile("w+", prefix="datasrc_", suffix=".json") as per_split_data_args_file:
            json.dump(content, per_split_data_args_file)
            per_split_data_args_file.seek(0)

            blend, blend_per_split, split = get_blend_fields_from_data_paths(
                per_split_data_args_path=per_split_data_args_file.name
            )

            assert blend is None
            assert blend_per_split == [
                (["/path/to/train1", "/path/to/train2"], [0.8, 0.2]),
                (["/path/to/valid1", "/path/to/valid2"], [0.7, 0.3]),
                (["/path/to/test1", "/path/to/test2"], [0.6, 0.4]),
            ]
            assert split is None

    def test_prioritize_blend_over_blend_per_split(self):
        """Test that data_paths takes priority over split data paths when both are provided."""

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["/path/to/data1", "/path/to/data2"],
            train_data_path=["/path/to/train1", "/path/to/train2"],
            valid_data_path=["/path/to/valid1", "/path/to/valid2"],
            test_data_path=["/path/to/test1", "/path/to/test2"],
        )

        # Should prioritize blend over blend_per_split
        assert blend == (["/path/to/data1", "/path/to/data2"], None)
        assert blend_per_split is None
        assert split == "9999,8,2"

    @patch("megatron.bridge.recipes.utils.dataset_utils.get_blend_and_blend_per_split")
    def test_fallback_to_mock_when_no_weights(self, mock_get_blend):
        """Test function falls back to mock mode when no weights are returned."""
        mock_get_blend.return_value = (None, None)

        blend, blend_per_split, split = get_blend_fields_from_data_paths(data_paths=["/some/path"])

        # Should fall back to mock mode
        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_blend_per_split_with_empty_paths(self):
        """Test blend_per_split with empty paths (should create None entries)."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            valid_data_path=["/path/to/valid1"],  # Only valid paths
            test_data_path=None,  # No test paths
        )

        assert blend is None
        assert blend_per_split == [
            None,  # train_paths is empty, so None
            (["/path/to/valid1"], None),  # valid_paths exists
            None,  # test_paths is None, so None
        ]
        assert split is None

    def test_edge_case_empty_lists(self):
        """Test edge case with empty lists for all path parameters."""

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=[],
            train_data_path=[],
            valid_data_path=[],
            test_data_path=[],
        )

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"
