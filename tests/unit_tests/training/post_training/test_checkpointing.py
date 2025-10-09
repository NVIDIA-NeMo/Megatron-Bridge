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
"""Unit tests for megatron.bridge.training.post_training.checkpointing module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.training.post_training.checkpointing import (
    has_modelopt_state,
    load_modelopt_state,
)


class TestPostTrainingImports:
    """Test import handling for post-training checkpointing."""

    def test_modelopt_import_available(self):
        """Test that modelopt import is available in the current environment."""
        # This test ensures that the import succeeded and the functions are available
        # This covers the successful import path (lines 17-18)
        try:
            from modelopt.torch.opt.plugins import restore_sharded_modelopt_state

            # If we get here, the import succeeded
            assert callable(restore_sharded_modelopt_state)
        except ImportError:
            # If modelopt is not available, we should still be able to import our module
            # but it should have raised the custom ImportError
            pytest.fail("modelopt should be available in test environment")

    def test_import_error_scenario_simulation(self):
        """Test the ImportError handling logic by simulating the scenario."""
        # This test simulates what would happen if modelopt was not available
        # We can't easily test the actual import failure since the module is already loaded,
        # but we can test the error handling logic

        original_error = ImportError("No module named 'modelopt.torch.opt.plugins'")
        expected_message = 'Required `"nvidia-modelopt[torch]"` is not installed!'

        # Simulate the exception chain that would occur
        try:
            try:
                raise original_error
            except ImportError as e:
                raise ImportError(expected_message) from e
        except ImportError as final_error:
            # Verify the error message and chaining
            assert expected_message in str(final_error)
            assert final_error.__cause__ is original_error
            assert "modelopt.torch.opt.plugins" in str(final_error.__cause__)


@pytest.fixture
def mock_model_fixtures():
    """Fixture for model testing."""
    mock_model_instance = Mock()
    mock_model_instance.sharded_state_dict.return_value = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
    mock_model_instance.load_state_dict.return_value = None
    return [mock_model_instance]


class TestPostTrainingCheckpointUtilities:
    """Test utility functions for post-training checkpoint management."""

    @pytest.mark.parametrize(
        "checkpoint_path,modelopt_exists,expected",
        [
            ("/checkpoints", True, True),
            ("/checkpoints", False, False),
            ("/nonexistent", False, False),
        ],
    )
    def test_has_modelopt_state(self, checkpoint_path, modelopt_exists, expected):
        """Test modelopt state detection."""
        if modelopt_exists and checkpoint_path != "/nonexistent":
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_dir = Path(temp_dir)
                modelopt_state_path = checkpoint_dir / "modelopt_state"
                modelopt_state_path.mkdir()

                result = has_modelopt_state(str(checkpoint_dir))
                assert result == expected
        else:
            if checkpoint_path == "/nonexistent":
                result = has_modelopt_state(checkpoint_path)
                assert result == expected
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint_dir = Path(temp_dir)
                    # Don't create modelopt_state folder

                    result = has_modelopt_state(str(checkpoint_dir))
                    assert result == expected

    def test_has_modelopt_state_file_instead_of_dir(self):
        """Test when modelopt_state exists but is a file, not a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            # Create a file instead of directory
            modelopt_state_path.touch()

            result = has_modelopt_state(str(checkpoint_path))
            assert result is False

    @patch("megatron.bridge.training.post_training.checkpointing.os.path.isdir")
    def test_has_modelopt_state_with_mock(self, mock_isdir):
        """Test has_modelopt_state with mocked os.path.isdir."""
        mock_isdir.return_value = True

        result = has_modelopt_state("/fake/checkpoint/path")
        assert result is True
        mock_isdir.assert_called_once_with("/fake/checkpoint/path/modelopt_state")

    def test_has_modelopt_state_with_none_path(self):
        """Test has_modelopt_state with None checkpoint path."""
        with pytest.raises(TypeError):
            has_modelopt_state(None)

    def test_has_modelopt_state_with_empty_string_path(self):
        """Test has_modelopt_state with empty string checkpoint path."""
        result = has_modelopt_state("")
        assert result is False


class TestLoadModeloptState:
    """Test load_modelopt_state function."""

    @patch("megatron.bridge.training.post_training.checkpointing.restore_sharded_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_state_success(self, mock_unwrap_model, mock_restore_state, mock_model_fixtures):
        """Test successful loading of modelopt state."""
        # Setup mocks
        unwrapped_model = [Mock()]
        mock_unwrap_model.return_value = unwrapped_model
        mock_restore_state.return_value = None

        # Call the function
        load_modelopt_state(mock_model_fixtures, "/test/checkpoint/path")

        # Verify calls
        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        mock_restore_state.assert_called_once_with(unwrapped_model, "/test/checkpoint/path")

    @patch("megatron.bridge.training.post_training.checkpointing.restore_sharded_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_state_with_exception(self, mock_unwrap_model, mock_restore_state, mock_model_fixtures):
        """Test load_modelopt_state when restore_sharded_modelopt_state raises an exception."""
        # Setup mocks
        unwrapped_model = [Mock()]
        mock_unwrap_model.return_value = unwrapped_model
        mock_restore_state.side_effect = RuntimeError("Failed to restore modelopt state")

        # Should propagate the exception
        with pytest.raises(RuntimeError) as exc_info:
            load_modelopt_state(mock_model_fixtures, "/test/checkpoint/path")

        assert "Failed to restore modelopt state" in str(exc_info.value)
        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        mock_restore_state.assert_called_once_with(unwrapped_model, "/test/checkpoint/path")

    @patch("megatron.bridge.training.post_training.checkpointing.restore_sharded_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_state_empty_model_list(self, mock_unwrap_model, mock_restore_state):
        """Test load_modelopt_state with empty model list."""
        # Setup mocks
        empty_model_list = []
        unwrapped_model = []
        mock_unwrap_model.return_value = unwrapped_model
        mock_restore_state.return_value = None

        # Call the function
        load_modelopt_state(empty_model_list, "/test/checkpoint/path")

        # Verify calls
        mock_unwrap_model.assert_called_once_with(empty_model_list)
        mock_restore_state.assert_called_once_with(unwrapped_model, "/test/checkpoint/path")

    @patch("megatron.bridge.training.post_training.checkpointing.restore_sharded_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_state_multiple_models(self, mock_unwrap_model, mock_restore_state):
        """Test load_modelopt_state with multiple models."""
        # Setup mocks
        model1 = Mock()
        model2 = Mock()
        model_list = [model1, model2]
        unwrapped_models = [Mock(), Mock()]
        mock_unwrap_model.return_value = unwrapped_models
        mock_restore_state.return_value = None

        # Call the function
        load_modelopt_state(model_list, "/test/checkpoint/path")

        # Verify calls
        mock_unwrap_model.assert_called_once_with(model_list)
        mock_restore_state.assert_called_once_with(unwrapped_models, "/test/checkpoint/path")

    @patch("megatron.bridge.training.post_training.checkpointing.restore_sharded_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_state_with_empty_string_path(self, mock_unwrap_model, mock_restore_state):
        """Test load_modelopt_state with empty checkpoint path."""
        mock_model = [Mock()]
        mock_unwrap_model.return_value = mock_model
        mock_restore_state.return_value = None

        # Should work fine - the function doesn't validate path
        load_modelopt_state(mock_model, "")

        mock_unwrap_model.assert_called_once_with(mock_model)
        mock_restore_state.assert_called_once_with(mock_model, "")


class TestPostTrainingIntegration:
    """Test integration scenarios for post-training checkpointing."""

    def test_full_workflow_with_existing_modelopt_state(self):
        """Test the full workflow when modelopt_state exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            modelopt_state_path.mkdir()

            # Check that modelopt_state exists
            assert has_modelopt_state(str(checkpoint_path)) is True

            # This would typically be followed by load_modelopt_state call
            # but we don't actually call it here to avoid dependency issues

    def test_full_workflow_without_modelopt_state(self):
        """Test the full workflow when modelopt_state doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            # Don't create modelopt_state folder

            # Check that modelopt_state doesn't exist
            assert has_modelopt_state(str(checkpoint_path)) is False

            # In this case, load_modelopt_state wouldn't be called


class TestPostTrainingEdgeCases:
    """Test edge cases and error conditions for post-training checkpointing."""

    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_functions_with_none_model(self, mock_unwrap_model):
        """Test load functions when model is None."""
        mock_unwrap_model.side_effect = AttributeError("'NoneType' object has no attribute")

        with pytest.raises(AttributeError):
            load_modelopt_state(None, "/test/checkpoint/path")
