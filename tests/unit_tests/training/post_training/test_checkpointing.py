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
    load_modelopt_checkpoint,
    load_modelopt_state,
)


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


class TestLoadModeloptCheckpoint:
    """Test load_modelopt_checkpoint function."""

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_checkpoint_success(self, mock_unwrap_model, mock_dist_checkpointing, mock_model_fixtures):
        """Test successful loading of modelopt checkpoint."""
        # Setup mocks
        unwrapped_model = mock_model_fixtures
        mock_unwrap_model.return_value = unwrapped_model

        loaded_state_dict = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        mock_dist_checkpointing.load.return_value = loaded_state_dict

        # Call the function
        load_modelopt_checkpoint(mock_model_fixtures, "/test/checkpoint/path")

        # Verify calls
        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        unwrapped_model[0].sharded_state_dict.assert_called_once()
        mock_dist_checkpointing.load.assert_called_once()
        # Verify the call arguments
        call_args = mock_dist_checkpointing.load.call_args
        assert call_args[0][1] == "/test/checkpoint/path"  # checkpoint_path
        assert call_args[1]["strict"] == "assume_ok_unexpected"  # strict parameter

        unwrapped_model[0].load_state_dict.assert_called_once_with(loaded_state_dict, strict=False)

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_checkpoint_dist_load_failure(
        self, mock_unwrap_model, mock_dist_checkpointing, mock_model_fixtures
    ):
        """Test load_modelopt_checkpoint when dist_checkpointing.load fails."""
        # Setup mocks
        unwrapped_model = mock_model_fixtures
        mock_unwrap_model.return_value = unwrapped_model

        mock_dist_checkpointing.load.side_effect = RuntimeError("Failed to load checkpoint")

        # Should propagate the exception
        with pytest.raises(RuntimeError) as exc_info:
            load_modelopt_checkpoint(mock_model_fixtures, "/test/checkpoint/path")

        assert "Failed to load checkpoint" in str(exc_info.value)
        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        unwrapped_model[0].sharded_state_dict.assert_called_once()
        mock_dist_checkpointing.load.assert_called_once()
        # Model's load_state_dict should not be called due to the exception
        unwrapped_model[0].load_state_dict.assert_not_called()

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_checkpoint_model_load_failure(
        self, mock_unwrap_model, mock_dist_checkpointing, mock_model_fixtures
    ):
        """Test load_modelopt_checkpoint when model.load_state_dict fails."""
        # Setup mocks
        unwrapped_model = mock_model_fixtures
        mock_unwrap_model.return_value = unwrapped_model

        loaded_state_dict = {"weight": torch.randn(10, 10)}
        mock_dist_checkpointing.load.return_value = loaded_state_dict

        unwrapped_model[0].load_state_dict.side_effect = RuntimeError("Failed to load state dict")

        # Should propagate the exception
        with pytest.raises(RuntimeError) as exc_info:
            load_modelopt_checkpoint(mock_model_fixtures, "/test/checkpoint/path")

        assert "Failed to load state dict" in str(exc_info.value)
        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        unwrapped_model[0].sharded_state_dict.assert_called_once()
        mock_dist_checkpointing.load.assert_called_once()
        unwrapped_model[0].load_state_dict.assert_called_once_with(loaded_state_dict, strict=False)

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_checkpoint_sharded_state_dict_failure(
        self, mock_unwrap_model, mock_dist_checkpointing, mock_model_fixtures
    ):
        """Test load_modelopt_checkpoint when sharded_state_dict fails."""
        # Setup mocks
        unwrapped_model = mock_model_fixtures
        mock_unwrap_model.return_value = unwrapped_model

        unwrapped_model[0].sharded_state_dict.side_effect = RuntimeError("Failed to get sharded state dict")

        # Should propagate the exception
        with pytest.raises(RuntimeError) as exc_info:
            load_modelopt_checkpoint(mock_model_fixtures, "/test/checkpoint/path")

        assert "Failed to get sharded state dict" in str(exc_info.value)
        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        unwrapped_model[0].sharded_state_dict.assert_called_once()
        # dist_checkpointing.load should not be called due to the exception
        mock_dist_checkpointing.load.assert_not_called()
        unwrapped_model[0].load_state_dict.assert_not_called()

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_checkpoint_empty_state_dict(
        self, mock_unwrap_model, mock_dist_checkpointing, mock_model_fixtures
    ):
        """Test load_modelopt_checkpoint with empty sharded_state_dict."""
        # Setup mocks
        unwrapped_model = mock_model_fixtures
        mock_unwrap_model.return_value = unwrapped_model

        # Return empty state dict
        unwrapped_model[0].sharded_state_dict.return_value = {}
        loaded_state_dict = {}
        mock_dist_checkpointing.load.return_value = loaded_state_dict

        # Call the function - should work fine with empty dicts
        load_modelopt_checkpoint(mock_model_fixtures, "/test/checkpoint/path")

        # Verify calls
        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        unwrapped_model[0].sharded_state_dict.assert_called_once()
        mock_dist_checkpointing.load.assert_called_once()
        unwrapped_model[0].load_state_dict.assert_called_once_with(loaded_state_dict, strict=False)

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_checkpoint_with_complex_state_dict(self, mock_unwrap_model, mock_dist_checkpointing):
        """Test load_modelopt_checkpoint with complex state dict structures."""
        # Create a more complex mock model
        mock_model_instance = Mock()
        complex_sharded_state_dict = {
            "layers.0.weight": torch.randn(512, 512),
            "layers.0.bias": torch.randn(512),
            "layers.1.weight": torch.randn(512, 256),
            "layers.1.bias": torch.randn(256),
            "norm.weight": torch.randn(256),
            "norm.bias": torch.randn(256),
        }
        mock_model_instance.sharded_state_dict.return_value = complex_sharded_state_dict
        mock_model_instance.load_state_dict.return_value = None
        mock_model = [mock_model_instance]

        # Setup mocks
        unwrapped_model = mock_model
        mock_unwrap_model.return_value = unwrapped_model

        # Return modified state dict (simulating checkpoint loading)
        loaded_state_dict = {
            "layers.0.weight": torch.randn(512, 512),
            "layers.0.bias": torch.randn(512),
            "layers.1.weight": torch.randn(512, 256),
            "layers.1.bias": torch.randn(256),
            "norm.weight": torch.randn(256),
            "norm.bias": torch.randn(256),
        }
        mock_dist_checkpointing.load.return_value = loaded_state_dict

        # Call the function
        load_modelopt_checkpoint(mock_model, "/test/checkpoint/path")

        # Verify calls
        mock_unwrap_model.assert_called_once_with(mock_model)
        unwrapped_model[0].sharded_state_dict.assert_called_once()
        mock_dist_checkpointing.load.assert_called_once()

        # Verify the arguments to dist_checkpointing.load
        call_args = mock_dist_checkpointing.load.call_args
        sharded_state_dict_arg = call_args[0][0]
        assert len(sharded_state_dict_arg) == 6  # All 6 parameters
        assert "layers.0.weight" in sharded_state_dict_arg
        assert "norm.bias" in sharded_state_dict_arg

        unwrapped_model[0].load_state_dict.assert_called_once_with(loaded_state_dict, strict=False)

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_modelopt_checkpoint_with_empty_string_path(self, mock_unwrap_model, mock_dist_ckpt):
        """Test load_modelopt_checkpoint with empty checkpoint path."""
        mock_model_instance = Mock()
        mock_model_instance.sharded_state_dict.return_value = {}
        mock_model_instance.load_state_dict.return_value = None
        mock_model = [mock_model_instance]

        mock_unwrap_model.return_value = mock_model
        mock_dist_ckpt.load.return_value = {}

        # Should work fine - the function doesn't validate path
        load_modelopt_checkpoint(mock_model, "")

        mock_unwrap_model.assert_called_once_with(mock_model)
        mock_model_instance.sharded_state_dict.assert_called_once()
        mock_dist_ckpt.load.assert_called_once()
        mock_model_instance.load_state_dict.assert_called_once()


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

    @patch("megatron.bridge.training.post_training.checkpointing.restore_sharded_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing")
    def test_both_load_functions_with_same_model(self, mock_dist_ckpt, mock_unwrap_model, mock_restore_state):
        """Test calling both load functions with the same model."""
        # Setup mock model
        mock_model_instance = Mock()
        mock_model_instance.sharded_state_dict.return_value = {"param": torch.randn(10, 10)}
        mock_model_instance.load_state_dict.return_value = None
        mock_model = [mock_model_instance]

        # Setup mocks
        unwrapped_model = mock_model
        mock_unwrap_model.return_value = unwrapped_model
        mock_restore_state.return_value = None
        mock_dist_ckpt.load.return_value = {"param": torch.randn(10, 10)}

        checkpoint_path = "/test/checkpoint/path"

        # Call both functions
        load_modelopt_state(mock_model, checkpoint_path)
        load_modelopt_checkpoint(mock_model, checkpoint_path)

        # Verify both functions were called correctly
        assert mock_unwrap_model.call_count == 2
        mock_restore_state.assert_called_once_with(unwrapped_model, checkpoint_path)
        mock_dist_ckpt.load.assert_called_once()
        mock_model_instance.sharded_state_dict.assert_called_once()
        mock_model_instance.load_state_dict.assert_called_once()


class TestPostTrainingEdgeCases:
    """Test edge cases and error conditions for post-training checkpointing."""

    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_load_functions_with_none_model(self, mock_unwrap_model):
        """Test load functions when model is None."""
        mock_unwrap_model.side_effect = AttributeError("'NoneType' object has no attribute")

        with pytest.raises(AttributeError):
            load_modelopt_state(None, "/test/checkpoint/path")

        with pytest.raises(AttributeError):
            load_modelopt_checkpoint(None, "/test/checkpoint/path")
