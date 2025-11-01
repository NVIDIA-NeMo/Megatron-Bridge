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
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME

from megatron.bridge.training.post_training.checkpointing import (
    _has_only_kd_state,
    has_modelopt_state,
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

    def test_has_only_kd_state_returns_true(self):
        """Test _has_only_kd_state when modelopt_state contains only kd_loss state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            modelopt_state_path = Path(temp_dir)
            common_state_file = modelopt_state_path / COMMON_STATE_FNAME

            # Create modelopt_state_dict with only kd_loss
            modelopt_state = {"modelopt_state_dict": [("kd_loss", {"some": "data"})]}
            torch.save(modelopt_state, common_state_file)

            result = _has_only_kd_state(str(modelopt_state_path))
            assert result is True

    def test_has_only_kd_state_returns_false_multiple_states(self):
        """Test _has_only_kd_state when modelopt_state contains multiple states."""
        with tempfile.TemporaryDirectory() as temp_dir:
            modelopt_state_path = Path(temp_dir)
            common_state_file = modelopt_state_path / COMMON_STATE_FNAME

            # Create modelopt_state_dict with multiple states including kd_loss
            modelopt_state = {
                "modelopt_state_dict": [
                    ("kd_loss", {"some": "data"}),
                    ("quantization", {"other": "data"}),
                ]
            }
            torch.save(modelopt_state, common_state_file)

            result = _has_only_kd_state(str(modelopt_state_path))
            assert result is False

    def test_has_only_kd_state_returns_false_different_state(self):
        """Test _has_only_kd_state when modelopt_state contains a single state that is not kd_loss."""
        with tempfile.TemporaryDirectory() as temp_dir:
            modelopt_state_path = Path(temp_dir)
            common_state_file = modelopt_state_path / COMMON_STATE_FNAME

            # Create modelopt_state_dict with only quantization state (not kd_loss)
            modelopt_state = {"modelopt_state_dict": [("quantization", {"some": "data"})]}
            torch.save(modelopt_state, common_state_file)

            result = _has_only_kd_state(str(modelopt_state_path))
            assert result is False

    def test_has_modelopt_state_with_ignore_kd_state_true_only_kd(self):
        """Test has_modelopt_state with ignore_kd_state=True when only kd_loss state exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            modelopt_state_path.mkdir()
            common_state_file = modelopt_state_path / COMMON_STATE_FNAME

            # Create modelopt_state_dict with only kd_loss
            modelopt_state = {"modelopt_state_dict": [("kd_loss", {"some": "data"})]}
            torch.save(modelopt_state, common_state_file)

            # When ignore_kd_state=True and only kd_loss exists, should return True
            result = has_modelopt_state(str(checkpoint_path), ignore_kd_state=True)
            assert result is True

    def test_has_modelopt_state_with_ignore_kd_state_true_multiple_states(self):
        """Test has_modelopt_state with ignore_kd_state=True when multiple states exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            modelopt_state_path.mkdir()
            common_state_file = modelopt_state_path / COMMON_STATE_FNAME

            # Create modelopt_state_dict with multiple states
            modelopt_state = {
                "modelopt_state_dict": [
                    ("kd_loss", {"some": "data"}),
                    ("quantization", {"other": "data"}),
                ]
            }
            torch.save(modelopt_state, common_state_file)

            # When ignore_kd_state=True but multiple states exist, should return False
            result = has_modelopt_state(str(checkpoint_path), ignore_kd_state=True)
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
