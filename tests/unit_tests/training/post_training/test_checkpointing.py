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

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME

from megatron.bridge.training.post_training.checkpointing import (
    _get_modelopt_checkpoint_path,
    _validate_modelopt_checkpointing,
    has_modelopt_state,
    load_modelopt_state,
)
from megatron.bridge.training.state import TrainState


@pytest.fixture
def mock_model_fixtures():
    """Fixture for model testing."""
    mock_model_instance = Mock()
    mock_model_instance.sharded_state_dict.return_value = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
    mock_model_instance.load_state_dict.return_value = None
    return [mock_model_instance]


def _write_modelopt_common_state(modelopt_state_dir: Path, states, modelopt_version: str | None = None):
    """Helper to write a common_state file with the given modelopt states."""
    with open(modelopt_state_dir / "metadata.json", "w") as metadata_file:
        json.dump({"sharded_backend": "torch_dist", "sharded_backend_version": 1}, metadata_file)
    modelopt_state = {"modelopt_state_dict": states}
    if modelopt_version is not None:
        modelopt_state["modelopt_version"] = modelopt_version
    common_state_file = modelopt_state_dir / COMMON_STATE_FNAME
    torch.save(modelopt_state, common_state_file)
    return common_state_file


@pytest.fixture
def single_rank_process_group(tmp_path):
    """Initialize a single-rank process group for real distributed-checkpoint saves."""
    initialized_here = not torch.distributed.is_initialized()
    if initialized_here:
        torch.distributed.init_process_group(
            "gloo",
            init_method=(tmp_path / "process_group_init").as_uri(),
            rank=0,
            world_size=1,
        )

    yield

    if initialized_here:
        torch.distributed.destroy_process_group()


class TestGetModeloptCheckpointPath:
    """Test _get_modelopt_checkpoint_path helper function."""

    def test_returns_path_when_no_directory(self):
        """Test that it returns the original path when it's not a directory."""
        result = _get_modelopt_checkpoint_path("/nonexistent/path")
        assert result == "/nonexistent/path"

    def test_returns_path_when_empty_string(self):
        """Test that it returns empty string when checkpoint_path is empty."""
        result = _get_modelopt_checkpoint_path("")
        assert result == ""

    def test_returns_path_when_none(self):
        """Test that it returns None when checkpoint_path is None."""
        result = _get_modelopt_checkpoint_path(None)
        assert result is None

    def test_returns_path_when_no_iter_folders(self):
        """Test that it returns the original path when there are no iter_* folders."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            # Create some non-iter folders
            (checkpoint_path / "other_folder").mkdir()
            (checkpoint_path / "model.pt").touch()

            result = _get_modelopt_checkpoint_path(str(checkpoint_path))
            assert result == str(checkpoint_path)

    def test_returns_latest_iter_folder_with_single_iter(self):
        """Test that it returns the iter_* folder when only one exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder = checkpoint_path / "iter_0000100"
            iter_folder.mkdir()

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:
                mock_load.return_value = {"iteration": 100}

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                assert result == str(iter_folder)
                mock_load.assert_called_once_with(str(iter_folder))

    def test_returns_latest_iter_folder_with_multiple_iters(self):
        """Test that it returns the folder with the highest iteration number."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_200 = checkpoint_path / "iter_0000200"
            iter_folder_150 = checkpoint_path / "iter_0000150"

            iter_folder_100.mkdir()
            iter_folder_200.mkdir()
            iter_folder_150.mkdir()

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:

                def load_side_effect(path):
                    if "iter_0000100" in path:
                        return {"iteration": 100}
                    elif "iter_0000200" in path:
                        return {"iteration": 200}
                    elif "iter_0000150" in path:
                        return {"iteration": 150}
                    return None

                mock_load.side_effect = load_side_effect

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                assert result == str(iter_folder_200)
                assert mock_load.call_count == 3

    def test_returns_tracked_iter_folder_before_higher_iter(self):
        """Test that ModelOpt state follows the checkpoint selected by the native tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_200 = checkpoint_path / "iter_0000200"
            iter_folder_100.mkdir()
            iter_folder_200.mkdir()
            (checkpoint_path / "latest_checkpointed_iteration.txt").write_text("100", encoding="utf-8")

            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:
                mock_load.side_effect = lambda path: {"iteration": 100 if path == str(iter_folder_100) else 200}

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))

            assert result == str(iter_folder_100)

    def test_explicit_step_overrides_higher_iter(self):
        """Test that ModelOpt state follows the explicit native checkpoint step."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_200 = checkpoint_path / "iter_0000200"
            iter_folder_100.mkdir()
            iter_folder_200.mkdir()

            result = _get_modelopt_checkpoint_path(str(checkpoint_path), ckpt_step=100)

            assert result == str(iter_folder_100)

    def test_handles_exception_during_state_dict_load(self):
        """Test that it skips checkpoints that fail to load and continues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_200 = checkpoint_path / "iter_0000200"
            iter_folder_corrupted = checkpoint_path / "iter_0000150"

            iter_folder_100.mkdir()
            iter_folder_200.mkdir()
            iter_folder_corrupted.mkdir()

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:

                def load_side_effect(path):
                    if "iter_0000100" in path:
                        return {"iteration": 100}
                    elif "iter_0000200" in path:
                        return {"iteration": 200}
                    elif "iter_0000150" in path:
                        # Simulate a corrupted checkpoint
                        raise RuntimeError("Failed to load checkpoint")
                    return None

                mock_load.side_effect = load_side_effect

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                # Should still return the highest valid iteration (200)
                assert result == str(iter_folder_200)
                assert mock_load.call_count == 3

    def test_returns_original_path_when_all_iters_fail_to_load(self):
        """Test that it returns the original path when all iter folders fail to load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_200 = checkpoint_path / "iter_0000200"

            iter_folder_100.mkdir()
            iter_folder_200.mkdir()

            # Mock the dist_checkpointing.load_common_state_dict to always fail
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:
                mock_load.side_effect = RuntimeError("Failed to load checkpoint")

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                # Should return the original path since no valid checkpoint was found
                assert result == str(checkpoint_path)
                assert mock_load.call_count == 2

    def test_handles_state_dict_without_iteration_key(self):
        """Test that it handles state dicts without 'iteration' key (defaults to 0)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_no_iter = checkpoint_path / "iter_0000050"

            iter_folder_100.mkdir()
            iter_folder_no_iter.mkdir()

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:

                def load_side_effect(path):
                    if "iter_0000100" in path:
                        return {"iteration": 100}
                    elif "iter_0000050" in path:
                        # State dict without iteration key
                        return {"model": "data"}
                    return None

                mock_load.side_effect = load_side_effect

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                # Should return iter_0000100 since it has the highest iteration
                assert result == str(iter_folder_100)
                assert mock_load.call_count == 2

    def test_handles_state_dict_returning_none(self):
        """Test that it handles when load_common_state_dict returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_none = checkpoint_path / "iter_0000200"

            iter_folder_100.mkdir()
            iter_folder_none.mkdir()

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:

                def load_side_effect(path):
                    if "iter_0000100" in path:
                        return {"iteration": 100}
                    elif "iter_0000200" in path:
                        # Return None
                        return None
                    return None

                mock_load.side_effect = load_side_effect

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                # Should return iter_0000100 since iter_0000200 returned None
                assert result == str(iter_folder_100)
                assert mock_load.call_count == 2

    def test_handles_oserror_during_listdir(self):
        """Test that it returns original path when OSError occurs during listdir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Mock os.listdir to raise OSError
            with patch("megatron.bridge.training.post_training.checkpointing.os.listdir") as mock_listdir:
                mock_listdir.side_effect = OSError("Permission denied")

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                assert result == str(checkpoint_path)

    def test_handles_filenotfounderror_during_listdir(self):
        """Test that it returns original path when FileNotFoundError occurs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Mock os.listdir to raise FileNotFoundError
            with patch("megatron.bridge.training.post_training.checkpointing.os.listdir") as mock_listdir:
                mock_listdir.side_effect = FileNotFoundError("Directory not found")

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                assert result == str(checkpoint_path)

    def test_handles_iter_folders_mixed_with_files(self):
        """Test that it only considers directories starting with iter_, not files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder = checkpoint_path / "iter_0000100"
            iter_folder.mkdir()

            # Create a file that starts with iter_ (should be ignored)
            iter_file = checkpoint_path / "iter_0000200.txt"
            iter_file.touch()

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:
                mock_load.return_value = {"iteration": 100}

                result = _get_modelopt_checkpoint_path(str(checkpoint_path))
                # Should only consider the directory, not the file
                assert result == str(iter_folder)
                # Should only be called once (for the directory, not the file)
                mock_load.assert_called_once_with(str(iter_folder))


class TestPostTrainingCheckpointUtilities:
    """Test utility functions for post-training checkpoint management."""

    @pytest.mark.parametrize("non_persistent_ckpt_type", ["global", "local"])
    def test_rejects_non_persistent_modelopt_restore(self, non_persistent_ckpt_type):
        """Test that ModelOpt restoration rejects ambiguous non-persistent checkpoint loading."""
        with pytest.raises(
            ValueError,
            match="ModelOpt checkpoint restoration is not supported with non-persistent checkpoint loading",
        ):
            _validate_modelopt_checkpointing(non_persistent_ckpt_type)

    def test_allows_persistent_modelopt_restore(self):
        """Test that persistent ModelOpt restoration remains supported."""
        _validate_modelopt_checkpointing(None)

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

                _write_modelopt_common_state(modelopt_state_path, [("quantization", {"value": 1})])

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

    @patch("megatron.bridge.training.post_training.checkpointing._load_extra_state_from_sharded_checkpoint")
    @patch("megatron.bridge.training.post_training.checkpointing.mto.restore_from_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.mto.ModeloptStateManager.is_converted")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_current_torch_dist_save_detect_restore_contract(
        self,
        mock_unwrap_model,
        mock_is_converted,
        mock_restore_from_state,
        mock_load_extra_state,
        mock_model_fixtures,
        tmp_path,
        single_rank_process_group,
    ):
        """Detect and route restoration for ModelOpt state saved in the current MCore layout."""
        checkpoint_path = tmp_path / "checkpoint"
        iteration_path = checkpoint_path / "iter_0000100"
        newer_iteration_path = checkpoint_path / "iter_0000200"
        modelopt_state_path = iteration_path / "modelopt_state"
        iteration_path.mkdir(parents=True)
        newer_iteration_path.mkdir()
        torch.save(TrainState(step=100).state_dict(), checkpoint_path / "latest_train_state.pt")

        # The torch_dist strategy synchronizes CUDA even though this fixture only stores objects.
        with patch("torch.cuda.synchronize"), patch("torch.cuda.current_device", return_value=torch.device("cpu")):
            dist_checkpointing.save({"iteration": 100}, str(iteration_path), ("torch_dist", 1))
            modelopt_state_path.mkdir()
            dist_checkpointing.save(
                {
                    "modelopt_version": "0.44.0rc5",
                    "modelopt_state_dict": [("quantization", {"value": 1})],
                },
                str(modelopt_state_path),
                ("torch_dist", 1),
            )

        assert not (modelopt_state_path / COMMON_STATE_FNAME).exists()
        assert not (newer_iteration_path / "modelopt_state").exists()
        assert has_modelopt_state(str(checkpoint_path)) is True

        original_model = Mock()
        restored_model = Mock()
        unwrapped_model = [original_model]
        mock_unwrap_model.return_value = unwrapped_model
        mock_is_converted.return_value = False
        mock_restore_from_state.return_value = restored_model
        load_modelopt_state(mock_model_fixtures, str(checkpoint_path))

        mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
        mock_restore_from_state.assert_called_once()
        assert mock_restore_from_state.call_args.args[0] is original_model
        assert mock_restore_from_state.call_args.args[1]["modelopt_state_dict"] == [("quantization", {"value": 1})]
        assert unwrapped_model == [restored_model]
        mock_load_extra_state.assert_called_once_with(restored_model, str(iteration_path), prefix="")

    def test_has_modelopt_state_file_instead_of_dir(self):
        """Test when modelopt_state exists but is a file, not a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            # Create a file instead of directory
            modelopt_state_path.touch()

            result = has_modelopt_state(str(checkpoint_path))
            assert result is False

    @patch("megatron.bridge.training.post_training.checkpointing.dist_checkpointing.load_common_state_dict")
    @patch("megatron.bridge.training.post_training.checkpointing.os.path.isdir")
    def test_has_modelopt_state_with_mock(self, mock_isdir, mock_load_common_state):
        """Test has_modelopt_state with a mocked distributed-checkpoint loader."""
        mock_isdir.return_value = True
        mock_load_common_state.return_value = {"modelopt_state_dict": [("quantization", {"foo": "bar"})]}

        result = has_modelopt_state("/fake/checkpoint/path")
        assert result is True
        # has_modelopt_state calls isdir twice:
        # 1. In _get_modelopt_checkpoint_path() to check if checkpoint_path exists
        # 2. In has_modelopt_state() to check if modelopt_state subfolder exists
        assert mock_isdir.call_count == 2
        mock_isdir.assert_any_call("/fake/checkpoint/path")
        mock_isdir.assert_any_call("/fake/checkpoint/path/modelopt_state")
        mock_load_common_state.assert_called_once_with("/fake/checkpoint/path/modelopt_state")

    def test_has_modelopt_state_with_none_path(self):
        """Test has_modelopt_state with None checkpoint path."""
        with pytest.raises(TypeError):
            has_modelopt_state(None)

    def test_has_modelopt_state_with_empty_string_path(self):
        """Test has_modelopt_state with empty string checkpoint path."""
        result = has_modelopt_state("")
        assert result is False

    def test_has_modelopt_state_returns_false_for_only_kd_loss(self):
        """Test has_modelopt_state ignores a sole kd_loss entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            modelopt_state_path.mkdir()

            _write_modelopt_common_state(
                modelopt_state_path,
                [("kd_loss", {"some": "data"})],
            )

            result = has_modelopt_state(str(checkpoint_path))
            assert result is False

    def test_has_modelopt_state_returns_true_with_other_states(self):
        """Test has_modelopt_state stays true when kd_loss is accompanied by other modes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            modelopt_state_path.mkdir()

            _write_modelopt_common_state(
                modelopt_state_path,
                [
                    ("kd_loss", {"some": "data"}),
                    ("quantization", {"other": "data"}),
                ],
            )

            result = has_modelopt_state(str(checkpoint_path))
            assert result is True

    def test_has_modelopt_state_with_iter_folders(self):
        """Test has_modelopt_state when modelopt_state is in an iter_* folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder = checkpoint_path / "iter_0000100"
            iter_folder.mkdir()
            modelopt_state_path = iter_folder / "modelopt_state"
            modelopt_state_path.mkdir()
            _write_modelopt_common_state(modelopt_state_path, [("quantization", {"value": 1})])

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:
                mock_load.side_effect = lambda path: (
                    {"modelopt_state_dict": [("quantization", {"value": 1})]}
                    if str(path).endswith("modelopt_state")
                    else {"iteration": 100}
                )

                result = has_modelopt_state(str(checkpoint_path))
                assert result is True

    def test_has_modelopt_state_with_iter_folders_no_modelopt_state(self):
        """Test has_modelopt_state when iter_* folders exist but no modelopt_state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder = checkpoint_path / "iter_0000100"
            iter_folder.mkdir()
            # Don't create modelopt_state folder

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:
                mock_load.return_value = {"iteration": 100}

                result = has_modelopt_state(str(checkpoint_path))
                assert result is False

    def test_has_modelopt_state_with_multiple_iter_folders(self):
        """Test has_modelopt_state finds modelopt_state in the latest iter folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_200 = checkpoint_path / "iter_0000200"

            iter_folder_100.mkdir()
            iter_folder_200.mkdir()

            # Only create modelopt_state in the iter_200 folder (the latest)
            modelopt_state_path = iter_folder_200 / "modelopt_state"
            modelopt_state_path.mkdir()
            _write_modelopt_common_state(modelopt_state_path, [("quantization", {"value": 2})])

            # Mock the dist_checkpointing.load_common_state_dict
            with patch("megatron.core.dist_checkpointing.load_common_state_dict") as mock_load:

                def load_side_effect(path):
                    if str(path).endswith("modelopt_state"):
                        return {"modelopt_state_dict": [("quantization", {"value": 2})]}
                    if "iter_0000100" in path:
                        return {"iteration": 100}
                    elif "iter_0000200" in path:
                        return {"iteration": 200}
                    return None

                mock_load.side_effect = load_side_effect

                result = has_modelopt_state(str(checkpoint_path))
                assert result is True


class TestLoadModeloptState:
    """Test load_modelopt_state function."""

    @patch("megatron.bridge.training.post_training.checkpointing._load_extra_state_from_sharded_checkpoint")
    @patch("megatron.bridge.training.post_training.checkpointing.mto.restore_from_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.mto.ModeloptStateManager.is_converted")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    def test_legacy_common_save_detect_restore_contract(
        self,
        mock_unwrap_model,
        mock_is_converted,
        mock_restore_from_state,
        mock_load_extra_state,
        mock_model_fixtures,
        tmp_path,
    ):
        """Detect and restore ModelOpt state saved in the legacy common.pt layout."""
        checkpoint_path = tmp_path / "checkpoint"
        modelopt_state_path = checkpoint_path / "modelopt_state"
        modelopt_state_path.mkdir(parents=True)
        _write_modelopt_common_state(
            modelopt_state_path,
            [("quantization", {"value": 1})],
            modelopt_version="0.44.0rc5",
        )

        assert has_modelopt_state(str(checkpoint_path)) is True

        original_model = Mock()
        restored_model = Mock()
        unwrapped_model = [original_model]
        mock_unwrap_model.return_value = unwrapped_model
        mock_is_converted.return_value = False
        mock_restore_from_state.return_value = restored_model
        load_modelopt_state(mock_model_fixtures, str(checkpoint_path))

        assert mock_restore_from_state.call_args.args[0] is original_model
        assert mock_restore_from_state.call_args.args[1]["modelopt_state_dict"] == [("quantization", {"value": 1})]
        assert unwrapped_model == [restored_model]
        mock_load_extra_state.assert_called_once_with(restored_model, str(checkpoint_path), prefix="")

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

    @patch("megatron.bridge.training.post_training.checkpointing.restore_sharded_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.unwrap_model")
    @patch("megatron.core.dist_checkpointing.load_common_state_dict")
    def test_load_modelopt_state_with_iter_folders(
        self, mock_load_state_dict, mock_unwrap_model, mock_restore_state, mock_model_fixtures
    ):
        """Test load_modelopt_state correctly uses the latest iter folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            iter_folder_100 = checkpoint_path / "iter_0000100"
            iter_folder_200 = checkpoint_path / "iter_0000200"

            iter_folder_100.mkdir()
            iter_folder_200.mkdir()

            # Mock the dist_checkpointing.load_common_state_dict
            def load_side_effect(path):
                if "iter_0000100" in path:
                    return {"iteration": 100}
                elif "iter_0000200" in path:
                    return {"iteration": 200}
                return None

            mock_load_state_dict.side_effect = load_side_effect

            # Setup mocks for load_modelopt_state
            unwrapped_model = [Mock()]
            mock_unwrap_model.return_value = unwrapped_model
            mock_restore_state.return_value = None

            # Call the function
            load_modelopt_state(mock_model_fixtures, str(checkpoint_path))

            # Verify that it used the latest iteration folder (iter_0000200)
            mock_unwrap_model.assert_called_once_with(mock_model_fixtures)
            mock_restore_state.assert_called_once_with(unwrapped_model, str(iter_folder_200))


class TestPostTrainingIntegration:
    """Test integration scenarios for post-training checkpointing."""

    def test_full_workflow_with_existing_modelopt_state(self):
        """Test the full workflow when modelopt_state exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            modelopt_state_path = checkpoint_path / "modelopt_state"
            modelopt_state_path.mkdir()
            _write_modelopt_common_state(modelopt_state_path, [("quantization", {"value": 4})])

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
