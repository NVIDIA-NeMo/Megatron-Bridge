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

import os
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import patch

import pytest
import yaml

from megatron.bridge.training.utils.checkpoint_utils import (
    CONFIG_FILE,
    TRACKER_PREFIX,
    TRAIN_STATE_FILE,
    checkpoint_exists,
    get_checkpoint_train_state_filename,
    get_hf_model_id_from_checkpoint,
    is_checkpoint_iteration_directory,
    is_hf_checkpoint_dir,
    read_train_state,
)


@dataclass
class MockTrainState:
    """Mock train state class for testing."""

    iteration: int = 0
    epoch: int = 0
    step: int = 0

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.iteration = state_dict.get("iteration", 0)
        self.epoch = state_dict.get("epoch", 0)
        self.step = state_dict.get("step", 0)


@dataclass
class ComplexTrainState:
    """More complex train state class for advanced testing."""

    iteration: int = 0
    epoch: int = 0
    step: int = 0
    learning_rate: float = 0.0
    loss: float = 0.0
    metrics: Dict[str, float] = None
    optimizer_state: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.optimizer_state is None:
            self.optimizer_state = {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.iteration = state_dict.get("iteration", 0)
        self.epoch = state_dict.get("epoch", 0)
        self.step = state_dict.get("step", 0)
        self.learning_rate = state_dict.get("learning_rate", 0.0)
        self.loss = state_dict.get("loss", 0.0)
        self.metrics = state_dict.get("metrics", {})
        self.optimizer_state = state_dict.get("optimizer_state", {})


class TestCheckpointUtils:
    """Test suite for checkpoint utility functions."""

    def test_checkpoint_exists_with_valid_path(self, tmp_path):
        """Test checkpoint_exists returns True when checkpoint tracker file exists."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create the tracker file
        tracker_file = checkpoint_dir / f"{TRACKER_PREFIX}_{TRAIN_STATE_FILE}"
        tracker_file.touch()

        assert checkpoint_exists(str(checkpoint_dir)) is True

    def test_checkpoint_exists_with_missing_tracker_file(self, tmp_path):
        """Test checkpoint_exists returns False when tracker file doesn't exist."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Directory exists but no tracker file
        assert checkpoint_exists(str(checkpoint_dir)) is False

    def test_checkpoint_exists_with_missing_directory(self):
        """Test checkpoint_exists returns False when directory doesn't exist."""
        assert checkpoint_exists("/nonexistent/path") is False

    def test_checkpoint_exists_with_none_path(self):
        """Test checkpoint_exists returns False when path is None."""
        assert checkpoint_exists(None) is False

    def test_get_hf_model_id_from_checkpoint_root_directory(self, tmp_path):
        """Test inferring HF model id when run_config.yaml lives at root."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        run_config_file = checkpoint_dir / CONFIG_FILE
        run_config_file.write_text(yaml.dump({"model": {"hf_model_id": "meta-llama/Meta-Llama-3-8B"}}))

        result = get_hf_model_id_from_checkpoint(str(checkpoint_dir))
        assert result == "meta-llama/Meta-Llama-3-8B"

    def test_get_hf_model_id_from_checkpoint_latest_iteration(self, tmp_path):
        """Test inferring HF model id selects latest iteration when multiple exist."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        older_iter = checkpoint_dir / "iter_0000001"
        newer_iter = checkpoint_dir / "iter_0000005"
        older_iter.mkdir()
        newer_iter.mkdir()

        (older_iter / CONFIG_FILE).write_text(yaml.dump({"model": {"hf_model_id": "older/model"}}))
        (newer_iter / CONFIG_FILE).write_text(yaml.dump({"model": {"hf_model_id": "newer/model"}}))

        result = get_hf_model_id_from_checkpoint(str(checkpoint_dir))
        assert result == "newer/model"

    def test_get_hf_model_id_from_checkpoint_missing_run_config(self, tmp_path):
        """Test inferring HF model id returns None when no run_config.yaml is present."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        result = get_hf_model_id_from_checkpoint(str(checkpoint_dir))
        assert result is None

    def test_get_hf_model_id_from_checkpoint_invalid_path(self, tmp_path):
        """Test inferring HF model id handles invalid paths."""
        with pytest.raises(FileNotFoundError):
            get_hf_model_id_from_checkpoint(tmp_path / "does_not_exist")

        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory")
        with pytest.raises(NotADirectoryError):
            get_hf_model_id_from_checkpoint(file_path)

    def test_get_checkpoint_train_state_filename_without_prefix(self, tmp_path):
        """Test get_checkpoint_train_state_filename without prefix."""
        checkpoint_dir = str(tmp_path / "checkpoints")
        expected_path = os.path.join(checkpoint_dir, TRAIN_STATE_FILE)

        result = get_checkpoint_train_state_filename(checkpoint_dir)
        assert result == expected_path

    def test_get_checkpoint_train_state_filename_with_prefix(self, tmp_path):
        """Test get_checkpoint_train_state_filename with prefix."""
        checkpoint_dir = str(tmp_path / "checkpoints")
        prefix = "custom_prefix"
        expected_path = os.path.join(checkpoint_dir, f"{prefix}_{TRAIN_STATE_FILE}")

        result = get_checkpoint_train_state_filename(checkpoint_dir, prefix)
        assert result == expected_path

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.load")
    def test_read_train_state_rank_0_success(self, mock_torch_load, mock_is_initialized, mock_get_rank):
        """Test read_train_state successful read on rank 0."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_is_initialized.return_value = False

        # Mock train state data
        state_dict = {"iteration": 100, "epoch": 5, "step": 1000}
        mock_torch_load.return_value = state_dict

        with patch("megatron.bridge.training.utils.checkpoint_utils.TrainState", return_value=MockTrainState()):
            result = read_train_state("train_state.pt")

        assert isinstance(result, MockTrainState)
        assert result.iteration == 100
        assert result.epoch == 5
        assert result.step == 1000
        mock_torch_load.assert_called_once_with("train_state.pt", map_location="cpu", weights_only=True)

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.get_world_size_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_rank")  # Mock the direct torch.distributed.get_rank call
    def test_read_train_state_distributed_success(
        self, mock_torch_get_rank, mock_broadcast, mock_is_initialized, mock_get_world_size, mock_get_rank
    ):
        """Test read_train_state with distributed broadcasting."""
        # Setup mocks for distributed scenario
        rank = 2  # Non-rank 0
        mock_get_rank.return_value = rank
        mock_torch_get_rank.return_value = rank  # Mock torch.distributed.get_rank as well
        mock_get_world_size.return_value = 4
        mock_is_initialized.return_value = True

        # Mock the broadcast to simulate receiving train state from rank 0
        train_state = MockTrainState()
        train_state.iteration = 200
        train_state.epoch = 10

        def broadcast_side_effect(obj_list, src):
            obj_list[0] = train_state

        mock_broadcast.side_effect = broadcast_side_effect

        result = read_train_state("train_state.pt")

        assert result == train_state
        assert result.iteration == 200
        assert result.epoch == 10
        mock_broadcast.assert_called_once()

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.load")
    def test_read_train_state_load_error(self, mock_torch_load, mock_is_initialized, mock_get_rank):
        """Test read_train_state handles torch.load error."""
        mock_get_rank.return_value = 0
        mock_is_initialized.return_value = False
        mock_torch_load.side_effect = RuntimeError("Corrupted file")

        with pytest.raises(RuntimeError, match="Unable to load train state file"):
            read_train_state("corrupted.pt")

    def test_caching_behavior_read_train_state(self):
        """Test that read_train_state uses caching properly."""
        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
            patch("megatron.bridge.training.utils.checkpoint_utils.torch.load") as mock_load,
        ):
            state_dict = {"iteration": 100, "epoch": 5}
            mock_load.return_value = state_dict

            # First call
            with patch("megatron.bridge.training.utils.checkpoint_utils.TrainState", return_value=MockTrainState()):
                result1 = read_train_state("train_state.pt")
            # Second call should use cache
            with patch("megatron.bridge.training.utils.checkpoint_utils.TrainState", return_value=MockTrainState()):
                result2 = read_train_state("train_state.pt")

            assert result1 == result2
            assert result1.iteration == 100
            # torch.load should only be called once due to caching
            assert mock_load.call_count == 1

    def test_constants_are_correct(self):
        """Test that module constants have expected values."""
        assert TRAIN_STATE_FILE == "train_state.pt"
        assert TRACKER_PREFIX == "latest"
        assert CONFIG_FILE == "run_config.yaml"

    @pytest.mark.parametrize(
        "checkpoint_path,expected",
        [
            ("", False),
            ("relative/path", False),
            ("/absolute/nonexistent", False),
        ],
    )
    def test_checkpoint_exists_edge_cases(self, checkpoint_path, expected):
        """Test checkpoint_exists with various edge case paths."""
        assert checkpoint_exists(checkpoint_path) == expected

    # ===== ADVANCED TEST SCENARIOS =====

    def test_memory_usage_with_complex_train_state(self):
        """Test memory efficiency with complex train state objects."""
        # Create a complex state with large nested data
        complex_state_dict = {
            "iteration": 10000,
            "epoch": 50,
            "step": 100000,
            "learning_rate": 0.0001,
            "loss": 1.23,
            "metrics": {f"metric_{i}": i * 0.1 for i in range(1000)},
            "optimizer_state": {
                "param_groups": [{"params": list(range(10000))} for _ in range(10)],
                "state": {i: {"momentum": [0.1] * 100} for i in range(1000)},
            },
        }

        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
            patch("megatron.bridge.training.utils.checkpoint_utils.torch.load", return_value=complex_state_dict),
            patch("megatron.bridge.training.utils.checkpoint_utils.TrainState", return_value=ComplexTrainState()),
        ):
            result = read_train_state("complex_state.pt")

            # Verify the complex state is loaded correctly
            assert result.iteration == 10000
            assert result.epoch == 50
            assert len(result.metrics) == 1000
            assert len(result.optimizer_state["param_groups"]) == 10
            assert len(result.optimizer_state["state"]) == 1000

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.load")
    def test_out_of_memory_error_handling(self, mock_torch_load, mock_is_initialized, mock_get_rank):
        """Test handling of out-of-memory errors during torch.load."""
        mock_get_rank.return_value = 0
        mock_is_initialized.return_value = False
        mock_torch_load.side_effect = RuntimeError("CUDA out of memory")

        with patch("megatron.bridge.training.utils.checkpoint_utils.TrainState", return_value=MockTrainState()):
            with pytest.raises(RuntimeError, match="Unable to load train state file"):
                read_train_state("large_state.pt")

    def test_is_iteration_dir_with_run_config(self, tmp_path):
        """Test detection via run_config.yaml (Bridge checkpoint)."""
        iter_dir = tmp_path / "iter_0001000"
        iter_dir.mkdir()
        (iter_dir / CONFIG_FILE).touch()

        assert is_checkpoint_iteration_directory(str(iter_dir)) is True

    def test_is_iteration_dir_with_train_state(self, tmp_path):
        """Test detection via train_state.pt (Bridge per-iteration state)."""
        iter_dir = tmp_path / "iter_0000800"
        iter_dir.mkdir()
        (iter_dir / TRAIN_STATE_FILE).touch()

        assert is_checkpoint_iteration_directory(str(iter_dir)) is True

    def test_is_iteration_dir_with_metadata_json(self, tmp_path):
        """Test detection via metadata.json (torch_dist checkpoint)."""
        iter_dir = tmp_path / "iter_0000500"
        iter_dir.mkdir()
        (iter_dir / "metadata.json").touch()

        assert is_checkpoint_iteration_directory(str(iter_dir)) is True

    def test_is_iteration_dir_with_dot_metadata(self, tmp_path):
        """Test detection via .metadata (fsdp_dtensor checkpoint)."""
        iter_dir = tmp_path / "iter_0000100"
        iter_dir.mkdir()
        (iter_dir / ".metadata").touch()

        assert is_checkpoint_iteration_directory(str(iter_dir)) is True

    def test_is_iteration_dir_empty_directory(self, tmp_path):
        """Test that an empty directory is not detected as an iteration dir."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        assert is_checkpoint_iteration_directory(str(empty_dir)) is False

    def test_is_iteration_dir_none(self):
        """Test that None returns False."""
        assert is_checkpoint_iteration_directory(None) is False

    def test_is_iteration_dir_nonexistent(self):
        """Test that a nonexistent path returns False."""
        assert is_checkpoint_iteration_directory("/nonexistent/iter_0000000") is False

    def test_hf_model_dir_is_not_iteration_dir(self, tmp_path):
        """A raw HF model directory should not be treated as a Megatron iteration directory."""
        hf_dir = tmp_path / "hf_model"
        hf_dir.mkdir()
        (hf_dir / "config.json").touch()
        (hf_dir / "model.safetensors").touch()

        assert is_hf_checkpoint_dir(str(hf_dir)) is True
        assert is_checkpoint_iteration_directory(str(hf_dir)) is False
        assert checkpoint_exists(str(hf_dir)) is False
        assert checkpoint_exists(str(hf_dir)) or is_hf_checkpoint_dir(str(hf_dir))

        adapter_dir = tmp_path / "adapter_only"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").touch()
        (adapter_dir / "adapter_model.safetensors").touch()

        assert is_hf_checkpoint_dir(str(adapter_dir)) is False

    def test_checkpoint_exists_with_iteration_directory(self, tmp_path):
        """Test checkpoint_exists detects a direct iteration directory."""
        iter_dir = tmp_path / "iter_0001000"
        iter_dir.mkdir()
        (iter_dir / CONFIG_FILE).touch()

        assert checkpoint_exists(str(iter_dir)) is True

    def test_checkpoint_exists_prefers_iteration_dir_over_tracker(self, tmp_path):
        """Test that a directory with both markers is still detected."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / CONFIG_FILE).touch()
        (ckpt_dir / f"{TRACKER_PREFIX}_{TRAIN_STATE_FILE}").touch()

        assert checkpoint_exists(str(ckpt_dir)) is True

    def test_checkpoint_exists_with_symlinks(self, tmp_path):
        """Test checkpoint_exists with symbolic links."""
        # Create actual checkpoint directory with tracker file
        real_checkpoint_dir = tmp_path / "real_checkpoints"
        real_checkpoint_dir.mkdir()
        tracker_file = real_checkpoint_dir / f"{TRACKER_PREFIX}_{TRAIN_STATE_FILE}"
        tracker_file.touch()

        # Create symbolic link to the checkpoint directory
        symlink_dir = tmp_path / "symlink_checkpoints"
        symlink_dir.symlink_to(real_checkpoint_dir)

        # Test that checkpoint_exists works with symlinks
        assert checkpoint_exists(str(symlink_dir)) is True
        assert checkpoint_exists(str(real_checkpoint_dir)) is True
