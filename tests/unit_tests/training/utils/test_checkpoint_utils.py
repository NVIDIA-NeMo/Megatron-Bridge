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
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import mock_open, patch

import pytest
import yaml

from megatron.bridge.training.utils.checkpoint_utils import (
    CONFIG_FILE,
    TRACKER_PREFIX,
    TRAIN_STATE_FILE,
    _get_latest_iteration_path,
    checkpoint_exists,
    get_checkpoint_run_config_filename,
    get_checkpoint_train_state_filename,
    get_hf_model_id_from_checkpoint,
    is_iteration_directory,
    read_run_config,
    read_train_state,
    resolve_checkpoint_path,
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

    def test_get_checkpoint_run_config_filename(self, tmp_path):
        """Test get_checkpoint_run_config_filename returns correct path."""
        checkpoint_dir = str(tmp_path / "checkpoints")
        expected_path = os.path.join(checkpoint_dir, CONFIG_FILE)

        result = get_checkpoint_run_config_filename(checkpoint_dir)
        assert result == expected_path

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
        """Test inferring HF model id returns None when run_config.yaml has no hf_model_id."""
        checkpoint_dir = tmp_path / "checkpoints"
        iter_dir = checkpoint_dir / "iter_0000001"
        iter_dir.mkdir(parents=True)

        # Create run_config.yaml without hf_model_id
        run_config = {"model": {"some_other_field": "value"}}
        run_config_path = iter_dir / "run_config.yaml"
        run_config_path.write_text(yaml.dump(run_config))

        result = get_hf_model_id_from_checkpoint(str(checkpoint_dir))
        assert result is None

    def test_get_hf_model_id_from_checkpoint_no_iterations(self, tmp_path):
        """Test inferring HF model id raises when no iteration directories exist."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No iteration checkpoints found"):
            get_hf_model_id_from_checkpoint(str(checkpoint_dir))

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
    def test_read_run_config_rank_0_success(self, mock_is_initialized, mock_get_rank):
        """Test read_run_config successful read on rank 0."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_is_initialized.return_value = False

        # Mock config data
        config_data = {"model": {"type": "gpt"}, "training": {"lr": 0.001}}
        config_yaml = yaml.dump(config_data)

        with patch("builtins.open", mock_open(read_data=config_yaml)):
            result = read_run_config("config.yaml")

        assert result == config_data

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.get_world_size_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_rank")  # Mock the direct torch.distributed.get_rank call
    def test_read_run_config_distributed_success(
        self, mock_torch_get_rank, mock_broadcast, mock_is_initialized, mock_get_world_size, mock_get_rank
    ):
        """Test read_run_config with distributed broadcasting."""
        # Setup mocks for distributed scenario
        rank = 1  # Non-rank 0
        mock_get_rank.return_value = rank
        mock_torch_get_rank.return_value = rank  # Mock torch.distributed.get_rank as well
        mock_get_world_size.return_value = 4
        mock_is_initialized.return_value = True

        # Mock the broadcast to simulate receiving data from rank 0
        config_data = {"model": {"type": "gpt"}}

        def broadcast_side_effect(obj_list, src):
            obj_list[0] = config_data

        mock_broadcast.side_effect = broadcast_side_effect

        result = read_run_config("config.yaml")

        assert result == config_data
        mock_broadcast.assert_called_once()

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    def test_read_run_config_file_not_found(self, mock_is_initialized, mock_get_rank):
        """Test read_run_config handles file not found error."""
        mock_get_rank.return_value = 0
        mock_is_initialized.return_value = False

        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(RuntimeError, match="Unable to load config file"):
                read_run_config("nonexistent.yaml")

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    def test_read_run_config_invalid_yaml(self, mock_is_initialized, mock_get_rank):
        """Test read_run_config handles invalid YAML."""
        mock_get_rank.return_value = 0
        mock_is_initialized.return_value = False

        invalid_yaml = "invalid: yaml: content: ["

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with pytest.raises(RuntimeError, match="Unable to load config file"):
                read_run_config("invalid.yaml")

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0)
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False)
    def test_read_run_config_sanitizes_runtime_only_targets(self, mock_is_initialized, mock_get_rank):
        """Run config should drop runtime-only objects such as timers."""
        raw_config = {
            "model": {
                "timers": {"_target_": "megatron.core.timers.Timers"},
                "keep": {"_target_": "some.other.Component", "value": 1},
                "nested": [
                    {"timers": {"_target_": "megatron.core.timers.Timers"}},
                    {"other": {"_target_": "another.Component", "value": 2}},
                ],
            },
            "tokenizer": {"type": "sentencepiece"},
        }
        config_yaml = yaml.dump(raw_config)

        with patch("builtins.open", mock_open(read_data=config_yaml)):
            result = read_run_config("config_with_timers.yaml")

        assert result["model"]["timers"] is None
        assert result["model"]["nested"][0]["timers"] is None
        assert result["model"]["keep"]["_target_"] == "some.other.Component"
        assert result["model"]["nested"][1]["other"]["_target_"] == "another.Component"

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
        mock_torch_load.assert_called_once_with("train_state.pt", map_location="cpu")

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

    def test_caching_behavior_read_run_config(self):
        """Test that read_run_config uses caching properly."""
        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
        ):
            config_data = {"test": "data"}
            config_yaml = yaml.dump(config_data)

            with patch("builtins.open", mock_open(read_data=config_yaml)) as mock_file:
                # First call
                result1 = read_run_config("config.yaml")
                # Second call should use cache
                result2 = read_run_config("config.yaml")

                assert result1 == result2 == config_data
                # File should only be opened once due to caching
                assert mock_file.call_count == 1

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

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.get_world_size_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_rank")  # Mock the direct torch.distributed.get_rank call
    def test_distributed_error_propagation(
        self, mock_torch_get_rank, mock_broadcast, mock_is_initialized, mock_get_world_size, mock_get_rank
    ):
        """Test that errors are properly propagated in distributed setting."""
        # Setup for rank 1 (non-master)
        rank = 1
        mock_get_rank.return_value = rank
        mock_torch_get_rank.return_value = rank  # Mock torch.distributed.get_rank as well
        mock_get_world_size.return_value = 2
        mock_is_initialized.return_value = True

        # Simulate rank 0 sending an error
        def broadcast_error(obj_list, src):
            obj_list[0] = {"error": True, "msg": "File not found on rank 0"}

        mock_broadcast.side_effect = broadcast_error

        with pytest.raises(RuntimeError, match="File not found on rank 0"):
            read_run_config("missing_file.yaml")

    def test_integration_with_real_files(self, tmp_path):
        """Integration test with real file I/O."""
        # Create a real config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "test_config.yaml"

        config_data = {"model": {"type": "gpt", "layers": 12}, "training": {"lr": 0.001, "batch_size": 32}}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Test reading the real file (mocking distributed to avoid complexity)
        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
        ):
            result = read_run_config(str(config_file))
            assert result == config_data

    def test_edge_case_empty_config_file(self, tmp_path):
        """Test handling of empty config file."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("")

        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
        ):
            result = read_run_config(str(config_file))
            assert result is None  # Empty YAML file returns None

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

    def test_concurrent_access_to_cached_functions(self):
        """Test concurrent access to cached functions for thread safety."""
        config_data = {"model": {"type": "concurrent_test"}}
        config_yaml = yaml.dump(config_data)

        results = []
        errors = []

        def read_config_worker():
            try:
                with (
                    patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
                    patch(
                        "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized",
                        return_value=False,
                    ),
                    patch("builtins.open", mock_open(read_data=config_yaml)),
                ):
                    result = read_run_config("concurrent_config.yaml")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=read_config_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred and all results are consistent
        assert len(errors) == 0, f"Errors occurred during concurrent access: {errors}"
        assert len(results) == 10
        assert all(result == config_data for result in results)

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

    def test_stress_test_many_different_files(self, tmp_path):
        """Stress test with many different config files to test cache behavior."""
        num_files = 50
        config_files = []
        expected_results = []

        # Create many different config files
        for i in range(num_files):
            config_data = {"model": {"type": f"model_{i}", "id": i}, "training": {"lr": 0.001 * (i + 1)}}
            config_file = tmp_path / f"config_{i}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            config_files.append(str(config_file))
            expected_results.append(config_data)

        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
        ):
            # Read all files
            results = []
            start_time = time.time()
            for config_file in config_files:
                result = read_run_config(config_file)
                results.append(result)
            end_time = time.time()

            # Verify all results are correct
            assert results == expected_results

            # Reading many files should be efficient due to caching
            avg_time_per_file = (end_time - start_time) / num_files
            assert avg_time_per_file < 0.01, f"Average time per file: {avg_time_per_file:.4f}s"

    @patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe")
    @patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized")
    def test_permission_error_handling(self, mock_is_initialized, mock_get_rank):
        """Test handling of permission errors when reading files."""
        mock_get_rank.return_value = 0
        mock_is_initialized.return_value = False

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(RuntimeError, match="Unable to load config file"):
                read_run_config("restricted_config.yaml")

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

    def test_unicode_and_special_characters_in_config(self, tmp_path):
        """Test handling of Unicode and special characters in config files."""
        config_data = {
            "model": {
                "name": "模型测试",  # Chinese characters
                "description": "Test with émojis 🚀 and spécial chars àçñü",
                "symbols": "¡¿¾×÷±∞≠≤≥",
            },
            "paths": {"data": "/path/with spaces/and-special_chars", "output": "~/user's folder/output"},
        }

        config_file = tmp_path / "unicode_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True)

        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
        ):
            result = read_run_config(str(config_file))
            assert result == config_data

    def test_very_deep_nested_config(self, tmp_path):
        """Test handling of deeply nested configuration structures."""
        # Create a deeply nested config
        deep_config = {"level0": {}}
        current_level = deep_config["level0"]

        for i in range(1, 20):  # 20 levels deep
            current_level[f"level{i}"] = {}
            current_level = current_level[f"level{i}"]

        current_level["final_value"] = "deep_value"

        config_file = tmp_path / "deep_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(deep_config, f)

        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
        ):
            result = read_run_config(str(config_file))
            assert result == deep_config

            # Verify the deep nesting worked
            current = result["level0"]
            for i in range(1, 20):
                current = current[f"level{i}"]
            assert current["final_value"] == "deep_value"

    @pytest.mark.parametrize(
        "world_size,rank",
        [
            (1, 0),  # Single process
            (2, 0),  # Rank 0 in 2-process setup
            (2, 1),  # Rank 1 in 2-process setup
            (8, 3),  # Mid-rank in larger setup
            (16, 15),  # Last rank in large setup
        ],
    )
    @patch("torch.distributed.get_rank")  # Mock the direct torch.distributed.get_rank call
    def test_various_distributed_configurations(self, mock_torch_get_rank, world_size, rank):
        """Test various distributed configurations."""
        config_data = {"test": f"rank_{rank}_of_{world_size}"}

        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=rank),
            patch("megatron.bridge.training.utils.checkpoint_utils.get_world_size_safe", return_value=world_size),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized",
                return_value=(world_size > 1),
            ),
        ):
            # Mock torch.distributed.get_rank as well
            mock_torch_get_rank.return_value = rank

            if rank == 0:
                # Rank 0 reads the file
                config_yaml = yaml.dump(config_data)
                with patch("builtins.open", mock_open(read_data=config_yaml)):
                    if world_size == 1:
                        # Single process, no broadcasting
                        result = read_run_config("config.yaml")
                        assert result == config_data
                    else:
                        # Multi-process, rank 0 reads and broadcasts
                        with patch(
                            "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.broadcast_object_list"
                        ) as mock_broadcast:
                            result = read_run_config("config.yaml")
                            assert result == config_data
                            mock_broadcast.assert_called_once()
            else:
                # Non-rank 0 receives from broadcast
                with patch(
                    "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.broadcast_object_list"
                ) as mock_broadcast:

                    def broadcast_side_effect(obj_list, src):
                        obj_list[0] = config_data

                    mock_broadcast.side_effect = broadcast_side_effect

                    result = read_run_config("config.yaml")
                    assert result == config_data
                    mock_broadcast.assert_called_once()

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

    def test_config_with_scientific_notation_and_special_types(self, tmp_path):
        """Test config files with scientific notation, special YAML types."""
        config_data = {
            "model": {
                "learning_rate": 1e-4,  # Scientific notation
                "dropout": 0.1,
                "epsilon": 1e-8,
                "large_number": 1.5e10,
            },
            "flags": {
                "enabled": True,  # Boolean
                "disabled": False,
                "debug": None,  # None/null
            },
            "mixed_list": [1, 2.5, "string", True, None],  # Mixed types
            "timestamps": {
                "start": "2024-01-01T00:00:00Z",  # ISO format
                "duration": "PT1H30M",  # Duration format
            },
        }

        config_file = tmp_path / "special_types_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with (
            patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
            patch(
                "megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False
            ),
        ):
            result = read_run_config(str(config_file))
            assert result == config_data

            # Verify specific type handling
            assert isinstance(result["model"]["learning_rate"], float)
            assert isinstance(result["flags"]["enabled"], bool)
            assert result["flags"]["debug"] is None


class TestIsIterationDirectory:
    """Test suite for is_iteration_directory function."""

    def test_valid_iteration_directory_standard_format(self):
        """Test that standard 7-digit iteration directory names are recognized."""
        assert is_iteration_directory("iter_0000000") is True
        assert is_iteration_directory("iter_0000001") is True
        assert is_iteration_directory("iter_0001000") is True
        assert is_iteration_directory("iter_9999999") is True

    def test_valid_iteration_directory_flexible_digit_count(self):
        """Test that iteration directories with various digit counts are recognized."""
        # Shorter than standard 7 digits
        assert is_iteration_directory("iter_1") is True
        assert is_iteration_directory("iter_12") is True
        assert is_iteration_directory("iter_123456") is True
        # Longer than standard 7 digits (for very long training runs)
        assert is_iteration_directory("iter_12345678") is True
        assert is_iteration_directory("iter_99999999999") is True

    def test_valid_iteration_directory_with_path(self):
        """Test that full paths with iteration directories are recognized."""
        assert is_iteration_directory("/path/to/checkpoint/iter_0000005") is True
        assert is_iteration_directory("/path/to/checkpoint/iter_0000005/") is True
        assert is_iteration_directory("relative/path/iter_0001000") is True
        assert is_iteration_directory("/path/to/checkpoint/iter_123") is True

    def test_invalid_iteration_directory(self):
        """Test that non-iteration directories are not recognized."""
        assert is_iteration_directory("checkpoint") is False
        assert is_iteration_directory("model") is False
        assert is_iteration_directory("release") is False
        assert is_iteration_directory("iter") is False
        assert is_iteration_directory("iteration_0000001") is False

    def test_malformed_iteration_directory(self):
        """Test that malformed iteration directories are rejected."""
        # Non-numeric suffix
        assert is_iteration_directory("iter_abc") is False
        assert is_iteration_directory("iter_abcdefg") is False
        assert is_iteration_directory("iter_000000a") is False
        # Mixed alphanumeric
        assert is_iteration_directory("iter_123abc") is False
        assert is_iteration_directory("iter_abc123") is False
        # Just prefix with no digits
        assert is_iteration_directory("iter_") is False

    def test_top_level_checkpoint_directory(self):
        """Test that top-level checkpoint directories are not iteration directories."""
        assert is_iteration_directory("/path/to/checkpoint") is False
        assert is_iteration_directory("/path/to/checkpoint/") is False


class TestGetLatestIterationPath:
    """Test suite for _get_latest_iteration_path function."""

    def test_returns_latest_iteration(self):
        """Test that the latest iteration path is returned."""
        iter_dirs = [
            ("iter_0000001", "/checkpoint/iter_0000001"),
            ("iter_0000005", "/checkpoint/iter_0000005"),
            ("iter_0000003", "/checkpoint/iter_0000003"),
        ]
        result = _get_latest_iteration_path(iter_dirs)
        assert result == "/checkpoint/iter_0000005"

    def test_single_iteration(self):
        """Test with a single iteration directory."""
        iter_dirs = [("iter_0000010", "/checkpoint/iter_0000010")]
        result = _get_latest_iteration_path(iter_dirs)
        assert result == "/checkpoint/iter_0000010"

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot get latest iteration from empty list"):
            _get_latest_iteration_path([])

    def test_high_iteration_numbers(self):
        """Test with high iteration numbers."""
        iter_dirs = [
            ("iter_0000999", "/checkpoint/iter_0000999"),
            ("iter_1000000", "/checkpoint/iter_1000000"),
            ("iter_0100000", "/checkpoint/iter_0100000"),
        ]
        result = _get_latest_iteration_path(iter_dirs)
        assert result == "/checkpoint/iter_1000000"


class TestResolveCheckpointPath:
    """Test suite for resolve_checkpoint_path function."""

    def test_resolve_specific_iteration_path(self, tmp_path):
        """Test resolving a specific iteration directory path."""
        checkpoint_dir = tmp_path / "checkpoints"
        iter_dir = checkpoint_dir / "iter_0000005"
        iter_dir.mkdir(parents=True)

        result = resolve_checkpoint_path(str(iter_dir))
        assert result == str(iter_dir)

    def test_resolve_top_level_to_latest_iteration(self, tmp_path):
        """Test resolving a top-level directory to the latest iteration."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create multiple iteration directories
        (checkpoint_dir / "iter_0000001").mkdir()
        (checkpoint_dir / "iter_0000005").mkdir()
        (checkpoint_dir / "iter_0000003").mkdir()

        result = resolve_checkpoint_path(str(checkpoint_dir))
        assert result == str(checkpoint_dir / "iter_0000005")

    def test_resolve_handles_single_iteration(self, tmp_path):
        """Test resolving when only one iteration exists."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "iter_0000010").mkdir()

        result = resolve_checkpoint_path(str(checkpoint_dir))
        assert result == str(checkpoint_dir / "iter_0000010")

    def test_resolve_nonexistent_path_raises(self, tmp_path):
        """Test that resolving a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_path(str(tmp_path / "nonexistent"))

    def test_resolve_file_path_raises(self, tmp_path):
        """Test that resolving a file path raises NotADirectoryError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory")

        with pytest.raises(NotADirectoryError):
            resolve_checkpoint_path(str(file_path))

    def test_resolve_empty_directory_raises(self, tmp_path):
        """Test that resolving an empty directory raises FileNotFoundError."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No iteration checkpoints found"):
            resolve_checkpoint_path(str(checkpoint_dir))

    def test_resolve_directory_without_iter_folders_raises(self, tmp_path):
        """Test that resolving a directory without iter_* folders raises."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "other_folder").mkdir()
        (checkpoint_dir / "another_folder").mkdir()

        with pytest.raises(FileNotFoundError, match="No iteration checkpoints found"):
            resolve_checkpoint_path(str(checkpoint_dir))

    def test_resolve_ignores_malformed_iter_directories(self, tmp_path):
        """Test that malformed iteration directories (non-numeric) are ignored."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create valid and invalid iteration directories
        (checkpoint_dir / "iter_0000005").mkdir()  # Valid: iteration 5
        (checkpoint_dir / "iter_12").mkdir()  # Valid: iteration 12 (flexible digit count)
        (checkpoint_dir / "iter_invalid").mkdir()  # Invalid: non-numeric
        (checkpoint_dir / "iter_abc").mkdir()  # Invalid: non-numeric
        (checkpoint_dir / "iter_").mkdir()  # Invalid: no digits

        # Should find iter_12 as latest (12 > 5)
        result = resolve_checkpoint_path(str(checkpoint_dir))
        assert result == str(checkpoint_dir / "iter_12")

    def test_resolve_with_only_malformed_iter_directories_raises(self, tmp_path):
        """Test that a directory with only malformed iter directories raises."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create only invalid iteration directories (non-numeric suffixes)
        (checkpoint_dir / "iter_invalid").mkdir()
        (checkpoint_dir / "iter_abc").mkdir()
        (checkpoint_dir / "iter_").mkdir()

        with pytest.raises(FileNotFoundError, match="No iteration checkpoints found"):
            resolve_checkpoint_path(str(checkpoint_dir))
