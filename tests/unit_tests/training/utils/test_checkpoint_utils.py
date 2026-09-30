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
from unittest.mock import mock_open, patch

import pytest
import yaml
from megatron.training.utils.checkpoint_utils import TRAIN_STATE_FILE

from megatron.bridge.training.utils.checkpoint_utils import (
    CONFIG_FILE,
    TRACKER_PREFIX,
    checkpoint_exists,
    get_checkpoint_run_config_filename,
    get_hf_model_id_from_checkpoint,
    is_checkpoint_iteration_directory,
    is_hf_checkpoint_dir,
    read_run_config,
)


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

    def test_constants_are_correct(self):
        """Test that module constants have expected values."""
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

    def test_concurrent_access_to_cached_functions(self, tmp_path):
        """Test concurrent access to cached functions for thread safety."""
        config_data = {"model": {"type": "concurrent_test"}}
        config_file = tmp_path / "concurrent_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

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
                ):
                    result = read_run_config(str(config_file))
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
