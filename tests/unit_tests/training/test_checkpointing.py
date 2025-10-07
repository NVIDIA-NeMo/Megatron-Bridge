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
"""Unit tests for megatron.bridge.training.checkpointing module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import torch
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.training.checkpointing import (
    CheckpointType,
    _extract_megatron_lm_args_from_state_dict,
    _get_checkpoint_format,
    _get_non_persistent_iteration,
    _load_base_checkpoint,
    _load_model_state_dict,
    checkpoint_exists,
    cleanup_old_non_persistent_checkpoint,
    delete_extra_state,
    ensure_directory_exists,
    find_checkpoint_rank_0,
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    get_checkpoint_tracker_filename,
    get_checkpoint_train_state_filename,
    get_rng_state,
    has_nvidia_modelopt,
    init_checkpointing_context,
    load_checkpoint,
    read_metadata,
    save_checkpoint,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer
from megatron.bridge.training.state import GlobalState, TrainState


class _DummyClass:
    save_sharded_modelopt_state = None


_dummy_obj = _DummyClass()


class TestCheckpointUtilities:
    """Test utility functions for checkpoint management."""

    @pytest.mark.parametrize(
        "checkpoints_path,iteration,release,expected",
        [
            ("/path/to/checkpoints", 1000, False, "/path/to/checkpoints/iter_0001000"),
            ("/path/to/checkpoints", 1000, True, "/path/to/checkpoints/release"),
            ("/base", 0, False, "/base/iter_0000000"),
        ],
    )
    def test_get_checkpoint_name(self, checkpoints_path, iteration, release, expected):
        """Test checkpoint name generation."""
        result = get_checkpoint_name(checkpoints_path, iteration, release=release)
        assert result == expected

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    def test_find_checkpoint_rank_0(self, mock_dist_ckpt):
        """Test finding distributed checkpoints."""
        # Test when checkpoint exists
        mock_dist_ckpt.check_is_distributed_checkpoint.return_value = True
        result = find_checkpoint_rank_0("/checkpoints", 1000)
        expected = "/checkpoints/iter_0001000"
        assert result == expected
        mock_dist_ckpt.check_is_distributed_checkpoint.assert_called_with(expected)

        # Test when checkpoint doesn't exist
        mock_dist_ckpt.check_is_distributed_checkpoint.return_value = False
        result = find_checkpoint_rank_0("/checkpoints", 1000)
        assert result is None

        # Test release checkpoint
        mock_dist_ckpt.check_is_distributed_checkpoint.return_value = True
        result = find_checkpoint_rank_0("/checkpoints", 1000, release=True)
        expected = "/checkpoints/release"
        assert result == expected

    @pytest.mark.parametrize(
        "checkpoints_path,prefix,expected",
        [
            ("/checkpoints", None, "/checkpoints/train_state.pt"),
            ("/checkpoints", "latest", "/checkpoints/latest_train_state.pt"),
        ],
    )
    def test_get_checkpoint_train_state_filename(self, checkpoints_path, prefix, expected):
        """Test train state filename generation."""
        result = get_checkpoint_train_state_filename(checkpoints_path, prefix)
        assert result == expected

    def test_get_checkpoint_run_config_filename(self):
        """Test run config filename generation."""
        result = get_checkpoint_run_config_filename("/checkpoints")
        expected = "/checkpoints/run_config.yaml"
        assert result == expected

    def test_get_checkpoint_tracker_filename(self):
        """Test tracker filename generation for Megatron-LM compatibility."""
        result = get_checkpoint_tracker_filename("/checkpoints")
        expected = "/checkpoints/latest_checkpointed_iteration.txt"
        assert result == expected

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.all_reduce")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("builtins.open", create=True)
    def test_read_metadata_mismatch_warns(
        self, mock_open, mock_print_rank_0, mock_all_reduce, mock_get_rank, mock_dist_init
    ):
        """When iterations differ across ranks, a warning should be printed via print_rank_0."""
        mock_dist_init.return_value = True
        mock_get_rank.return_value = 0
        mock_file = mock_open.return_value.__enter__.return_value
        mock_file.read.return_value = "10"

        # Mock tensor semantics: iters_cuda[0].item() -> 20
        mock_tensor_item = Mock()
        mock_tensor_item.item.return_value = 20
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=mock_tensor_item)

        with patch("torch.tensor", return_value=mock_tensor):
            _ = read_metadata("/path/to/tracker")

        assert mock_print_rank_0.called

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.all_reduce")
    @patch("builtins.open", create=True)
    def test_read_metadata_iteration(self, mock_open, mock_all_reduce, mock_get_rank, mock_dist_init):
        """Test reading iteration from Megatron-LM tracker file."""
        mock_dist_init.return_value = True
        mock_get_rank.return_value = 0
        mock_file = mock_open.return_value.__enter__.return_value
        mock_file.read.return_value = "1000"

        # Mock tensor operations - need to make it subscriptable
        mock_tensor_item = Mock()
        mock_tensor_item.item.return_value = 1000
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=mock_tensor_item)  # Make it subscriptable

        with patch("torch.tensor", return_value=mock_tensor):
            iteration, release = read_metadata("/path/to/tracker")

        assert iteration == 1000
        assert release is False

    @patch("torch.distributed.is_initialized")
    @patch("builtins.open", create=True)
    def test_read_metadata_release(self, mock_open, mock_dist_init):
        """Test reading release flag from Megatron-LM tracker file."""
        mock_dist_init.return_value = False  # Simplify by not using distributed
        mock_file = mock_open.return_value.__enter__.return_value
        mock_file.read.return_value = "release"

        iteration, release = read_metadata("/path/to/tracker")

        assert iteration == 0
        assert release is True

    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_checkpoint_exists_fallback(self, mock_isfile, mock_exists):
        """Test checkpoint existence checking with fallback to Megatron-LM tracker."""
        # NeMo-LM tracker doesn't exist, but Megatron-LM tracker does
        mock_exists.return_value = False  # latest_train_state.pt doesn't exist
        mock_isfile.return_value = True  # latest_checkpointed_iteration.txt exists

        result = checkpoint_exists("/checkpoints")
        assert result is True

        # Verify both files were checked
        mock_exists.assert_called_with("/checkpoints/latest_train_state.pt")
        mock_isfile.assert_called_with("/checkpoints/latest_checkpointed_iteration.txt")

    @patch("os.path.exists")
    def test_checkpoint_exists_normal(self, mock_exists):
        """Test checkpoint existence checking for normal checkpoints."""
        # Test when NeMo-LM checkpoint exists
        mock_exists.return_value = True
        result = checkpoint_exists("/checkpoints")
        assert result is True
        mock_exists.assert_called_with("/checkpoints/latest_train_state.pt")

        # Test when no checkpoint exists
        mock_exists.return_value = False
        with patch("os.path.isfile", return_value=False):
            result = checkpoint_exists("/checkpoints")
            assert result is False

        # Test with None path
        result = checkpoint_exists(None)
        assert result is False

    @patch("os.makedirs")
    def test_ensure_directory_exists(self, mock_makedirs):
        """Test directory creation."""
        # Test with parent directory
        ensure_directory_exists("/path/to/file.txt", check_parent=True)
        mock_makedirs.assert_called_with("/path/to", exist_ok=True)

        # Test with full path as directory
        ensure_directory_exists("/path/to/dir", check_parent=False)
        mock_makedirs.assert_called_with("/path/to/dir", exist_ok=True)

    def test_ensure_directory_exists_with_msc_url(self):
        """Test directory creation with MSC URL."""
        MultiStorageClientFeature.enable()

        # Verify that the parent directory is created
        with tempfile.TemporaryDirectory() as temp_dir:
            ensure_directory_exists(f"msc://default{temp_dir}/checkpoints/iter_0000001", check_parent=True)
            assert os.path.exists(f"{temp_dir}/checkpoints")

            ensure_directory_exists(f"msc://default{temp_dir}/checkpoints/iter_0000001", check_parent=False)
            assert os.path.exists(f"{temp_dir}/checkpoints/iter_0000001")


class TestCheckpointTypes:
    """Test CheckpointType enum and related logic."""

    def test_checkpoint_type_enum(self):
        """Test CheckpointType enum values."""
        assert len(CheckpointType) == 3
        assert CheckpointType.LOCAL in CheckpointType
        assert CheckpointType.GLOBAL in CheckpointType
        assert CheckpointType.FSDP_DTENSOR in CheckpointType


class TestRNGState:
    """Test RNG state collection."""

    @patch("megatron.bridge.training.checkpointing.mpu")
    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("torch.distributed.is_initialized")
    @patch("torch.cuda.get_rng_state")
    @patch("torch.get_rng_state")
    @patch("numpy.random.get_state")
    @patch("random.getstate")
    def test_get_rng_state(self, mock_random, mock_np, mock_torch, mock_cuda, mock_dist_init, mock_tp, mock_mpu):
        """Test RNG state collection."""
        # Setup mocks
        mock_dist_init.return_value = False
        mock_random.return_value = "random_state"
        mock_np.return_value = "np_state"
        mock_torch.return_value = torch.tensor([1, 2, 3])
        mock_cuda.return_value = torch.tensor([4, 5, 6])
        mock_tracker = Mock()
        mock_tracker.get_states.return_value = "tracker_states"
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_world_size.return_value = 1
        mock_mpu.get_data_parallel_rank.return_value = 0

        result = get_rng_state(data_parallel_random_init=False)

        # Verify the result is a ShardedObject
        assert result.key == "rng_state"
        assert len(result.data) == 1

        # Verify RNG state structure
        rng_state = result.data[0]
        assert rng_state["random_rng_state"] == "random_state"
        assert rng_state["np_rng_state"] == "np_state"
        assert rng_state["rng_tracker_states"] == "tracker_states"


class TestDeleteExtraState:
    """Tests for delete_extra_state utility added for cleanup of extraneous keys."""

    def test_delete_extra_state_with_model_section(self):
        sd = {"model": {"layer.weight": 1, "te_extra_state": 2, "_extra_state.foo": 3}}
        result = delete_extra_state(sd)
        assert "te_extra_state" not in result["model"]
        assert "_extra_state.foo" not in result["model"]
        assert result["model"]["layer.weight"] == 1

    def test_delete_extra_state_direct_model_state(self):
        sd = {"layer.weight": 1, "something_extra_state": 2}
        result = delete_extra_state(sd)
        assert "something_extra_state" not in result
        assert result["layer.weight"] == 1

    def test_delete_extra_state_non_mapping_noop(self):
        class NotMapping:
            pass

        # Should not throw and should return the original object wrapper
        sd = {"model": NotMapping()}
        result = delete_extra_state(sd)
        assert result is sd


@pytest.fixture
def save_checkpoint_fixtures():
    """Fixture for save checkpoint tests."""
    mock_state = Mock(spec=GlobalState)
    mock_state.train_state = Mock(spec=TrainState)
    mock_state.train_state.step = 1000
    # Make state_dict() return a real dictionary that supports item assignment
    mock_state.train_state.state_dict.return_value = {
        "step": torch.tensor(1000),
        "floating_point_operations_so_far": torch.tensor(500000, dtype=torch.float32),
    }
    mock_state.rank_monitor_client = Mock()  # Add missing attribute for fault tolerance

    mock_cfg = Mock(spec=ConfigContainer)
    mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
    mock_cfg.checkpoint.save = "/checkpoints"
    mock_cfg.checkpoint.async_save = False
    mock_cfg.checkpoint.save_optim = True
    mock_cfg.checkpoint.save_rng = True
    mock_cfg.checkpoint.ckpt_format = "torch_dist"
    mock_cfg.checkpoint.non_persistent_ckpt_type = "global"

    # Create nested mock attributes
    mock_cfg.optimizer = Mock()
    mock_cfg.optimizer.use_distributed_optimizer = False
    mock_cfg.rng = Mock()
    mock_cfg.rng.data_parallel_random_init = False
    mock_cfg.dataset = Mock()
    mock_cfg.dataset.dataloader_save = None
    mock_cfg.to_yaml = Mock()  # Mock config YAML export
    mock_cfg.logger = Mock()
    mock_cfg.logger.log_progress = False

    mock_state.cfg = mock_cfg

    mock_model = [Mock()]
    mock_optimizer = Mock()
    mock_scheduler = Mock()

    return {
        "mock_state": mock_state,
        "mock_cfg": mock_cfg,
        "mock_model": mock_model,
        "mock_optimizer": mock_optimizer,
        "mock_scheduler": mock_scheduler,
    }


def _patch_modelopt_state_saver():
    """Conditionally patch modelopt state saving function."""
    if has_nvidia_modelopt:
        return patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state")
    return patch.object(_dummy_obj, "save_sharded_modelopt_state")


class TestSaveCheckpoint:
    """Test checkpoint saving functionality."""

    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("builtins.open", new_callable=mock_open)
    @patch("torch.save")
    @patch("shutil.copy")
    @_patch_modelopt_state_saver()
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.mpu")
    @patch("megatron.bridge.training.checkpointing.fault_tolerance")
    @patch("megatron.bridge.training.checkpointing.is_empty_async_queue")
    @patch("megatron.bridge.training.checkpointing.get_rank_safe")
    @patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state")
    @patch("megatron.bridge.training.checkpointing.ensure_directory_exists")
    @patch("megatron.bridge.training.checkpointing.get_default_save_sharded_strategy")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.barrier")
    def test_save_checkpoint_global(
        self,
        mock_barrier,
        mock_get_dist_rank,
        mock_dist_init,
        mock_print_rank_0,
        mock_get_strategy,
        mock_ensure_dir,
        mock_save_dataloader,
        mock_get_rank_safe,
        mock_empty_queue,
        mock_ft,
        mock_mpu,
        mock_dist_ckpt,
        mock_gen_state,
        mock_rerun,
        mock_get_rng,
        mock_unwrap,
        mock_save_modelopt,
        mock_shutil_copy,
        mock_torch_save,
        mock_file_open,
        mock_is_last_rank,
        mock_wandb,
        save_checkpoint_fixtures,
    ):
        """Test saving a global checkpoint."""
        # Setup mocks
        mock_dist_init.return_value = True
        mock_get_dist_rank.return_value = 0
        mock_get_rank_safe.return_value = 0
        mock_empty_queue.return_value = True
        mock_unwrap.return_value = save_checkpoint_fixtures["mock_model"]
        mock_get_rng.return_value = Mock()
        mock_rerun.return_value.state_dict.return_value = {}
        mock_gen_state.return_value = {"model": {"param1": "value1", "param2": "value2"}}
        mock_mpu.get_expert_data_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_world_size.return_value = 1
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_get_strategy.return_value = Mock()
        mock_dist_ckpt.save.return_value = None  # Synchronous save
        mock_save_modelopt.return_value = None  # Mock ModelOpt save
        mock_is_last_rank.return_value = False  # Disable wandb logging for simplicity
        mock_torch_save.return_value = None  # Mock file save
        mock_shutil_copy.return_value = None  # Mock file copy

        # Add wandb logger to state
        save_checkpoint_fixtures["mock_state"].wandb_logger = Mock()
        save_checkpoint_fixtures["mock_state"].cfg.checkpoint.most_recent_k = -1

        # Call save_checkpoint
        save_checkpoint(
            save_checkpoint_fixtures["mock_state"],
            save_checkpoint_fixtures["mock_model"],
            save_checkpoint_fixtures["mock_optimizer"],
            save_checkpoint_fixtures["mock_scheduler"],
            1000000,
            checkpointing_context={},
            non_persistent_ckpt=False,
        )

        # Verify calls
        mock_ft.on_checkpointing_start.assert_called_once()
        mock_gen_state.assert_called_once()
        mock_dist_ckpt.save.assert_called_once()

        # Verify that the tracker file was written with the correct iteration
        tracker_calls = [
            call
            for call in mock_file_open.call_args_list
            if len(call[0]) > 0 and "latest_checkpointed_iteration.txt" in call[0][0]
        ]
        assert len(tracker_calls) > 0, "Tracker file should be written"

        # Verify the iteration was written to the file
        mock_file_handle = mock_file_open()
        write_calls = [call for call in mock_file_handle.write.call_args_list]
        assert len(write_calls) > 0, "Should write iteration to tracker file"
        # Check that the iteration (1000) was written
        written_content = "".join([str(call[0][0]) for call in write_calls if len(call[0]) > 0])
        assert "1000" in written_content, f"Expected '1000' in written content, got: {written_content}"

    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    def test_save_checkpoint_invalid_non_persistent_type(self, mock_print_rank_0, save_checkpoint_fixtures):
        """Test error handling for invalid non_persistent_ckpt_type."""
        save_checkpoint_fixtures["mock_cfg"].checkpoint.non_persistent_ckpt_type = "invalid"

        with pytest.raises(ValueError) as exc_info:
            save_checkpoint(
                save_checkpoint_fixtures["mock_state"],
                save_checkpoint_fixtures["mock_model"],
                save_checkpoint_fixtures["mock_optimizer"],
                save_checkpoint_fixtures["mock_scheduler"],
                1000000,
                checkpointing_context={},
                non_persistent_ckpt=True,
            )

        assert "Invalid non_persistent_ckpt_type" in str(exc_info.value)
        assert "Must be 'local' or 'global'" in str(exc_info.value)


@pytest.fixture
def load_checkpoint_fixtures():
    """Fixture for load checkpoint tests."""
    mock_state = Mock(spec=GlobalState)
    mock_state.train_state = Mock(spec=TrainState)
    mock_state.train_state.consumed_train_samples = 0
    mock_state.train_state.skipped_train_samples = 0
    mock_state.train_state.consumed_valid_samples = 0

    mock_cfg = Mock(spec=ConfigContainer)
    mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
    mock_cfg.checkpoint.load = "/checkpoints"
    mock_cfg.checkpoint.pretrained_checkpoint = None
    mock_cfg.checkpoint.finetune = False
    mock_cfg.checkpoint.load_optim = True
    mock_cfg.checkpoint.load_rng = True

    # Create nested mock attributes that might be accessed during loading
    mock_cfg.model = Mock()
    mock_cfg.model.fp16 = False
    mock_cfg.model.bf16 = False
    mock_cfg.model.tensor_model_parallel_size = 1
    mock_cfg.model.pipeline_model_parallel_size = 1
    mock_cfg.rng = Mock()
    mock_cfg.rng.data_parallel_random_init = False
    mock_cfg.optimizer = Mock()
    mock_cfg.optimizer.use_distributed_optimizer = False
    mock_cfg.checkpoint.ckpt_format = "torch_dist"
    mock_cfg.checkpoint.non_persistent_save_interval = None

    mock_state.cfg = mock_cfg

    mock_model = [Mock()]
    mock_optimizer = Mock()
    mock_scheduler = Mock()

    return {
        "mock_state": mock_state,
        "mock_cfg": mock_cfg,
        "mock_model": mock_model,
        "mock_optimizer": mock_optimizer,
        "mock_scheduler": mock_scheduler,
    }


class TestLoadCheckpoint:
    """Test checkpoint loading functionality."""

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.barrier")
    def test_load_checkpoint_not_found(
        self,
        mock_barrier,
        mock_dist_init,
        mock_print_rank_0,
        mock_exists,
        mock_unwrap,
        mock_read_config,
        mock_read_state,
        mock_load_base,
        load_checkpoint_fixtures,
    ):
        """Test loading when no checkpoint is found."""
        # Setup mocks
        mock_dist_init.return_value = False  # Disable distributed for simpler testing
        mock_exists.return_value = False
        mock_unwrap.return_value = load_checkpoint_fixtures["mock_model"]
        mock_load_base.return_value = (None, "", False, None)
        mock_read_config.return_value = {}

        result = load_checkpoint(
            load_checkpoint_fixtures["mock_state"],
            load_checkpoint_fixtures["mock_model"],
            load_checkpoint_fixtures["mock_optimizer"],
            load_checkpoint_fixtures["mock_scheduler"],
        )

        # Should return default values when no checkpoint found
        assert result == (0, 0)

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.set_checkpoint_version")
    @patch("megatron.bridge.training.checkpointing.update_num_microbatches")
    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("megatron.bridge.training.checkpointing.mpu")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("random.setstate")
    @patch("numpy.random.set_state")
    @patch("torch.set_rng_state")
    @patch("torch.cuda.set_rng_state")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.barrier")
    @patch("torch.cuda.empty_cache")
    @patch("os.path.exists")  # Add patch for train state file existence check
    def test_load_checkpoint_found(
        self,
        mock_exists_os,
        mock_empty_cache,
        mock_barrier,
        mock_dist_init,
        mock_torch_cuda_set_rng,
        mock_torch_set_rng,
        mock_np_set_state,
        mock_random_setstate,
        mock_get_rng_state,
        mock_generate_state_dict,
        mock_tensor_parallel,
        mock_rerun_machine,
        mock_mpu,
        mock_print_rank_0,
        mock_is_last_rank,
        mock_wandb,
        mock_update_microbatches,
        mock_set_version,
        mock_exists,
        mock_unwrap,
        mock_read_config,
        mock_read_state,
        mock_load_base,
        load_checkpoint_fixtures,
    ):
        """Test successful checkpoint loading."""
        # Setup mocks
        mock_dist_init.return_value = False  # Disable distributed for simpler testing
        mock_is_last_rank.return_value = False
        mock_exists.return_value = True
        mock_unwrap.return_value = load_checkpoint_fixtures["mock_model"]

        # Mock train state file existence (for train_state.pt check)
        mock_exists_os.return_value = True  # train_state.pt exists (normal case)

        mock_train_state = Mock()
        mock_train_state.step = 1000
        mock_train_state.floating_point_operations_so_far = 500000
        mock_read_state.return_value = mock_train_state

        # Mock utility functions
        mock_generate_state_dict.return_value = {"test": "state"}
        mock_get_rng_state.return_value = Mock()

        # Mock RNG functions (no-op)
        mock_random_setstate.return_value = None
        mock_np_set_state.return_value = None
        mock_torch_set_rng.return_value = None
        mock_torch_cuda_set_rng.return_value = None

        # Mock tensor parallel
        mock_rng_tracker = Mock()
        mock_rng_tracker.set_states = Mock()
        mock_tensor_parallel.get_cuda_rng_tracker.return_value = mock_rng_tracker

        # Mock MPU functions
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_world_size.return_value = 1
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.get_data_parallel_rank.return_value = 0

        # Mock rerun state machine
        mock_rerun_machine.return_value.load_state_dict = Mock()

        # Mock run config to avoid file I/O
        mock_run_config = {
            "model": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
            "checkpoint": {
                "save_rng": True,
                "save_optim": True,
                "fully_parallel_save": False,
            },
        }
        mock_read_config.return_value = mock_run_config

        mock_state_dict = {
            "checkpoint_version": 3.0,
            "model": {"param": "value"},
            "optimizer": {"param_groups": []},  # Mock optimizer state
            "opt_param_scheduler": {"scheduler_state": "test"},  # Mock scheduler state
            "rerun_state_machine": {"state": "test"},  # Mock rerun state
            "rng_state": [
                {
                    "random_rng_state": ("test", [1, 2, 3]),
                    "np_rng_state": ("MT19937", [1, 2, 3], 4, 0, 0.0),
                    "torch_rng_state": torch.tensor([1, 2, 3]),
                    "cuda_rng_state": torch.tensor([4, 5, 6]),
                    "rng_tracker_states": {"test_tracker": "state"},
                }
            ],  # Mock RNG state
        }
        mock_load_base.return_value = (mock_state_dict, "/ckpt/path", False, CheckpointType.GLOBAL)

        result = load_checkpoint(
            load_checkpoint_fixtures["mock_state"],
            load_checkpoint_fixtures["mock_model"],
            load_checkpoint_fixtures["mock_optimizer"],
            load_checkpoint_fixtures["mock_scheduler"],
        )

        # Verify results
        assert result[0] == 1000  # iteration
        assert result[1] == 500000  # FLOPs
        mock_set_version.assert_called_with(3.0)
        # Verify that train_state.pt was read (not megatron-lm fallback)
        mock_read_state.assert_called_once()


@pytest.fixture
def mock_config():
    """Fixture for config-based tests."""
    mock_cfg = Mock(spec=ConfigContainer)
    mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
    return mock_cfg


class TestNonPersistentCheckpoints:
    """Test non-persistent checkpoint functionality."""

    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("os.path.exists")
    def test_get_non_persistent_iteration_global(self, mock_isfile, mock_read_state):
        """Test getting iteration from global non-persistent checkpoint."""
        non_persistent_ckpt_type = "global"
        mock_isfile.return_value = True

        mock_train_state = Mock()
        mock_train_state.step = 1500
        mock_read_state.return_value = mock_train_state

        result = _get_non_persistent_iteration("/np_dir", non_persistent_ckpt_type)

        assert result == 1500
        mock_read_state.assert_called_once()

    def test_get_non_persistent_iteration_none(self):
        """Test when non_persistent_ckpt_type is None."""
        non_persistent_ckpt_type = None

        result = _get_non_persistent_iteration("/np_dir", non_persistent_ckpt_type)

        assert result == -1

    def test_get_non_persistent_iteration_local(self):
        """Test getting iteration from local non-persistent checkpoint."""
        non_persistent_ckpt_type = "local"
        mock_context = {"local_checkpoint_manager": Mock()}
        mock_context["local_checkpoint_manager"].find_latest.return_value = 2000

        result = _get_non_persistent_iteration("/np_dir", non_persistent_ckpt_type, mock_context)

        assert result == 2000

    def test_get_non_persistent_iteration_invalid_type(self):
        """Test error for invalid non_persistent_ckpt_type."""
        non_persistent_ckpt_type = "invalid"

        with pytest.raises(ValueError):
            _get_non_persistent_iteration("/np_dir", non_persistent_ckpt_type)


class TestCheckpointingContext:
    """Test checkpointing context initialization."""

    def test_init_checkpointing_context_non_local(self):
        """Test context initialization for non-local checkpointing."""
        mock_config = Mock(spec=CheckpointConfig)
        mock_config.non_persistent_ckpt_type = "global"

        result = init_checkpointing_context(mock_config)

        assert result == {}

    @patch("megatron.bridge.training.checkpointing.HAVE_RESIL", True)
    @patch("nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager.LocalCheckpointManager")
    @patch("nvidia_resiliency_ext.checkpointing.local.replication.strategies.CliqueReplicationStrategy")
    def test_init_checkpointing_context_local(self, mock_strategy, mock_manager):
        """Test context initialization for local checkpointing."""
        mock_config = Mock(spec=CheckpointConfig)
        mock_config.non_persistent_ckpt_type = "local"
        mock_config.replication = True
        mock_config.replication_jump = 2
        mock_config.replication_factor = 3
        mock_config.non_persistent_local_ckpt_dir = "/local/ckpt"

        mock_strategy.from_replication_params.return_value = "mock_strategy"
        mock_manager.return_value = "mock_manager"

        result = init_checkpointing_context(mock_config)

        assert "local_checkpoint_manager" in result
        assert result["local_checkpoint_manager"] == "mock_manager"
        mock_strategy.from_replication_params.assert_called_with(2, 3)

    @patch("megatron.bridge.training.checkpointing.HAVE_RESIL", False)
    def test_init_checkpointing_context_local_no_resil(self):
        """Test error when nvidia_resiliency_ext is not available."""
        mock_config = Mock(spec=CheckpointConfig)
        mock_config.non_persistent_ckpt_type = "local"

        with pytest.raises(RuntimeError) as exc_info:
            init_checkpointing_context(mock_config)

        assert "nvidia_resiliency_ext" in str(exc_info.value)


class TestCleanupNonPersistentCheckpoints:
    """Test cleanup of old non-persistent checkpoints."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("shutil.rmtree")
    def test_cleanup_old_non_persistent_checkpoint(self, mock_rmtree, mock_get_rank, mock_dist_init):
        """Test cleanup of old checkpoints."""
        mock_dist_init.return_value = True
        mock_get_rank.return_value = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock checkpoint directories
            save_dir = Path(temp_dir)
            old_ckpt1 = save_dir / "iter_0001000"
            old_ckpt2 = save_dir / "iter_0002000"
            new_ckpt = save_dir / "iter_0003000"

            old_ckpt1.mkdir()
            old_ckpt2.mkdir()
            new_ckpt.mkdir()

            cleanup_old_non_persistent_checkpoint(str(save_dir), leave_ckpt_num=1, do_async=False)

            # Should remove the two older checkpoints
            assert mock_rmtree.call_count == 2

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_cleanup_old_non_persistent_checkpoint_non_rank0(self, mock_get_rank, mock_dist_init):
        """Test that non-rank0 processes don't perform cleanup."""
        mock_dist_init.return_value = True
        mock_get_rank.return_value = 1

        with patch("shutil.rmtree") as mock_rmtree:
            cleanup_old_non_persistent_checkpoint("/fake/dir", leave_ckpt_num=1)
            mock_rmtree.assert_not_called()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_cleanup_old_non_persistent_checkpoint_retain_interval(self, mock_get_rank, mock_dist_init):
        """Test cleanup of old checkpoints."""
        mock_dist_init.return_value = True
        mock_get_rank.return_value = 0

        def get_ckpt_nums(path, return_ckpts=False):
            count = 0
            ckpts = []
            for root, dirs, files in os.walk(path):
                count += len(dirs)
                ckpts.append(dirs)

            if return_ckpts:
                return count, sorted(ckpts[0])
            else:
                return count

        # save ckpt every 10 steps with max_steps=200
        save_interval = 10
        max_steps = 200
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock checkpoint directories
            save_dir = Path(temp_dir)
            for i in range(10, (max_steps + 10), save_interval):
                old_ckpt = save_dir / "iter_{:07d}".format(i)
                old_ckpt.mkdir()

            # save last top 5 ckpts
            cleanup_old_non_persistent_checkpoint(str(save_dir), leave_ckpt_num=5, do_async=False)
            assert get_ckpt_nums(str(save_dir), return_ckpts=True) == (
                5,
                ["iter_0000160", "iter_0000170", "iter_0000180", "iter_0000190", "iter_0000200"],
            )


class TestLoadBaseCheckpoint:
    """Test base checkpoint loading logic."""

    @pytest.fixture
    def base_config(self):
        """Fixture for base checkpoint tests."""
        mock_cfg = Mock(spec=CheckpointConfig)
        mock_cfg.exit_on_missing_checkpoint = False
        return mock_cfg

    @patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("os.path.isfile")
    def test_load_base_checkpoint_no_checkpoint(self, mock_isfile, mock_read_state, mock_get_np_iter, base_config):
        """Test when no checkpoint is found."""
        mock_get_np_iter.return_value = -1
        mock_isfile.return_value = False

        result = _load_base_checkpoint("/fake/dir", base_config)

        assert result == (None, "", False, None)

    @patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("os.path.exists")
    def test_load_base_checkpoint_non_distributed_error(
        self, mock_os_exists, mock_file_exists, mock_dist_ckpt, mock_read_state, mock_get_np_iter, base_config
    ):
        """Test error when trying to load non-distributed checkpoint."""
        mock_get_np_iter.return_value = -1
        mock_file_exists.return_value = True  # train_state file exists

        mock_train_state = Mock()
        mock_train_state.step = 1000
        mock_read_state.return_value = mock_train_state

        mock_dist_ckpt.check_is_distributed_checkpoint.return_value = False
        # Mock that .metadata file does NOT exist (so it's not fsdp_dtensor)
        mock_os_exists.return_value = False

        with pytest.raises(RuntimeError) as exc_info:
            _load_base_checkpoint("/fake/dir", base_config)

        assert "Unknown checkpoint format" in str(exc_info.value)


class TestLoadModelWeightsFromCheckpoint:
    """Test the _load_model_weights_from_checkpoint function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.sharded_state_dict.return_value = {"weight": torch.randn(10, 10)}
        return [model]

    @pytest.fixture
    def mock_multiple_models(self):
        """Create multiple mock models for testing."""
        model1 = Mock()
        model1.sharded_state_dict.return_value = {"weight1": torch.randn(10, 10)}
        model2 = Mock()
        model2.sharded_state_dict.return_value = {"weight2": torch.randn(5, 5)}
        return [model1, model2]

    @pytest.fixture
    def mock_common_state_dict(self):
        """Create a mock state dict for testing."""
        return {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "optimizer": {"optimizer": {"param_groups": []}},
            "opt_param_scheduler": {"max_lr": 0.001},
        }

    @pytest.fixture
    def mock_full_state_dict(self):
        """Create a mock state dict for testing."""
        return {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "optimizer": {"optimizer": {"param_groups": []}},
            "opt_param_scheduler": {"max_lr": 0.001},
            "model": {"weight": torch.randn(10, 10)},
            "model0": {"weight1": torch.randn(10, 10)},
            "model1": {"weight2": torch.randn(5, 5)},
        }

    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata for testing."""
        return {"distrib_optim_sharding_type": "fully_sharded_model_space"}

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing._load_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_default_load_sharded_strategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.mpu")
    def test_load_model_weights_single_model_success(
        self,
        mock_mpu,
        mock_fully_parallel_wrapper,
        mock_get_strategy,
        mock_load_state_dict,
        mock_generate_state_dict,
        mock_unwrap_model,
        mock_dist_ckpt,
        mock_model,
        mock_common_state_dict,
        mock_full_state_dict,
        mock_metadata,
    ):
        """Test successful loading of weights for a single model."""
        # Setup mocks
        mock_dist_ckpt.load_common_state_dict.return_value = mock_common_state_dict
        mock_dist_ckpt.load_content_metadata.return_value = mock_metadata
        mock_dist_ckpt.load.return_value = mock_full_state_dict
        mock_get_strategy.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model

        # Call the function
        from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

        _load_model_weights_from_checkpoint(
            checkpoint_path="/test/checkpoint",
            model=mock_model,
            fully_parallel_load=False,
            dist_ckpt_strictness="assume_ok_unexpected",
            strict=True,
        )

        # Verify calls
        mock_dist_ckpt.load_common_state_dict.assert_called_once_with("/test/checkpoint")
        mock_dist_ckpt.load_content_metadata.assert_called_once_with(preloaded_state_dict=mock_common_state_dict)
        mock_unwrap_model.assert_called_once_with(mock_model)
        mock_generate_state_dict.assert_called_once()
        call_args = mock_generate_state_dict.call_args
        assert call_args[0][1] == {"metadata": mock_metadata}
        mock_get_strategy.assert_called_once_with("/test/checkpoint")
        mock_load_state_dict.assert_called_once_with(mock_model[0], mock_full_state_dict["model"], True)

    @patch("megatron.bridge.training.checkpointing.delete_extra_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_default_load_sharded_strategy")
    def test_load_model_weights_calls_delete_extra_state(
        self,
        mock_get_strategy,
        mock_generate_state_dict,
        mock_unwrap_model,
        mock_dist_ckpt,
        mock_delete_extra_state,
        mock_model,
        mock_common_state_dict,
        mock_full_state_dict,
        mock_metadata,
    ):
        """Ensure extra state cleanup is invoked on the loaded state dict."""
        mock_dist_ckpt.load_common_state_dict.return_value = mock_common_state_dict
        mock_dist_ckpt.load_content_metadata.return_value = mock_metadata
        mock_dist_ckpt.load.return_value = mock_full_state_dict
        mock_get_strategy.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(1)}}
        mock_unwrap_model.return_value = mock_model

        from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

        _load_model_weights_from_checkpoint(
            checkpoint_path="/ckpt",
            model=mock_model,
            fully_parallel_load=False,
            dist_ckpt_strictness="assume_ok_unexpected",
            strict=True,
        )

        mock_delete_extra_state.assert_called_once_with(mock_full_state_dict)

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing._load_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_default_load_sharded_strategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.mpu")
    def test_load_model_weights_multiple_models_success(
        self,
        mock_mpu,
        mock_fully_parallel_wrapper,
        mock_get_strategy,
        mock_load_state_dict,
        mock_generate_state_dict,
        mock_unwrap_model,
        mock_dist_ckpt,
        mock_multiple_models,
        mock_common_state_dict,
        mock_full_state_dict,
        mock_metadata,
    ):
        """Test successful loading of weights for multiple models."""
        # Setup mocks
        mock_dist_ckpt.load_common_state_dict.return_value = mock_common_state_dict
        mock_dist_ckpt.load_content_metadata.return_value = mock_metadata
        mock_dist_ckpt.load.return_value = mock_full_state_dict
        mock_get_strategy.return_value = Mock()
        mock_generate_state_dict.return_value = {
            "model0": {"weight1": torch.randn(10, 10)},
            "model1": {"weight2": torch.randn(5, 5)},
        }
        mock_unwrap_model.return_value = mock_multiple_models

        # Call the function
        from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

        _load_model_weights_from_checkpoint(
            checkpoint_path="/test/checkpoint",
            model=mock_multiple_models,
            fully_parallel_load=False,
            dist_ckpt_strictness="assume_ok_unexpected",
            strict=True,
        )

        # Verify calls
        mock_dist_ckpt.load_common_state_dict.assert_called_once_with("/test/checkpoint")
        mock_dist_ckpt.load_content_metadata.assert_called_once_with(preloaded_state_dict=mock_common_state_dict)
        mock_unwrap_model.assert_called_once_with(mock_multiple_models)
        mock_generate_state_dict.assert_called_once()
        mock_get_strategy.assert_called_once_with("/test/checkpoint")

        # Verify both models were loaded
        assert mock_load_state_dict.call_count == 2
        mock_load_state_dict.assert_any_call(mock_multiple_models[0], mock_full_state_dict["model0"], True)
        mock_load_state_dict.assert_any_call(mock_multiple_models[1], mock_full_state_dict["model1"], True)

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing._load_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_default_load_sharded_strategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.mpu")
    def test_load_model_weights_fully_parallel_load(
        self,
        mock_mpu,
        mock_fully_parallel_wrapper,
        mock_get_strategy,
        mock_load_state_dict,
        mock_generate_state_dict,
        mock_unwrap_model,
        mock_dist_ckpt,
        mock_model,
        mock_common_state_dict,
        mock_metadata,
    ):
        """Test loading with fully parallel load enabled."""
        # Setup mocks
        mock_dist_ckpt.load_common_state_dict.return_value = mock_common_state_dict
        mock_dist_ckpt.load_content_metadata.return_value = mock_metadata
        mock_strategy = Mock()
        mock_get_strategy.return_value = mock_strategy
        mock_fully_parallel_wrapper.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model
        mock_mpu.get_data_parallel_group.return_value = Mock()

        # Call the function
        from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

        _load_model_weights_from_checkpoint(
            checkpoint_path="/test/checkpoint",
            model=mock_model,
            fully_parallel_load=True,
            dist_ckpt_strictness="assume_ok_unexpected",
            strict=True,
        )

        # Verify fully parallel wrapper was used
        mock_fully_parallel_wrapper.assert_called_once_with(
            mock_strategy, mock_mpu.get_data_parallel_group.return_value
        )
        mock_mpu.get_data_parallel_group.assert_called_once_with(with_context_parallel=True)

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing._load_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_default_load_sharded_strategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.mpu")
    def test_load_model_weights_none_state_dict(
        self,
        mock_mpu,
        mock_fully_parallel_wrapper,
        mock_get_strategy,
        mock_load_state_dict,
        mock_generate_state_dict,
        mock_unwrap_model,
        mock_dist_ckpt,
        mock_model,
        mock_metadata,
    ):
        """Test loading when checkpoint returns None state dict."""
        # Setup mocks
        mock_dist_ckpt.load_common_state_dict.return_value = None
        mock_dist_ckpt.load_content_metadata.return_value = mock_metadata
        mock_get_strategy.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model

        # Call the function and expect assertion error
        from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

        with pytest.raises(AssertionError):
            _load_model_weights_from_checkpoint(
                checkpoint_path="/test/checkpoint",
                model=mock_model,
                fully_parallel_load=False,
                dist_ckpt_strictness="assume_ok_unexpected",
                strict=True,
            )

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing._load_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_default_load_sharded_strategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.mpu")
    def test_return_state_dict(
        self,
        mock_mpu,
        mock_fully_parallel_wrapper,
        mock_get_strategy,
        mock_load_state_dict,
        mock_generate_state_dict,
        mock_unwrap_model,
        mock_dist_ckpt,
        mock_model,
        mock_common_state_dict,
        mock_full_state_dict,
        mock_metadata,
    ):
        """Test skip loading weights and return state dict."""
        # Setup mocks
        mock_dist_ckpt.load_common_state_dict.return_value = mock_common_state_dict
        mock_dist_ckpt.load_content_metadata.return_value = mock_metadata
        mock_dist_ckpt.load.return_value = mock_full_state_dict
        mock_get_strategy.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model

        # Call the function
        from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

        returned_sd = _load_model_weights_from_checkpoint(
            checkpoint_path="/test/checkpoint",
            model=mock_model,
            fully_parallel_load=False,
            return_state_dict=True,
            dist_ckpt_strictness="assume_ok_unexpected",
            strict=True,
        )

        # Verify calls
        assert returned_sd == mock_full_state_dict
        mock_dist_ckpt.load.assert_called_once()
        mock_load_state_dict.assert_not_called()


class TestLoadModelStateDictHelper:
    """Tests for _load_model_state_dict strict fallback behavior and logging."""

    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    def test_load_model_state_dict_strict_fallback(self, mock_print_rank_0):
        module = Mock()
        # First call raises, second (non-strict) call succeeds
        module.load_state_dict.side_effect = [Exception("boom"), "ok"]

        _load_model_state_dict(module, {"w": 1}, strict=True)

        # Should have been called twice: strict=True then strict=False
        assert module.load_state_dict.call_count == 2
        first_args, first_kwargs = module.load_state_dict.call_args_list[0]
        second_args, second_kwargs = module.load_state_dict.call_args_list[1]
        assert first_kwargs.get("strict") is True
        assert second_kwargs.get("strict") is False
        assert mock_print_rank_0.called

    def test_load_model_state_dict_non_strict_raises(self):
        module = Mock()
        module.load_state_dict.side_effect = Exception("fail")

        with pytest.raises(Exception):
            _load_model_state_dict(module, {"w": 1}, strict=False)


class TestMegatronLMCompatibility:
    """Test Megatron-LM checkpoint compatibility features."""

    def test_extract_megatron_lm_args_from_state_dict_success(self):
        """Test successful extraction of Megatron-LM args."""
        # Create a mock args object that mimics Megatron-LM argparse Namespace
        mock_args = Mock()
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 4
        mock_args.encoder_tensor_model_parallel_size = 1
        mock_args.encoder_pipeline_model_parallel_size = 2
        mock_args.no_save_optim = False  # Will become save_optim = True
        mock_args.no_save_rng = True  # Will become save_rng = False
        mock_args.ckpt_fully_parallel_save = True

        state_dict = {"args": mock_args}

        result = _extract_megatron_lm_args_from_state_dict(state_dict)

        expected = {
            "model": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 4,
                "encoder_tensor_model_parallel_size": 1,
                "encoder_pipeline_model_parallel_size": 2,
            },
            "checkpoint": {
                "save_optim": True,  # Inverted from no_save_optim=False
                "save_rng": False,  # Inverted from no_save_rng=True
                "fully_parallel_save": True,
            },
        }

        assert result == expected

    def test_extract_megatron_lm_args_from_state_dict_defaults(self):
        """Test extraction with default values when args are missing."""

        # Create a simple object that behaves like argparse.Namespace
        # Only set the tensor_model_parallel_size, other attributes will be missing
        class MinimalArgs:
            def __init__(self):
                self.tensor_model_parallel_size = 1
                # Don't set other attributes - they will trigger AttributeError
                # which makes getattr() return the default value

        mock_args = MinimalArgs()
        state_dict = {"args": mock_args}

        result = _extract_megatron_lm_args_from_state_dict(state_dict)

        expected = {
            "model": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,  # default
                "encoder_tensor_model_parallel_size": 0,  # default
                "encoder_pipeline_model_parallel_size": 0,  # default
            },
            "checkpoint": {
                "save_optim": True,  # default (no_save_optim=False)
                "save_rng": True,  # default (no_save_rng=False)
                "fully_parallel_save": False,  # default
            },
        }

        assert result == expected

    def test_extract_megatron_lm_args_from_state_dict_missing_args(self):
        """Test error when args are missing from state_dict."""
        state_dict = {"model": "some_model"}  # No 'args' key

        with pytest.raises(RuntimeError) as exc_info:
            _extract_megatron_lm_args_from_state_dict(state_dict)

        assert "Legacy checkpoint missing 'args' field" in str(exc_info.value)

    @patch("megatron.bridge.training.checkpointing.read_metadata")
    @patch("os.path.exists")
    def test_load_base_checkpoint_legacy_tracker(self, mock_isfile, mock_read_metadata):
        """Test loading checkpoint with legacy Megatron-LM tracker file."""
        mock_cfg = Mock()
        mock_cfg.checkpoint = Mock()
        mock_cfg.checkpoint.non_persistent_ckpt_type = None
        mock_cfg.checkpoint.exit_on_missing_checkpoint = False

        # Mock file existence: NeMo-LM tracker doesn't exist, legacy tracker does
        def mock_isfile_side_effect(path):
            if "latest_train_state.pt" in path:
                return False
            elif "latest_checkpointed_iteration.txt" in path:
                return True
            return False

        mock_isfile.side_effect = mock_isfile_side_effect
        mock_read_metadata.return_value = (1000, False)

        with patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration", return_value=-1):
            with patch("megatron.bridge.training.checkpointing.dist_checkpointing") as mock_dist_ckpt:
                mock_dist_ckpt.check_is_distributed_checkpoint.return_value = True
                with patch("megatron.bridge.training.checkpointing._load_global_dist_base_checkpoint") as mock_load:
                    mock_load.return_value = ({"test": "data"}, "/ckpt/path", False, CheckpointType.GLOBAL)

                    result = _load_base_checkpoint("/test/dir", mock_cfg, rank0=True)

                    state_dict, checkpoint_name, release, ckpt_type = result
                    assert state_dict == {"test": "data"}
                    assert release is False
                    assert ckpt_type == CheckpointType.GLOBAL

                    # Verify legacy tracker was read
                    mock_read_metadata.assert_called_once()

    @patch("megatron.bridge.training.checkpointing._extract_megatron_lm_args_from_state_dict")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("os.path.exists")
    def test_load_checkpoint_legacy_config_extraction(self, mock_exists, mock_read_config, mock_extract_args):
        """Test checkpoint loading with legacy config extraction."""
        # Mock that run_config.yaml doesn't exist (legacy checkpoint)
        mock_exists.return_value = False

        # Mock the extracted legacy config
        mock_extract_args.return_value = {
            "model": {"tensor_model_parallel_size": 2},
            "checkpoint": {"save_optim": True, "save_rng": True},
        }

        state_dict = {"args": Mock(), "iteration": 1000}

        # This would be called in the actual loading flow
        with patch("megatron.bridge.training.checkpointing.print_rank_0"):
            # Simulate the config loading logic
            run_config_filename = "/fake/run_config.yaml"
            if mock_exists(run_config_filename):
                config = mock_read_config(run_config_filename)
            else:
                config = mock_extract_args(state_dict)

            assert config["model"]["tensor_model_parallel_size"] == 2
            assert config["checkpoint"]["save_optim"] is True
            mock_extract_args.assert_called_once_with(state_dict)
            mock_read_config.assert_not_called()

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.set_checkpoint_version")
    @patch("megatron.bridge.training.checkpointing.update_num_microbatches")
    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("megatron.bridge.training.checkpointing.mpu")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.barrier")
    @patch("torch.cuda.empty_cache")
    @patch("os.path.exists")
    def test_load_checkpoint_full_legacy_integration(
        self,
        mock_exists,
        mock_empty_cache,
        mock_barrier,
        mock_dist_init,
        mock_get_rng_state,
        mock_generate_state_dict,
        mock_rerun_machine,
        mock_mpu,
        mock_print_rank_0,
        mock_is_last_rank,
        mock_wandb,
        mock_update_microbatches,
        mock_set_version,
        mock_exists_checkpoint,
        mock_unwrap,
        mock_load_base,
    ):
        """Test complete integration of loading a Megatron-LM checkpoint."""
        # Setup for legacy checkpoint loading
        mock_dist_init.return_value = False
        mock_is_last_rank.return_value = False
        mock_exists_checkpoint.return_value = True
        mock_unwrap.return_value = [Mock()]

        # Mock file existence checks
        def mock_exists_side_effect(path):
            if "run_config.yaml" in path:
                return False  # No run_config.yaml (legacy)
            elif "train_state.pt" in path:
                return False  # No train_state.pt (legacy)
            return True

        mock_exists.side_effect = mock_exists_side_effect

        # Create a complete legacy Megatron-LM state_dict
        mock_args = Mock()
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 1
        mock_args.encoder_tensor_model_parallel_size = 0
        mock_args.encoder_pipeline_model_parallel_size = 0
        mock_args.no_save_optim = False
        mock_args.no_save_rng = False
        mock_args.ckpt_fully_parallel_save = True
        mock_args.consumed_train_samples = 100000
        mock_args.skipped_train_samples = 50
        mock_args.consumed_valid_samples = 10000

        legacy_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 2000,
            "args": mock_args,
            "num_floating_point_operations_so_far": 5000000,
            "model": {"param": "value"},
            "optimizer": {"param_groups": []},
            "opt_param_scheduler": {"scheduler_state": "test"},  # Add scheduler state
        }

        mock_load_base.return_value = (legacy_state_dict, "/legacy/ckpt/path", False, CheckpointType.GLOBAL)

        # Mock other required functions
        mock_generate_state_dict.return_value = {"test": "state"}
        mock_get_rng_state.return_value = Mock()
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_world_size.return_value = 2
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_rerun_machine.return_value.load_state_dict = Mock()

        # Create test fixtures
        mock_state = Mock()
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.wandb_logger = Mock()

        mock_cfg = Mock()
        mock_cfg.checkpoint = Mock()
        mock_cfg.checkpoint.load = "/legacy/checkpoint"
        mock_cfg.checkpoint.pretrained_checkpoint = None
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_optim = True
        mock_cfg.checkpoint.load_rng = False  # Skip RNG loading for this test
        mock_cfg.model = Mock()
        mock_cfg.model.fp16 = False
        mock_cfg.model.bf16 = False
        mock_cfg.model.tensor_model_parallel_size = 2  # Should match checkpoint
        mock_cfg.model.pipeline_model_parallel_size = 1  # Should match checkpoint
        mock_cfg.rng = Mock()
        mock_cfg.rng.data_parallel_random_init = False
        mock_cfg.optimizer = Mock()
        mock_cfg.optimizer.use_distributed_optimizer = False
        mock_cfg.peft = None  # No PEFT for this test

        mock_state.cfg = mock_cfg

        # Create mocks with necessary methods
        mock_model = Mock()
        mock_model.load_state_dict = Mock()

        mock_optimizer = Mock()
        mock_optimizer.load_state_dict = Mock()
        mock_optimizer.is_stub_optimizer = False

        mock_scheduler = Mock()
        mock_scheduler.load_state_dict = Mock()

        # Call load_checkpoint
        result = load_checkpoint(
            mock_state,
            [mock_model],  # model
            mock_optimizer,  # optimizer
            mock_scheduler,  # scheduler
        )

        # Verify the results
        iteration, flops = result
        assert iteration == 2000
        assert flops == 5000000

        # Verify that the legacy train state was created correctly
        train_state = mock_state.train_state
        assert train_state.step == 2000
        assert train_state.consumed_train_samples == 100000
        assert train_state.skipped_train_samples == 50
        assert train_state.consumed_valid_samples == 10000
        assert train_state.floating_point_operations_so_far == 5000000
        assert train_state.do_train is False
        assert train_state.do_valid is False
        assert train_state.do_test is False

        # Verify checkpoint version was set
        mock_set_version.assert_called_with(3.0)


class TestGetTrainStateFromStateDict:
    """Test _get_train_state_from_state_dict function."""

    def test_get_train_state_complete_state_dict(self):
        """Test creating TrainState from a complete state_dict."""
        # Create a mock args object
        mock_args = Mock()
        mock_args.consumed_train_samples = 150000
        mock_args.skipped_train_samples = 250
        mock_args.consumed_valid_samples = 12000

        state_dict = {
            "iteration": 3000,
            "args": mock_args,
            "num_floating_point_operations_so_far": 7500000,
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Verify all fields are set correctly
        assert result.step == 3000
        assert result.consumed_train_samples == 150000
        assert result.skipped_train_samples == 250
        assert result.consumed_valid_samples == 12000
        assert result.floating_point_operations_so_far == 7500000
        assert result.do_train is False
        assert result.do_valid is False
        assert result.do_test is False

    def test_get_train_state_missing_iteration(self):
        """Test creating TrainState when iteration is missing."""
        mock_args = Mock()
        mock_args.consumed_train_samples = 100000
        mock_args.skipped_train_samples = 50
        mock_args.consumed_valid_samples = 8000

        state_dict = {
            "args": mock_args,
            "num_floating_point_operations_so_far": 5000000,
            # No 'iteration' key
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should use default value of 0 for missing iteration
        assert result.step == 0
        assert result.consumed_train_samples == 100000
        assert result.skipped_train_samples == 50
        assert result.consumed_valid_samples == 8000
        assert result.floating_point_operations_so_far == 5000000

    def test_get_train_state_missing_args(self):
        """Test creating TrainState when args is missing."""
        state_dict = {
            "iteration": 2000,
            "num_floating_point_operations_so_far": 4000000,
            # No 'args' key
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should use fallback default values for sample counts
        assert result.step == 2000
        assert result.consumed_train_samples == 0  # fallback
        assert result.skipped_train_samples == 0  # fallback
        assert result.consumed_valid_samples == 0  # fallback
        assert result.floating_point_operations_so_far == 4000000

    def test_get_train_state_missing_flops(self):
        """Test creating TrainState when floating point operations count is missing."""
        mock_args = Mock()
        mock_args.consumed_train_samples = 75000
        mock_args.skipped_train_samples = 30
        mock_args.consumed_valid_samples = 6000

        state_dict = {
            "iteration": 1500,
            "args": mock_args,
            # No 'num_floating_point_operations_so_far' key
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should use default value of 0 for missing FLOPS
        assert result.step == 1500
        assert result.consumed_train_samples == 75000
        assert result.skipped_train_samples == 30
        assert result.consumed_valid_samples == 6000
        assert result.floating_point_operations_so_far == 0  # default

    def test_get_train_state_partial_args(self):
        """Test creating TrainState when args has only some attributes."""

        # Create args object with only some attributes set
        class PartialArgs:
            def __init__(self):
                self.consumed_train_samples = 200000
                # Don't set skipped_train_samples or consumed_valid_samples
                # getattr() will return default values

        partial_args = PartialArgs()

        state_dict = {
            "iteration": 4000,
            "args": partial_args,
            "num_floating_point_operations_so_far": 9000000,
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should use available attribute and defaults for missing ones
        assert result.step == 4000
        assert result.consumed_train_samples == 200000  # from args
        assert result.skipped_train_samples == 0  # default from getattr
        assert result.consumed_valid_samples == 0  # default from getattr
        assert result.floating_point_operations_so_far == 9000000

    def test_get_train_state_empty_state_dict(self):
        """Test creating TrainState from an empty state_dict."""
        state_dict = {}

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should use all default values
        assert result.step == 0
        assert result.consumed_train_samples == 0
        assert result.skipped_train_samples == 0
        assert result.consumed_valid_samples == 0
        assert result.floating_point_operations_so_far == 0
        assert result.do_train is False
        assert result.do_valid is False
        assert result.do_test is False

    def test_get_train_state_args_none(self):
        """Test creating TrainState when args is explicitly None."""
        state_dict = {
            "iteration": 500,
            "args": None,
            "num_floating_point_operations_so_far": 1000000,
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should trigger the fallback branch (args is None)
        assert result.step == 500
        assert result.consumed_train_samples == 0  # fallback
        assert result.skipped_train_samples == 0  # fallback
        assert result.consumed_valid_samples == 0  # fallback
        assert result.floating_point_operations_so_far == 1000000

    def test_get_train_state_large_values(self):
        """Test creating TrainState with large numerical values."""
        mock_args = Mock()
        mock_args.consumed_train_samples = 999999999
        mock_args.skipped_train_samples = 1000000
        mock_args.consumed_valid_samples = 50000000

        state_dict = {
            "iteration": 100000,
            "args": mock_args,
            "num_floating_point_operations_so_far": 999999999999,
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should handle large values correctly
        assert result.step == 100000
        assert result.consumed_train_samples == 999999999
        assert result.skipped_train_samples == 1000000
        assert result.consumed_valid_samples == 50000000
        assert result.floating_point_operations_so_far == 999999999999

    def test_get_train_state_zero_values(self):
        """Test creating TrainState with zero values."""
        mock_args = Mock()
        mock_args.consumed_train_samples = 0
        mock_args.skipped_train_samples = 0
        mock_args.consumed_valid_samples = 0

        state_dict = {
            "iteration": 0,
            "args": mock_args,
            "num_floating_point_operations_so_far": 0,
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Should handle zero values correctly
        assert result.step == 0
        assert result.consumed_train_samples == 0
        assert result.skipped_train_samples == 0
        assert result.consumed_valid_samples == 0
        assert result.floating_point_operations_so_far == 0
        # Boolean flags should still be False
        assert result.do_train is False
        assert result.do_valid is False
        assert result.do_test is False

    def test_get_train_state_boolean_flags_always_true(self):
        """Test that boolean flags are always set to False regardless of input."""
        # Even with different inputs, the boolean flags should always be False
        state_dict = {
            "iteration": 1000,
            "do_train": False,  # This should be ignored
            "do_valid": False,  # This should be ignored
            "do_test": False,  # This should be ignored
        }

        from megatron.bridge.training.checkpointing import _get_train_state_from_state_dict

        result = _get_train_state_from_state_dict(state_dict)

        # Boolean flags should always be False (hardcoded in the function)
        assert result.do_train is False
        assert result.do_valid is False
        assert result.do_test is False


class TestFSDPDTensorFunctionality:
    """Test new FSDP DTensor checkpointing functionality."""

    @patch("megatron.core.dist_checkpointing.check_is_distributed_checkpoint")
    @patch("os.path.exists")
    def test_get_checkpoint_format_torch_dist(self, mock_exists, mock_check_dist_ckpt):
        """Test _get_checkpoint_format detects torch_dist format."""
        mock_check_dist_ckpt.return_value = True
        mock_exists.return_value = False

        result = _get_checkpoint_format("/path/to/checkpoint")

        assert result == "torch_dist"
        mock_check_dist_ckpt.assert_called_once_with("/path/to/checkpoint")

    @patch("megatron.core.dist_checkpointing.check_is_distributed_checkpoint")
    @patch("os.path.exists")
    def test_get_checkpoint_format_fsdp_dtensor(self, mock_exists, mock_check_dist_ckpt):
        """Test _get_checkpoint_format detects fsdp_dtensor format."""
        mock_check_dist_ckpt.return_value = False
        mock_exists.return_value = True  # .metadata file exists

        result = _get_checkpoint_format("/path/to/checkpoint")

        assert result == "fsdp_dtensor"
        mock_check_dist_ckpt.assert_called_once_with("/path/to/checkpoint")
        mock_exists.assert_called_once_with("/path/to/checkpoint/.metadata")

    @patch("megatron.core.dist_checkpointing.check_is_distributed_checkpoint")
    @patch("os.path.exists")
    def test_get_checkpoint_format_unknown(self, mock_exists, mock_check_dist_ckpt):
        """Test _get_checkpoint_format raises error for unknown format."""
        mock_check_dist_ckpt.return_value = False
        mock_exists.return_value = False  # No .metadata file

        with pytest.raises(NotImplementedError, match="Unknown checkpoint format"):
            _get_checkpoint_format("/path/to/checkpoint")

    @patch("megatron.core.mpu.get_pipeline_model_parallel_rank")
    @patch("megatron.core.mpu.get_tensor_model_parallel_rank")
    @patch("megatron.core.mpu.get_data_parallel_world_size")
    @patch("torch.distributed.is_initialized")
    def test_get_rng_state_fsdp_dtensor_format(self, mock_dist_init, mock_dp_world_size, mock_tp_rank, mock_pp_rank):
        """Test get_rng_state returns correct format for fsdp_dtensor."""
        mock_dist_init.return_value = False  # Simplify
        mock_dp_world_size.return_value = 1
        mock_tp_rank.return_value = 0
        mock_pp_rank.return_value = 0

        with (
            patch("random.getstate"),
            patch("numpy.random.get_state"),
            patch("torch.get_rng_state"),
            patch("torch.cuda.get_rng_state"),
            patch("megatron.core.tensor_parallel.get_cuda_rng_tracker"),
        ):
            result = get_rng_state(data_parallel_random_init=False, ckpt_format="fsdp_dtensor")

        # Should return dict format for fsdp_dtensor
        assert isinstance(result, dict)
        assert "(0, 0)" in result

    @patch("megatron.core.mpu.get_pipeline_model_parallel_rank")
    @patch("megatron.core.mpu.get_pipeline_model_parallel_world_size")
    @patch("megatron.core.mpu.get_tensor_model_parallel_rank")
    @patch("megatron.core.mpu.get_tensor_model_parallel_world_size")
    @patch("megatron.core.mpu.get_data_parallel_rank")
    @patch("megatron.core.mpu.get_data_parallel_world_size")
    @patch("torch.distributed.is_initialized")
    def test_get_rng_state_torch_dist_format(
        self,
        mock_dist_init,
        mock_dp_world_size,
        mock_dp_rank,
        mock_tp_world_size,
        mock_tp_rank,
        mock_pp_world_size,
        mock_pp_rank,
    ):
        """Test get_rng_state returns ShardedObject for torch_dist."""
        # The ShardedObject is only created for torch_dist format regardless of distributed state
        mock_dp_world_size.return_value = 1
        mock_dp_rank.return_value = 0
        mock_tp_rank.return_value = 0
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 0
        mock_pp_world_size.return_value = 1
        mock_dist_init.return_value = False

        with (
            patch("random.getstate"),
            patch("numpy.random.get_state"),
            patch("torch.get_rng_state"),
            patch("torch.cuda.get_rng_state"),
            patch("megatron.core.tensor_parallel.get_cuda_rng_tracker"),
            patch("megatron.bridge.training.checkpointing.ShardedObject") as mock_sharded_obj,
        ):
            _ = get_rng_state(data_parallel_random_init=False, ckpt_format="torch_dist")

        # Should create ShardedObject for torch_dist format
        # The exact arguments depend on the RNG state, but we just verify it was called
        mock_sharded_obj.assert_called_once()

    def test_build_sharded_state_dict_metadata_fsdp_dtensor(self):
        """Test _build_sharded_state_dict_metadata for fsdp_dtensor format."""
        from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata

        result = _build_sharded_state_dict_metadata(
            use_distributed_optimizer=True, ckpt_fully_parallel_save=False, ckpt_format="fsdp_dtensor"
        )

        assert result["distrib_optim_sharding_type"] == "fsdp_dtensor"
        assert result["chained_optim_avoid_prefix"] is True
        assert result["singleton_local_shards"] is False

    def test_build_sharded_state_dict_metadata_torch_dist_fully_parallel(self):
        """Test _build_sharded_state_dict_metadata for torch_dist with fully parallel save."""
        from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata

        result = _build_sharded_state_dict_metadata(
            use_distributed_optimizer=True, ckpt_fully_parallel_save=True, ckpt_format="torch_dist"
        )

        assert result["distrib_optim_sharding_type"] == "fully_sharded_model_space"

    def test_build_sharded_state_dict_metadata_torch_dist_dp_zero(self):
        """Test _build_sharded_state_dict_metadata for torch_dist with dp_zero_gather_scatter."""
        from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata

        result = _build_sharded_state_dict_metadata(
            use_distributed_optimizer=True, ckpt_fully_parallel_save=False, ckpt_format="torch_dist"
        )

        assert result["distrib_optim_sharding_type"] == "dp_zero_gather_scatter"

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", True)
    @patch("torch.distributed.checkpoint.FileSystemReader")
    @patch("torch.distributed.checkpoint.load_state_dict")
    @patch("torch.distributed.checkpoint.default_planner.DefaultLoadPlanner")
    def test_load_fsdp_dtensor_base_checkpoint_rank0(self, mock_planner, mock_load_state_dict, mock_reader):
        """Test _load_fsdp_dtensor_base_checkpoint for rank0."""
        from megatron.bridge.training.checkpointing import _load_fsdp_dtensor_base_checkpoint
        from megatron.bridge.training.config import CheckpointConfig

        ckpt_cfg = CheckpointConfig()

        state_dict, checkpoint_name, release, ckpt_type = _load_fsdp_dtensor_base_checkpoint(
            load_dir="/test/dir",
            ckpt_cfg=ckpt_cfg,
            rank0=True,
            sharded_state_dict=None,
            iteration=1000,
            release=False,
        )

        # For rank0, should return empty state dict
        assert state_dict == {}
        assert ckpt_type == CheckpointType.FSDP_DTENSOR
        assert release is False
        # Should not call any loading functions for rank0
        mock_reader.assert_not_called()
        mock_load_state_dict.assert_not_called()

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", False)
    def test_load_fsdp_dtensor_base_checkpoint_no_fsdp(self):
        """Test _load_fsdp_dtensor_base_checkpoint raises error when FSDP not available."""
        from megatron.bridge.training.checkpointing import _load_fsdp_dtensor_base_checkpoint
        from megatron.bridge.training.config import CheckpointConfig

        ckpt_cfg = CheckpointConfig()

        with pytest.raises(RuntimeError, match="Megatron FSDP is required but not available"):
            _load_fsdp_dtensor_base_checkpoint(
                load_dir="/test/dir",
                ckpt_cfg=ckpt_cfg,
                rank0=False,
                sharded_state_dict={},
                iteration=1000,
                release=False,
            )

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", True)
    def test_generate_state_dict_fsdp_dtensor_preprocessing(self):
        """Test generate_state_dict applies FSDP DTensor preprocessing."""
        from unittest.mock import Mock

        from megatron.bridge.training.checkpointing import generate_state_dict
        from megatron.bridge.training.config import CheckpointConfig

        # Create mock model with no swiglu flag (to avoid SWiGLU handler)
        mock_model = Mock()
        mock_model.state_dict_for_save_checkpoint.return_value = {"test_param": torch.tensor([1.0])}

        # Mock get_model_config to return config without swiglu
        with patch("megatron.core.utils.get_model_config") as mock_get_config:
            mock_config = Mock()
            mock_config.swiglu = False  # No swiglu flag
            mock_get_config.return_value = mock_config

            with (
                patch("megatron.bridge.training.checkpointing.handle_fp8_extra_state_case") as mock_fp8,
                patch(
                    "megatron.bridge.training.checkpointing.preprocess_state_dict_for_uneven_dtensor"
                ) as mock_uneven,
            ):
                ckpt_cfg = CheckpointConfig(ckpt_format="fsdp_dtensor", save_rng=False)
                result = generate_state_dict(
                    ckpt_cfg=ckpt_cfg,
                    model=[mock_model],
                    optimizer=None,
                    opt_param_scheduler=None,
                    rng_state=None,
                )

                # Should call FSDP preprocessing functions
                mock_fp8.assert_called_once()
                mock_uneven.assert_called_once()
                # Should use state_dict_for_save_checkpoint for fsdp_dtensor
                mock_model.state_dict_for_save_checkpoint.assert_called_once()
                assert "model" in result
                assert result["checkpoint_version"] == 3.0

    def test_generate_state_dict_torch_dist_no_preprocessing(self):
        """Test generate_state_dict skips FSDP preprocessing for torch_dist."""
        from unittest.mock import Mock

        from megatron.bridge.training.checkpointing import generate_state_dict
        from megatron.bridge.training.config import CheckpointConfig

        mock_model = Mock()
        mock_model.sharded_state_dict.return_value = {"test_param": torch.tensor([1.0])}

        with (
            patch("megatron.bridge.training.checkpointing.handle_fp8_extra_state_case") as mock_fp8,
            patch("megatron.bridge.training.checkpointing.preprocess_state_dict_for_uneven_dtensor") as mock_uneven,
        ):
            ckpt_cfg = CheckpointConfig(ckpt_format="torch_dist", save_rng=False)
            result = generate_state_dict(
                ckpt_cfg=ckpt_cfg,
                model=[mock_model],
                optimizer=None,
                opt_param_scheduler=None,
                rng_state=None,
            )

            # Should NOT call FSDP preprocessing functions for torch_dist
            mock_fp8.assert_not_called()
            mock_uneven.assert_not_called()
            # Should use sharded_state_dict for torch_dist
            mock_model.sharded_state_dict.assert_called_once()
            assert "model" in result
