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

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.training.checkpointing import (
    CheckpointType,
    _get_non_persistent_iteration,
    _load_base_checkpoint,
    checkpoint_exists,
    cleanup_old_non_persistent_checkpoint,
    ensure_directory_exists,
    find_checkpoint_rank_0,
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    get_checkpoint_train_state_filename,
    get_rng_state,
    has_nvidia_modelopt,
    init_checkpointing_context,
    load_checkpoint,
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

    @patch("os.path.exists")
    def test_checkpoint_exists(self, mock_exists):
        """Test checkpoint existence checking."""
        # Test when checkpoint exists
        mock_exists.return_value = True
        result = checkpoint_exists("/checkpoints")
        assert result is True
        mock_exists.assert_called_with("/checkpoints/latest_train_state.pt")

        # Test when checkpoint doesn't exist
        mock_exists.return_value = False
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


class TestCheckpointTypes:
    """Test CheckpointType enum and related logic."""

    def test_checkpoint_type_enum(self):
        """Test CheckpointType enum values."""
        assert len(CheckpointType) == 2
        assert CheckpointType.LOCAL in CheckpointType
        assert CheckpointType.GLOBAL in CheckpointType


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
    def test_load_checkpoint_found(
        self,
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


@pytest.fixture
def mock_config():
    """Fixture for config-based tests."""
    mock_cfg = Mock(spec=ConfigContainer)
    mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
    return mock_cfg


class TestNonPersistentCheckpoints:
    """Test non-persistent checkpoint functionality."""

    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("os.path.isfile")
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
    @patch("os.path.isfile")
    def test_load_base_checkpoint_non_distributed_error(
        self, mock_isfile, mock_dist_ckpt, mock_read_state, mock_get_np_iter, base_config
    ):
        """Test error when trying to load non-distributed checkpoint."""
        mock_get_np_iter.return_value = -1
        mock_isfile.return_value = True

        mock_train_state = Mock()
        mock_train_state.step = 1000
        mock_read_state.return_value = mock_train_state

        mock_dist_ckpt.check_is_distributed_checkpoint.return_value = False

        with pytest.raises(RuntimeError) as exc_info:
            _load_base_checkpoint("/fake/dir", base_config)

        assert "non-distributed checkpoints is no longer supported" in str(exc_info.value)


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
        assert call_args[1]["model_sd_kwargs"] == {"metadata": mock_metadata}
        mock_get_strategy.assert_called_once_with("/test/checkpoint")
        mock_load_state_dict.assert_called_once_with(mock_model[0], mock_full_state_dict["model"], True)

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
