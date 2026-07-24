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
    _DIRECT_ITERATION_DIR_SENTINEL,
    DATALOADER_STATE_SUBDIR,
    CheckpointLoadContext,
    CheckpointManager,
    CheckpointSaveContext,
    CheckpointType,
    DefaultCheckpointManager,
    _build_auto_bridge_for_save,
    _clear_auto_bridge_cache,
    _extract_megatron_lm_args_from_state_dict,
    _get_checkpoint_format,
    _get_non_persistent_iteration,
    _load_base_checkpoint,
    _load_checkpoint_from_path,
    _load_hf_pretrained_checkpoint,
    _load_model_state_dict,
    _record_dataloader_state_dir,
    _save_hf_adapter_weights,
    _save_hf_weights,
    checkpoint_exists,
    cleanup_old_non_persistent_checkpoint,
    create_checkpoint_manager,
    delete_extra_state,
    ensure_directory_exists,
    find_checkpoint_rank_0,
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    get_checkpoint_tracker_filename,
    get_checkpoint_train_state_filename,
    get_rng_state,
    init_checkpointing_context,
    load_checkpoint,
    maybe_load_dataloader_state,
    maybe_save_dataloader_state,
    read_metadata,
    save_checkpoint,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer
from megatron.bridge.training.state import GlobalState, TrainState
from megatron.bridge.utils.instantiate_utils import InstantiationException


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

    @patch("megatron.bridge.models.conversion.auto_bridge.AutoBridge.from_hf_pretrained")
    def test_clear_auto_bridge_cache_resets_cached_bridge(self, mock_from_hf_pretrained):
        """Test that the AutoBridge cache can be reset between multi-stage runs."""
        cfg = Mock()
        cfg.checkpoint.hf_source_path = "/hf/source"
        cfg.checkpoint.hf_trust_remote_code = False
        cfg.model.hf_model_id = None
        cfg.tokenizer.tokenizer_model = None

        first_bridge = Mock(name="first_bridge")
        second_bridge = Mock(name="second_bridge")
        mock_from_hf_pretrained.side_effect = [first_bridge, second_bridge]

        _clear_auto_bridge_cache()
        assert _build_auto_bridge_for_save(cfg) is first_bridge
        assert _build_auto_bridge_for_save(cfg) is first_bridge

        _clear_auto_bridge_cache()
        assert _build_auto_bridge_for_save(cfg) is second_bridge

        assert mock_from_hf_pretrained.call_count == 2

    def test_get_checkpoint_tracker_filename(self):
        """Test tracker filename generation for Megatron-LM compatibility."""
        result = get_checkpoint_tracker_filename("/checkpoints")
        expected = "/checkpoints/latest_checkpointed_iteration.txt"
        assert result == expected

    @patch("megatron.bridge.training.checkpointing.ensure_directory_exists")
    @patch("megatron.bridge.training.checkpointing._build_auto_bridge_for_save")
    def test_save_hf_weights_rejects_unsupported_bridge_before_writing(self, mock_build_bridge, mock_ensure_dir):
        state = Mock()
        state.cfg.peft = None
        state.cfg.checkpoint.hf_source_path = "/hf/source"
        bridge = mock_build_bridge.return_value
        bridge._model_bridge.SUPPORTS_HF_PRETRAINED_EXPORT = False

        with pytest.raises(NotImplementedError, match="standalone Hugging Face"):
            _save_hf_weights(state, [], "/checkpoints/iter_0000001/hf")

        mock_ensure_dir.assert_not_called()
        bridge.export_hf_weights.assert_not_called()
        bridge.hf_pretrained.save_artifacts.assert_not_called()

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
        # A parent checkpoint directory does NOT contain iteration-dir markers
        # (run_config.yaml, train_state.pt, etc.) — only tracker files.
        _iter_markers = {"run_config.yaml", "train_state.pt", "metadata.json", ".metadata"}

        def parent_dir_exists(path):
            if os.path.basename(path) in _iter_markers:
                return False
            return True

        mock_exists.side_effect = parent_dir_exists
        result = checkpoint_exists("/checkpoints")
        assert result is True
        mock_exists.assert_called_with("/checkpoints/latest_train_state.pt")

        # Test when no checkpoint exists
        mock_exists.side_effect = None
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

    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("torch.distributed.is_initialized")
    @patch("torch.cuda.get_rng_state")
    @patch("torch.get_rng_state")
    @patch("numpy.random.get_state")
    @patch("random.getstate")
    def test_get_rng_state(self, mock_random, mock_np, mock_torch, mock_cuda, mock_dist_init, mock_tp):
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

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_pg_collection.dp_cp.size.return_value = 1
        mock_pg_collection.ep.size.return_value = 1  # EP = 1 (no expert parallelism)

        result = get_rng_state(
            data_parallel_random_init=False, ckpt_format="torch_dist", pg_collection=mock_pg_collection
        )

        # Verify the result is a ShardedObject
        assert result.key == "rng_state"
        assert len(result.data) == 1

        # Verify RNG state structure
        rng_state = result.data[0]
        assert rng_state["random_rng_state"] == "random_state"
        assert rng_state["np_rng_state"] == "np_state"
        assert rng_state["rng_tracker_states"] == "tracker_states"

    @patch("megatron.bridge.training.checkpointing.get_pg_size")
    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("torch.distributed.is_initialized")
    @patch("torch.cuda.get_rng_state")
    @patch("torch.get_rng_state")
    @patch("numpy.random.get_state")
    @patch("random.getstate")
    def test_get_rng_state_with_expert_parallelism(
        self, mock_random, mock_np, mock_torch, mock_cuda, mock_dist_init, mock_tp, mock_get_pg_size
    ):
        """Test RNG state collection with Expert Parallelism (EP > 1).

        When EP > 1, RNG state should be sharded by (PP, TP, DP) dimensions
        with replica_id=0, since different EP ranks may have different RNG states.
        """
        # Setup mocks
        mock_dist_init.return_value = False
        mock_random.return_value = "random_state"
        mock_np.return_value = "np_state"
        mock_torch.return_value = torch.tensor([1, 2, 3])
        mock_cuda.return_value = torch.tensor([4, 5, 6])
        mock_tracker = Mock()
        mock_tracker.get_states.return_value = "tracker_states"
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        # Mock get_pg_size to return EP size > 1
        mock_get_pg_size.return_value = 8  # EP > 1

        # Create mock pg_collection with EP > 1 configuration
        mock_pg_collection = Mock()
        mock_pg_collection.pp.rank.return_value = 1
        mock_pg_collection.pp.size.return_value = 2
        mock_pg_collection.tp.rank.return_value = 3
        mock_pg_collection.tp.size.return_value = 4
        mock_pg_collection.dp_cp.rank.return_value = 5
        mock_pg_collection.dp_cp.size.return_value = 6

        result = get_rng_state(
            data_parallel_random_init=False, ckpt_format="torch_dist", pg_collection=mock_pg_collection
        )

        # Verify get_pg_size was called with pg_collection.ep
        mock_get_pg_size.assert_called_once_with(mock_pg_collection.ep)

        # Verify the result is a ShardedObject with correct sharding
        assert result.key == "rng_state"
        # Shape should be (pp_size, tp_size, dp_size) when EP > 1
        assert result.global_shape == (2, 4, 6)
        # Global offset should include dp_rank
        assert result.global_offset == (1, 3, 5)
        # replica_id should be 0 (not dp_rank) when EP > 1
        assert result.replica_id == 0

    @patch("megatron.bridge.training.checkpointing.get_pg_size")
    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("torch.distributed.is_initialized")
    @patch("torch.cuda.get_rng_state")
    @patch("torch.get_rng_state")
    @patch("numpy.random.get_state")
    @patch("random.getstate")
    def test_get_rng_state_without_expert_parallelism(
        self, mock_random, mock_np, mock_torch, mock_cuda, mock_dist_init, mock_tp, mock_get_pg_size
    ):
        """Test RNG state collection without Expert Parallelism (EP = 1).

        When EP = 1, RNG state should be sharded by (PP, TP) dimensions
        with replica_id=dp_rank (standard behavior).
        """
        # Setup mocks
        mock_dist_init.return_value = False
        mock_random.return_value = "random_state"
        mock_np.return_value = "np_state"
        mock_torch.return_value = torch.tensor([1, 2, 3])
        mock_cuda.return_value = torch.tensor([4, 5, 6])
        mock_tracker = Mock()
        mock_tracker.get_states.return_value = "tracker_states"
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        # Mock get_pg_size to return EP size = 1
        mock_get_pg_size.return_value = 1  # EP = 1

        # Create mock pg_collection with EP = 1 configuration
        mock_pg_collection = Mock()
        mock_pg_collection.pp.rank.return_value = 1
        mock_pg_collection.pp.size.return_value = 2
        mock_pg_collection.tp.rank.return_value = 3
        mock_pg_collection.tp.size.return_value = 4
        mock_pg_collection.dp_cp.rank.return_value = 5
        mock_pg_collection.dp_cp.size.return_value = 1

        result = get_rng_state(
            data_parallel_random_init=False, ckpt_format="torch_dist", pg_collection=mock_pg_collection
        )

        # Verify get_pg_size was called with pg_collection.ep
        mock_get_pg_size.assert_called_once_with(mock_pg_collection.ep)

        # Verify the result is a ShardedObject with correct sharding
        assert result.key == "rng_state"
        # Shape should be (pp_size, tp_size) when EP = 1
        assert result.global_shape == (2, 4)
        # Global offset should NOT include dp_rank
        assert result.global_offset == (1, 3)
        # replica_id should be dp_rank when EP = 1
        assert result.replica_id == 5

    @patch("megatron.bridge.training.checkpointing.get_pg_size")
    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("torch.distributed.is_initialized")
    @patch("torch.cuda.get_rng_state")
    @patch("torch.get_rng_state")
    @patch("numpy.random.get_state")
    @patch("random.getstate")
    def test_get_rng_state_with_none_ep_group(
        self, mock_random, mock_np, mock_torch, mock_cuda, mock_dist_init, mock_tp, mock_get_pg_size
    ):
        """Test RNG state collection when EP group is None (not initialized).

        When pg_collection.ep is None, get_pg_size returns 1, so this should
        behave the same as EP=1 (sharded by PP, TP with replica_id=dp_rank).
        """
        # Setup mocks
        mock_dist_init.return_value = False
        mock_random.return_value = "random_state"
        mock_np.return_value = "np_state"
        mock_torch.return_value = torch.tensor([1, 2, 3])
        mock_cuda.return_value = torch.tensor([4, 5, 6])
        mock_tracker = Mock()
        mock_tracker.get_states.return_value = "tracker_states"
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        # Mock get_pg_size to return 1 (what it does for None groups)
        mock_get_pg_size.return_value = 1

        # Create mock pg_collection with ep=None
        mock_pg_collection = Mock()
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 2
        mock_pg_collection.tp.rank.return_value = 1
        mock_pg_collection.tp.size.return_value = 4
        mock_pg_collection.dp_cp.rank.return_value = 3
        mock_pg_collection.dp_cp.size.return_value = 1
        mock_pg_collection.ep = None  # Explicitly None

        result = get_rng_state(
            data_parallel_random_init=False, ckpt_format="torch_dist", pg_collection=mock_pg_collection
        )

        # Verify get_pg_size was called with None
        mock_get_pg_size.assert_called_once_with(None)

        # Verify the result is a ShardedObject with correct sharding (same as EP=1)
        assert result.key == "rng_state"
        assert result.global_shape == (2, 4)  # (pp_size, tp_size)
        assert result.global_offset == (0, 1)  # (pp_rank, tp_rank)
        assert result.replica_id == 3  # dp_rank


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
    mock_cfg.checkpoint.save_tokenizer_assets = False  # Disable for unit tests
    mock_cfg.checkpoint.also_save_hf_checkpoint = False

    # Create nested mock attributes
    mock_cfg.optimizer = Mock()
    mock_cfg.optimizer.use_distributed_optimizer = False
    mock_cfg.rng = Mock()
    mock_cfg.rng.data_parallel_random_init = False
    mock_cfg.dataset = Mock()
    mock_cfg.dataset.dataloader_save = None
    mock_cfg.dataset.tokenizer = None  # No tokenizer in unit tests
    mock_cfg.peft = None
    mock_cfg.to_yaml = Mock()  # Mock config YAML export
    mock_cfg.logger = Mock()
    mock_cfg.logger.log_progress = False
    mock_cfg.dist = Mock()
    mock_cfg.dist.use_decentralized_pg = False
    mock_cfg.ddp = Mock()
    mock_cfg.ddp.use_megatron_fsdp = False

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


class TestSaveCheckpoint:
    """Test checkpoint saving functionality."""

    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("builtins.open", new_callable=mock_open)
    @patch("torch.save")
    @patch("shutil.copy")
    @patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.fault_tolerance")
    @patch("megatron.bridge.training.checkpointing.is_empty_async_queue")
    @patch("megatron.bridge.training.checkpointing.get_rank_safe")
    @patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state")
    @patch("megatron.bridge.training.checkpointing.ensure_directory_exists")
    @patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy")
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
        mock_strategy_cls,
        mock_ensure_dir,
        mock_save_dataloader,
        mock_get_rank_safe,
        mock_empty_queue,
        mock_ft,
        mock_get_pg_collection,
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

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.expt_dp.rank.return_value = 0
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_get_pg_collection.return_value = mock_pg_collection

        mock_strategy_cls.return_value = Mock()
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

    def test_async_retention_keeps_tracker_checkpoint_until_finalize(self, tmp_path, save_checkpoint_fixtures):
        """The tracker-selected checkpoint must survive until its async replacement is durable."""
        old_checkpoint = tmp_path / "iter_0000500"
        old_checkpoint.mkdir()
        torch.save({"step": torch.tensor(500)}, old_checkpoint / "train_state.pt")
        latest_train_state = tmp_path / "latest_train_state.pt"
        torch.save({"step": torch.tensor(500)}, latest_train_state)

        state = save_checkpoint_fixtures["mock_state"]
        state.cfg.checkpoint.save = str(tmp_path)
        state.cfg.checkpoint.async_save = True
        state.cfg.checkpoint.most_recent_k = 1
        state.wandb_logger = Mock()

        pg_collection = Mock()
        pg_collection.expt_dp.rank.return_value = 0
        pg_collection.tp.rank.return_value = 0
        pg_collection.tp.size.return_value = 1
        pg_collection.pp.rank.return_value = 0
        pg_collection.pp.size.return_value = 1

        finalize_fns = []
        call_order = []
        async_request = Mock()

        def add_finalize_fn(finalize_fn):
            finalize_fns.append(finalize_fn)
            if finalize_fn.__name__ == "cleanup_old_checkpoints_finalize_fn":
                call_order.append("register_cleanup")

        async_request.add_finalize_fn.side_effect = add_finalize_fn

        def start_async_save(*args, **kwargs):
            (tmp_path / "iter_0001000").mkdir(exist_ok=True)
            return async_request

        def run_cleanup_immediately(*, target, args):
            thread = Mock()
            thread.start.side_effect = lambda: target(*args)
            return thread

        with (
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.save", side_effect=start_async_save),
            patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_pg_collection", return_value=pg_collection),
            patch("megatron.bridge.training.checkpointing.get_rng_state", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_rerun_state_machine") as mock_rerun,
            patch("megatron.bridge.training.checkpointing._get_model_glu_interleave_sizes", return_value=(None, None)),
            patch(
                "megatron.bridge.training.checkpointing.generate_state_dict",
                return_value={"model": {"weight": Mock()}},
            ),
            patch(
                "megatron.bridge.training.checkpointing.unwrap_model",
                return_value=save_checkpoint_fixtures["mock_model"],
            ),
            patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state"),
            patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state"),
            patch(
                "megatron.bridge.training.checkpointing.schedule_async_save",
                side_effect=lambda *args, **kwargs: call_order.append("schedule"),
            ),
            patch("megatron.bridge.training.checkpointing.fault_tolerance"),
            patch("megatron.bridge.training.checkpointing.is_empty_async_queue", return_value=True),
            patch("megatron.bridge.training.checkpointing.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.checkpointing.is_last_rank", return_value=False),
            patch("megatron.bridge.training.checkpointing.threading.Thread", side_effect=run_cleanup_immediately),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.barrier"),
        ):
            mock_rerun.return_value.state_dict.return_value = {}
            save_checkpoint(
                state,
                save_checkpoint_fixtures["mock_model"],
                save_checkpoint_fixtures["mock_optimizer"],
                save_checkpoint_fixtures["mock_scheduler"],
                1000000,
                checkpointing_context={},
            )

            assert call_order == ["register_cleanup", "schedule"]
            assert old_checkpoint.is_dir()
            assert torch.load(latest_train_state, weights_only=True)["step"].item() == 500

            for finalize_fn in finalize_fns:
                finalize_fn()

        assert not old_checkpoint.exists()
        assert torch.load(latest_train_state, weights_only=True)["step"].item() == 1000

    @pytest.mark.parametrize("most_recent_k", [0, 1])
    def test_async_global_non_persistent_overlap_preserves_tracker_checkpoint(
        self, tmp_path, save_checkpoint_fixtures, most_recent_k
    ):
        """An incomplete async replacement must not evict the tracker-selected checkpoint."""
        save_dir = tmp_path / "persistent"
        non_persistent_dir = tmp_path / "non_persistent"
        durable_checkpoint = non_persistent_dir / "iter_0000010"
        incomplete_checkpoint = non_persistent_dir / "iter_0000020"
        current_checkpoint = non_persistent_dir / "iter_0000030"
        future_incomplete_checkpoint = non_persistent_dir / "iter_0000040"
        durable_checkpoint.mkdir(parents=True)
        incomplete_checkpoint.mkdir()
        future_incomplete_checkpoint.mkdir()
        latest_train_state = non_persistent_dir / "latest_train_state.pt"
        torch.save({"step": torch.tensor(10)}, latest_train_state)

        state = save_checkpoint_fixtures["mock_state"]
        state.train_state.step = 30
        state.train_state.state_dict.return_value = {"step": torch.tensor(30)}
        state.cfg.checkpoint.save = str(save_dir)
        state.cfg.checkpoint.non_persistent_ckpt_type = "global"
        state.cfg.checkpoint.non_persistent_global_ckpt_dir = str(non_persistent_dir)
        state.cfg.checkpoint.async_save = True
        state.cfg.checkpoint.most_recent_k = most_recent_k
        state.wandb_logger = Mock()

        pg_collection = Mock()
        pg_collection.expt_dp.rank.return_value = 0
        pg_collection.tp.rank.return_value = 0
        pg_collection.tp.size.return_value = 1
        pg_collection.pp.rank.return_value = 0
        pg_collection.pp.size.return_value = 1

        finalize_fns = []
        async_request = Mock()
        async_request.add_finalize_fn.side_effect = finalize_fns.append

        def run_cleanup_immediately(*, target, args):
            thread = Mock()
            thread.start.side_effect = lambda: target(*args)
            return thread

        with (
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.save", return_value=async_request),
            patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_pg_collection", return_value=pg_collection),
            patch("megatron.bridge.training.checkpointing.get_rng_state", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_rerun_state_machine") as mock_rerun,
            patch("megatron.bridge.training.checkpointing._get_model_glu_interleave_sizes", return_value=(None, None)),
            patch(
                "megatron.bridge.training.checkpointing.generate_state_dict",
                return_value={"model": {"weight": Mock()}},
            ),
            patch(
                "megatron.bridge.training.checkpointing.unwrap_model",
                return_value=save_checkpoint_fixtures["mock_model"],
            ),
            patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state"),
            patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state"),
            patch("megatron.bridge.training.checkpointing.schedule_async_save"),
            patch("megatron.bridge.training.checkpointing.fault_tolerance"),
            patch("megatron.bridge.training.checkpointing.is_empty_async_queue", return_value=False),
            patch("megatron.bridge.training.checkpointing.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.checkpointing.is_last_rank", return_value=False),
            patch("megatron.bridge.training.checkpointing.threading.Thread", side_effect=run_cleanup_immediately),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.barrier"),
        ):
            mock_rerun.return_value.state_dict.return_value = {}
            save_checkpoint(
                state,
                save_checkpoint_fixtures["mock_model"],
                save_checkpoint_fixtures["mock_optimizer"],
                save_checkpoint_fixtures["mock_scheduler"],
                1000000,
                checkpointing_context={},
                non_persistent_ckpt=True,
            )

            assert durable_checkpoint.is_dir()
            assert incomplete_checkpoint.is_dir()
            assert future_incomplete_checkpoint.is_dir()
            assert torch.load(latest_train_state, weights_only=True)["step"].item() == 10

            for finalize_fn in finalize_fns:
                finalize_fn()

        assert not durable_checkpoint.exists()
        assert not incomplete_checkpoint.exists()
        assert current_checkpoint.is_dir()
        assert future_incomplete_checkpoint.is_dir()
        assert torch.load(latest_train_state, weights_only=True)["step"].item() == 30

    def test_sync_global_non_persistent_honors_configured_retention(self, tmp_path, save_checkpoint_fixtures):
        """Synchronous global non-persistent cleanup must retain the configured checkpoint count."""
        save_dir = tmp_path / "persistent"
        non_persistent_dir = tmp_path / "non_persistent"
        checkpoint_steps = (10, 20, 30, 40, 50)
        for step in checkpoint_steps:
            (non_persistent_dir / f"iter_{step:07d}").mkdir(parents=True, exist_ok=True)
        current_checkpoint = non_persistent_dir / "iter_0000060"
        future_incomplete_checkpoint = non_persistent_dir / "iter_0000070"
        future_incomplete_checkpoint.mkdir()

        state = save_checkpoint_fixtures["mock_state"]
        state.train_state.step = 60
        state.train_state.state_dict.return_value = {"step": torch.tensor(60)}
        state.cfg.checkpoint.save = str(save_dir)
        state.cfg.checkpoint.non_persistent_ckpt_type = "global"
        state.cfg.checkpoint.non_persistent_global_ckpt_dir = str(non_persistent_dir)
        state.cfg.checkpoint.async_save = False
        state.cfg.checkpoint.most_recent_k = 5
        state.wandb_logger = Mock()

        pg_collection = Mock()
        pg_collection.expt_dp.rank.return_value = 0
        pg_collection.tp.rank.return_value = 0
        pg_collection.tp.size.return_value = 1
        pg_collection.pp.rank.return_value = 0
        pg_collection.pp.size.return_value = 1

        with (
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.save", return_value=None),
            patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_pg_collection", return_value=pg_collection),
            patch("megatron.bridge.training.checkpointing.get_rng_state", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_rerun_state_machine") as mock_rerun,
            patch("megatron.bridge.training.checkpointing._get_model_glu_interleave_sizes", return_value=(None, None)),
            patch(
                "megatron.bridge.training.checkpointing.generate_state_dict",
                return_value={"model": {"weight": Mock()}},
            ),
            patch(
                "megatron.bridge.training.checkpointing.unwrap_model",
                return_value=save_checkpoint_fixtures["mock_model"],
            ),
            patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state"),
            patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state"),
            patch("megatron.bridge.training.checkpointing.fault_tolerance"),
            patch("megatron.bridge.training.checkpointing.is_empty_async_queue", return_value=True),
            patch("megatron.bridge.training.checkpointing.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.checkpointing.is_last_rank", return_value=False),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.barrier"),
        ):
            mock_rerun.return_value.state_dict.return_value = {}
            save_checkpoint(
                state,
                save_checkpoint_fixtures["mock_model"],
                save_checkpoint_fixtures["mock_optimizer"],
                save_checkpoint_fixtures["mock_scheduler"],
                1000000,
                checkpointing_context={},
                non_persistent_ckpt=True,
            )

        assert not (non_persistent_dir / "iter_0000010").exists()
        for step in (20, 30, 40, 50):
            assert (non_persistent_dir / f"iter_{step:07d}").is_dir()
        assert current_checkpoint.is_dir()
        assert future_incomplete_checkpoint.is_dir()

    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("builtins.open", new_callable=mock_open)
    @patch("torch.save")
    @patch("shutil.copy")
    @patch("megatron.bridge.training.checkpointing.schedule_async_save")
    @patch("megatron.bridge.training.checkpointing._save_hf_weights")
    @patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.fault_tolerance")
    @patch("megatron.bridge.training.checkpointing.is_empty_async_queue")
    @patch("megatron.bridge.training.checkpointing.get_rank_safe")
    @patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state")
    @patch("megatron.bridge.training.checkpointing.ensure_directory_exists")
    @patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.barrier")
    def test_save_checkpoint_hf_exports_extra_weights_without_rewriting_megatron_checkpoint(
        self,
        mock_barrier,
        mock_get_dist_rank,
        mock_dist_init,
        mock_print_rank_0,
        mock_strategy_cls,
        mock_ensure_dir,
        mock_save_dataloader,
        mock_get_rank_safe,
        mock_empty_queue,
        mock_ft,
        mock_get_pg_collection,
        mock_dist_ckpt,
        mock_gen_state,
        mock_rerun,
        mock_get_rng,
        mock_unwrap,
        mock_save_modelopt,
        mock_save_hf_weights,
        mock_schedule_async_save,
        mock_shutil_copy,
        mock_torch_save,
        mock_file_open,
        mock_is_last_rank,
        mock_wandb,
        save_checkpoint_fixtures,
    ):
        """HF weight export should be an extra artifact, not a replacement for Megatron state."""
        mock_dist_init.return_value = True
        mock_get_dist_rank.return_value = 0
        mock_get_rank_safe.return_value = 0
        mock_empty_queue.return_value = True
        mock_unwrap.return_value = save_checkpoint_fixtures["mock_model"]
        mock_get_rng.return_value = Mock()
        mock_rerun.return_value.state_dict.return_value = {}
        full_state_dict = {
            "model": {"param1": "value1"},
            "optimizer": {"state": "value2"},
            "rng_state": ["value3"],
        }
        mock_gen_state.return_value = full_state_dict

        mock_pg_collection = Mock()
        mock_pg_collection.expt_dp.rank.return_value = 0
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_get_pg_collection.return_value = mock_pg_collection

        mock_strategy_cls.return_value = Mock()
        mock_dist_ckpt.save.return_value = None
        mock_save_modelopt.return_value = None
        mock_is_last_rank.return_value = False
        mock_torch_save.return_value = None
        mock_shutil_copy.return_value = None

        state = save_checkpoint_fixtures["mock_state"]
        state.wandb_logger = Mock()
        state.cfg.checkpoint.most_recent_k = -1
        state.cfg.checkpoint.also_save_hf_checkpoint = True

        save_checkpoint(
            state,
            save_checkpoint_fixtures["mock_model"],
            save_checkpoint_fixtures["mock_optimizer"],
            save_checkpoint_fixtures["mock_scheduler"],
            1000000,
            checkpointing_context={},
            non_persistent_ckpt=False,
        )

        iter_dir = "/checkpoints/iter_0001000"
        mock_dist_ckpt.save.assert_called_once()
        saved_state_dict, save_target = mock_dist_ckpt.save.call_args.args[:2]
        assert save_target == iter_dir
        assert "model" in saved_state_dict
        assert saved_state_dict["model"] == {"param1": "value1"}
        mock_save_hf_weights.assert_called_once_with(state, save_checkpoint_fixtures["mock_model"], f"{iter_dir}/hf")
        mock_schedule_async_save.assert_not_called()

    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("builtins.open", new_callable=mock_open)
    @patch("torch.save")
    @patch("shutil.copy")
    @patch("megatron.bridge.training.checkpointing.schedule_async_save")
    @patch("megatron.bridge.training.checkpointing._save_hf_weights")
    @patch("megatron.bridge.training.checkpointing.apply_peft_adapter_filter_to_state_dict")
    @patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.fault_tolerance")
    @patch("megatron.bridge.training.checkpointing.is_empty_async_queue")
    @patch("megatron.bridge.training.checkpointing.get_rank_safe")
    @patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state")
    @patch("megatron.bridge.training.checkpointing.ensure_directory_exists")
    @patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.barrier")
    def test_save_checkpoint_peft_hf_keeps_adapter_only_megatron_state_and_exports_hf_sidecar(
        self,
        mock_barrier,
        mock_get_dist_rank,
        mock_dist_init,
        mock_print_rank_0,
        mock_strategy_cls,
        mock_ensure_dir,
        mock_save_dataloader,
        mock_get_rank_safe,
        mock_empty_queue,
        mock_ft,
        mock_get_pg_collection,
        mock_dist_ckpt,
        mock_gen_state,
        mock_rerun,
        mock_get_rng,
        mock_unwrap,
        mock_save_modelopt,
        mock_apply_peft_filter,
        mock_save_hf_weights,
        mock_schedule_async_save,
        mock_shutil_copy,
        mock_torch_save,
        mock_file_open,
        mock_is_last_rank,
        mock_wandb,
        save_checkpoint_fixtures,
    ):
        """PEFT + HF sidecar should preserve adapter-only Megatron checkpoint behavior."""
        mock_dist_init.return_value = True
        mock_get_dist_rank.return_value = 0
        mock_get_rank_safe.return_value = 0
        mock_empty_queue.return_value = True
        mock_unwrap.return_value = save_checkpoint_fixtures["mock_model"]
        mock_get_rng.return_value = Mock()
        mock_rerun.return_value.state_dict.return_value = {}
        full_state_dict = {
            "model": {
                "decoder.layers.0.self_attention.linear_qkv.weight": "base",
                "decoder.layers.0.self_attention.linear_qkv.adapter.weight": "adapter",
            },
            "optimizer": {"state": "value2"},
            "rng_state": ["value3"],
        }
        filtered_state_dict = {
            "model": {
                "decoder.layers.0.self_attention.linear_qkv.adapter.weight": "adapter",
            },
            "optimizer": {"state": "value2"},
            "rng_state": ["value3"],
        }
        mock_gen_state.return_value = full_state_dict
        mock_apply_peft_filter.return_value = filtered_state_dict

        mock_pg_collection = Mock()
        mock_pg_collection.expt_dp.rank.return_value = 0
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_get_pg_collection.return_value = mock_pg_collection

        mock_strategy_cls.return_value = Mock()
        mock_dist_ckpt.save.return_value = None
        mock_save_modelopt.return_value = None
        mock_is_last_rank.return_value = False
        mock_torch_save.return_value = None
        mock_shutil_copy.return_value = None

        state = save_checkpoint_fixtures["mock_state"]
        state.wandb_logger = Mock()
        state.cfg.peft = Mock()
        state.cfg.checkpoint.most_recent_k = -1
        state.cfg.checkpoint.also_save_hf_checkpoint = True

        save_checkpoint(
            state,
            save_checkpoint_fixtures["mock_model"],
            save_checkpoint_fixtures["mock_optimizer"],
            save_checkpoint_fixtures["mock_scheduler"],
            1000000,
            checkpointing_context={},
            non_persistent_ckpt=False,
        )

        iter_dir = "/checkpoints/iter_0001000"
        mock_apply_peft_filter.assert_called_once_with(full_state_dict, state.cfg.peft)
        mock_dist_ckpt.save.assert_called_once()
        saved_state_dict, save_target = mock_dist_ckpt.save.call_args.args[:2]
        assert save_target == iter_dir
        assert saved_state_dict == filtered_state_dict
        assert "decoder.layers.0.self_attention.linear_qkv.weight" not in saved_state_dict["model"]
        mock_save_hf_weights.assert_called_once_with(state, save_checkpoint_fixtures["mock_model"], f"{iter_dir}/hf")
        mock_schedule_async_save.assert_not_called()

    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("builtins.open", new_callable=mock_open)
    @patch("torch.save")
    @patch("shutil.copy")
    @patch("megatron.bridge.training.checkpointing.schedule_async_save")
    @patch("megatron.bridge.training.checkpointing._save_hf_weights")
    @patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.fault_tolerance")
    @patch("megatron.bridge.training.checkpointing.is_empty_async_queue")
    @patch("megatron.bridge.training.checkpointing.get_rank_safe")
    @patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state")
    @patch("megatron.bridge.training.checkpointing.ensure_directory_exists")
    @patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.barrier")
    def test_save_checkpoint_async_hf_schedules_megatron_before_hf_export(
        self,
        mock_barrier,
        mock_get_dist_rank,
        mock_dist_init,
        mock_print_rank_0,
        mock_strategy_cls,
        mock_ensure_dir,
        mock_save_dataloader,
        mock_get_rank_safe,
        mock_empty_queue,
        mock_ft,
        mock_get_pg_collection,
        mock_dist_ckpt,
        mock_gen_state,
        mock_rerun,
        mock_get_rng,
        mock_unwrap,
        mock_save_modelopt,
        mock_save_hf_weights,
        mock_schedule_async_save,
        mock_shutil_copy,
        mock_torch_save,
        mock_file_open,
        mock_is_last_rank,
        mock_wandb,
        save_checkpoint_fixtures,
    ):
        """Async Megatron checkpoint I/O should be scheduled before extra HF export starts."""
        call_order = []

        mock_dist_init.return_value = True
        mock_get_dist_rank.return_value = 0
        mock_get_rank_safe.return_value = 0
        mock_empty_queue.return_value = True
        mock_unwrap.return_value = save_checkpoint_fixtures["mock_model"]
        mock_get_rng.return_value = Mock()
        mock_rerun.return_value.state_dict.return_value = {}
        mock_gen_state.return_value = {
            "model": {"param1": "value1"},
            "optimizer": {"state": "value2"},
            "rng_state": ["value3"],
        }

        mock_pg_collection = Mock()
        mock_pg_collection.expt_dp.rank.return_value = 0
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_get_pg_collection.return_value = mock_pg_collection

        async_request = Mock()
        mock_strategy_cls.return_value = Mock()
        mock_dist_ckpt.save.return_value = async_request
        mock_save_modelopt.return_value = None
        mock_is_last_rank.return_value = False
        mock_torch_save.return_value = None
        mock_shutil_copy.return_value = None

        def record_schedule(*args, **kwargs):
            call_order.append("schedule")

        def record_hf_save(*args, **kwargs):
            call_order.append("hf")

        mock_schedule_async_save.side_effect = record_schedule
        mock_save_hf_weights.side_effect = record_hf_save

        state = save_checkpoint_fixtures["mock_state"]
        state.wandb_logger = Mock()
        state.cfg.checkpoint.most_recent_k = -1
        state.cfg.checkpoint.async_save = True
        state.cfg.checkpoint.also_save_hf_checkpoint = True

        save_checkpoint(
            state,
            save_checkpoint_fixtures["mock_model"],
            save_checkpoint_fixtures["mock_optimizer"],
            save_checkpoint_fixtures["mock_scheduler"],
            1000000,
            checkpointing_context={},
            non_persistent_ckpt=False,
        )

        assert call_order == ["schedule", "hf"]
        mock_schedule_async_save.assert_called_once_with(state, async_request)
        mock_save_hf_weights.assert_called_once_with(
            state, save_checkpoint_fixtures["mock_model"], "/checkpoints/iter_0001000/hf"
        )

    @patch("megatron.bridge.training.checkpointing._resolve_hf_source")
    @patch("megatron.bridge.training.checkpointing._build_auto_bridge_for_save")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.is_available")
    def test_save_hf_adapter_weights_uses_synchronous_bridge_save(
        self,
        mock_dist_available,
        mock_dist_init,
        mock_build_bridge,
        mock_resolve_source,
        save_checkpoint_fixtures,
    ):
        """PEFT HF adapter export should not create a separate HF async save path."""
        mock_dist_available.return_value = False
        mock_dist_init.return_value = False
        mock_resolve_source.return_value = "/hf/base"

        bridge = Mock()
        mock_build_bridge.return_value = bridge

        state = save_checkpoint_fixtures["mock_state"]
        state.cfg.peft = Mock()

        _save_hf_adapter_weights(state, save_checkpoint_fixtures["mock_model"], "/checkpoints/iter_0001000/hf")

        bridge.save_hf_adapter.assert_called_once_with(
            save_checkpoint_fixtures["mock_model"],
            "/checkpoints/iter_0001000/hf",
            peft_config=state.cfg.peft,
            base_model_name_or_path="/hf/base",
            show_progress=False,
        )

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
    mock_cfg.dist = Mock()
    mock_cfg.dist.use_decentralized_pg = False
    mock_cfg.ddp = Mock()
    mock_cfg.ddp.use_megatron_fsdp = False

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

    @patch("megatron.bridge.training.checkpointing._load_hf_pretrained_checkpoint")
    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing.is_hf_checkpoint_dir")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing.check_is_distributed_checkpoint")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    def test_load_checkpoint_from_parent_save_dir_ignores_iter_hf_export(
        self,
        mock_unwrap,
        mock_get_pg_collection,
        mock_check_dist_ckpt,
        mock_is_hf_checkpoint_dir,
        mock_file_exists,
        mock_read_train_state,
        mock_load_base,
        mock_load_hf,
        load_checkpoint_fixtures,
    ):
        """Resume from a parent save dir should use Megatron state, not the extra iter_*/hf export."""
        mock_unwrap.return_value = load_checkpoint_fixtures["mock_model"]
        mock_get_pg_collection.return_value = Mock()
        mock_check_dist_ckpt.return_value = False
        mock_file_exists.return_value = True
        mock_read_train_state.return_value = Mock(step=1000)
        mock_is_hf_checkpoint_dir.side_effect = lambda path: str(path).endswith("/hf")
        mock_load_base.return_value = (None, "", False, None)

        result = _load_checkpoint_from_path(
            "/checkpoints",
            load_checkpoint_fixtures["mock_state"],
            load_checkpoint_fixtures["mock_model"],
            load_checkpoint_fixtures["mock_optimizer"],
            load_checkpoint_fixtures["mock_scheduler"],
        )

        assert result == (0, 0)
        mock_load_hf.assert_not_called()
        mock_load_base.assert_called_once()

    @patch("megatron.bridge.training.checkpointing._load_hf_pretrained_checkpoint")
    @patch("megatron.bridge.training.checkpointing.is_hf_checkpoint_dir")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    def test_load_checkpoint_rejects_hf_directory_for_resume_load(
        self,
        mock_unwrap,
        mock_get_pg_collection,
        mock_is_hf_checkpoint_dir,
        mock_load_hf,
        load_checkpoint_fixtures,
    ):
        """checkpoint.load should resume from native Megatron checkpoints, not HF directories."""
        mock_unwrap.return_value = load_checkpoint_fixtures["mock_model"]
        mock_get_pg_collection.return_value = Mock()
        mock_is_hf_checkpoint_dir.return_value = True
        load_checkpoint_fixtures["mock_cfg"].checkpoint.load = "/hf/full-model"
        load_checkpoint_fixtures["mock_cfg"].checkpoint.pretrained_checkpoint = None

        with pytest.raises(ValueError, match="checkpoint.load does not support HuggingFace-format"):
            _load_checkpoint_from_path(
                "/hf/full-model",
                load_checkpoint_fixtures["mock_state"],
                load_checkpoint_fixtures["mock_model"],
                load_checkpoint_fixtures["mock_optimizer"],
                load_checkpoint_fixtures["mock_scheduler"],
            )

        mock_load_hf.assert_not_called()

    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.training.checkpointing._build_auto_bridge_for_save")
    def test_load_hf_pretrained_checkpoint_initializes_from_hf_source(
        self,
        mock_build_bridge,
        mock_dist_init,
        load_checkpoint_fixtures,
    ):
        """HF pretrained initialization should not require a separate hf_source_path."""
        mock_dist_init.return_value = False
        bridge = Mock()
        mock_build_bridge.return_value = bridge

        _load_hf_pretrained_checkpoint(
            "/hf/full-model",
            load_checkpoint_fixtures["mock_state"],
            load_checkpoint_fixtures["mock_model"],
            load_checkpoint_fixtures["mock_optimizer"],
            skip_load_to_model_and_opt=False,
        )

        mock_build_bridge.assert_called_once_with(load_checkpoint_fixtures["mock_cfg"], hf_source="/hf/full-model")
        bridge.load_hf_weights.assert_called_once_with(
            load_checkpoint_fixtures["mock_model"], hf_path="/hf/full-model"
        )

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

    @patch("megatron.bridge.training.checkpointing.is_hf_checkpoint_dir", return_value=False)
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
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
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
        mock_dist_ckpt,
        mock_get_rng_state,
        mock_generate_state_dict,
        mock_tensor_parallel,
        mock_rerun_machine,
        mock_get_pg_collection,
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
        mock_is_hf_checkpoint_dir,
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

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_pg_collection.dp.rank.return_value = 0
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_get_pg_collection.return_value = mock_pg_collection

        # Mock dist_checkpointing
        mock_dist_ckpt.load_content_metadata.return_value = {}
        mock_dist_ckpt.load.return_value = {}

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
    @patch("shutil.rmtree")
    def test_cleanup_old_non_persistent_checkpoint_leave_zero(self, mock_rmtree, mock_get_rank, mock_dist_init):
        """leave_ckpt_num=0 should remove all checkpoints, not keep them (guards the [:-0] pitfall)."""
        mock_dist_init.return_value = True
        mock_get_rank.return_value = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            for name in ("iter_0001000", "iter_0002000", "iter_0003000"):
                (save_dir / name).mkdir()

            cleanup_old_non_persistent_checkpoint(str(save_dir), leave_ckpt_num=0, do_async=False)

            # All three checkpoints must be scheduled for removal.
            assert mock_rmtree.call_count == 3

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_cleanup_old_non_persistent_checkpoint_max_iteration(self, mock_get_rank, mock_dist_init, tmp_path):
        """Checkpoints newer than max_iteration must not participate in retention cleanup."""
        old_checkpoint = tmp_path / "iter_0000010"
        retained_checkpoint = tmp_path / "iter_0000020"
        future_incomplete_checkpoint = tmp_path / "iter_0000030"
        for checkpoint in (old_checkpoint, retained_checkpoint, future_incomplete_checkpoint):
            checkpoint.mkdir()

        cleanup_old_non_persistent_checkpoint(
            str(tmp_path),
            leave_ckpt_num=1,
            do_async=False,
            max_iteration=20,
        )

        assert not old_checkpoint.exists()
        assert retained_checkpoint.is_dir()
        assert future_incomplete_checkpoint.is_dir()

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
        mock_cfg.ckpt_step = None
        mock_cfg.non_persistent_ckpt_type = None
        return mock_cfg

    @pytest.fixture
    def mock_pg_collection(self):
        """Fixture for mock pg_collection."""
        mock_pg = Mock()
        mock_pg.dp_cp.rank.return_value = 0
        mock_pg.dp_cp.size.return_value = 1
        mock_pg.pp.rank.return_value = 0
        mock_pg.pp.size.return_value = 1
        mock_pg.tp.rank.return_value = 0
        mock_pg.tp.size.return_value = 1
        return mock_pg

    @patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration")
    @patch("megatron.bridge.training.checkpointing.file_exists")
    def test_load_base_checkpoint_no_checkpoint(
        self, mock_file_exists, mock_get_np_iter, base_config, mock_pg_collection
    ):
        """Test when no checkpoint is found."""
        mock_get_np_iter.return_value = -1
        mock_file_exists.return_value = False

        result = _load_base_checkpoint("/fake/dir", base_config, pg_collection=mock_pg_collection)

        assert result == (None, "", False, None)

    @patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("os.path.exists")
    def test_load_base_checkpoint_non_distributed_error(
        self,
        mock_os_exists,
        mock_file_exists,
        mock_dist_ckpt,
        mock_read_state,
        mock_get_np_iter,
        base_config,
        mock_pg_collection,
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

        with pytest.raises(NotImplementedError) as exc_info:
            _load_base_checkpoint("/fake/dir", base_config, pg_collection=mock_pg_collection)

        assert "Unknown checkpoint format" in str(exc_info.value)

    @patch("megatron.bridge.training.checkpointing._load_global_dist_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing._get_checkpoint_format")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_load_base_checkpoint_direct_iteration_dir_torch_dist(
        self,
        mock_resolve,
        mock_get_format,
        mock_load_global,
        base_config,
        mock_pg_collection,
    ):
        """Direct iteration directory with torch_dist format delegates to _load_global_dist_base_checkpoint."""
        mock_resolve.return_value = (_DIRECT_ITERATION_DIR_SENTINEL, False)
        mock_get_format.return_value = "torch_dist"
        mock_load_global.return_value = ({"model": "data"}, "/ckpt/iter_0001000", False, CheckpointType.GLOBAL)

        result = _load_base_checkpoint("/ckpt/iter_0001000", base_config, rank0=True, pg_collection=mock_pg_collection)

        state_dict, _, _, ckpt_type = result
        assert state_dict == {"model": "data"}
        assert ckpt_type == CheckpointType.GLOBAL

        mock_load_global.assert_called_once()
        call_kwargs = mock_load_global.call_args
        assert (
            call_kwargs[1].get("checkpoint_path_override") == "/ckpt/iter_0001000"
            or call_kwargs[0][6] == "/ckpt/iter_0001000"
        )

    @patch("megatron.bridge.training.checkpointing.load_fsdp_dtensor_checkpoint")
    @patch("megatron.bridge.training.checkpointing._get_checkpoint_format")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_load_base_checkpoint_direct_iteration_dir_fsdp_dtensor(
        self,
        mock_resolve,
        mock_get_format,
        mock_load_fsdp,
        base_config,
        mock_pg_collection,
    ):
        """Direct iteration directory with fsdp_dtensor format delegates to load_fsdp_dtensor_checkpoint."""
        mock_resolve.return_value = (_DIRECT_ITERATION_DIR_SENTINEL, False)
        mock_get_format.return_value = "fsdp_dtensor"
        mock_load_fsdp.return_value = ({}, "/ckpt/iter_0001000", False, CheckpointType.FSDP_DTENSOR)

        result = _load_base_checkpoint("/ckpt/iter_0001000", base_config, rank0=True, pg_collection=mock_pg_collection)

        _, _, _, ckpt_type = result
        assert ckpt_type == CheckpointType.FSDP_DTENSOR

        mock_load_fsdp.assert_called_once()

    @patch("megatron.bridge.training.checkpointing._get_checkpoint_format")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_load_base_checkpoint_direct_iteration_dir_unsupported_format(
        self,
        mock_resolve,
        mock_get_format,
        base_config,
        mock_pg_collection,
    ):
        """Direct iteration directory with unsupported format raises NotImplementedError."""
        mock_resolve.return_value = (_DIRECT_ITERATION_DIR_SENTINEL, False)
        mock_get_format.return_value = "zarr"

        with pytest.raises(NotImplementedError, match="not supported"):
            _load_base_checkpoint("/ckpt/iter_0001000", base_config, pg_collection=mock_pg_collection)

    @patch("megatron.bridge.training.checkpointing._load_global_dist_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing._get_checkpoint_format")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_load_base_checkpoint_direct_iteration_dir_skips_non_persistent(
        self,
        mock_resolve,
        mock_get_format,
        mock_load_global,
        base_config,
        mock_pg_collection,
    ):
        """Direct iteration directory path skips non-persistent checkpoint lookup entirely."""
        mock_resolve.return_value = (_DIRECT_ITERATION_DIR_SENTINEL, False)
        mock_get_format.return_value = "torch_dist"
        mock_load_global.return_value = ({"model": "data"}, "/ckpt/iter_0001000", False, CheckpointType.GLOBAL)

        with patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration") as mock_get_np_iter:
            _load_base_checkpoint("/ckpt/iter_0001000", base_config, pg_collection=mock_pg_collection)
            mock_get_np_iter.assert_not_called()

    @patch("megatron.bridge.training.checkpointing._load_global_dist_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing._get_checkpoint_format")
    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_records_dataloader_state_dir_under_load_dir_for_persistent(
        self,
        mock_resolve,
        mock_get_np_iter,
        mock_file_exists,
        mock_get_format,
        mock_load_global,
        base_config,
        mock_pg_collection,
    ):
        """A persistent checkpoint records its dataloader-state dir under load_dir, for setup() to restore."""
        mock_resolve.return_value = (100, False)
        mock_get_np_iter.return_value = -1  # no non-persistent checkpoint
        mock_file_exists.return_value = True
        mock_get_format.return_value = "torch_dist"
        base_config.ckpt_format = "torch_dist"
        mock_load_global.return_value = ({"model": "x"}, "/ckpt/iter_0000100", False, CheckpointType.GLOBAL)

        ctx = {}
        _load_base_checkpoint("/ckpt", base_config, checkpointing_context=ctx, pg_collection=mock_pg_collection)

        assert ctx["dataloader_state_dir"] == os.path.join("/ckpt", DATALOADER_STATE_SUBDIR)

    @patch("megatron.bridge.training.checkpointing._load_non_persistent_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_records_dataloader_state_dir_under_save_for_non_persistent(
        self,
        mock_resolve,
        mock_get_np_iter,
        mock_file_exists,
        mock_load_np,
        base_config,
        mock_pg_collection,
    ):
        """A non-persistent/local checkpoint records its dataloader-state dir under checkpoint.save, not load."""
        base_config.save = "/save_dir"
        mock_resolve.return_value = (50, False)  # older persistent
        mock_get_np_iter.return_value = 100  # newer non-persistent wins
        mock_file_exists.return_value = False
        mock_load_np.return_value = ({"model": "x"}, "np_name", False, CheckpointType.GLOBAL)

        ctx = {}
        _load_base_checkpoint("/load_dir", base_config, checkpointing_context=ctx, pg_collection=mock_pg_collection)

        assert ctx["dataloader_state_dir"] == os.path.join("/save_dir", DATALOADER_STATE_SUBDIR)

    @patch("megatron.bridge.training.checkpointing._load_global_dist_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing._get_checkpoint_format")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_records_dataloader_state_dir_one_level_up_for_direct_iteration(
        self,
        mock_resolve,
        mock_get_format,
        mock_load_global,
        base_config,
        mock_pg_collection,
    ):
        """A direct iter_N directory records its dataloader-state dir one level up (the energon sibling)."""
        mock_resolve.return_value = (_DIRECT_ITERATION_DIR_SENTINEL, False)
        mock_get_format.return_value = "torch_dist"
        mock_load_global.return_value = ({"model": "x"}, "/ckpt/iter_0001000", False, CheckpointType.GLOBAL)

        ctx = {}
        _load_base_checkpoint(
            "/ckpt/iter_0001000", base_config, checkpointing_context=ctx, pg_collection=mock_pg_collection
        )

        assert ctx["dataloader_state_dir"] == os.path.join("/ckpt", DATALOADER_STATE_SUBDIR)

    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration")
    @patch("megatron.bridge.training.checkpointing._resolve_checkpoint_iteration")
    def test_records_none_when_no_checkpoint_found(
        self,
        mock_resolve,
        mock_get_np_iter,
        mock_file_exists,
        base_config,
        mock_pg_collection,
    ):
        """No checkpoint selected => dataloader_state_dir is None so resume restores nothing."""
        mock_resolve.return_value = (-1, False)
        mock_get_np_iter.return_value = -1
        mock_file_exists.return_value = False

        ctx = {}
        _load_base_checkpoint("/ckpt", base_config, checkpointing_context=ctx, pg_collection=mock_pg_collection)

        assert ctx["dataloader_state_dir"] is None


class TestRecordDataloaderStateDir:
    """Tests for _record_dataloader_state_dir, which captures where setup() restores dataloader state."""

    def test_joins_energon_subdir(self):
        ctx = {}
        _record_dataloader_state_dir(ctx, "/ckpt")
        assert ctx["dataloader_state_dir"] == os.path.join("/ckpt", DATALOADER_STATE_SUBDIR)

    def test_none_base_records_none(self):
        ctx = {}
        _record_dataloader_state_dir(ctx, None)
        assert ctx["dataloader_state_dir"] is None

    def test_none_context_is_noop(self):
        """Must not raise when there is no context to record into."""
        _record_dataloader_state_dir(None, "/ckpt")


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
    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    def test_load_model_weights_single_model_success(
        self,
        mock_get_pg_collection,
        mock_fully_parallel_wrapper,
        mock_strategy_cls,
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
        mock_strategy_cls.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.dp_cp = Mock()
        mock_get_pg_collection.return_value = mock_pg_collection

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
        mock_strategy_cls.assert_called_once_with()
        mock_load_state_dict.assert_called_once_with(mock_model[0], mock_full_state_dict["model"], True)

    @patch("megatron.bridge.training.checkpointing.delete_extra_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    def test_load_model_weights_calls_delete_extra_state(
        self,
        mock_strategy_cls,
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
        mock_strategy_cls.return_value = Mock()
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
    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    def test_load_model_weights_multiple_models_success(
        self,
        mock_get_pg_collection,
        mock_fully_parallel_wrapper,
        mock_strategy_cls,
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
        mock_strategy_cls.return_value = Mock()
        mock_generate_state_dict.return_value = {
            "model0": {"weight1": torch.randn(10, 10)},
            "model1": {"weight2": torch.randn(5, 5)},
        }
        mock_unwrap_model.return_value = mock_multiple_models

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.dp_cp = Mock()
        mock_get_pg_collection.return_value = mock_pg_collection

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
        mock_strategy_cls.assert_called_once_with()

        # Verify both models were loaded
        assert mock_load_state_dict.call_count == 2
        mock_load_state_dict.assert_any_call(mock_multiple_models[0], mock_full_state_dict["model0"], True)
        mock_load_state_dict.assert_any_call(mock_multiple_models[1], mock_full_state_dict["model1"], True)

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing._load_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    def test_load_model_weights_fully_parallel_load(
        self,
        mock_get_pg_collection,
        mock_fully_parallel_wrapper,
        mock_strategy_cls,
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
        mock_strategy_cls.return_value = mock_strategy
        mock_fully_parallel_wrapper.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_dp_cp_group = Mock()
        mock_pg_collection.dp_cp = mock_dp_cp_group
        mock_get_pg_collection.return_value = mock_pg_collection

        # Call the function
        from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

        _load_model_weights_from_checkpoint(
            checkpoint_path="/test/checkpoint",
            model=mock_model,
            fully_parallel_load=True,
            dist_ckpt_strictness="assume_ok_unexpected",
            strict=True,
        )

        # Verify fully parallel wrapper was used with pg_collection.dp_cp
        mock_fully_parallel_wrapper.assert_called_once_with(mock_strategy, mock_dp_cp_group)

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing._generate_model_state_dict")
    @patch("megatron.bridge.training.checkpointing._load_model_state_dict")
    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    def test_load_model_weights_none_state_dict(
        self,
        mock_get_pg_collection,
        mock_fully_parallel_wrapper,
        mock_strategy_cls,
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
        mock_strategy_cls.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.dp_cp = Mock()
        mock_get_pg_collection.return_value = mock_pg_collection

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
    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.FullyParallelLoadStrategyWrapper")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    def test_return_state_dict(
        self,
        mock_get_pg_collection,
        mock_fully_parallel_wrapper,
        mock_strategy_cls,
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
        mock_strategy_cls.return_value = Mock()
        mock_generate_state_dict.return_value = {"model": {"weight": torch.randn(10, 10)}}
        mock_unwrap_model.return_value = mock_model

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.dp_cp = Mock()
        mock_get_pg_collection.return_value = mock_pg_collection

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
        load_return = Mock(missing_keys=["layer.weight"], unexpected_keys=[])
        module.load_state_dict.side_effect = [Exception("boom"), load_return]

        _load_model_state_dict(module, {"w": 1}, strict=True)

        assert module.load_state_dict.call_count == 2
        first_args, first_kwargs = module.load_state_dict.call_args_list[0]
        second_args, second_kwargs = module.load_state_dict.call_args_list[1]
        assert first_kwargs.get("strict") is True
        assert second_kwargs.get("strict") is False
        assert mock_print_rank_0.called

    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    def test_load_model_state_dict_only_extra_state_keys_no_warning(self, mock_print_rank_0):
        """When every mismatched key ends with '._extra_state', no warning is printed."""
        module = Mock()
        load_return = Mock(
            missing_keys=["layer.self_attention._extra_state", "layer.mlp._extra_state"],
            unexpected_keys=["encoder.norm._extra_state"],
        )
        module.load_state_dict.side_effect = [Exception("strict mismatch"), load_return]

        _load_model_state_dict(module, {"w": 1}, strict=True)

        assert module.load_state_dict.call_count == 2
        mock_print_rank_0.assert_not_called()

    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    def test_load_model_state_dict_mixed_keys_warns_non_extra_only(self, mock_print_rank_0):
        """When some keys don't end with '._extra_state', warn with only those keys."""
        module = Mock()
        load_return = Mock(
            missing_keys=["layer.self_attention._extra_state", "layer.weight"],
            unexpected_keys=["encoder.norm._extra_state", "decoder.bias"],
        )
        err = Exception("strict mismatch")
        module.load_state_dict.side_effect = [err, load_return]

        _load_model_state_dict(module, {"w": 1}, strict=True)

        assert module.load_state_dict.call_count == 2
        assert mock_print_rank_0.call_count == 2
        warning_call = mock_print_rank_0.call_args_list[0][0][0]
        keys_call = mock_print_rank_0.call_args_list[1][0][0]
        assert "Warning: Exception during strict loading:" in warning_call
        assert "strict mismatch" in warning_call
        assert "layer.weight" in keys_call
        assert "decoder.bias" in keys_call
        assert "._extra_state" not in keys_call

    def test_load_model_state_dict_non_strict_raises(self):
        module = Mock()
        module.load_state_dict.side_effect = Exception("fail")

        with pytest.raises(Exception):
            _load_model_state_dict(module, {"w": 1}, strict=False)


class TestLoadModelWeights:
    """Test the load_model_weights function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.sharded_state_dict.return_value = {"weight": torch.randn(10, 10)}
        return [model]

    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    def test_load_model_weights_delegates_to_loader(
        self,
        mock_load_weights,
        mock_model,
    ):
        """Test load_model_weights delegates to _load_model_weights_from_checkpoint."""
        from megatron.bridge.training.checkpointing import load_model_weights

        load_model_weights(mock_model, "/checkpoint/iter_0000005")

        mock_load_weights.assert_called_once_with(
            "/checkpoint/iter_0000005",
            mock_model,
            fully_parallel_load=False,
            strict=True,
            return_state_dict=False,
        )

    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    def test_load_model_weights_with_fully_parallel_load(
        self,
        mock_load_weights,
        mock_model,
    ):
        """Test load_model_weights with fully_parallel_load enabled."""
        from megatron.bridge.training.checkpointing import load_model_weights

        load_model_weights(mock_model, "/checkpoint/iter_0000005", fully_parallel_load=True)

        mock_load_weights.assert_called_once_with(
            "/checkpoint/iter_0000005",
            mock_model,
            fully_parallel_load=True,
            strict=True,
            return_state_dict=False,
        )

    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    def test_load_model_weights_return_state_dict(
        self,
        mock_load_weights,
        mock_model,
    ):
        """Test load_model_weights with return_state_dict=True."""
        from megatron.bridge.training.checkpointing import load_model_weights

        mock_load_weights.return_value = {"model": {"weight": torch.randn(10, 10)}}

        result = load_model_weights(mock_model, "/checkpoint/iter_0000005", return_state_dict=True)

        mock_load_weights.assert_called_once_with(
            "/checkpoint/iter_0000005",
            mock_model,
            fully_parallel_load=False,
            strict=True,
            return_state_dict=True,
        )
        assert result is not None


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
    @patch("megatron.bridge.training.checkpointing.file_exists")
    def test_load_base_checkpoint_legacy_tracker(self, mock_file_exists, mock_read_metadata):
        """Test loading checkpoint with legacy Megatron-LM tracker file."""
        mock_cfg = Mock(spec=CheckpointConfig)
        mock_cfg.non_persistent_ckpt_type = None
        mock_cfg.exit_on_missing_checkpoint = False
        mock_cfg.ckpt_step = None
        mock_cfg.ckpt_format = "torch_dist"

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_pg_collection.dp_cp.size.return_value = 1

        # Mock file existence: NeMo-LM tracker doesn't exist, legacy tracker does
        def mock_file_exists_side_effect(path):
            if "latest_train_state.pt" in path:
                return False
            elif "latest_checkpointed_iteration.txt" in path:
                return True
            return False

        mock_file_exists.side_effect = mock_file_exists_side_effect
        mock_read_metadata.return_value = (1000, False)

        with patch("megatron.bridge.training.checkpointing._get_non_persistent_iteration", return_value=-1):
            with patch("megatron.bridge.training.checkpointing.dist_checkpointing") as mock_dist_ckpt:
                mock_dist_ckpt.check_is_distributed_checkpoint.return_value = True
                with patch("megatron.bridge.training.checkpointing._load_global_dist_base_checkpoint") as mock_load:
                    mock_load.return_value = ({"test": "data"}, "/ckpt/path", False, CheckpointType.GLOBAL)

                    result = _load_base_checkpoint("/test/dir", mock_cfg, rank0=True, pg_collection=mock_pg_collection)

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

    @patch("megatron.bridge.training.checkpointing.is_hf_checkpoint_dir", return_value=False)
    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.set_checkpoint_version")
    @patch("megatron.bridge.training.checkpointing.update_num_microbatches")
    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
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
        mock_dist_ckpt,
        mock_get_rng_state,
        mock_generate_state_dict,
        mock_rerun_machine,
        mock_get_pg_collection,
        mock_print_rank_0,
        mock_is_last_rank,
        mock_wandb,
        mock_update_microbatches,
        mock_set_version,
        mock_exists_checkpoint,
        mock_unwrap,
        mock_load_base,
        mock_is_hf_checkpoint_dir,
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

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 2
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_get_pg_collection.return_value = mock_pg_collection

        # Legacy checkpoints predate content metadata in the common state.
        mock_dist_ckpt.load_content_metadata.return_value = None
        mock_dist_ckpt.load.return_value = {}

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
        mock_cfg.checkpoint.ckpt_format = "torch_dist"  # Set format explicitly
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
        mock_cfg.dist = Mock()
        mock_cfg.dist.use_decentralized_pg = False

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


class TestCheckpointIterationResolution:
    """Test checkpoint iteration resolution logic."""

    def test_inprocess_restart_refreshes_mutable_bridge_tracker(self, tmp_path):
        """Restart teardown must invalidate a cached tracker overwritten by a later save."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        tracker = get_checkpoint_train_state_filename(str(tmp_path), prefix="latest")
        first_state = TrainState(step=5)
        torch.save(first_state.state_dict(), tracker)
        assert _resolve_checkpoint_iteration(str(tmp_path), None) == (5, False)

        latest_state = TrainState(step=10)
        torch.save(latest_state.state_dict(), tracker)
        GlobalState().reset_for_restart()

        assert _resolve_checkpoint_iteration(str(tmp_path), None) == (10, False)

    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    def test_loads_from_bridge_tracker(self, mock_read_train_state, mock_file_exists):
        """Should read iteration from Bridge format tracker file."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        mock_file_exists.return_value = True
        train_state = TrainState()
        train_state.step = 1000
        mock_read_train_state.return_value = train_state

        iteration, release = _resolve_checkpoint_iteration(
            load_dir="/checkpoints",
            ckpt_step_override=None,
        )

        assert iteration == 1000
        assert release is False

    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing.read_metadata")
    def test_fallback_to_legacy_tracker(self, mock_read_metadata, mock_file_exists):
        """Should fallback to legacy format when Bridge format not found."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        def file_exists_side_effect(path):
            # Bridge format doesn't exist, legacy does
            return "latest_checkpointed_iteration.txt" in path

        mock_file_exists.side_effect = file_exists_side_effect
        mock_read_metadata.return_value = (2000, False)

        iteration, release = _resolve_checkpoint_iteration(
            load_dir="/checkpoints",
            ckpt_step_override=None,
        )

        assert iteration == 2000
        assert release is False

    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    def test_ckpt_step_validates_existence(self, mock_read_train_state, mock_file_exists):
        """ckpt_step should validate checkpoint directory exists."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        mock_file_exists.return_value = True  # Checkpoint dir exists

        iteration, release = _resolve_checkpoint_iteration(
            load_dir="/checkpoints",
            ckpt_step_override=5000,
        )

        assert iteration == 5000
        assert release is False
        # Verify we checked if checkpoint dir exists (but not tracker file)
        mock_file_exists.assert_called_once()
        assert "/checkpoints/iter_0005000" in str(mock_file_exists.call_args)
        mock_read_train_state.assert_not_called()

    @patch("megatron.bridge.training.checkpointing.file_exists")
    def test_ckpt_step_raises_if_not_exists(self, mock_file_exists):
        """ckpt_step should raise FileNotFoundError if checkpoint directory doesn't exist."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        mock_file_exists.return_value = False  # Checkpoint dir doesn't exist

        with pytest.raises(FileNotFoundError) as exc_info:
            _resolve_checkpoint_iteration(
                load_dir="/checkpoints",
                ckpt_step_override=3000,
            )

        assert "ckpt_step=3000 specified but checkpoint directory does not exist" in str(exc_info.value)
        assert "/checkpoints/iter_0003000" in str(exc_info.value)
        assert "ls /checkpoints/iter_*" in str(exc_info.value)

    def test_ckpt_step_validation_in_config_finalize(self):
        """ckpt_step without load should raise ValueError during config finalize."""
        from megatron.bridge.training.config import CheckpointConfig

        config = CheckpointConfig(
            ckpt_step=3000,
            load=None,  # Missing load directory
        )

        with pytest.raises(ValueError) as exc_info:
            config.finalize()

        assert "ckpt_step=3000 specified but checkpoint.load is None" in str(exc_info.value)
        assert "Please set checkpoint.load to the base checkpoint directory" in str(exc_info.value)

    def test_ckpt_step_with_only_pretrained_raises(self):
        """ckpt_step with only pretrained_checkpoint should raise because pretrained path does not exist."""
        from megatron.bridge.training.config import CheckpointConfig

        config = CheckpointConfig(
            ckpt_step=5000,
            load=None,
            pretrained_checkpoint="/pretrained/model",  # Has pretrained but no load
        )

        with pytest.raises(AssertionError, match="Pretrained checkpoint /pretrained/model does not exist"):
            config.finalize()

    def test_no_load_dir_returns_default(self):
        """When load_dir is None, should return iteration=-1."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        iteration, release = _resolve_checkpoint_iteration(
            load_dir=None,
            ckpt_step_override=None,
        )

        assert iteration == -1
        assert release is False

    @patch("megatron.bridge.training.checkpointing.file_exists")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    def test_ignore_ckpt_step_loads_latest(self, mock_read_train_state, mock_file_exists):
        """When ignore_ckpt_step=True via load path, should load latest even if ckpt_step is set in config."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        mock_file_exists.return_value = True
        train_state = TrainState()
        train_state.step = 9999  # Latest checkpoint
        mock_read_train_state.return_value = train_state

        # Pass None to _resolve_checkpoint_iteration (simulating ignore_ckpt_step=True)
        iteration, release = _resolve_checkpoint_iteration(
            load_dir="/pretrained",
            ckpt_step_override=None,  # Passed as None when ignore_ckpt_step=True
        )

        # Should load latest (9999), not ckpt_step value
        assert iteration == 9999
        assert release is False

    @patch("megatron.bridge.training.checkpointing.is_checkpoint_iteration_directory")
    def test_direct_iteration_directory_returns_sentinel(self, mock_is_iter_dir):
        """When load_dir is already an iteration directory, should return the sentinel value."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        mock_is_iter_dir.return_value = True

        iteration, release = _resolve_checkpoint_iteration(
            load_dir="/checkpoints/iter_0001000",
            ckpt_step_override=None,
        )

        assert iteration == _DIRECT_ITERATION_DIR_SENTINEL
        assert release is False
        mock_is_iter_dir.assert_called_once_with("/checkpoints/iter_0001000")

    @patch("megatron.bridge.training.checkpointing.is_checkpoint_iteration_directory")
    def test_direct_iteration_directory_ignores_ckpt_step(self, mock_is_iter_dir):
        """Iteration directory detection takes priority over ckpt_step_override."""
        from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration

        mock_is_iter_dir.return_value = True

        iteration, release = _resolve_checkpoint_iteration(
            load_dir="/checkpoints/iter_0001000",
            ckpt_step_override=5000,
        )

        assert iteration == _DIRECT_ITERATION_DIR_SENTINEL
        assert release is False


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

    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("torch.distributed.is_initialized")
    def test_get_rng_state_fsdp_dtensor_format(self, mock_dist_init, mock_tp):
        """Test get_rng_state returns correct format for fsdp_dtensor."""
        mock_dist_init.return_value = False  # Simplify
        mock_tracker = Mock()
        mock_tracker.get_states.return_value = "tracker_states"
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_pg_collection.dp_cp.size.return_value = 1
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1

        with (
            patch("random.getstate"),
            patch("numpy.random.get_state"),
            patch("torch.get_rng_state"),
            patch("torch.cuda.get_rng_state"),
        ):
            result = get_rng_state(
                data_parallel_random_init=False, ckpt_format="fsdp_dtensor", pg_collection=mock_pg_collection
            )

        # Should return dict format for fsdp_dtensor
        assert isinstance(result, dict)
        assert "(0, 0)" in result

    @patch("megatron.bridge.training.checkpointing.tensor_parallel")
    @patch("torch.distributed.is_initialized")
    def test_get_rng_state_torch_dist_format(self, mock_dist_init, mock_tp):
        """Test get_rng_state returns ShardedObject for torch_dist."""
        mock_dist_init.return_value = False
        mock_tracker = Mock()
        mock_tracker.get_states.return_value = "tracker_states"
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        # Create mock pg_collection
        mock_pg_collection = Mock()
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_pg_collection.dp_cp.size.return_value = 1
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1

        with (
            patch("random.getstate"),
            patch("numpy.random.get_state"),
            patch("torch.get_rng_state"),
            patch("torch.cuda.get_rng_state"),
            patch("megatron.bridge.training.checkpointing.ShardedObject") as mock_sharded_obj,
        ):
            _ = get_rng_state(
                data_parallel_random_init=False, ckpt_format="torch_dist", pg_collection=mock_pg_collection
            )

        # Should create ShardedObject for torch_dist format
        # The exact arguments depend on the RNG state, but we just verify it was called
        mock_sharded_obj.assert_called_once()

    def test_build_sharded_state_dict_metadata_fsdp_dtensor(self):
        """Test _build_sharded_state_dict_metadata for fsdp_dtensor format."""
        from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata

        ckpt_cfg = CheckpointConfig(fully_parallel_save=False, ckpt_format="fsdp_dtensor")
        result = _build_sharded_state_dict_metadata(use_distributed_optimizer=True, cfg=ckpt_cfg)

        assert result["distrib_optim_sharding_type"] == "fsdp_dtensor"
        assert result["chained_optim_avoid_prefix"] is True
        assert result["singleton_local_shards"] is False

    def test_build_sharded_state_dict_metadata_torch_dist_fully_parallel(self):
        """Test _build_sharded_state_dict_metadata for torch_dist with fully parallel save."""
        from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata

        ckpt_cfg = CheckpointConfig(fully_parallel_save=True, ckpt_format="torch_dist")
        result = _build_sharded_state_dict_metadata(use_distributed_optimizer=True, cfg=ckpt_cfg)

        assert result["distrib_optim_sharding_type"] == "dp_reshardable"

    def test_build_sharded_state_dict_metadata_torch_dist_dp_zero(self):
        """Test _build_sharded_state_dict_metadata for torch_dist with dp_zero_gather_scatter."""
        from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata

        ckpt_cfg = CheckpointConfig(fully_parallel_save=False, ckpt_format="torch_dist")
        result = _build_sharded_state_dict_metadata(use_distributed_optimizer=True, cfg=ckpt_cfg)

        assert result["distrib_optim_sharding_type"] == "dp_reshardable"

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", True)
    @patch("torch.distributed.checkpoint.FileSystemReader")
    @patch("torch.distributed.checkpoint.load_state_dict")
    @patch("torch.distributed.checkpoint.default_planner.DefaultLoadPlanner")
    def test_load_fsdp_dtensor_checkpoint_rank0(self, mock_planner, mock_load_state_dict, mock_reader):
        """Test load_fsdp_dtensor_checkpoint for rank0."""
        from megatron.bridge.training.checkpointing import load_fsdp_dtensor_checkpoint
        from megatron.bridge.training.config import CheckpointConfig

        ckpt_cfg = CheckpointConfig()

        state_dict, checkpoint_name, release, ckpt_type = load_fsdp_dtensor_checkpoint(
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
    def test_load_fsdp_dtensor_checkpoint_no_fsdp(self):
        """Test load_fsdp_dtensor_checkpoint raises error when FSDP not available."""
        from megatron.bridge.training.checkpointing import load_fsdp_dtensor_checkpoint
        from megatron.bridge.training.config import CheckpointConfig

        ckpt_cfg = CheckpointConfig()

        with pytest.raises(RuntimeError, match="Megatron FSDP is required but not available"):
            load_fsdp_dtensor_checkpoint(
                load_dir="/test/dir",
                ckpt_cfg=ckpt_cfg,
                rank0=False,
                sharded_state_dict={},
                iteration=1000,
                release=False,
            )

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", True)
    def test_generate_state_dict_fsdp_dtensor_no_preprocessing(self):
        """Test generate_state_dict does NOT apply FSDP DTensor preprocessing."""
        from unittest.mock import Mock

        from megatron.bridge.training.checkpointing import generate_state_dict
        from megatron.bridge.training.config import CheckpointConfig

        # Create mock model
        mock_model = Mock()
        mock_model.state_dict_for_save_checkpoint.return_value = {"test_param": torch.tensor([1.0])}

        with (
            patch("megatron.bridge.training.checkpointing.handle_fp8_extra_state_case") as mock_fp8,
            patch("megatron.bridge.training.checkpointing.preprocess_state_dict_for_uneven_dtensor") as mock_uneven,
        ):
            ckpt_cfg = CheckpointConfig(ckpt_format="fsdp_dtensor", save_rng=False)
            result = generate_state_dict(
                ckpt_cfg=ckpt_cfg,
                model=[mock_model],
                optimizer=None,
                opt_param_scheduler=None,
                rng_state=None,
            )

            # Should NOT call FSDP preprocessing functions (moved to preprocess_fsdp_dtensor_state_dict)
            mock_fp8.assert_not_called()
            mock_uneven.assert_not_called()
            # Should use state_dict_for_save_checkpoint for fsdp_dtensor
            mock_model.state_dict_for_save_checkpoint.assert_called_once()
            assert "model" in result
            assert result["checkpoint_version"] == 3.0

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", True)
    def test_preprocess_fsdp_dtensor_state_dict(self):
        """Test preprocess_fsdp_dtensor_state_dict applies all preprocessing steps."""
        from unittest.mock import Mock

        from megatron.bridge.training.checkpointing import preprocess_fsdp_dtensor_state_dict

        # Create mock model and config
        mock_model = Mock()
        mock_cfg = Mock()

        # Mock model config without swiglu or experts
        with patch("megatron.core.utils.get_model_config") as mock_get_config:
            mock_model_config = Mock()
            mock_model_config.gated_linear_unit = False
            mock_model_config.num_moe_experts = None
            mock_get_config.return_value = mock_model_config

            with (
                patch("megatron.bridge.training.checkpointing.handle_fp8_extra_state_case") as mock_fp8,
                patch(
                    "megatron.bridge.training.checkpointing.preprocess_state_dict_for_uneven_dtensor"
                ) as mock_uneven,
                patch("megatron.bridge.training.checkpointing.handle_gdn_in_state_dict", None),
            ):
                raw_state_dict = {"model": {"test_param": torch.tensor([1.0])}}
                result = preprocess_fsdp_dtensor_state_dict(mock_cfg, raw_state_dict, mock_model)

                # Should call FP8 and uneven dtensor preprocessing
                mock_fp8.assert_called_once()
                mock_uneven.assert_called_once()
                assert "model" in result

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", True)
    def test_save_fsdp_dtensor_checkpoint_preprocesses_and_saves(self, tmp_path):
        """Test public FSDP DTensor save helper preprocesses and calls PyTorch DCP."""
        from megatron.bridge.training.checkpointing import save_fsdp_dtensor_checkpoint

        checkpoint_path = tmp_path / "iter_0000001"
        raw_state_dict = {"model": {"test_param": torch.tensor([1.0])}}
        preprocessed_state_dict = {"model": {"test_param": torch.tensor([2.0])}}
        cfg = Mock()
        model = Mock()
        writer = Mock()
        save_result = object()

        with (
            patch(
                "megatron.bridge.training.checkpointing.preprocess_fsdp_dtensor_state_dict",
                return_value=preprocessed_state_dict,
            ) as mock_preprocess,
            patch("megatron.bridge.training.checkpointing.MultiStorageClientFeature.is_enabled", return_value=False),
            patch("torch.distributed.checkpoint.FileSystemWriter", return_value=writer) as mock_writer_cls,
            patch("torch.distributed.checkpoint.save", return_value=save_result) as mock_save,
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.barrier") as mock_barrier,
        ):
            result = save_fsdp_dtensor_checkpoint(checkpoint_path, raw_state_dict, cfg=cfg, model=model)

        assert result is save_result
        mock_preprocess.assert_called_once_with(cfg, raw_state_dict, model)
        mock_writer_cls.assert_called_once_with(str(checkpoint_path))
        mock_save.assert_called_once_with(state_dict=preprocessed_state_dict, storage_writer=writer)
        mock_barrier.assert_called_once()

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", False)
    def test_save_fsdp_dtensor_checkpoint_requires_megatron_fsdp(self, tmp_path):
        """Test public FSDP DTensor save helper reports missing Megatron FSDP."""
        from megatron.bridge.training.checkpointing import save_fsdp_dtensor_checkpoint

        with pytest.raises(RuntimeError, match="Megatron FSDP is required but not available"):
            save_fsdp_dtensor_checkpoint(tmp_path, {"model": {}}, cfg=Mock(), model=Mock())

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

    def test_generate_state_dict_includes_optimizer_scaffold_when_loading(self):
        """Load-time optimizer scaffolding should not depend on save_optim."""
        from unittest.mock import Mock

        from megatron.bridge.training.checkpointing import generate_state_dict
        from megatron.bridge.training.config import CheckpointConfig

        mock_model = Mock()
        mock_model.sharded_state_dict.return_value = {"test_param": torch.tensor([1.0])}
        mock_optimizer = Mock()
        mock_optimizer.is_stub_optimizer = False
        mock_optimizer.sharded_state_dict.return_value = {"optimizer": {"param_groups": []}}
        mock_scheduler = Mock()
        mock_scheduler.state_dict.return_value = {"scheduler": "state"}

        ckpt_cfg = CheckpointConfig(ckpt_format="torch_dist", save_optim=False, save_rng=False)
        result = generate_state_dict(
            ckpt_cfg=ckpt_cfg,
            model=[mock_model],
            optimizer=mock_optimizer,
            opt_param_scheduler=mock_scheduler,
            rng_state=None,
            optim_sd_kwargs={
                "is_loading": True,
                "metadata": {"distrib_optim_sharding_type": "dp_zero_gather_scatter"},
            },
        )

        assert result["optimizer"] == {"optimizer": {"param_groups": []}}
        assert result["opt_param_scheduler"] == {"scheduler": "state"}
        mock_optimizer.sharded_state_dict.assert_called_once()

    @pytest.mark.parametrize("ckpt_format", ["torch_dist", "fsdp_dtensor"])
    @pytest.mark.parametrize(
        "is_loading,has_rng_scaffold,expected_rng_state",
        [(True, True, True), (True, False, False), (False, True, False)],
    )
    def test_generate_state_dict_includes_rng_scaffold_only_when_loading(
        self, ckpt_format, is_loading, has_rng_scaffold, expected_rng_state
    ):
        """Load-time RNG scaffolding should not depend on save_rng."""
        from megatron.bridge.training.checkpointing import generate_state_dict

        mock_model = Mock()
        mock_model.sharded_state_dict.return_value = {"test_param": torch.tensor([1.0])}
        mock_model.state_dict_for_save_checkpoint.return_value = {"test_param": torch.tensor([1.0])}
        mock_rng_state = Mock() if has_rng_scaffold else None
        optim_sd_kwargs = {"is_loading": True} if is_loading else None

        ckpt_cfg = CheckpointConfig(ckpt_format=ckpt_format, save_rng=False)
        result = generate_state_dict(
            ckpt_cfg=ckpt_cfg,
            model=[mock_model],
            optimizer=None,
            opt_param_scheduler=None,
            rng_state=mock_rng_state,
            optim_sd_kwargs=optim_sd_kwargs,
        )

        assert ("rng_state" in result) is expected_rng_state
        if expected_rng_state:
            assert result["rng_state"] is mock_rng_state


class TestCheckpointPathOverride:
    """Test checkpoint_path_override parameter in loading functions."""

    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    def test_load_global_dist_uses_override_rank0(self, mock_dist_ckpt, mock_strategy_cls):
        """rank0 path should use checkpoint_path_override instead of find_checkpoint_rank_0."""
        from megatron.bridge.training.checkpointing import _load_global_dist_base_checkpoint

        mock_dist_ckpt.load_common_state_dict.return_value = {"test": "data"}
        mock_pg = Mock()

        state_dict, checkpoint_name, release, ckpt_type = _load_global_dist_base_checkpoint(
            load_dir="/should/not/be/used",
            ckpt_cfg=CheckpointConfig(),
            rank0=True,
            sharded_state_dict=None,
            iteration=None,
            release=False,
            checkpoint_path_override="/direct/iter_0001000",
            pg_collection=mock_pg,
        )

        assert checkpoint_name == "/direct/iter_0001000"
        assert ckpt_type == CheckpointType.GLOBAL
        mock_dist_ckpt.load_common_state_dict.assert_called_once_with("/direct/iter_0001000")
        mock_strategy_cls.assert_not_called()

    @patch("megatron.bridge.training.checkpointing.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    def test_load_global_dist_uses_override_non_rank0(self, mock_dist_ckpt, mock_strategy_cls):
        """Non-rank0 path should use checkpoint_path_override instead of get_checkpoint_name."""
        from megatron.bridge.training.checkpointing import _load_global_dist_base_checkpoint

        mock_strategy_cls.return_value = Mock()
        mock_dist_ckpt.load.return_value = {"model": "sharded_data"}
        mock_pg = Mock()
        mock_pg.dp_cp = Mock()

        sharded_sd = {"weight": "placeholder"}
        state_dict, checkpoint_name, release, ckpt_type = _load_global_dist_base_checkpoint(
            load_dir="/should/not/be/used",
            ckpt_cfg=CheckpointConfig(),
            rank0=False,
            sharded_state_dict=sharded_sd,
            iteration=None,
            release=False,
            checkpoint_path_override="/direct/iter_0001000",
            pg_collection=mock_pg,
        )

        assert checkpoint_name == "/direct/iter_0001000"
        mock_strategy_cls.assert_called_once_with()
        mock_dist_ckpt.load.assert_called_once()
        load_call_args = mock_dist_ckpt.load.call_args
        assert load_call_args[0][1] == "/direct/iter_0001000"

    @patch("megatron.bridge.training.checkpointing.HAVE_MEGATRON_FSDP", True)
    def test_load_fsdp_dtensor_uses_override_rank0(self):
        """rank0 path should use checkpoint_path_override instead of get_checkpoint_name."""
        from megatron.bridge.training.checkpointing import load_fsdp_dtensor_checkpoint

        state_dict, checkpoint_name, release, ckpt_type = load_fsdp_dtensor_checkpoint(
            load_dir="/should/not/be/used",
            ckpt_cfg=CheckpointConfig(),
            rank0=True,
            sharded_state_dict=None,
            iteration=None,
            release=False,
            checkpoint_path_override="/direct/iter_0001000",
        )

        assert checkpoint_name == "/direct/iter_0001000"
        assert state_dict == {}
        assert ckpt_type == CheckpointType.FSDP_DTENSOR


class TestLoadCheckpointFromPathDirectIterDir:
    """Test _load_checkpoint_from_path with a direct iteration directory (fsdp_dtensor path).

    The fsdp_dtensor branch in _load_checkpoint_from_path has its own
    is_checkpoint_iteration_directory check to resolve the checkpoint path
    before constructing the sharded state dict.  We verify that when the
    load_dir is an iteration directory the FileSystemReader receives the
    directory directly (no tracker-file indirection).
    """

    @patch("megatron.bridge.training.checkpointing.is_checkpoint_iteration_directory")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.core.dist_checkpointing.strategies.torch.FileSystemReader")
    def test_fsdp_dtensor_skips_tracker_resolution(self, mock_reader, mock_unwrap, mock_get_pg, mock_is_iter_dir):
        """When load_dir is an iteration directory, FileSystemReader should receive it directly."""
        from megatron.core.msc_utils import MultiStorageClientFeature

        from megatron.bridge.training.checkpointing import _load_checkpoint_from_path

        MultiStorageClientFeature.disable()

        mock_is_iter_dir.return_value = True

        mock_metadata = Mock()
        mock_metadata.state_dict_metadata = {}
        mock_reader_instance = Mock()
        mock_reader_instance.read_metadata.return_value = mock_metadata
        mock_reader.return_value = mock_reader_instance

        mock_model = Mock()
        mock_unwrap.return_value = [mock_model]
        mock_pg = Mock()
        mock_pg.dp_cp = Mock()
        mock_get_pg.return_value = mock_pg

        mock_cfg = Mock()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.ckpt_format = "fsdp_dtensor"
        mock_cfg.checkpoint.finetune = True
        mock_cfg.checkpoint.load_rng = False
        mock_cfg.checkpoint.load_optim = False
        mock_cfg.checkpoint.ckpt_step = None
        mock_cfg.checkpoint.load = None
        mock_cfg.checkpoint.pretrained_checkpoint = None
        mock_cfg.optimizer = Mock()
        mock_cfg.optimizer.use_distributed_optimizer = False
        mock_cfg.peft = None
        mock_cfg.rng = Mock()

        mock_state = Mock(spec=GlobalState)
        mock_state.cfg = mock_cfg

        with (
            patch("megatron.bridge.training.checkpointing.generate_state_dict", return_value={"model": {}}),
            patch("megatron.bridge.training.checkpointing._build_sharded_state_dict_metadata", return_value={}),
            patch("megatron.bridge.training.checkpointing._load_base_checkpoint") as mock_load_base,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
        ):
            mock_load_base.return_value = (
                {"model": {}, "checkpoint_version": 3.0},
                "/ckpt/iter_0001000",
                False,
                CheckpointType.FSDP_DTENSOR,
            )

            _load_checkpoint_from_path(
                load_dir="/ckpt/iter_0001000",
                state=mock_state,
                model=[mock_model],
                optimizer=None,
                opt_param_scheduler=None,
                skip_load_to_model_and_opt=True,
            )

            # The fsdp_dtensor prep block should have called FileSystemReader
            # with the direct path (not a tracker-resolved path).
            mock_reader.assert_called_once_with("/ckpt/iter_0001000")
            mock_is_iter_dir.assert_called_once_with("/ckpt/iter_0001000")

    @patch("megatron.bridge.training.checkpointing.is_checkpoint_iteration_directory")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("multistorageclient.torch.MultiStorageFileSystemReader")
    def test_fsdp_dtensor_skips_tracker_resolution_with_msc(
        self, mock_reader, mock_unwrap, mock_get_pg, mock_is_iter_dir
    ):
        """When load_dir is an iteration directory, FileSystemReader should receive it directly."""
        from megatron.core.msc_utils import MultiStorageClientFeature

        from megatron.bridge.training.checkpointing import _load_checkpoint_from_path

        MultiStorageClientFeature.enable()

        mock_is_iter_dir.return_value = True

        mock_metadata = Mock()
        mock_metadata.state_dict_metadata = {}
        mock_reader_instance = Mock()
        mock_reader_instance.read_metadata.return_value = mock_metadata
        mock_reader.return_value = mock_reader_instance

        mock_model = Mock()
        mock_unwrap.return_value = [mock_model]
        mock_pg = Mock()
        mock_pg.dp_cp = Mock()
        mock_get_pg.return_value = mock_pg

        mock_cfg = Mock()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.ckpt_format = "fsdp_dtensor"
        mock_cfg.checkpoint.finetune = True
        mock_cfg.checkpoint.load_rng = False
        mock_cfg.checkpoint.load_optim = False
        mock_cfg.checkpoint.ckpt_step = None
        mock_cfg.checkpoint.load = None
        mock_cfg.checkpoint.pretrained_checkpoint = None
        mock_cfg.optimizer = Mock()
        mock_cfg.optimizer.use_distributed_optimizer = False
        mock_cfg.peft = None
        mock_cfg.rng = Mock()

        mock_state = Mock(spec=GlobalState)
        mock_state.cfg = mock_cfg

        with (
            patch("megatron.bridge.training.checkpointing.generate_state_dict", return_value={"model": {}}),
            patch("megatron.bridge.training.checkpointing._build_sharded_state_dict_metadata", return_value={}),
            patch("megatron.bridge.training.checkpointing._load_base_checkpoint") as mock_load_base,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
        ):
            mock_load_base.return_value = (
                {"model": {}, "checkpoint_version": 3.0},
                "/ckpt/iter_0001000",
                False,
                CheckpointType.FSDP_DTENSOR,
            )

            _load_checkpoint_from_path(
                load_dir="/ckpt/iter_0001000",
                state=mock_state,
                model=[mock_model],
                optimizer=None,
                opt_param_scheduler=None,
                skip_load_to_model_and_opt=True,
            )

            # The fsdp_dtensor prep block should have called FileSystemReader
            # with the direct path (not a tracker-resolved path).
            mock_reader.assert_called_once_with("/ckpt/iter_0001000", thread_count=2)
            mock_is_iter_dir.assert_called_once_with("/ckpt/iter_0001000")


class TestCheckpointManager:
    """Tests for the CheckpointManager interface and DefaultCheckpointManager."""

    def test_checkpoint_save_context_creation(self):
        """Test CheckpointSaveContext dataclass can be created with required fields."""
        mock_state = Mock(spec=GlobalState)
        mock_model = [Mock()]
        mock_optimizer = Mock()
        mock_scheduler = Mock()

        ctx = CheckpointSaveContext(
            state=mock_state,
            model=mock_model,
            optimizer=mock_optimizer,
            opt_param_scheduler=mock_scheduler,
            num_floating_point_operations_so_far=1000,
        )

        assert ctx.state is mock_state
        assert ctx.model is mock_model
        assert ctx.optimizer is mock_optimizer
        assert ctx.opt_param_scheduler is mock_scheduler
        assert ctx.num_floating_point_operations_so_far == 1000
        assert ctx.train_data_iterator is None
        assert ctx.non_persistent_ckpt is False

    def test_checkpoint_save_context_with_optional_fields(self):
        """Test CheckpointSaveContext with optional fields set."""
        mock_state = Mock(spec=GlobalState)
        mock_iterator = Mock()

        ctx = CheckpointSaveContext(
            state=mock_state,
            model=[Mock()],
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            train_data_iterator=mock_iterator,
            non_persistent_ckpt=True,
        )

        assert ctx.train_data_iterator is mock_iterator
        assert ctx.non_persistent_ckpt is True

    def test_checkpoint_load_context_creation(self):
        """Test CheckpointLoadContext dataclass can be created with required fields."""
        mock_state = Mock(spec=GlobalState)
        mock_model = [Mock()]
        mock_optimizer = Mock()
        mock_scheduler = Mock()

        ctx = CheckpointLoadContext(
            state=mock_state,
            model=mock_model,
            optimizer=mock_optimizer,
            opt_param_scheduler=mock_scheduler,
        )

        assert ctx.state is mock_state
        assert ctx.model is mock_model
        assert ctx.optimizer is mock_optimizer
        assert ctx.opt_param_scheduler is mock_scheduler
        assert ctx.strict is True
        assert ctx.skip_load_to_model_and_opt is False

    def test_checkpoint_load_context_with_optional_fields(self):
        """Test CheckpointLoadContext with optional fields set."""
        mock_state = Mock(spec=GlobalState)

        ctx = CheckpointLoadContext(
            state=mock_state,
            model=[Mock()],
            optimizer=None,
            opt_param_scheduler=None,
            strict=False,
            skip_load_to_model_and_opt=True,
        )

        assert ctx.strict is False
        assert ctx.skip_load_to_model_and_opt is True

    def test_create_checkpoint_manager_returns_default(self):
        """Test create_checkpoint_manager returns DefaultCheckpointManager when no custom class."""
        config = CheckpointConfig()

        manager = create_checkpoint_manager(config)

        assert isinstance(manager, DefaultCheckpointManager)
        assert manager.checkpoint_config is config

    def test_create_checkpoint_manager_invalid_format_raises(self):
        """Test create_checkpoint_manager raises ValueError for invalid class format."""
        config = CheckpointConfig(custom_manager_class="InvalidClassName")

        with pytest.raises(ValueError, match="Invalid custom_manager_class format"):
            create_checkpoint_manager(config)

    def test_create_checkpoint_manager_missing_module_raises(self):
        """Test create_checkpoint_manager raises ImportError for non-existent module."""
        config = CheckpointConfig(custom_manager_class="megatron.bridge.nonexistent_module.ClassName")

        with pytest.raises(ImportError, match="Could not import module"):
            create_checkpoint_manager(config)

    def test_create_checkpoint_manager_missing_class_raises(self):
        """Test create_checkpoint_manager raises AttributeError for non-existent class."""
        # Use a real module but non-existent class
        config = CheckpointConfig(custom_manager_class="megatron.bridge.training.checkpointing.NonExistentClass")

        with pytest.raises(AttributeError, match="does not have class"):
            create_checkpoint_manager(config)

    def test_create_checkpoint_manager_rejects_unallowed_custom_class_before_import(self):
        """Test custom manager class paths are allowlist-validated before import."""
        config = CheckpointConfig(custom_manager_class="attacker_pkg.manager.PayloadManager")

        with patch("importlib.import_module") as mock_import:
            with pytest.raises(InstantiationException, match="not in the allowlist"):
                create_checkpoint_manager(config)

        mock_import.assert_not_called()

    def test_create_checkpoint_manager_custom_class(self):
        """Test create_checkpoint_manager loads and instantiates a custom class."""

        # Create a simple custom manager class for testing
        class CustomTestManager:
            def __init__(self, checkpoint_config):
                self.checkpoint_config = checkpoint_config
                self.initialized = True

            def save(self, _ctx):
                pass

            def load(self, _ctx):
                return (0, 0)

            def finalize_async_saves(self, state, blocking=False, terminate=False):
                pass

        # Patch the import to return our test class
        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.CustomTestManager = CustomTestManager
            mock_import.return_value = mock_module

            config = CheckpointConfig(custom_manager_class="megatron.bridge.test.CustomTestManager")
            manager = create_checkpoint_manager(config)

            assert isinstance(manager, CustomTestManager)
            assert manager.checkpoint_config is config
            assert manager.initialized is True

    def test_default_checkpoint_manager_init(self):
        """Test DefaultCheckpointManager initialization."""
        config = CheckpointConfig()

        with patch("megatron.bridge.training.checkpointing.init_checkpointing_context") as mock_init_ctx:
            mock_init_ctx.return_value = {"test_key": "test_value"}
            manager = DefaultCheckpointManager(config)

            assert manager.checkpoint_config is config
            mock_init_ctx.assert_called_once_with(config)
            assert manager.checkpointing_context == {"test_key": "test_value"}

    def test_default_checkpoint_manager_save_delegates(self):
        """Test DefaultCheckpointManager.save() delegates to save_checkpoint."""
        config = CheckpointConfig()
        mock_state = Mock(spec=GlobalState)
        mock_model = [Mock()]
        mock_optimizer = Mock()
        mock_scheduler = Mock()

        ctx = CheckpointSaveContext(
            state=mock_state,
            model=mock_model,
            optimizer=mock_optimizer,
            opt_param_scheduler=mock_scheduler,
            num_floating_point_operations_so_far=5000,
            train_data_iterator=Mock(),
            non_persistent_ckpt=True,
        )

        with (
            patch("megatron.bridge.training.checkpointing.init_checkpointing_context") as mock_init_ctx,
            patch("megatron.bridge.training.checkpointing.save_checkpoint") as mock_save,
        ):
            mock_init_ctx.return_value = {"context": "data"}
            manager = DefaultCheckpointManager(config)
            manager.save(ctx, None)

            mock_save.assert_called_once_with(
                state=mock_state,
                model=mock_model,
                optimizer=mock_optimizer,
                opt_param_scheduler=mock_scheduler,
                num_floating_point_operations_so_far=5000,
                checkpointing_context={"context": "data"},
                non_persistent_ckpt=True,
                train_data_iterator=ctx.train_data_iterator,
                pg_collection=None,
                callback_manager=None,
                module_name=None,
            )

    def test_default_checkpoint_manager_load_delegates(self):
        """Test DefaultCheckpointManager.load() delegates to load_checkpoint."""
        config = CheckpointConfig()
        mock_state = Mock(spec=GlobalState)
        mock_model = [Mock()]
        mock_optimizer = Mock()
        mock_scheduler = Mock()

        ctx = CheckpointLoadContext(
            state=mock_state,
            model=mock_model,
            optimizer=mock_optimizer,
            opt_param_scheduler=mock_scheduler,
            strict=False,
            skip_load_to_model_and_opt=True,
        )

        with (
            patch("megatron.bridge.training.checkpointing.init_checkpointing_context") as mock_init_ctx,
            patch("megatron.bridge.training.checkpointing.load_checkpoint") as mock_load,
        ):
            mock_init_ctx.return_value = {"context": "data"}
            mock_load.return_value = (100, 50000)

            manager = DefaultCheckpointManager(config)
            result = manager.load(ctx)

            mock_load.assert_called_once_with(
                state=mock_state,
                model=mock_model,
                optimizer=mock_optimizer,
                opt_param_scheduler=mock_scheduler,
                strict=False,
                checkpointing_context={"context": "data"},
                skip_load_to_model_and_opt=True,
                pg_collection=None,
                module_name=None,
            )
            assert result == (100, 50000)

    def test_default_checkpoint_manager_finalize_async_saves(self):
        """Test DefaultCheckpointManager.finalize_async_saves() delegates correctly."""
        config = CheckpointConfig()
        mock_state = Mock(spec=GlobalState)

        with (
            patch("megatron.bridge.training.checkpointing.init_checkpointing_context") as mock_init_ctx,
            patch("megatron.bridge.training.checkpointing.maybe_finalize_async_save") as mock_finalize,
        ):
            mock_init_ctx.return_value = {}

            manager = DefaultCheckpointManager(config)
            manager.finalize_async_saves(state=mock_state, blocking=True, terminate=True)

            mock_finalize.assert_called_once_with(
                global_state=mock_state,
                ckpt_cfg=config,
                blocking=True,
                terminate=True,
            )

    def test_checkpoint_manager_protocol_compliance(self):
        """Test that DefaultCheckpointManager satisfies the CheckpointManager protocol."""
        config = CheckpointConfig()

        with patch("megatron.bridge.training.checkpointing.init_checkpointing_context"):
            manager = DefaultCheckpointManager(config)

            # Protocol check using isinstance (works because @runtime_checkable)
            assert isinstance(manager, CheckpointManager)

    def test_checkpointing_context_property(self):
        """Test DefaultCheckpointManager.checkpointing_context property."""
        config = CheckpointConfig()
        expected_context = {"local_checkpoint_manager": Mock(), "save_strategy": Mock()}

        with patch("megatron.bridge.training.checkpointing.init_checkpointing_context") as mock_init_ctx:
            mock_init_ctx.return_value = expected_context

            manager = DefaultCheckpointManager(config)

            assert manager.checkpointing_context is expected_context
            assert manager._context is expected_context


@pytest.mark.unit
class TestLayerWiseOptimizerCheckpointing:
    """Tests for LayerWiseDistributedOptimizer checkpoint save/load guards.

    Covers the fix for https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2805:
    - save_checkpoint must NOT write layer_wise_optimizer_*.pt for torch_dist/fsdp_dtensor
      formats because the optimizer state is already embedded in the distributed checkpoint.
    - load_checkpoint must load layer_wise_optimizer_*.pt from the local checkpoint base
      directory for LOCAL checkpoints.
    """

    # -----------------------------------------------------------------------
    # Save-side tests
    # -----------------------------------------------------------------------

    @patch("megatron.bridge.training.checkpointing.wandb_utils")
    @patch("megatron.bridge.training.checkpointing.is_last_rank")
    @patch("builtins.open", new_callable=mock_open)
    @patch("torch.save")
    @patch("shutil.copy")
    @patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.fault_tolerance")
    @patch("megatron.bridge.training.checkpointing.is_empty_async_queue")
    @patch("megatron.bridge.training.checkpointing.get_rank_safe")
    @patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state")
    @patch("megatron.bridge.training.checkpointing.ensure_directory_exists")
    @patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.barrier")
    def test_save_global_torch_dist_skips_layer_wise_files(
        self,
        mock_barrier,
        mock_get_dist_rank,
        mock_dist_init,
        mock_print_rank_0,
        mock_strategy_cls,
        mock_ensure_dir,
        mock_save_dataloader,
        mock_get_rank_safe,
        mock_empty_queue,
        mock_ft,
        mock_get_pg_collection,
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
        """save_checkpoint must NOT write layer_wise_optimizer_*.pt for torch_dist format.

        When ckpt_format is 'torch_dist' the optimizer state is already embedded in the
        distributed checkpoint via optimizer.sharded_state_dict().  Writing separate per-rank
        files is wasteful and they are never loaded on resume.
        """
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        mock_layer_wise_optim = Mock(spec=LayerWiseDistributedOptimizer)
        mock_layer_wise_optim.__class__ = LayerWiseDistributedOptimizer
        mock_layer_wise_optim.is_stub_optimizer = False

        mock_dist_init.return_value = True
        mock_get_dist_rank.return_value = 0
        mock_get_rank_safe.return_value = 0
        mock_empty_queue.return_value = True
        mock_unwrap.return_value = save_checkpoint_fixtures["mock_model"]
        mock_get_rng.return_value = Mock()
        mock_rerun.return_value.state_dict.return_value = {}
        mock_gen_state.return_value = {"model": {"p": "v"}}
        mock_is_last_rank.return_value = False
        mock_strategy_cls.return_value = Mock()
        mock_dist_ckpt.save.return_value = None
        save_checkpoint_fixtures["mock_state"].wandb_logger = Mock()
        save_checkpoint_fixtures["mock_state"].cfg.checkpoint.most_recent_k = -1

        mock_pg_collection = Mock()
        mock_pg_collection.expt_dp.rank.return_value = 0
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_pg_collection.dp.rank.return_value = 0
        mock_get_pg_collection.return_value = mock_pg_collection

        # ckpt_format is "torch_dist" (set in save_checkpoint_fixtures)
        assert save_checkpoint_fixtures["mock_cfg"].checkpoint.ckpt_format == "torch_dist"

        save_checkpoint(
            save_checkpoint_fixtures["mock_state"],
            save_checkpoint_fixtures["mock_model"],
            mock_layer_wise_optim,
            save_checkpoint_fixtures["mock_scheduler"],
            1000000,
            checkpointing_context={},
            non_persistent_ckpt=False,
        )

        # For torch_dist format, save_state_dict_to_file must NOT be called.
        mock_layer_wise_optim.save_state_dict_to_file.assert_not_called()

    # -----------------------------------------------------------------------
    # Load-side tests
    # -----------------------------------------------------------------------

    @patch("megatron.bridge.training.checkpointing.is_hf_checkpoint_dir", return_value=False)
    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.set_checkpoint_version")
    @patch("megatron.bridge.training.checkpointing.update_num_microbatches")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing._build_sharded_state_dict_metadata")
    @patch("os.path.exists")
    @patch("torch.distributed.is_initialized")
    def test_load_local_checkpoint_calls_load_state_dict_from_file(
        self,
        mock_dist_init,
        mock_exists_os,
        mock_build_metadata,
        mock_dist_ckpt,
        mock_get_rng_state,
        mock_generate_state_dict,
        mock_rerun_machine,
        mock_get_pg_collection,
        mock_print_rank_0,
        mock_update_microbatches,
        mock_set_version,
        mock_exists,
        mock_unwrap,
        mock_read_config,
        mock_read_state,
        mock_load_base,
        mock_is_hf_checkpoint_dir,
        load_checkpoint_fixtures,
    ):
        """load_checkpoint must call load_state_dict_from_file for LOCAL checkpoints.

        For local (non-persistent) checkpoints the LayerWiseDistributedOptimizer state is
        saved to a separate per-rank file in the local checkpoint base directory.
        The load path must read that file back from local_checkpoint_manager.local_ckpt_dir.
        """
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        mock_layer_wise_optim = Mock(spec=LayerWiseDistributedOptimizer)
        mock_layer_wise_optim.__class__ = LayerWiseDistributedOptimizer
        mock_layer_wise_optim.is_stub_optimizer = False

        mock_dist_init.return_value = False
        mock_exists.return_value = True
        mock_exists_os.return_value = True
        mock_unwrap.return_value = load_checkpoint_fixtures["mock_model"]
        mock_generate_state_dict.return_value = {"test": "state"}
        mock_get_rng_state.return_value = Mock()
        mock_build_metadata.return_value = {}
        mock_dist_ckpt.load_content_metadata.return_value = {}
        mock_dist_ckpt.load.return_value = {}
        mock_rerun_machine.return_value.state_dict.return_value = {}

        mock_pg_collection = Mock()
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_pg_collection.dp.rank.return_value = 2  # non-zero to verify path construction
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_get_pg_collection.return_value = mock_pg_collection

        mock_train_state = Mock()
        mock_train_state.step = 500
        mock_train_state.floating_point_operations_so_far = 0
        mock_read_state.return_value = mock_train_state

        mock_read_config.return_value = {
            "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
            "checkpoint": {"save_rng": False, "save_optim": True, "fully_parallel_save": False},
        }

        mock_state_dict = {
            "checkpoint_version": 3.0,
            "model": {"param": "value"},
            "opt_param_scheduler": {"max_lr": 0.001},
        }
        # Return CheckpointType.LOCAL to trigger the LayerWise load path
        mock_load_base.return_value = (mock_state_dict, "/ckpts/local_ckpt_id", False, CheckpointType.LOCAL)

        local_ckpt_dir = "/ckpts/local_nonpersistent"
        mock_local_ckpt_manager = Mock()
        mock_local_ckpt_manager.local_ckpt_dir = local_ckpt_dir
        checkpointing_context = {"local_checkpoint_manager": mock_local_ckpt_manager}

        load_checkpoint_fixtures["mock_cfg"].checkpoint.load = "/ckpts"
        # For LOCAL checkpoints, run_config is built from cfg directly (not read_run_config).
        # Set save_rng=False so the test doesn't attempt to restore RNG state from the
        # minimal mock state_dict (which has no rng_state key and would hit sys.exit()).
        load_checkpoint_fixtures["mock_cfg"].checkpoint.save_rng = False
        # Ensure _is_megatron_mimo is False so the optimizer load path is not skipped.
        # Mock() auto-creates attributes, making hasattr(..., "megatron_mimo_parallelism_config") True.
        del load_checkpoint_fixtures["mock_cfg"].model.megatron_mimo_parallelism_config

        load_checkpoint(
            load_checkpoint_fixtures["mock_state"],
            load_checkpoint_fixtures["mock_model"],
            mock_layer_wise_optim,
            load_checkpoint_fixtures["mock_scheduler"],
            checkpointing_context=checkpointing_context,
            pg_collection=mock_pg_collection,
        )

        expected_path = f"{local_ckpt_dir}/layer_wise_optimizer_2.pt"
        mock_layer_wise_optim.load_state_dict_from_file.assert_called_once_with(expected_path)

    @patch("megatron.bridge.training.checkpointing.is_hf_checkpoint_dir", return_value=False)
    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.read_train_state")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("megatron.bridge.training.checkpointing.unwrap_model")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.set_checkpoint_version")
    @patch("megatron.bridge.training.checkpointing.update_num_microbatches")
    @patch("megatron.bridge.training.checkpointing.print_rank_0")
    @patch("megatron.bridge.training.checkpointing.get_pg_collection")
    @patch("megatron.bridge.training.checkpointing.get_rerun_state_machine")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.get_rng_state")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing._build_sharded_state_dict_metadata")
    @patch("os.path.exists")
    @patch("torch.distributed.is_initialized")
    def test_load_global_checkpoint_uses_standard_load_state_dict(
        self,
        mock_dist_init,
        mock_exists_os,
        mock_build_metadata,
        mock_dist_ckpt,
        mock_get_rng_state,
        mock_generate_state_dict,
        mock_rerun_machine,
        mock_get_pg_collection,
        mock_print_rank_0,
        mock_update_microbatches,
        mock_set_version,
        mock_exists,
        mock_unwrap,
        mock_read_config,
        mock_read_state,
        mock_load_base,
        mock_is_hf_checkpoint_dir,
        load_checkpoint_fixtures,
    ):
        """For GLOBAL checkpoints, optimizer state is loaded via standard load_state_dict.

        Even with a LayerWiseDistributedOptimizer, global (torch_dist) checkpoints embed
        optimizer state in the distributed checkpoint.  load_state_dict_from_file must NOT
        be called; the regular load_state_dict path is used instead.
        """
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        mock_layer_wise_optim = Mock(spec=LayerWiseDistributedOptimizer)
        mock_layer_wise_optim.__class__ = LayerWiseDistributedOptimizer
        mock_layer_wise_optim.is_stub_optimizer = False

        mock_dist_init.return_value = False
        mock_exists.return_value = True
        mock_exists_os.return_value = True
        mock_unwrap.return_value = load_checkpoint_fixtures["mock_model"]
        mock_generate_state_dict.return_value = {"test": "state"}
        mock_get_rng_state.return_value = Mock()
        mock_build_metadata.return_value = {}
        mock_dist_ckpt.load_content_metadata.return_value = {}
        mock_dist_ckpt.load.return_value = {}
        mock_rerun_machine.return_value.state_dict.return_value = {}

        mock_pg_collection = Mock()
        mock_pg_collection.tp.rank.return_value = 0
        mock_pg_collection.tp.size.return_value = 1
        mock_pg_collection.pp.rank.return_value = 0
        mock_pg_collection.pp.size.return_value = 1
        mock_pg_collection.dp.rank.return_value = 0
        mock_pg_collection.dp_cp.rank.return_value = 0
        mock_get_pg_collection.return_value = mock_pg_collection

        mock_train_state = Mock()
        mock_train_state.step = 1000
        mock_train_state.floating_point_operations_so_far = 0
        mock_read_state.return_value = mock_train_state

        mock_read_config.return_value = {
            "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
            "checkpoint": {"save_rng": False, "save_optim": True, "fully_parallel_save": False},
        }

        mock_state_dict = {
            "checkpoint_version": 3.0,
            "model": {"param": "value"},
            "optimizer": {"param_groups": []},
            "opt_param_scheduler": {"max_lr": 0.001},
        }
        # GLOBAL checkpoint - optimizer is in state_dict["optimizer"]
        mock_load_base.return_value = (mock_state_dict, "/ckpts/iter_0001000", False, CheckpointType.GLOBAL)

        load_checkpoint_fixtures["mock_cfg"].checkpoint.load = "/ckpts"
        # Ensure _is_megatron_mimo is False so the optimizer load path is not skipped.
        del load_checkpoint_fixtures["mock_cfg"].model.megatron_mimo_parallelism_config

        load_checkpoint(
            load_checkpoint_fixtures["mock_state"],
            load_checkpoint_fixtures["mock_model"],
            mock_layer_wise_optim,
            load_checkpoint_fixtures["mock_scheduler"],
            pg_collection=mock_pg_collection,
        )

        # Standard load_state_dict must be called; per-rank file loader must NOT be called.
        mock_layer_wise_optim.load_state_dict.assert_called_once_with(mock_state_dict["optimizer"])
        mock_layer_wise_optim.load_state_dict_from_file.assert_not_called()


class TestMaybeLoadDataloaderState:
    """Tests for restoring Energon dataloader stream-position state on resume."""

    @pytest.fixture(autouse=True)
    def _dist_initialized(self):
        """megatron.core.utils.get_pg_rank returns 0 for every group unless torch.distributed is
        initialized. Stub is_initialized=True so the real get_pg_rank resolves each mock group's rank
        (while still mapping a None group to 0), and stub get_rank=0 so print_rank_0 -> safe_get_rank
        does not hit the uninitialized default process group."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            yield

    @staticmethod
    def _pg(cp: int | None = 0, pp: int = 0, tp: int = 0, dp: int = 0):
        """Mock a ProcessGroupCollection with the given per-dimension ranks. Only cp may be None —
        modeling a collection configured without context parallelism (tp/pp/dp are always
        populated); the real get_pg_rank then reads the absent cp group as rank 0."""
        pg = Mock()
        if cp is None:
            pg.cp = None
        else:
            pg.cp.rank.return_value = cp
        pg.pp.rank.return_value = pp
        pg.tp.rank.return_value = tp
        pg.dp.rank.return_value = dp
        return pg

    def test_noop_when_no_path(self):
        """No load path => nothing restored."""
        train_iterator = Mock()
        maybe_load_dataloader_state(train_iterator, 10, None, pg_collection=self._pg())
        train_iterator.iterable.restore_state.assert_not_called()

    def test_noop_when_iterator_is_none(self):
        """No iterator => no-op (must not raise)."""
        maybe_load_dataloader_state(None, 10, "/some/path", pg_collection=self._pg())

    def test_noop_when_train_iterator_is_list(self, tmp_path):
        """VPP yields a list of iterators; the single-iterator case is the only one handled."""
        maybe_load_dataloader_state([Mock()], 10, str(tmp_path), pg_collection=self._pg())

    def test_noop_when_iterable_lacks_restore_state(self, tmp_path):
        """Non-Energon iterables (no restore_state) are skipped without error."""
        train_iterator = Mock(spec=["iterable"])
        train_iterator.iterable = Mock(spec=[])  # hasattr(..., "restore_state") is False
        maybe_load_dataloader_state(train_iterator, 10, str(tmp_path), pg_collection=self._pg())

    def test_missing_dir_warns_and_skips(self, tmp_path):
        """A missing state directory is tolerated so pre-feature checkpoints still resume."""
        train_iterator = Mock()
        missing = tmp_path / "energon"  # never created
        maybe_load_dataloader_state(train_iterator, 10, str(missing), pg_collection=self._pg())
        train_iterator.iterable.restore_state.assert_not_called()

    def test_existing_dir_missing_file_raises(self, tmp_path):
        """If the state dir exists but this rank's file does not, fail loudly (likely a DP-size change)."""
        train_iterator = Mock()
        # tmp_path exists (the energon root) but contains no per-rank file for this iteration.
        with pytest.raises(RuntimeError, match="data-parallel size"):
            maybe_load_dataloader_state(train_iterator, 10, str(tmp_path), pg_collection=self._pg())

    @patch("megatron.bridge.training.checkpointing.torch.load")
    def test_restores_on_every_cp_tp_pp_rank(self, mock_load, tmp_path):
        """Unlike save, restore must run on every cp/tp/pp because
        each rank pulls from its own data iterator. Keyed by the pure DP rank."""
        train_iterator = Mock()
        iter_dir = get_checkpoint_name(str(tmp_path), 10)
        os.makedirs(iter_dir, exist_ok=True)
        Path(iter_dir, "train_dataloader_dprank000.pt").touch()
        mock_load.return_value = {"dataloader_state_dict": {"dummy_energon_state": "xyz"}}

        # cp=1, pp=1, tp=1 would have been skipped by save-style (cp0,pp0,tp0) gating; restore must not skip.
        maybe_load_dataloader_state(
            train_iterator,
            10,
            str(tmp_path),
            pg_collection=self._pg(cp=1, pp=1, tp=1, dp=0),
        )

        train_iterator.iterable.restore_state.assert_called_once_with({"dummy_energon_state": "xyz"})

    @patch("megatron.bridge.training.checkpointing.torch.load")
    def test_restores_from_file(self, mock_load, tmp_path):
        """Happy path: the per-DP-rank file is loaded and restore_state is called with the saved dict."""
        train_iterator = Mock()
        iter_dir = get_checkpoint_name(str(tmp_path), 10)
        os.makedirs(iter_dir, exist_ok=True)
        # dp_rank defaults to 0 => dprank000.
        Path(iter_dir, "train_dataloader_dprank000.pt").touch()
        mock_load.return_value = {"dataloader_state_dict": {"dummy_energon_state": "xyz"}}

        maybe_load_dataloader_state(train_iterator, 10, str(tmp_path), pg_collection=self._pg())

        train_iterator.iterable.restore_state.assert_called_once_with({"dummy_energon_state": "xyz"})


class TestMaybeSaveDataloaderState:
    """Tests for gating which rank writes the Energon dataloader stream-position state."""

    @pytest.fixture(autouse=True)
    def _dist_initialized(self):
        """megatron.core.utils.get_pg_rank returns 0 for every group unless torch.distributed is
        initialized. Stub is_initialized=True so the real get_pg_rank resolves each mock group's rank
        (while still mapping a None group to 0), and stub get_rank=0 so print_rank_0 -> safe_get_rank
        does not hit the uninitialized default process group."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            yield

    @staticmethod
    def _pg(cp: int | None = 0, pp: int = 0, tp: int = 0, dp: int = 0):
        """Mock a ProcessGroupCollection with the given per-dimension ranks. Only cp may be None —
        modeling a collection configured without context parallelism (tp/pp/dp are always
        populated); the real get_pg_rank then reads the absent cp group as rank 0."""
        pg = Mock()
        if cp is None:
            pg.cp = None
        else:
            pg.cp.rank.return_value = cp
        pg.pp.rank.return_value = pp
        pg.tp.rank.return_value = tp
        pg.dp.rank.return_value = dp
        return pg

    @staticmethod
    def _iterator():
        """Mock train iterator whose iterable.save_state() returns a sentinel state dict."""
        train_iterator = Mock()
        train_iterator.iterable.save_state.return_value = {"dummy_energon_state": "xyz"}
        return train_iterator

    def test_noop_when_no_path(self):
        """No save path => nothing is saved."""
        train_iterator = self._iterator()
        maybe_save_dataloader_state(Mock(), train_iterator, 10, None, pg_collection=self._pg())
        train_iterator.iterable.save_state.assert_not_called()

    @pytest.mark.parametrize("ranks", [{"cp": 1}, {"tp": 1}, {"pp": 1}], ids=["cp", "tp", "pp"])
    def test_skips_nonzero_cp_tp_pp_rank(self, ranks):
        """Tensor/pipeline/context-parallel ranks all replicate the per-DP-rank state, so only the
        tp0/pp0/cp0 leader writes preventing racing on the same train_dataloader_dprank{dp}.pt file."""
        train_iterator = self._iterator()
        maybe_save_dataloader_state(Mock(), train_iterator, 10, "/some/path", pg_collection=self._pg(**ranks))
        train_iterator.iterable.save_state.assert_not_called()

    @patch("megatron.bridge.training.checkpointing.torch.save")
    def test_writes_on_zero_cp_tp_pp_rank(self, mock_save, tmp_path):
        """The single tp0/pp0/cp0 writer saves the state under the per-DP-rank filename."""
        train_iterator = self._iterator()
        # patch barrier: no real process group exists in a unit test.
        with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
            maybe_save_dataloader_state(Mock(), train_iterator, 10, str(tmp_path), pg_collection=self._pg(dp=2))

        train_iterator.iterable.save_state.assert_called_once_with()
        saved_dict, saved_path = mock_save.call_args[0]
        assert saved_dict == {"dataloader_state_dict": {"dummy_energon_state": "xyz"}}
        assert saved_path.endswith("train_dataloader_dprank002.pt")

    @patch("megatron.bridge.training.checkpointing.torch.save")
    def test_writes_when_cp_group_is_none(self, mock_save, tmp_path):
        """A run without context parallelism leaves cp=None. get_pg_rank reads a
        None group as rank 0, so the (pp0, tp0, cp0) leader still writes instead of raising on
        None.rank()."""
        train_iterator = self._iterator()
        with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
            maybe_save_dataloader_state(
                Mock(), train_iterator, 10, str(tmp_path), pg_collection=self._pg(cp=None, dp=2)
            )

        train_iterator.iterable.save_state.assert_called_once_with()
        _, saved_path = mock_save.call_args[0]
        assert saved_path.endswith("train_dataloader_dprank002.pt")
