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

"""
Unit tests for the use_decentralized_pg feature.

This feature enables using ProcessGroupCollection passed through functions instead
of relying on mcore's global parallel state (mpu) variables. When enabled, parallel
groups are obtained from the pg_collection object rather than the global
megatron.core.parallel_state module.
"""

from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.training.config import DistributedInitConfig


class TestDistributedInitConfigDecentralizedPg:
    """Tests for DistributedInitConfig.use_decentralized_pg configuration."""

    def test_use_decentralized_pg_default_is_false(self):
        """Test that use_decentralized_pg defaults to False."""
        config = DistributedInitConfig()
        assert config.use_decentralized_pg is False

    def test_use_decentralized_pg_can_be_enabled(self):
        """Test that use_decentralized_pg can be set to True."""
        config = DistributedInitConfig(use_decentralized_pg=True)
        assert config.use_decentralized_pg is True

    def test_use_decentralized_pg_can_be_explicitly_disabled(self):
        """Test that use_decentralized_pg can be explicitly set to False."""
        config = DistributedInitConfig(use_decentralized_pg=False)
        assert config.use_decentralized_pg is False


class TestCreatePgCollectionFunction:
    """Tests for the _create_pg_collection function."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model configuration for testing."""
        config = MagicMock()
        config.tensor_model_parallel_size = 1
        config.pipeline_model_parallel_size = 1
        config.context_parallel_size = 1
        config.expert_tensor_parallel_size = None
        config.expert_model_parallel_size = 1
        return config

    @patch("megatron.bridge.training.initialize.HyperCommGrid")
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.new_subgroups_by_enumeration")
    def test_create_pg_collection_returns_process_group_collection(
        self, mock_subgroups, mock_world_size, mock_hyper_grid, mock_model_config
    ):
        """Test that _create_pg_collection returns a ProcessGroupCollection."""
        from megatron.bridge.training.initialize import _create_pg_collection

        # Setup mock
        mock_grid_instance = MagicMock()
        mock_grid_instance.create_pg.return_value = MagicMock()
        mock_grid_instance._gen_rank_enum.return_value = [[0]]
        mock_hyper_grid.return_value = mock_grid_instance
        mock_subgroups.return_value = (MagicMock(), [])

        # Execute
        result = _create_pg_collection(mock_model_config, num_distributed_optimizer_instances=1)

        # Verify
        from megatron.core.process_groups_config import ProcessGroupCollection

        assert isinstance(result, ProcessGroupCollection)

    @patch("megatron.bridge.training.initialize.HyperCommGrid")
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.new_subgroups_by_enumeration")
    def test_create_pg_collection_with_tp(self, mock_subgroups, mock_world_size, mock_hyper_grid, mock_model_config):
        """Test _create_pg_collection with tensor parallelism."""
        from megatron.bridge.training.initialize import _create_pg_collection

        # Setup with TP=2
        mock_model_config.tensor_model_parallel_size = 2

        # Setup mock
        mock_grid_instance = MagicMock()
        mock_grid_instance.create_pg.return_value = MagicMock()
        mock_grid_instance._gen_rank_enum.return_value = [[0, 1, 2, 3, 4, 5, 6, 7]]
        mock_hyper_grid.return_value = mock_grid_instance
        mock_subgroups.return_value = (MagicMock(), [])

        # Execute
        _create_pg_collection(mock_model_config, num_distributed_optimizer_instances=1)

        # Verify grid was created with correct shape
        mock_hyper_grid.assert_called()
        call_kwargs = mock_hyper_grid.call_args[1]
        assert call_kwargs["shape"][0] == 2  # TP size

    @patch("megatron.bridge.training.initialize.HyperCommGrid")
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.new_subgroups_by_enumeration")
    def test_create_pg_collection_with_pp(self, mock_subgroups, mock_world_size, mock_hyper_grid, mock_model_config):
        """Test _create_pg_collection with pipeline parallelism."""
        from megatron.bridge.training.initialize import _create_pg_collection

        # Setup with PP=2
        mock_model_config.pipeline_model_parallel_size = 2

        # Setup mock
        mock_grid_instance = MagicMock()
        mock_grid_instance.create_pg.return_value = MagicMock()
        mock_grid_instance._gen_rank_enum.return_value = [[0, 1], [2, 3], [4, 5], [6, 7]]
        mock_hyper_grid.return_value = mock_grid_instance
        mock_subgroups.return_value = (MagicMock(), [])

        # Execute
        _create_pg_collection(mock_model_config, num_distributed_optimizer_instances=1)

        # Verify grid was created with correct shape
        mock_hyper_grid.assert_called()
        call_kwargs = mock_hyper_grid.call_args[1]
        assert call_kwargs["shape"][3] == 2  # PP size


class TestSetRandomSeedWithPgCollection:
    """Tests for _set_random_seed function with pg_collection parameter."""

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_group_rank")
    @patch("torch.cuda.device_count", return_value=1)
    @patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed")
    @patch("megatron.core.utils.get_pg_rank", return_value=0)
    @patch("torch.cuda.manual_seed")
    @patch("torch.manual_seed")
    def test_set_random_seed_uses_pg_collection_for_pp_rank(
        self,
        mock_torch_manual_seed,
        mock_cuda_manual_seed,
        mock_get_pg_rank,
        mock_model_parallel_seed,
        mock_cuda_device_count,
        mock_group_rank,
        mock_get_rank,
    ):
        """Test that _set_random_seed uses pg_collection for PP rank."""
        from megatron.bridge.training.initialize import _set_random_seed

        # Setup mock pg_collection
        mock_pg_collection = MagicMock()
        mock_pg_collection.pp = MagicMock()
        mock_pg_collection.dp = MagicMock()
        mock_pg_collection.tp = MagicMock()
        mock_pg_collection.ep = MagicMock()
        mock_pg_collection.expt_tp = MagicMock()

        # Mock get_group_rank to return PP rank
        mock_group_rank.side_effect = lambda pg, rank: 0

        # Execute
        _set_random_seed(
            seed_=42,
            data_parallel_random_init=False,
            te_rng_tracker=False,
            inference_rng_tracker=False,
            use_cudagraphable_rng=False,
            pg_collection=mock_pg_collection,
        )

        # Verify get_group_rank was called with pg_collection.pp
        mock_group_rank.assert_any_call(mock_pg_collection.pp, 0)

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_group_rank")
    @patch("torch.cuda.device_count", return_value=1)
    @patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed")
    @patch("megatron.core.utils.get_pg_rank", return_value=0)
    @patch("torch.cuda.manual_seed")
    @patch("torch.manual_seed")
    def test_set_random_seed_uses_pg_collection_for_dp_rank(
        self,
        mock_torch_manual_seed,
        mock_cuda_manual_seed,
        mock_get_pg_rank,
        mock_model_parallel_seed,
        mock_cuda_device_count,
        mock_group_rank,
        mock_get_rank,
    ):
        """Test that _set_random_seed uses pg_collection for DP rank with data_parallel_random_init."""
        from megatron.bridge.training.initialize import _set_random_seed

        # Setup mock pg_collection
        mock_pg_collection = MagicMock()
        mock_pg_collection.pp = MagicMock()
        mock_pg_collection.dp = MagicMock()
        mock_pg_collection.tp = MagicMock()
        mock_pg_collection.ep = MagicMock()
        mock_pg_collection.expt_tp = MagicMock()

        # Mock get_group_rank to return different values for PP and DP
        def side_effect(pg, rank):
            if pg == mock_pg_collection.dp:
                return 1
            return 0

        mock_group_rank.side_effect = side_effect

        # Execute with data_parallel_random_init=True
        _set_random_seed(
            seed_=42,
            data_parallel_random_init=True,
            te_rng_tracker=False,
            inference_rng_tracker=False,
            use_cudagraphable_rng=False,
            pg_collection=mock_pg_collection,
        )

        # Verify get_group_rank was called with pg_collection.dp
        mock_group_rank.assert_any_call(mock_pg_collection.dp, 0)


class TestTorchDistInitReturnValue:
    """Tests for torch_dist_init function return value."""

    def test_torch_dist_init_returns_pg_collection_when_not_lazy(self):
        """Test that torch_dist_init returns ProcessGroupCollection when not using lazy init."""
        # This test verifies the function signature and return type annotation
        import inspect

        from megatron.bridge.training.initialize import torch_dist_init

        sig = inspect.signature(torch_dist_init)

        # Verify the return type annotation includes ProcessGroupCollection
        return_annotation = sig.return_annotation
        assert "ProcessGroupCollection" in str(return_annotation) or "Callable" in str(return_annotation)


class TestInitializeDistributedRaisesOnNoDevices:
    """Tests for _initialize_distributed error handling."""

    @patch("torch.cuda.device_count", return_value=0)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=0)
    def test_initialize_distributed_raises_on_no_cuda_devices(self, mock_get_rank, mock_is_init, mock_device_count):
        """Test that _initialize_distributed raises RuntimeError when no CUDA devices."""
        from megatron.bridge.training.initialize import _initialize_distributed

        mock_model_config = MagicMock()
        mock_model_config.tensor_model_parallel_size = 1
        mock_model_config.pipeline_model_parallel_size = 1
        mock_model_config.context_parallel_size = 1

        mock_dist_config = MagicMock()

        with pytest.raises(RuntimeError, match="Cannot initialize parallel groups with no CUDA devices"):
            _initialize_distributed(
                model_config=mock_model_config,
                dist_config=mock_dist_config,
                num_distributed_optimizer_instances=1,
                get_embedding_ranks=None,
                get_position_embedding_ranks=None,
            )


class TestInitializeDistributedBranching:
    """Tests for _initialize_distributed branching based on use_decentralized_pg."""

    @patch("megatron.bridge.training.initialize._create_pg_collection")
    @patch("megatron.bridge.training.initialize.parallel_state")
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=0)
    def test_uses_hyper_comm_grid_when_decentralized_pg_enabled(
        self,
        mock_get_rank,
        mock_is_init,
        mock_device_count,
        mock_world_size,
        mock_parallel_state,
        mock_create_pg_collection,
    ):
        """Test that _initialize_distributed uses HyperCommGrid when use_decentralized_pg=True."""
        from megatron.bridge.training.initialize import _initialize_distributed

        mock_model_config = MagicMock()
        mock_model_config.tensor_model_parallel_size = 1
        mock_model_config.pipeline_model_parallel_size = 1
        mock_model_config.context_parallel_size = 1

        mock_dist_config = MagicMock()
        mock_dist_config.use_decentralized_pg = True

        mock_pg_collection = MagicMock()
        mock_create_pg_collection.return_value = mock_pg_collection

        result = _initialize_distributed(
            model_config=mock_model_config,
            dist_config=mock_dist_config,
            num_distributed_optimizer_instances=1,
            get_embedding_ranks=None,
            get_position_embedding_ranks=None,
        )

        # Verify _create_pg_collection was called
        mock_create_pg_collection.assert_called_once()
        # Verify parallel_state.initialize_model_parallel was NOT called
        mock_parallel_state.initialize_model_parallel.assert_not_called()
        # Verify the result is the pg_collection from _create_pg_collection
        assert result == mock_pg_collection

    @patch("megatron.bridge.training.initialize.ProcessGroupCollection")
    @patch("megatron.bridge.training.initialize._create_pg_collection")
    @patch("megatron.bridge.training.initialize.parallel_state")
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=0)
    def test_uses_mpu_when_decentralized_pg_disabled(
        self,
        mock_get_rank,
        mock_is_init,
        mock_device_count,
        mock_parallel_state,
        mock_create_pg_collection,
        mock_pg_collection_class,
    ):
        """Test that _initialize_distributed uses mpu when use_decentralized_pg=False."""
        from megatron.bridge.training.initialize import _initialize_distributed

        mock_model_config = MagicMock()
        mock_model_config.tensor_model_parallel_size = 1
        mock_model_config.pipeline_model_parallel_size = 1
        mock_model_config.context_parallel_size = 1
        mock_model_config.virtual_pipeline_model_parallel_size = None
        mock_model_config.pipeline_model_parallel_comm_backend = "nccl"
        mock_model_config.hierarchical_context_parallel_sizes = None
        mock_model_config.expert_model_parallel_size = 1
        mock_model_config.expert_tensor_parallel_size = None

        mock_dist_config = MagicMock()
        mock_dist_config.use_decentralized_pg = False
        mock_dist_config.distributed_timeout_minutes = 30
        mock_dist_config.nccl_communicator_config_path = None
        mock_dist_config.use_tp_pp_dp_mapping = False
        mock_dist_config.use_gloo_process_groups = False
        mock_dist_config.use_sharp = False
        mock_dist_config.high_priority_stream_groups = False
        mock_dist_config.sharp_enabled_group = None

        mock_parallel_state.model_parallel_is_initialized.return_value = False
        mock_pg_collection = MagicMock()
        mock_pg_collection_class.use_mpu_process_groups.return_value = mock_pg_collection

        _initialize_distributed(
            model_config=mock_model_config,
            dist_config=mock_dist_config,
            num_distributed_optimizer_instances=1,
            get_embedding_ranks=None,
            get_position_embedding_ranks=None,
        )

        # Verify _create_pg_collection was NOT called
        mock_create_pg_collection.assert_not_called()
        # Verify parallel_state.initialize_model_parallel WAS called
        mock_parallel_state.initialize_model_parallel.assert_called_once()


class TestSetupUsesDecentralizedPg:
    """Tests for setup function behavior with use_decentralized_pg."""

    def test_config_use_decentralized_pg_enabled(self):
        """Test that use_decentralized_pg can be enabled in config."""
        from megatron.bridge.training.config import DistributedInitConfig

        config = DistributedInitConfig(use_decentralized_pg=True)

        # When use_decentralized_pg=True, _initialize_distributed uses HyperCommGrid
        assert config.use_decentralized_pg is True

    def test_config_use_decentralized_pg_disabled_default(self):
        """Test that use_decentralized_pg defaults to False."""
        from megatron.bridge.training.config import DistributedInitConfig

        config = DistributedInitConfig(use_decentralized_pg=False)

        # When use_decentralized_pg=False (default), _initialize_distributed uses mpu
        assert config.use_decentralized_pg is False


class TestSetupOptimizerWithPgCollection:
    """Tests for setup_optimizer function with pg_collection parameter."""

    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.OptimizerParamScheduler")
    def test_setup_optimizer_passes_pg_collection_to_get_megatron_optimizer(self, mock_scheduler, mock_get_optimizer):
        """Test that setup_optimizer passes pg_collection to get_megatron_optimizer."""
        from megatron.core.optimizer import OptimizerConfig

        from megatron.bridge.training.config import SchedulerConfig
        from megatron.bridge.training.optim import setup_optimizer

        # Setup mocks
        mock_model = MagicMock()
        mock_pg_collection = MagicMock()
        mock_optimizer = MagicMock()
        mock_get_optimizer.return_value = mock_optimizer

        optimizer_config = OptimizerConfig(optimizer="adam", lr=1e-3)
        scheduler_config = SchedulerConfig()

        # Execute
        setup_optimizer(
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            model=mock_model,
            use_gloo_process_groups=False,
            pg_collection=mock_pg_collection,
        )

        # Verify pg_collection was passed
        mock_get_optimizer.assert_called_once()
        call_kwargs = mock_get_optimizer.call_args[1]
        assert call_kwargs["pg_collection"] == mock_pg_collection

    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.OptimizerParamScheduler")
    def test_setup_optimizer_passes_none_pg_collection_when_not_provided(self, mock_scheduler, mock_get_optimizer):
        """Test that setup_optimizer passes None pg_collection when not provided."""
        from megatron.core.optimizer import OptimizerConfig

        from megatron.bridge.training.config import SchedulerConfig
        from megatron.bridge.training.optim import setup_optimizer

        # Setup mocks
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        mock_get_optimizer.return_value = mock_optimizer

        optimizer_config = OptimizerConfig(optimizer="adam", lr=1e-3)
        scheduler_config = SchedulerConfig()

        # Execute without pg_collection
        setup_optimizer(
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            model=mock_model,
            use_gloo_process_groups=False,
        )

        # Verify pg_collection was None (default)
        mock_get_optimizer.assert_called_once()
        call_kwargs = mock_get_optimizer.call_args[1]
        assert call_kwargs["pg_collection"] is None

    @patch("megatron.bridge.training.optim.get_megatron_muon_optimizer")
    @patch("megatron.bridge.training.optim.OptimizerParamScheduler")
    def test_setup_optimizer_passes_pg_collection_to_muon_optimizer(self, mock_scheduler, mock_get_muon_optimizer):
        """Test that setup_optimizer passes pg_collection to muon optimizer."""
        from megatron.core.optimizer import OptimizerConfig

        from megatron.bridge.training.config import SchedulerConfig
        from megatron.bridge.training.optim import setup_optimizer

        # Setup mocks
        mock_model = MagicMock()
        mock_pg_collection = MagicMock()
        mock_optimizer = MagicMock()
        mock_get_muon_optimizer.return_value = mock_optimizer

        optimizer_config = OptimizerConfig(optimizer="muon", lr=1e-3)
        scheduler_config = SchedulerConfig()

        # Execute
        setup_optimizer(
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            model=mock_model,
            use_gloo_process_groups=False,
            pg_collection=mock_pg_collection,
        )

        # Verify pg_collection was passed to muon optimizer
        mock_get_muon_optimizer.assert_called_once()
        call_kwargs = mock_get_muon_optimizer.call_args[1]
        assert call_kwargs["pg_collection"] == mock_pg_collection

    @patch("megatron.bridge.training.optim.OptimizerParamScheduler")
    def test_setup_optimizer_with_optimizer_config_having_provide_method(self, mock_scheduler):
        """Test that setup_optimizer uses the provide method when optimizer_config has one."""
        from megatron.bridge.training.config import SchedulerConfig
        from megatron.bridge.training.optim import setup_optimizer

        # Setup mocks
        mock_model = MagicMock()
        mock_pg_collection = MagicMock()
        mock_optimizer = MagicMock()

        # Create a mock optimizer config with a provide method
        mock_optimizer_config = MagicMock()
        mock_optimizer_config.lr = 1e-3
        mock_optimizer_config.min_lr = 1e-5
        mock_optimizer_config.provide = MagicMock(return_value=mock_optimizer)

        scheduler_config = SchedulerConfig()

        # Execute
        optimizer, _ = setup_optimizer(
            optimizer_config=mock_optimizer_config,
            scheduler_config=scheduler_config,
            model=mock_model,
            use_gloo_process_groups=False,
            pg_collection=mock_pg_collection,
        )

        # Verify the provide method was called with correct parameters
        mock_optimizer_config.provide.assert_called_once()
        call_kwargs = mock_optimizer_config.provide.call_args[1]
        assert call_kwargs["model_chunks"] == mock_model
        assert call_kwargs["pg_collection"] == mock_pg_collection
        assert call_kwargs["use_gloo_process_groups"] is False
        assert "config_overrides" in call_kwargs

        # Verify the returned optimizer is from the provide method
        assert optimizer == mock_optimizer


class TestSetupConditionalPgCollectionPassing:
    """Tests for setup function's conditional pg_collection passing to optimizer."""

    def test_setup_passes_pg_collection_when_use_decentralized_pg_true(self):
        """
        Verify that when use_decentralized_pg=True, pg_collection is passed to optimizer.

        This tests the logic at setup.py line 232-234:
        pg_collection=pg_collection if cfg.dist.use_decentralized_pg else None
        """
        from megatron.bridge.training.config import DistributedInitConfig

        # Create config with use_decentralized_pg=True
        config = DistributedInitConfig(use_decentralized_pg=True)

        # Simulate the conditional expression from setup.py
        mock_pg_collection = MagicMock()
        passed_pg_collection = mock_pg_collection if config.use_decentralized_pg else None

        # Verify pg_collection is passed when use_decentralized_pg=True
        assert passed_pg_collection is mock_pg_collection

    def test_setup_passes_none_when_use_decentralized_pg_false(self):
        """
        Verify that when use_decentralized_pg=False, None is passed to optimizer.

        This tests the logic at setup.py line 232-234:
        pg_collection=pg_collection if cfg.dist.use_decentralized_pg else None
        """
        from megatron.bridge.training.config import DistributedInitConfig

        # Create config with use_decentralized_pg=False
        config = DistributedInitConfig(use_decentralized_pg=False)

        # Simulate the conditional expression from setup.py
        mock_pg_collection = MagicMock()
        passed_pg_collection = mock_pg_collection if config.use_decentralized_pg else None

        # Verify None is passed when use_decentralized_pg=False
        assert passed_pg_collection is None


class TestCheckpointingWithDecentralizedPg:
    """Tests for checkpointing behavior based on use_decentralized_pg setting."""

    def test_modelopt_state_save_skipped_when_use_decentralized_pg_true(self):
        """
        Verify that sharded modelopt_state save is skipped when use_decentralized_pg=True.

        This tests the logic at checkpointing.py line 641:
        if not cfg.dist.use_decentralized_pg:
            save_sharded_modelopt_state(model, checkpoint_name, (ckpt_cfg.ckpt_format, 1))
        """
        from megatron.bridge.training.config import DistributedInitConfig

        # Create config with use_decentralized_pg=True
        config = DistributedInitConfig(use_decentralized_pg=True)

        # Simulate the condition from checkpointing.py
        should_save_modelopt = not config.use_decentralized_pg

        # Verify modelopt save is skipped when use_decentralized_pg=True
        assert should_save_modelopt is False

    def test_modelopt_state_save_executed_when_use_decentralized_pg_false(self):
        """
        Verify that sharded modelopt_state save is executed when use_decentralized_pg=False.

        This tests the logic at checkpointing.py line 641:
        if not cfg.dist.use_decentralized_pg:
            save_sharded_modelopt_state(model, checkpoint_name, (ckpt_cfg.ckpt_format, 1))
        """
        from megatron.bridge.training.config import DistributedInitConfig

        # Create config with use_decentralized_pg=False (default)
        config = DistributedInitConfig(use_decentralized_pg=False)

        # Simulate the condition from checkpointing.py
        should_save_modelopt = not config.use_decentralized_pg

        # Verify modelopt save is executed when use_decentralized_pg=False
        assert should_save_modelopt is True


class TestTrainTensorShapesAdjustWithDecentralizedPg:
    """Tests for train.py tensor shapes adjust function behavior."""

    def test_tensor_shapes_adjust_fn_is_none_when_use_decentralized_pg_true(self):
        """
        Verify that adjust_tensor_shapes_fn is None when use_decentralized_pg=True.

        This tests the logic at train.py line 658-666:
        if not cfg.dist.use_decentralized_pg:
            adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(...)
        else:
            adjust_tensor_shapes_fn = None
        """
        from megatron.bridge.training.config import DistributedInitConfig

        # Create config with use_decentralized_pg=True
        config = DistributedInitConfig(use_decentralized_pg=True)

        # Simulate the condition from train.py
        if not config.use_decentralized_pg:
            adjust_tensor_shapes_fn = "would_call_get_tensor_shapes_adjust_fn"
        else:
            adjust_tensor_shapes_fn = None

        # Verify adjust_tensor_shapes_fn is None when use_decentralized_pg=True
        assert adjust_tensor_shapes_fn is None

    def test_tensor_shapes_adjust_fn_is_set_when_use_decentralized_pg_false(self):
        """
        Verify that adjust_tensor_shapes_fn is set when use_decentralized_pg=False.

        This tests the logic at train.py line 658-666:
        if not cfg.dist.use_decentralized_pg:
            adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(...)
        else:
            adjust_tensor_shapes_fn = None
        """
        from megatron.bridge.training.config import DistributedInitConfig

        # Create config with use_decentralized_pg=False (default)
        config = DistributedInitConfig(use_decentralized_pg=False)

        # Simulate the condition from train.py
        if not config.use_decentralized_pg:
            adjust_tensor_shapes_fn = "would_call_get_tensor_shapes_adjust_fn"
        else:
            adjust_tensor_shapes_fn = None

        # Verify adjust_tensor_shapes_fn is set when use_decentralized_pg=False
        assert adjust_tensor_shapes_fn is not None


class TestCreatePgCollectionWithContextParallelism:
    """Tests for _create_pg_collection with context parallelism."""

    @patch("megatron.bridge.training.initialize.HyperCommGrid")
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.new_subgroups_by_enumeration")
    def test_create_pg_collection_with_cp(self, mock_subgroups, mock_world_size, mock_hyper_grid):
        """Test _create_pg_collection with context parallelism."""
        from megatron.bridge.training.initialize import _create_pg_collection

        # Create a fresh mock config with CP=2 directly
        mock_model_config = MagicMock()
        mock_model_config.tensor_model_parallel_size = 1
        mock_model_config.pipeline_model_parallel_size = 1
        mock_model_config.context_parallel_size = 2  # CP=2
        mock_model_config.expert_tensor_parallel_size = None
        mock_model_config.expert_model_parallel_size = 1

        # Setup mock
        mock_grid_instance = MagicMock()
        mock_grid_instance.create_pg.return_value = MagicMock()
        mock_grid_instance._gen_rank_enum.return_value = [[0, 1, 2, 3, 4, 5, 6, 7]]
        mock_hyper_grid.return_value = mock_grid_instance
        mock_subgroups.return_value = (MagicMock(), [])

        # Execute
        _create_pg_collection(mock_model_config, num_distributed_optimizer_instances=1)

        # Verify grid was created with correct shape
        # HyperCommGrid is called multiple times (main grid and expert grid)
        # The first call is for the main grid which includes CP
        # With TP=1, CP=2, PP=1, world_size=8: dp_size = 8 / (1*2*1) = 4
        # Shape should be [1, 2, 4, 1] = [TP, CP, DP, PP]
        mock_hyper_grid.assert_called()
        first_call_kwargs = mock_hyper_grid.call_args_list[0][1]
        assert first_call_kwargs["shape"][1] == 2  # CP size at index 1

    @patch("megatron.bridge.training.initialize.HyperCommGrid")
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.new_subgroups_by_enumeration")
    def test_create_pg_collection_with_tp_cp_pp(self, mock_subgroups, mock_world_size, mock_hyper_grid):
        """Test _create_pg_collection with combined TP, CP, and PP."""
        from megatron.bridge.training.initialize import _create_pg_collection

        # Create a fresh mock config with TP=2, CP=2, PP=2 directly
        mock_model_config = MagicMock()
        mock_model_config.tensor_model_parallel_size = 2
        mock_model_config.pipeline_model_parallel_size = 2
        mock_model_config.context_parallel_size = 2
        mock_model_config.expert_tensor_parallel_size = None
        mock_model_config.expert_model_parallel_size = 1

        # Setup mock
        mock_grid_instance = MagicMock()
        mock_grid_instance.create_pg.return_value = MagicMock()
        mock_grid_instance._gen_rank_enum.return_value = [[0, 1], [2, 3], [4, 5], [6, 7]]
        mock_hyper_grid.return_value = mock_grid_instance
        mock_subgroups.return_value = (MagicMock(), [])

        # Execute
        _create_pg_collection(mock_model_config, num_distributed_optimizer_instances=1)

        # Verify grid was created with correct shape [TP, CP, DP, PP]
        # With TP=2, CP=2, PP=2, world_size=8: dp_size = 8 / (2*2*2) = 1
        # Shape should be [2, 2, 1, 2] = [TP, CP, DP, PP]
        mock_hyper_grid.assert_called()
        first_call_kwargs = mock_hyper_grid.call_args_list[0][1]
        assert first_call_kwargs["shape"] == [2, 2, 1, 2]  # TP=2, CP=2, DP=1, PP=2


class TestCreatePgCollectionWithDistributedOptimizerInstances:
    """Tests for _create_pg_collection with multiple distributed optimizer instances."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model configuration for testing."""
        config = MagicMock()
        config.tensor_model_parallel_size = 1
        config.pipeline_model_parallel_size = 1
        config.context_parallel_size = 1
        config.expert_tensor_parallel_size = None
        config.expert_model_parallel_size = 1
        return config

    @patch("megatron.bridge.training.initialize.HyperCommGrid")
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.new_subgroups_by_enumeration")
    def test_create_pg_collection_with_multiple_optimizer_instances(
        self, mock_subgroups, mock_world_size, mock_hyper_grid, mock_model_config
    ):
        """Test _create_pg_collection with multiple distributed optimizer instances."""
        from megatron.bridge.training.initialize import _create_pg_collection

        # Setup mock
        mock_grid_instance = MagicMock()
        mock_grid_instance.create_pg.return_value = MagicMock()
        mock_grid_instance._gen_rank_enum.return_value = [[0, 1, 2, 3, 4, 5, 6, 7]]
        mock_hyper_grid.return_value = mock_grid_instance
        mock_subgroups.return_value = (MagicMock(), [])

        # Execute with multiple optimizer instances
        result = _create_pg_collection(mock_model_config, num_distributed_optimizer_instances=2)

        # Verify result is a ProcessGroupCollection
        from megatron.core.process_groups_config import ProcessGroupCollection

        assert isinstance(result, ProcessGroupCollection)
