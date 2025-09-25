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

import unittest.mock as mock
from functools import partial

import pytest
import torch

from megatron.bridge.training.utils.train_utils import (
    check_forward_step_func_num_args,
    maybe_inject_state,
    training_log,
)


class TestTrainingLog:
    """Test suite for the training_log function."""

    @pytest.fixture(scope="function")
    def mock_config(self):
        """Create a mock configuration object."""
        config = mock.MagicMock()
        # Logger config
        config.logger.log_timers_to_tensorboard = True
        config.logger.tensorboard_log_interval = 10
        config.logger.log_interval = 5
        config.logger.log_loss_scale_to_tensorboard = True
        config.logger.log_world_size_to_tensorboard = True
        config.logger.log_memory_to_tensorboard = False
        config.logger.log_throughput = False

        # Training config
        config.train.micro_batch_size = 2
        config.train.train_iters = 1000

        # Model config
        config.model.num_moe_experts = None
        config.model.mtp_num_layers = None

        # Optimizer config
        config.optimizer.decoupled_lr = None

        # Data parallel size
        config.data_parallel_size = 4

        # Profiling config
        config.profiling = None

        return config

    @pytest.fixture(scope="function")
    def mock_global_state(self):
        """Create a mock global state object."""
        global_state = mock.MagicMock()

        # Mock train state
        global_state.train_state.step = 100
        global_state.train_state.consumed_train_samples = 12800
        global_state.train_state.skipped_train_samples = 0

        # Mock timers
        mock_timers = mock.MagicMock()
        mock_timers.return_value.elapsed.return_value = 0.5  # 500ms per iteration
        global_state.timers = mock_timers

        # Mock loggers
        global_state.tensorboard_logger = mock.MagicMock()
        global_state.wandb_logger = mock.MagicMock()
        global_state.energy_monitor = None

        return global_state

    @pytest.fixture(scope="function")
    def loss_dict(self):
        """Create a sample loss dictionary."""
        return {
            "lm_loss": torch.tensor([2.5], device="cuda", dtype=torch.float32),
            "total_loss": torch.tensor([2.5], device="cuda", dtype=torch.float32),
        }

    def get_fresh_total_loss_dict(self):
        """Create a fresh empty total loss dictionary for accumulation."""
        return {}

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_basic_logging_without_skip(
        self,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test basic logging functionality without skipped iterations."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Override iteration to avoid log interval reset (101 % 5 != 0)
        mock_global_state.train_state.step = 101

        # Call the function
        result = training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Assertions
        assert result is False  # report_memory_flag should remain False

        # Check that losses were accumulated correctly
        assert "advanced iterations" in total_loss_dict
        assert total_loss_dict["advanced iterations"] == 1
        assert "skipped iterations" in total_loss_dict
        assert total_loss_dict["skipped iterations"] == 0
        assert "nan iterations" in total_loss_dict
        assert total_loss_dict["nan iterations"] == 0

        # Check that losses were added to total_loss_dict
        for key in loss_dict:
            assert key in total_loss_dict
            torch.testing.assert_close(total_loss_dict[key], loss_dict[key])

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_skipped_iterations(
        self,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test logging behavior with skipped iterations."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Override iteration to avoid log interval reset (101 % 5 != 0)
        mock_global_state.train_state.step = 101

        # Call the function with skipped iteration
        result = training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=1,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Assertions
        assert result is False

        # Check iteration counters
        assert total_loss_dict["advanced iterations"] == 0  # No advanced iterations
        assert total_loss_dict["skipped iterations"] == 1

        # When skipped, losses should not be accumulated in the usual way
        for key in loss_dict:
            assert key not in total_loss_dict or total_loss_dict[key] == torch.tensor(
                [0.0], dtype=torch.float, device="cuda"
            )

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_nan_detection(
        self,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
    ):
        """Test NaN detection in loss values."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Override iteration to avoid log interval reset (101 % 5 != 0)
        mock_global_state.train_state.step = 101

        # Create loss dict with NaN values
        nan_loss_dict = {
            "lm_loss": torch.tensor([float("nan")], device="cuda", dtype=torch.float32),
            "total_loss": torch.tensor([2.5], device="cuda", dtype=torch.float32),
        }

        training_log(
            loss_dict=nan_loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=1,  # Must be skipped for NaN detection
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Assertions
        assert total_loss_dict["nan iterations"] == 1

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_tensorboard_logging_interval(
        self,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test tensorboard logging at specified intervals."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Set iteration to match tensorboard logging interval
        mock_global_state.train_state.step = 100  # Should trigger tensorboard logging (100 % 10 == 0)
        mock_config.logger.tensorboard_log_interval = 10

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Verify tensorboard logging was called
        mock_global_state.tensorboard_logger.add_scalar.assert_called()
        mock_global_state.timers.write.assert_called()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_memory")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_theoretical_memory")
    @mock.patch("torch.distributed.get_rank")
    def test_memory_reporting(
        self,
        mock_get_rank,
        mock_report_theoretical,
        mock_report_memory,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test memory reporting functionality."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True
        mock_get_rank.return_value = 0

        # Set iteration to match log interval for memory reporting
        mock_global_state.train_state.step = 5
        mock_config.logger.log_interval = 5

        # Call the function with memory reporting enabled
        result = training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=True,  # Enable memory reporting
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Memory reporting should disable the flag
        assert result is False

        # Verify memory reporting functions were called
        mock_report_theoretical.assert_called_once()
        mock_report_memory.assert_called_once()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.track_moe_metrics")
    def test_moe_logging(
        self,
        mock_track_moe,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test MoE (Mixture of Experts) logging when enabled."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Enable MoE configuration
        mock_config.model.num_moe_experts = 8
        mock_config.model.moe_router_load_balancing_type = "aux_loss"
        mock_config.model.moe_z_loss_coeff = 0.1
        mock_config.model.moe_per_layer_logging = True
        mock_config.model.num_layers = 12
        mock_config.model.moe_layer_freq = 2
        mock_config.model.mtp_num_layers = None

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Verify MoE tracking was called
        mock_track_moe.assert_called_once()
        call_args = mock_track_moe.call_args
        assert "load_balancing_loss" in call_args.kwargs["track_names"]
        assert "z_loss" in call_args.kwargs["track_names"]

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.MTPLossLoggingHelper")
    def test_mtp_logging(
        self,
        mock_mtp_helper,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test MTP (Multi-Token Prediction) logging when enabled."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Enable MTP configuration
        mock_config.model.mtp_num_layers = 4
        mock_config.model.num_moe_experts = None

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Verify MTP tracking was called
        mock_mtp_helper.track_mtp_metrics.assert_called_once()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.core.parallel_state.is_pipeline_first_stage")
    @mock.patch("megatron.core.parallel_state.is_pipeline_last_stage")
    def test_decoupled_learning_rate(
        self,
        mock_is_pipeline_last,
        mock_is_pipeline_first,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test decoupled learning rate logging."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True
        mock_is_pipeline_first.return_value = True
        mock_is_pipeline_last.return_value = False

        # Enable decoupled learning rate
        mock_config.optimizer.decoupled_lr = 0.01

        # Set iteration to match log interval
        mock_global_state.train_state.step = 5
        mock_config.logger.log_interval = 5

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=2e-5,  # Different from regular LR
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Check that the log string includes decoupled learning rate
        mock_print_rank_last.assert_called()
        log_call_args = mock_print_rank_last.call_args[0][0]
        assert "decoupled learning rate" in log_call_args

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_energy_monitoring(
        self,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test energy monitoring functionality."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Enable energy monitoring
        mock_energy_monitor = mock.MagicMock()
        mock_energy_monitor.lap.return_value = 100.0  # 100 Joules
        mock_global_state.energy_monitor = mock_energy_monitor

        # Set iteration to match log interval
        mock_global_state.train_state.step = 5
        mock_config.logger.log_interval = 5

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Verify energy monitoring was called
        mock_energy_monitor.lap.assert_called_once()

        # Check that energy metrics appear in the log string
        mock_print_rank_last.assert_called()
        log_call_args = mock_print_rank_last.call_args[0][0]
        assert "energy per GPU" in log_call_args
        assert "power per GPU" in log_call_args

        # Verify tensorboard logging for energy metrics
        mock_global_state.tensorboard_logger.add_scalar.assert_any_call(
            "iter-energy/gpu", mock.ANY, mock_global_state.train_state.step
        )
        mock_global_state.tensorboard_logger.add_scalar.assert_any_call(
            "power/gpu", mock.ANY, mock_global_state.train_state.step
        )

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("torch.cuda.memory._snapshot")
    @mock.patch("builtins.open")
    @mock.patch("pickle.dump")
    def test_profiling_memory_snapshot(
        self,
        mock_pickle_dump,
        mock_open,
        mock_memory_snapshot,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test memory snapshot functionality when profiling is enabled."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True
        mock_memory_snapshot.return_value = {"mock": "snapshot"}
        mock_file_handle = mock.MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle

        # Enable profiling with memory history
        mock_profiling_config = mock.MagicMock()
        mock_profiling_config.record_memory_history = True
        mock_profiling_config.memory_snapshot_path = "/tmp/memory_snapshot.pkl"
        mock_config.profiling = mock_profiling_config

        # Set iteration to match tensorboard logging interval
        mock_global_state.train_state.step = 10
        mock_config.logger.tensorboard_log_interval = 10

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Verify memory snapshot was taken and saved
        mock_memory_snapshot.assert_called_once()
        mock_open.assert_called_once_with("/tmp/memory_snapshot.pkl", "wb")
        mock_pickle_dump.assert_called_once_with({"mock": "snapshot"}, mock_file_handle)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_wandb_specific_logging(
        self,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test WandB-specific logging functionality."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Set iteration to match tensorboard logging interval
        mock_global_state.train_state.step = 10
        mock_config.logger.tensorboard_log_interval = 10

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Verify WandB logging was called for various metrics
        wandb_writer = mock_global_state.wandb_logger
        wandb_writer.log.assert_any_call(
            {"samples vs steps": mock_global_state.train_state.consumed_train_samples}, 10
        )
        wandb_writer.log.assert_any_call({"learning-rate": 1e-4}, 10)
        wandb_writer.log.assert_any_call({"batch-size": mock.ANY}, 10)

        # Check loss logging to WandB
        for key in loss_dict:
            wandb_writer.log.assert_any_call({key: loss_dict[key]}, 10)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_no_loggers_present(
        self,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test behavior when no loggers are present."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        # Remove loggers
        mock_global_state.tensorboard_logger = None
        mock_global_state.wandb_logger = None

        # Set iteration to match logging intervals
        mock_global_state.train_state.step = 10
        mock_config.logger.tensorboard_log_interval = 10
        mock_config.logger.log_interval = 5

        result = training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        assert result is False

        # Should still print log string even without loggers
        mock_print_rank_last.assert_called()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.is_last_rank")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("torch.cuda.memory_stats")
    def test_memory_tensorboard_logging(
        self,
        mock_memory_stats,
        mock_print_rank_last,
        mock_is_last_rank,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test CUDA memory logging to tensorboard."""
        # Get fresh total_loss_dict for this test
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_is_last_rank.return_value = True

        mock_memory_stats.return_value = {
            "reserved_bytes.all.current": 2048000000,
            "allocated_bytes.all.current": 1536000000,
            "allocated_bytes.all.peak": 1792000000,
            "allocation.all.current": 5000,
        }

        # Enable memory logging
        mock_config.logger.log_memory_to_tensorboard = True

        # Set iteration to match tensorboard logging interval
        mock_global_state.train_state.step = 10
        mock_config.logger.tensorboard_log_interval = 10

        training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=1e-4,
            decoupled_learning_rate=None,
            loss_scale=1024.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=2.5,
            params_norm=15.2,
            num_zeros_in_grad=0,
            config=mock_config,
            global_state=mock_global_state,
        )

        # Verify memory stats were logged to tensorboard
        writer = mock_global_state.tensorboard_logger
        writer.add_scalar.assert_any_call("mem-reserved-bytes", 2048000000, 10)
        writer.add_scalar.assert_any_call("mem-allocated-bytes", 1536000000, 10)
        writer.add_scalar.assert_any_call("mem-max-allocated-bytes", 1792000000, 10)
        writer.add_scalar.assert_any_call("mem-allocated-count", 5000, 10)


class TestCheckForwardStepFuncNumArgs:
    """Test suite for the check_forward_step_func_num_args function."""

    def test_two_args_function(self):
        """Test function with 2 arguments."""

        def forward_step_func_2_args(data_iterator, model):
            return None

        result = check_forward_step_func_num_args(forward_step_func_2_args)
        assert result == 2

    def test_three_args_function(self):
        """Test function with 3 arguments."""

        def forward_step_func_3_args(data_iterator, model, return_schedule_plan=False):
            return None

        result = check_forward_step_func_num_args(forward_step_func_3_args)
        assert result == 3

    def test_four_args_function(self):
        """Test function with 4 arguments."""

        def forward_step_func_4_args(state, data_iterator, model, return_schedule_plan=False):
            return None

        result = check_forward_step_func_num_args(forward_step_func_4_args)
        assert result == 4

    def test_one_arg_function_raises_assertion_error(self):
        """Test function with 1 argument raises AssertionError."""

        def forward_step_func_1_arg(data_iterator):
            return None

        with pytest.raises(AssertionError) as exc_info:
            check_forward_step_func_num_args(forward_step_func_1_arg)

        error_message = str(exc_info.value)
        assert "forward_step_func has 1 arguments" in error_message
        assert "Only the following signatures are supported" in error_message
        assert "2 args:" in error_message
        assert "3 args:" in error_message
        assert "4 args:" in error_message

    def test_five_args_function_raises_assertion_error(self):
        """Test function with 5 arguments raises AssertionError."""

        def forward_step_func_5_args(state, data_iterator, model, return_schedule_plan, extra_arg):
            return None

        with pytest.raises(AssertionError) as exc_info:
            check_forward_step_func_num_args(forward_step_func_5_args)

        error_message = str(exc_info.value)
        assert "forward_step_func has 5 arguments" in error_message
        assert "Only the following signatures are supported" in error_message

    def test_zero_args_function_raises_assertion_error(self):
        """Test function with 0 arguments raises AssertionError."""

        def forward_step_func_0_args():
            return None

        with pytest.raises(AssertionError) as exc_info:
            check_forward_step_func_num_args(forward_step_func_0_args)

        error_message = str(exc_info.value)
        assert "forward_step_func has 0 arguments" in error_message

    def test_lambda_function_two_args(self):
        """Test lambda function with 2 arguments."""
        forward_step_func = lambda data_iterator, model: None

        result = check_forward_step_func_num_args(forward_step_func)
        assert result == 2

    def test_lambda_function_four_args(self):
        """Test lambda function with 4 arguments."""
        forward_step_func = lambda state, data_iterator, model, return_schedule_plan=False: None

        result = check_forward_step_func_num_args(forward_step_func)
        assert result == 4

    def test_partial_function(self):
        """Test partial function (should count remaining parameters)."""

        def original_func(state, data_iterator, model, return_schedule_plan=False):
            return None

        # Create partial function with state bound
        partial_func = partial(original_func, mock.MagicMock())

        result = check_forward_step_func_num_args(partial_func)
        assert result == 3  # 4 original args - 1 bound arg = 3 remaining


class TestMaybeInjectState:
    """Test suite for the maybe_inject_state function."""

    def test_inject_state_four_args_function(self):
        """Test state injection for 4-argument function."""

        def forward_step_func_4_args(state, data_iterator, model, return_schedule_plan=False):
            return f"Called with state: {state.name}"

        mock_state = mock.MagicMock()
        mock_state.name = "test_state"

        result_func = maybe_inject_state(forward_step_func_4_args, mock_state)

        # Result should be a partial function
        assert isinstance(result_func, partial)

        # Test calling the partial function
        mock_data_iterator = mock.MagicMock()
        mock_model = mock.MagicMock()

        result = result_func(mock_data_iterator, mock_model, return_schedule_plan=True)
        assert result == "Called with state: test_state"

    def test_inject_state_four_args_with_explicit_num_args(self):
        """Test state injection when num_fw_args is explicitly provided."""

        def forward_step_func_4_args(state, data_iterator, model, return_schedule_plan=False):
            return f"Called with state: {state.name}"

        mock_state = mock.MagicMock()
        mock_state.name = "test_state"

        result_func = maybe_inject_state(forward_step_func_4_args, mock_state, num_fw_args=4)

        # Result should be a partial function
        assert isinstance(result_func, partial)

    def test_no_injection_three_args_function(self):
        """Test no state injection for 3-argument function."""

        def forward_step_func_3_args(data_iterator, model, return_schedule_plan=False):
            return "Called without state injection"

        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(forward_step_func_3_args, mock_state)

        # Result should be the original function
        assert result_func is forward_step_func_3_args
        assert not isinstance(result_func, partial)

    def test_no_injection_two_args_function(self):
        """Test no state injection for 2-argument function."""

        def forward_step_func_2_args(data_iterator, model):
            return "Called without state injection"

        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(forward_step_func_2_args, mock_state)

        # Result should be the original function
        assert result_func is forward_step_func_2_args
        assert not isinstance(result_func, partial)

    def test_no_injection_three_args_with_explicit_num_args(self):
        """Test no state injection when num_fw_args is explicitly provided as 3."""

        def forward_step_func_3_args(data_iterator, model, return_schedule_plan=False):
            return "Called without state injection"

        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(forward_step_func_3_args, mock_state, num_fw_args=3)

        # Result should be the original function
        assert result_func is forward_step_func_3_args

    def test_no_injection_two_args_with_explicit_num_args(self):
        """Test no state injection when num_fw_args is explicitly provided as 2."""

        def forward_step_func_2_args(data_iterator, model):
            return "Called without state injection"

        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(forward_step_func_2_args, mock_state, num_fw_args=2)

        # Result should be the original function
        assert result_func is forward_step_func_2_args

    def test_inject_state_with_partial_function(self):
        """Test state injection with a function that's already partial."""

        def original_func(arg1, arg2, data_iterator, model):
            return f"Called with {arg1}, {arg2}"

        # Create partial function (simulating pre-bound arguments)
        partial_func = partial(original_func, "bound_arg1", "bound_arg2")

        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(partial_func, mock_state)

        # Should return original partial since it has 2 remaining args
        assert result_func is partial_func
