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

import time
import unittest.mock as mock
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.train_utils import (
    calc_params_l2_norm,
    maybe_inject_state,
    needs_global_state_injection,
    param_is_not_shared,
    prepare_forward_step_func,
    report_l2_norm_grad,
    report_memory,
    report_runtime,
    report_throughput,
    training_log,
)


@dataclass
class MockTrainState:
    step: int = None
    consumed_train_samples: int = None


@dataclass
class MockTrainConfig:
    global_batch_size: int = None
    micro_batch_size: int = None


@dataclass
class MockParam:
    requires_grad: bool = True
    main_grad: float = None


class MockModelChunk:
    def __init__(self, layer_name, param):
        self.layer_name = layer_name
        self.param = param

    def named_parameters(self):
        yield self.layer_name, self.param


def make_default_model_config():
    """Create a SimpleNamespace with sane defaults for model attributes."""
    return SimpleNamespace(
        num_moe_experts=None,
        moe_router_load_balancing_type="",
        moe_z_loss_coeff=None,
        moe_per_layer_logging=False,
        num_layers=24,
        moe_layer_freq=1,
        mtp_num_layers=None,
        kv_channels=128,
        num_attention_heads=32,
        hidden_size=4096,
        num_query_groups=None,
        moe_router_topk=1,
        ffn_hidden_size=16384,
        moe_ffn_hidden_size=None,
        moe_shared_expert_intermediate_size=None,
        gated_linear_unit=False,
        activation_func=None,
        multi_latent_attention=False,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_head_dim=64,
        qk_pos_emb_head_dim=0,
        v_head_dim=64,
        seq_length=2048,
        vocab_size=51200,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        group_query_attention=False,
        num_moe_experts_routed_to=None,
        moe_router_load_balancing_threshold=None,
        moe_z_loss_scale=None,
        is_hybrid_model=False,
    )


class TestTrainingLog:
    """Test suite for the training_log function."""

    @pytest.fixture(autouse=True)
    def _patch_pg_collection(self, monkeypatch):
        class _PG:
            def __init__(self):
                self.dp = object()
                self.dp_cp = object()
                self.mp = object()
                self.pp = object()

        monkeypatch.setattr(
            "megatron.bridge.training.utils.train_utils.get_pg_collection",
            lambda model: _PG(),
            raising=True,
        )

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

        # Model config as a simple namespace to avoid auto-mocking methods
        config.model = make_default_model_config()

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
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_basic_logging_without_skip(
        self,
        mock_print_rank_last,
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
            history_wct=None,
            model=None,
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
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_skipped_iterations(
        self,
        mock_print_rank_last,
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
            history_wct=None,
            model=None,
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
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_nan_detection(
        self,
        mock_print_rank_last,
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
            history_wct=None,
            model=None,
        )

        # Assertions
        assert total_loss_dict["nan iterations"] == 1

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_tensorboard_logging_interval(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_print_rank_last,
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
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

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
            history_wct=None,
            model=None,
        )

        # Verify tensorboard logging was called
        mock_global_state.tensorboard_logger.add_scalar.assert_called()
        mock_global_state.timers.write.assert_called()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
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
            history_wct=None,
            model=None,
        )

        # Memory reporting should disable the flag
        assert result is False

        # Verify memory reporting functions were called
        mock_report_theoretical.assert_called_once()
        mock_report_memory.assert_called_once()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.track_moe_metrics")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_moe_logging(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_track_moe,
        mock_print_rank_last,
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
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

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
            history_wct=None,
            model=None,
        )

        # Verify MoE tracking was called
        mock_track_moe.assert_called_once()
        call_args = mock_track_moe.call_args
        assert "load_balancing_loss" in call_args.kwargs["track_names"]
        assert "z_loss" in call_args.kwargs["track_names"]

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.track_moe_metrics")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_moe_logging_seq_aux_loss(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_track_moe,
        mock_print_rank_last,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test MoE logging with seq_aux_loss router load balancing type."""
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

        # Enable MoE with seq_aux_loss
        mock_config.model.num_moe_experts = 8
        mock_config.model.moe_router_load_balancing_type = "seq_aux_loss"
        mock_config.model.moe_z_loss_coeff = None
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
            history_wct=None,
            model=None,
        )

        # Verify correct track names
        # Note: "seq_aux_loss" contains "aux_loss" substring, so both are matched
        mock_track_moe.assert_called_once()
        call_args = mock_track_moe.call_args
        track_names = call_args.kwargs["track_names"]
        assert "seq_load_balancing_loss" in track_names
        assert "load_balancing_loss" in track_names  # Also matched because "aux_loss" in "seq_aux_loss"
        assert "z_loss" not in track_names
        assert len(track_names) == 2

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.track_moe_metrics")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_moe_logging_global_aux_loss(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_track_moe,
        mock_print_rank_last,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test MoE logging with global_aux_loss router load balancing type."""
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

        # Enable MoE with global_aux_loss
        mock_config.model.num_moe_experts = 8
        mock_config.model.moe_router_load_balancing_type = "global_aux_loss"
        mock_config.model.moe_z_loss_coeff = None
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
            history_wct=None,
            model=None,
        )

        # Verify correct track names
        # Note: "global_aux_loss" contains "aux_loss" substring, so both are matched
        mock_track_moe.assert_called_once()
        call_args = mock_track_moe.call_args
        track_names = call_args.kwargs["track_names"]
        assert "global_load_balancing_loss" in track_names
        assert "load_balancing_loss" in track_names  # Also matched because "aux_loss" in "global_aux_loss"
        assert "z_loss" not in track_names
        assert len(track_names) == 2

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.track_moe_metrics")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_moe_logging_combined_aux_losses(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_track_moe,
        mock_print_rank_last,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test MoE logging with multiple aux loss types combined."""
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

        # Enable MoE with combined aux losses (string contains multiple types)
        mock_config.model.num_moe_experts = 8
        mock_config.model.moe_router_load_balancing_type = "aux_loss,seq_aux_loss,global_aux_loss"
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
            history_wct=None,
            model=None,
        )

        # Verify all track names are present
        mock_track_moe.assert_called_once()
        call_args = mock_track_moe.call_args
        track_names = call_args.kwargs["track_names"]
        assert "load_balancing_loss" in track_names
        assert "seq_load_balancing_loss" in track_names
        assert "global_load_balancing_loss" in track_names
        assert "z_loss" in track_names
        # Should have all 4 types
        assert len(track_names) == 4

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.track_moe_metrics")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_moe_logging_with_z_loss_only(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_track_moe,
        mock_print_rank_last,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test MoE logging with only z_loss enabled (no aux loss types)."""
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

        # Enable MoE with only z_loss
        mock_config.model.num_moe_experts = 8
        mock_config.model.moe_router_load_balancing_type = "none"  # No aux loss
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
            history_wct=None,
            model=None,
        )

        # Verify only z_loss is tracked
        mock_track_moe.assert_called_once()
        call_args = mock_track_moe.call_args
        track_names = call_args.kwargs["track_names"]
        assert "z_loss" in track_names
        assert "load_balancing_loss" not in track_names
        assert "seq_load_balancing_loss" not in track_names
        assert "global_load_balancing_loss" not in track_names
        assert len(track_names) == 1

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.track_moe_metrics")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_moe_logging_without_z_loss(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_track_moe,
        mock_print_rank_last,
        mock_get_world_size,
        mock_reduce_lr,
        mock_get_microbatches,
        mock_config,
        mock_global_state,
        loss_dict,
    ):
        """Test MoE logging with aux_loss but without z_loss."""
        total_loss_dict = self.get_fresh_total_loss_dict()

        # Setup mocks
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

        # Enable MoE with aux_loss but no z_loss
        mock_config.model.num_moe_experts = 8
        mock_config.model.moe_router_load_balancing_type = "aux_loss"
        mock_config.model.moe_z_loss_coeff = None  # No z_loss
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
            history_wct=None,
            model=None,
        )

        # Verify only load_balancing_loss is tracked
        mock_track_moe.assert_called_once()
        call_args = mock_track_moe.call_args
        track_names = call_args.kwargs["track_names"]
        assert "load_balancing_loss" in track_names
        assert "z_loss" not in track_names
        assert len(track_names) == 1

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.MTPLossLoggingHelper")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_mtp_logging(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_mtp_helper,
        mock_print_rank_last,
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
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

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
            history_wct=None,
            model=None,
        )

        # Verify MTP tracking was called
        mock_mtp_helper.track_mtp_metrics.assert_called_once()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_energy_monitoring(
        self,
        mock_print_rank_last,
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
            history_wct=None,
            model=None,
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
    @mock.patch("megatron.bridge.training.utils.train_utils.get_rank_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_0")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("torch.cuda.memory._snapshot")
    @mock.patch("builtins.open")
    @mock.patch("pickle.dump")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_profiling_memory_snapshot(
        self,
        mock_report_runtime,
        mock_report_throughput,
        mock_report_l2_norm_grad,
        mock_pickle_dump,
        mock_open,
        mock_memory_snapshot,
        mock_print_rank_last,
        mock_print_rank_0,
        mock_get_rank_safe,
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
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32
        mock_get_rank_safe.return_value = 7
        mock_memory_snapshot.return_value = {"mock": "snapshot"}
        mock_file_handle = mock.MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle

        # Enable profiling with memory history
        mock_profiling_config = mock.MagicMock()
        mock_profiling_config.record_memory_history = True
        mock_profiling_config.memory_snapshot_path = "/tmp/memory_snapshot.pkl"
        mock_profiling_config.profile_ranks = [7]
        mock_config.profiling = mock_profiling_config
        mock_config.logger.tensorboard_dir = "/tmp/tb"

        # Set iteration (snapshot itself is not gated by tensorboard log interval anymore)
        mock_global_state.train_state.step = 10

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
            history_wct=None,
            model=None,
        )

        # Verify memory snapshot was taken and saved
        mock_memory_snapshot.assert_called_once()
        mock_open.assert_called_once_with("/tmp/memory_snapshot_7.pkl", "wb")
        mock_pickle_dump.assert_called_once_with({"mock": "snapshot"}, mock_file_handle)
        mock_print_rank_0.assert_any_call("Saved memory snapshot to /tmp/memory_snapshot_7.pkl")

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    def test_wandb_specific_logging(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_print_rank_last,
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
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

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
            history_wct=None,
            model=None,
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
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    def test_no_loggers_present(
        self,
        mock_print_rank_last,
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
            history_wct=None,
            model=None,
        )

        assert result is False

        # Should still print log string even without loggers
        mock_print_rank_last.assert_called()

    @mock.patch("megatron.bridge.training.utils.train_utils.get_num_microbatches")
    @mock.patch("megatron.bridge.training.utils.train_utils.reduce_max_stat_across_model_parallel_group")
    @mock.patch("megatron.bridge.training.utils.train_utils.get_world_size_safe")
    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_last")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_memory")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_runtime")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_throughput")
    @mock.patch("megatron.bridge.training.utils.train_utils.report_l2_norm_grad")
    @mock.patch("torch.cuda.memory_stats")
    def test_memory_tensorboard_logging(
        self,
        mock_report_l2_norm_grad,
        mock_report_throughput,
        mock_report_runtime,
        mock_report_memory,
        mock_memory_stats,
        mock_print_rank_last,
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
        mock_report_l2_norm_grad.return_value = {}
        mock_report_throughput.return_value = {}
        mock_report_runtime.return_value = {}
        mock_get_microbatches.return_value = 8
        mock_reduce_lr.return_value = 1e-4
        mock_get_world_size.return_value = 32

        mock_memory_stats.return_value = {
            "mem-reserved-gigabytes": 2.048,
            "mem-allocated-gigabytes": 1.536000000,
            "mem-max-allocated-gigabytes": 1.792,
            "mem-allocated-count": 5000,
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
            history_wct=None,
            model=None,
        )

        # Verify memory stats were logged to tensorboard
        writer = mock_global_state.tensorboard_logger
        writer.add_scalar.assert_any_call("memory/mem-reserved-gigabytes", 2.048, 10)
        writer.add_scalar.assert_any_call("memory/mem-allocated-gigabytes", 1.536, 10)
        writer.add_scalar.assert_any_call("memory/mem-max-allocated-gigabytes", 1.792, 10)
        writer.add_scalar.assert_any_call("memory/mem-allocated-count", 5000, 10)

    def test_report_memory(self):
        """Test memory metrics."""
        memory_report = report_memory(memory_keys=None)
        assert len(memory_report) == 10

        memory_keys = {
            "reserved_bytes.all.current": "mem-reserved-bytes",
            "reserved_bytes.all.peak": "mem-max-reserved-bytes",
        }
        expected_keys = ["mem-reserved-gigabytes", "mem-max-reserved-gigabytes"]
        memory_report = report_memory(memory_keys=memory_keys)
        assert list(memory_report.keys()) == expected_keys

    def test_report_runtime(self):
        """Test runtime metrics."""
        start_time = time.time()

        step = 100
        consumed_train_samples = 1000
        seq_length = 2048
        train_iters = 1000

        train_state = MockTrainState(step=step, consumed_train_samples=consumed_train_samples)
        runtime_report = report_runtime(
            train_state=train_state,
            start_time=start_time,
            seq_length=seq_length,
            train_iters=train_iters,
        )

        assert runtime_report["time/tokens"] == consumed_train_samples * seq_length
        assert runtime_report["time/samples"] == consumed_train_samples

    def test_report_throughput(self):
        """Test throughput metrics."""
        global_batch_size = 64
        micro_batch_size = 4
        iteration = 100
        seq_length = 4096
        history_wct = [0.9, 1.7, 2.9, 4.2, 5.9]
        window_size = len(history_wct)
        train_config = MockTrainConfig(global_batch_size=global_batch_size, micro_batch_size=micro_batch_size)

        throughput_report = report_throughput(
            train_config=train_config,
            iteration=iteration,
            seq_length=seq_length,
            history_wct=history_wct,
            window_size=window_size,
        )

        assert throughput_report["throughput/tokens_per_sec"] == 209715.2
        assert throughput_report["throughput/batches_per_sec"] == 0.8
        assert throughput_report["throughput/micro_batch_size"] == 4
        assert throughput_report["throughput/device/samples_per_sec"] == 51.2

    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_0")
    def test_report_throughput_zero_elapsed_wct(self, mock_print_rank_0):
        """Test throughput metrics when elapsed_wct is zero (identical timestamps).

        This can happen during checkpoint resumption when history_wct is reinitialized
        and the first few iterations have identical timestamps.
        """
        global_batch_size = 64
        micro_batch_size = 4
        iteration = 100
        seq_length = 4096
        # All timestamps are identical - elapsed_wct will be 0
        history_wct = [1.5, 1.5, 1.5, 1.5, 1.5]
        window_size = len(history_wct)
        train_config = MockTrainConfig(global_batch_size=global_batch_size, micro_batch_size=micro_batch_size)

        throughput_report = report_throughput(
            train_config=train_config,
            iteration=iteration,
            seq_length=seq_length,
            history_wct=history_wct,
            window_size=window_size,
        )

        # Should return empty dict when elapsed_wct is 0
        assert throughput_report == {}

        # Verify warning was printed
        mock_print_rank_0.assert_called_once()
        warning_message = mock_print_rank_0.call_args[0][0]
        assert "Warning: elapsed_wct is 0" in warning_message
        assert "skipping throughput calculation" in warning_message
        assert f"iteration {iteration}" in warning_message

    @mock.patch("megatron.bridge.training.utils.train_utils.print_rank_0")
    def test_report_throughput_negative_elapsed_wct(self, mock_print_rank_0):
        """Test throughput metrics when elapsed_wct is negative.

        This shouldn't happen in normal operation, but the code guards against it
        to prevent division by zero or negative throughput values.
        """
        global_batch_size = 64
        micro_batch_size = 4
        iteration = 100
        seq_length = 4096
        # Timestamps go backwards - elapsed_wct will be negative
        history_wct = [5.9, 4.2, 2.9, 1.7, 0.9]
        window_size = len(history_wct)
        train_config = MockTrainConfig(global_batch_size=global_batch_size, micro_batch_size=micro_batch_size)

        throughput_report = report_throughput(
            train_config=train_config,
            iteration=iteration,
            seq_length=seq_length,
            history_wct=history_wct,
            window_size=window_size,
        )

        # Should return empty dict when elapsed_wct is negative
        assert throughput_report == {}

        # Verify warning was printed with negative value
        mock_print_rank_0.assert_called_once()
        warning_message = mock_print_rank_0.call_args[0][0]
        assert "Warning: elapsed_wct is -5.0" in warning_message
        assert "skipping throughput calculation" in warning_message
        assert f"iteration {iteration}" in warning_message

    def test_l2_norm_grad(self):
        """Test l2 norm grad metrics."""
        num_chunks = 10
        layer_name = "layer"
        model = []
        # generate mock model
        for i in range(num_chunks):
            chunk_name = f"{layer_name}_{i}"
            main_grad = torch.tensor(i).float()
            param = MockParam(main_grad=main_grad)
            model_chunk = MockModelChunk(chunk_name, param)
            model.append(model_chunk)

        l2_norm_report = report_l2_norm_grad(model)
        assert np.round(l2_norm_report["l2_norm/grad/global"], 2) == 74.92
        assert l2_norm_report["l2_norm/grad/layer_2"] == 2.0
        assert l2_norm_report["l2_norm/grad/layer_9"] == 9.0


class TestNeedsGlobalStateInjection:
    """Test suite for the needs_global_state_injection function."""

    def test_function_with_globalstate_type_hint_needs_injection(self):
        """Test function with GlobalState type hint needs injection."""
        from megatron.bridge.training.state import GlobalState

        def forward_step_func(state: GlobalState, data_iterator, model):
            return None

        result = needs_global_state_injection(forward_step_func)
        assert result is True

    def test_function_with_string_globalstate_annotation_needs_injection(self):
        """Test function with string GlobalState annotation needs injection."""

        def forward_step_func(state: "GlobalState", data_iterator, model):
            return None

        result = needs_global_state_injection(forward_step_func)
        assert result is True

    def test_function_with_state_name_needs_injection(self):
        """Test function with 'state' parameter name needs injection."""

        def forward_step_func(state, data_iterator, model):
            return None

        result = needs_global_state_injection(forward_step_func)
        assert result is True

    def test_function_with_global_state_name_needs_injection(self):
        """Test function with 'global_state' parameter name needs injection."""

        def forward_step_func(global_state, data_iterator, model):
            return None

        result = needs_global_state_injection(forward_step_func)
        assert result is True

    def test_function_without_state_no_injection(self):
        """Test function without state parameter doesn't need injection."""

        def forward_step_func(data_iterator, model, return_schedule_plan=False):
            return None

        result = needs_global_state_injection(forward_step_func)
        assert result is False

    def test_lambda_function_with_state_name(self):
        """Test lambda function with state parameter name."""
        forward_step_func = lambda state, data_iterator, model: None

        result = needs_global_state_injection(forward_step_func)
        assert result is True

    def test_lambda_function_without_state(self):
        """Test lambda function without state parameter."""
        forward_step_func = lambda data_iterator, model: None

        result = needs_global_state_injection(forward_step_func)
        assert result is False

    def test_callable_class_with_globalstate_type_hint(self):
        """Test callable class with GlobalState type hint."""
        from megatron.bridge.training.state import GlobalState

        class ForwardFunctor:
            def __call__(self, state: GlobalState, data_iterator, model):
                return None

        result = needs_global_state_injection(ForwardFunctor())
        assert result is True

    def test_callable_class_with_state_name(self):
        """Test callable class with state parameter name."""

        class ForwardFunctor:
            def __call__(self, state, data_iterator, model, return_schedule_plan=False):
                return None

        result = needs_global_state_injection(ForwardFunctor())
        assert result is True

    def test_callable_class_without_state(self):
        """Test callable class without state parameter."""

        class ForwardFunctor:
            def __call__(self, data_iterator, model, return_schedule_plan=False):
                return None

        result = needs_global_state_injection(ForwardFunctor())
        assert result is False


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

        result_func = maybe_inject_state(forward_step_func_4_args, mock_state, needs_injection=True)

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

        result_func = maybe_inject_state(forward_step_func_3_args, mock_state, needs_injection=False)

        # Result should be the original function
        assert result_func is forward_step_func_3_args

    def test_no_injection_two_args_with_explicit_num_args(self):
        """Test no state injection when num_fw_args is explicitly provided as 2."""

        def forward_step_func_2_args(data_iterator, model):
            return "Called without state injection"

        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(forward_step_func_2_args, mock_state, needs_injection=False)

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

    def test_callable_class_four_args_injects_state(self):
        """Test state injection for callable class with 4 arguments."""

        class ForwardFunctor:
            def __init__(self):
                self.seen_state = None

            def __call__(self, state, data_iterator, model, return_schedule_plan=False):
                self.seen_state = state
                return "called"

        functor = ForwardFunctor()
        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(functor, mock_state)

        assert isinstance(result_func, partial)

        mock_data_iterator = mock.MagicMock()
        mock_model = mock.MagicMock()
        result = result_func(mock_data_iterator, mock_model, return_schedule_plan=True)

        assert result == "called"
        assert functor.seen_state is mock_state

    def test_callable_class_three_args_no_injection(self):
        """Test callable class with 3 arguments does not inject state."""

        class ForwardFunctor:
            def __call__(self, data_iterator, model, return_schedule_plan=False):
                return "no state"

        functor = ForwardFunctor()
        mock_state = mock.MagicMock()

        result_func = maybe_inject_state(functor, mock_state)

        assert result_func is functor
        assert not isinstance(result_func, partial)


class TestPrepareForwardStepFunc:
    """Tests for prepare_forward_step_func convenience function."""

    def test_prepare_with_state_parameter_injects(self):
        """Test prepare_forward_step_func with function that needs state injection."""

        def forward_with_state(state: GlobalState, data_iterator, model):
            return state.train_state.step

        mock_state = mock.MagicMock()
        mock_state.train_state.step = 42

        result = prepare_forward_step_func(forward_with_state, mock_state)

        # Should be wrapped
        assert isinstance(result, partial)
        # Should work correctly
        assert result(None, None) == 42

    def test_prepare_without_state_parameter_returns_original(self):
        """Test prepare_forward_step_func with function that doesn't need state injection."""

        def forward_no_state(data_iterator, model):
            return "no state needed"

        mock_state = mock.MagicMock()

        result = prepare_forward_step_func(forward_no_state, mock_state)

        # Should return original function
        assert result is forward_no_state
        assert not isinstance(result, partial)

    def test_prepare_with_functor_needing_state(self):
        """Test prepare_forward_step_func with functor that needs state injection."""

        class ForwardFunctor:
            def __init__(self):
                self.call_count = 0

            def __call__(self, state: GlobalState, data_iterator, model):
                self.call_count += 1
                return state.train_state.step + self.call_count

        functor = ForwardFunctor()
        mock_state = mock.MagicMock()
        mock_state.train_state.step = 10

        result = prepare_forward_step_func(functor, mock_state)

        # Should be wrapped
        assert isinstance(result, partial)

        # Call multiple times - verify functor's internal state still works
        assert result(None, None) == 11  # step=10 + call_count=1
        assert result(None, None) == 12  # step=10 + call_count=2
        assert functor.call_count == 2

    def test_prepare_with_functor_not_needing_state(self):
        """Test prepare_forward_step_func with functor that doesn't need state."""

        class ForwardFunctor:
            def __init__(self):
                self.call_count = 0

            def __call__(self, data_iterator, model):
                self.call_count += 1
                return self.call_count

        functor = ForwardFunctor()
        mock_state = mock.MagicMock()

        result = prepare_forward_step_func(functor, mock_state)

        # Should return original functor
        assert result is functor
        assert not isinstance(result, partial)

        # Functor should still work
        assert result(None, None) == 1
        assert result(None, None) == 2

    def test_prepare_sees_state_mutations(self):
        """Test that prepared function sees mutations to GlobalState."""

        def forward_with_state(state: GlobalState, data_iterator, model):
            return state.train_state.step

        mock_state = mock.MagicMock()
        mock_state.train_state.step = 10

        # Prepare once
        wrapped = prepare_forward_step_func(forward_with_state, mock_state)

        # Call with initial state
        assert wrapped(None, None) == 10

        # Mutate state (simulates training loop incrementing step)
        mock_state.train_state.step = 20

        # Call again - should see mutated value
        assert wrapped(None, None) == 20

        # Further mutation
        mock_state.train_state.step = 100

        # Still sees current value
        assert wrapped(None, None) == 100


class TestParamIsNotShared:
    """Test suite for the param_is_not_shared function."""

    def test_param_without_shared_attribute(self):
        """Test parameter without 'shared' attribute returns True."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        assert param_is_not_shared(param) is True

    def test_param_with_shared_false(self):
        """Test parameter with shared=False returns True."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.shared = False
        assert param_is_not_shared(param) is True

    def test_param_with_shared_true(self):
        """Test parameter with shared=True returns False."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.shared = True
        assert param_is_not_shared(param) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this test")
class TestCalcParamsL2Norm:
    """Test suite for the calc_params_l2_norm function."""

    @pytest.fixture(autouse=True)
    def _patch_pg_collection(self, monkeypatch):
        class _PG:
            def __init__(self):
                # Minimal set of groups used by calc_params_l2_norm
                self.dp_cp = object()
                self.mp = object()
                self.tp_ep_pp = object()
                self.pp = object()

                # Provide dp with size() to satisfy any incidental calls
                class _DP:
                    def size(self_inner):
                        return 1

                self.dp = _DP()

        monkeypatch.setattr(
            "megatron.bridge.training.utils.train_utils.get_pg_collection",
            lambda model: _PG(),
            raising=True,
        )

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20, bias=False),
            torch.nn.Linear(20, 10, bias=False),
        ).cuda()
        return model

    @pytest.fixture
    def mock_model_config_fp32(self):
        """Create a mock model config for FP32 mode."""
        config = mock.MagicMock()
        config.bf16 = False
        return config

    @pytest.fixture
    def mock_model_config_bf16(self):
        """Create a mock model config for BF16 mode."""
        config = mock.MagicMock()
        config.bf16 = True
        return config

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_single_model_fp32(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        simple_model,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm with a single model in FP32 mode."""
        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x  # Return input unchanged
        mock_get_ranks.return_value = [0]

        # Initialize model parameters to known values
        for param in simple_model.parameters():
            torch.nn.init.constant_(param, 1.0)

        # Expected L2 norm: sqrt(sum of squares of all parameters)
        # Model has 10*20 + 20*10 = 400 parameters, each = 1.0
        # L2 norm = sqrt(400 * 1.0^2) = 20.0
        expected_norm = 20.0

        result = calc_params_l2_norm(simple_model, mock_model_config_fp32)

        assert isinstance(result, float)
        assert result == pytest.approx(expected_norm, rel=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_list_of_models(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm with a list of models."""
        # Create two simple models
        model1 = torch.nn.Linear(5, 5, bias=False).cuda()
        model2 = torch.nn.Linear(5, 5, bias=False).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Initialize to known values
        torch.nn.init.constant_(model1.weight, 1.0)
        torch.nn.init.constant_(model2.weight, 1.0)

        # Expected: 2 models * 25 params each = 50 params
        # L2 norm = sqrt(50 * 1.0^2) = sqrt(50) ≈ 7.071
        expected_norm = torch.sqrt(torch.tensor(50.0)).item()

        result = calc_params_l2_norm([model1, model2], mock_model_config_fp32)

        assert result == pytest.approx(expected_norm, rel=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_bf16_mode_without_main_param(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_bf16,
    ):
        """Test calc_params_l2_norm in BF16 mode without main_param attribute."""
        model = torch.nn.Linear(5, 5, bias=False, dtype=torch.bfloat16).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        torch.nn.init.constant_(model.weight, 1.0)

        expected_norm = torch.sqrt(torch.tensor(25.0)).item()

        result = calc_params_l2_norm(model, mock_model_config_bf16, force_create_fp32_copy=False)

        assert result == pytest.approx(expected_norm, rel=1e-3)  # BF16 has lower precision

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_bf16_mode_with_main_param(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_bf16,
    ):
        """Test calc_params_l2_norm in BF16 mode with main_param attribute."""
        model = torch.nn.Linear(5, 5, bias=False, dtype=torch.bfloat16).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Add main_param attribute (FP32 copy)
        for param in model.parameters():
            torch.nn.init.constant_(param, 1.0)
            param.main_param = torch.ones_like(param, dtype=torch.float32).cuda()
            param.main_param_sharded = False

        expected_norm = 5.0  # sqrt(25)

        result = calc_params_l2_norm(model, mock_model_config_bf16, force_create_fp32_copy=False)

        assert result == pytest.approx(expected_norm, rel=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_bf16_mode_with_sharded_main_param(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_bf16,
    ):
        """Test calc_params_l2_norm with sharded main params (distributed optimizer)."""
        model = torch.nn.Linear(5, 5, bias=False, dtype=torch.bfloat16).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Add sharded main_param attribute
        for param in model.parameters():
            torch.nn.init.constant_(param, 1.0)
            param.main_param = torch.ones(13, dtype=torch.float32).cuda()  # Sharded to 13 elements
            param.main_param_sharded = True

        result = calc_params_l2_norm(model, mock_model_config_bf16, force_create_fp32_copy=False)

        # Should use sharded params path and call all_reduce
        assert isinstance(result, float)
        assert result > 0

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_force_create_fp32_copy(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_bf16,
    ):
        """Test force_create_fp32_copy flag ignores main_param."""
        model = torch.nn.Linear(5, 5, bias=False, dtype=torch.bfloat16).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Add main_param but it should be ignored with force_create_fp32_copy=True
        for param in model.parameters():
            torch.nn.init.constant_(param, 1.0)
            # Set main_param to different value to verify it's not used
            param.main_param = torch.zeros_like(param, dtype=torch.float32).cuda()
            param.main_param_sharded = False

        result = calc_params_l2_norm(model, mock_model_config_bf16, force_create_fp32_copy=True)

        # Should create FP32 copy from bf16 params (value 1.0), not use main_param (value 0.0)
        expected_norm = 5.0  # sqrt(25 * 1.0^2)
        assert result == pytest.approx(expected_norm, rel=1e-3)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_moe_params(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm with MoE parameters (allreduce=False)."""
        model = torch.nn.Linear(5, 5, bias=False).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Mark parameters as MoE (allreduce=False)
        for param in model.parameters():
            torch.nn.init.constant_(param, 1.0)
            param.allreduce = False

        result = calc_params_l2_norm(model, mock_model_config_fp32)

        expected_norm = 5.0  # sqrt(25)
        assert result == pytest.approx(expected_norm, rel=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_shared_params(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm skips shared parameters."""
        model = torch.nn.Linear(5, 5, bias=False).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Mark parameters as shared (should be skipped)
        for param in model.parameters():
            torch.nn.init.constant_(param, 1.0)
            param.shared = True

        result = calc_params_l2_norm(model, mock_model_config_fp32)

        # Should be 0 since all params are shared
        assert result == pytest.approx(0.0, abs=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    def test_tp_duplicate_params(
        self,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm skips TP duplicate parameters."""
        model = torch.nn.Linear(5, 5, bias=False).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        # Mark all params as TP duplicates
        mock_is_not_tp_dup.return_value = False

        torch.nn.init.constant_(model.weight, 1.0)

        with (
            mock.patch("megatron.core.parallel_state.get_data_parallel_group"),
            mock.patch("megatron.core.parallel_state.get_model_parallel_group"),
            mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group"),
            mock.patch("torch.distributed.get_process_group_ranks", return_value=[0]),
            mock.patch("torch.distributed.all_reduce"),
        ):
            result = calc_params_l2_norm(model, mock_model_config_fp32)

            # Should be 0 since all params are TP duplicates
            assert result == pytest.approx(0.0, abs=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.calc_dtensor_params_l2_norm")
    def test_megatron_fsdp_path(self, mock_calc_dtensor_norm, mock_model_config_fp32):
        """Test calc_params_l2_norm with use_megatron_fsdp=True."""
        # Create a mock model with DTensor parameters
        model = mock.MagicMock()
        model.stop_communication = mock.MagicMock()

        # Mock parameter with DTensor attribute
        mock_param = mock.MagicMock()
        mock_param._local_tensor = torch.randn(5, 5).cuda()
        model.named_parameters.return_value = [("weight", mock_param)]

        mock_calc_dtensor_norm.return_value = 7.5

        result = calc_params_l2_norm(model, mock_model_config_fp32, use_megatron_fsdp=True)

        # Verify stop_communication was called
        model.stop_communication.assert_called_once()

        # Verify calc_dtensor_params_l2_norm was called
        mock_calc_dtensor_norm.assert_called_once()

        assert result == 7.5

    def test_megatron_fsdp_missing_dtensor(self, mock_model_config_fp32):
        """Test error when FSDP is enabled but parameter is not DTensor."""
        model = mock.MagicMock()
        model.stop_communication = mock.MagicMock()

        # Mock parameter without DTensor attribute
        mock_param = mock.MagicMock(spec=torch.nn.Parameter)
        mock_param.__class__ = torch.nn.Parameter
        del mock_param._local_tensor  # Ensure attribute doesn't exist
        model.named_parameters.return_value = [("weight", mock_param)]

        with pytest.raises(RuntimeError, match="Megatron FSDP requires parameters are PyTorch DTensor"):
            calc_params_l2_norm(model, mock_model_config_fp32, use_megatron_fsdp=True)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_mixed_dense_and_moe_params(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm with mixed dense and MoE parameters."""
        # Create a model with multiple layers
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 5, bias=False),
            torch.nn.Linear(5, 5, bias=False),
        ).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Initialize all params to 1.0
        params = list(model.parameters())
        for param in params:
            torch.nn.init.constant_(param, 1.0)

        # Mark first layer as dense, second as MoE
        params[0].allreduce = True
        params[1].allreduce = False

        result = calc_params_l2_norm(model, mock_model_config_fp32)

        # Both layers contribute: 2 * 25 params = 50 total
        expected_norm = torch.sqrt(torch.tensor(50.0)).item()
        assert result == pytest.approx(expected_norm, rel=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_empty_model(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm with a model that has no parameters."""
        model = torch.nn.Sequential().cuda()  # Empty model

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        result = calc_params_l2_norm(model, mock_model_config_fp32)

        # Empty model should have norm of 0
        assert result == pytest.approx(0.0, abs=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_different_reduce_groups(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_fp32,
    ):
        """Test calc_params_l2_norm with different dense and expert reduce groups."""
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 3, bias=False),
            torch.nn.Linear(3, 3, bias=False),
        ).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x

        # Mock different groups for dense and expert params
        mock_get_ranks.side_effect = [
            [0, 1],  # dense_reduce_group ranks
            [0, 1, 2, 3],  # expert_reduce_group ranks (different)
        ]

        params = list(model.parameters())
        torch.nn.init.constant_(params[0], 1.0)
        params[0].allreduce = True  # Dense

        torch.nn.init.constant_(params[1], 1.0)
        params[1].allreduce = False  # MoE

        result = calc_params_l2_norm(model, mock_model_config_fp32)

        # Verify all_reduce was called separately for each group
        assert mock_all_reduce.call_count >= 2
        assert isinstance(result, float)
        assert result > 0

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_main_param_none_with_sharded(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_bf16,
    ):
        """Test calc_params_l2_norm when main_param is None with main_param_sharded=True.

        When main_param_sharded=True but main_param is None, the parameter is skipped
        (nothing is added to sharded_params_data list).
        """
        model = torch.nn.Linear(5, 5, bias=False, dtype=torch.bfloat16).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Add main_param_sharded attribute but set main_param to None
        # This causes the parameter to be skipped entirely
        for param in model.parameters():
            torch.nn.init.constant_(param, 1.0)
            param.main_param = None
            param.main_param_sharded = True

        result = calc_params_l2_norm(model, mock_model_config_bf16, force_create_fp32_copy=False)

        # Parameter is skipped, so norm should be 0
        assert result == pytest.approx(0.0, abs=1e-5)

    @mock.patch("megatron.bridge.training.utils.train_utils.get_data_parallel_group_if_dtensor")
    @mock.patch("megatron.bridge.training.utils.train_utils.param_is_not_tensor_parallel_duplicate")
    @mock.patch("megatron.bridge.training.utils.train_utils.to_local_if_dtensor")
    @mock.patch("megatron.core.parallel_state.get_data_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_model_parallel_group")
    @mock.patch("megatron.core.parallel_state.get_expert_tensor_model_pipeline_parallel_group")
    @mock.patch("torch.distributed.get_process_group_ranks")
    @mock.patch("torch.distributed.all_reduce")
    def test_main_param_none_without_sharded(
        self,
        mock_all_reduce,
        mock_get_ranks,
        mock_get_expert_group,
        mock_get_model_group,
        mock_get_dp_group,
        mock_to_local,
        mock_is_not_tp_dup,
        mock_get_dp_group_if_dtensor,
        mock_model_config_bf16,
    ):
        """Test calc_params_l2_norm when main_param is None with main_param_sharded=False.

        This is an edge case that currently causes an error because None is added to
        params_data, and multi_tensor_l2norm doesn't accept None values. This test
        documents the current behavior - ideally the code should handle this more
        gracefully (e.g., skip None values or fallback to creating FP32 copy).
        """
        model = torch.nn.Linear(5, 5, bias=False, dtype=torch.bfloat16).cuda()

        # Setup mocks
        mock_get_dp_group_if_dtensor.return_value = None
        mock_is_not_tp_dup.return_value = True
        mock_to_local.side_effect = lambda x: x
        mock_get_ranks.return_value = [0]

        # Add main_param attribute set to None with main_param_sharded=False
        for param in model.parameters():
            torch.nn.init.constant_(param, 1.0)
            param.main_param = None
            param.main_param_sharded = False

        # This currently raises a TypeError because None is passed to multi_tensor_l2norm
        with pytest.raises(TypeError, match="incompatible function arguments"):
            calc_params_l2_norm(model, mock_model_config_bf16, force_create_fp32_copy=False)
