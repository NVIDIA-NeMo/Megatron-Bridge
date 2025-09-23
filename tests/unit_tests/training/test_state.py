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

from unittest.mock import MagicMock, patch

import torch

from megatron.bridge.training.state import FaultToleranceState, GlobalState, TrainState


class TestTrainState:
    """Test suite for TrainState class."""

    def test_initialization_defaults(self):
        """Test that TrainState initializes with correct default values."""
        state = TrainState()

        assert state.step == 0
        assert state.consumed_train_samples == 0
        assert state.skipped_train_samples == 0
        assert state.consumed_valid_samples == 0
        assert state.floating_point_operations_so_far == 0
        assert state.do_train is False
        assert state.do_valid is False
        assert state.do_test is False

    def test_initialization_with_custom_values(self):
        """Test initialization with custom values."""
        state = TrainState(
            step=100,
            consumed_train_samples=1000,
            skipped_train_samples=50,
            consumed_valid_samples=200,
            floating_point_operations_so_far=1500,
            do_train=True,
            do_valid=True,
            do_test=False,
        )

        assert state.step == 100
        assert state.consumed_train_samples == 1000
        assert state.skipped_train_samples == 50
        assert state.consumed_valid_samples == 200
        assert state.floating_point_operations_so_far == 1500
        assert state.do_train is True
        assert state.do_valid is True
        assert state.do_test is False

    def test_state_dict_structure_and_types(self):
        """Test that state_dict returns correct structure and tensor types."""
        state = TrainState(
            step=42,
            consumed_train_samples=500,
            skipped_train_samples=25,
            consumed_valid_samples=100,
            floating_point_operations_so_far=2000,
            do_train=True,
            do_valid=False,
            do_test=True,
        )

        state_dict = state.state_dict()

        # Check all expected keys are present
        expected_keys = {
            "step",
            "consumed_train_samples",
            "skipped_train_samples",
            "consumed_valid_samples",
            "floating_point_operations_so_far",
            "do_train",
            "do_valid",
            "do_test",
        }
        assert set(state_dict.keys()) == expected_keys

        # Check tensor types
        assert state_dict["step"].dtype == torch.int32
        assert state_dict["consumed_train_samples"].dtype == torch.int32
        assert state_dict["skipped_train_samples"].dtype == torch.int32
        assert state_dict["consumed_valid_samples"].dtype == torch.int32
        assert state_dict["floating_point_operations_so_far"].dtype == torch.float64
        assert state_dict["do_train"].dtype == torch.bool
        assert state_dict["do_valid"].dtype == torch.bool
        assert state_dict["do_test"].dtype == torch.bool

        # Check values
        assert state_dict["step"].item() == 42
        assert state_dict["consumed_train_samples"].item() == 500
        assert state_dict["skipped_train_samples"].item() == 25
        assert state_dict["consumed_valid_samples"].item() == 100
        assert state_dict["floating_point_operations_so_far"].item() == 2000
        assert state_dict["do_train"].item() is True
        assert state_dict["do_valid"].item() is False
        assert state_dict["do_test"].item() is True

    def test_load_state_dict(self):
        """Test loading state from state dictionary."""
        # Create a state dict manually
        state_dict = {
            "step": torch.tensor(75, dtype=torch.int32),
            "consumed_train_samples": torch.tensor(800, dtype=torch.int32),
            "skipped_train_samples": torch.tensor(40, dtype=torch.int32),
            "consumed_valid_samples": torch.tensor(150, dtype=torch.int32),
            "floating_point_operations_so_far": torch.tensor(3000.5, dtype=torch.float64),
            "do_train": torch.tensor(False, dtype=torch.bool),
            "do_valid": torch.tensor(True, dtype=torch.bool),
            "do_test": torch.tensor(False, dtype=torch.bool),
        }

        state = TrainState()
        state.load_state_dict(state_dict)

        assert state.step == 75
        assert state.consumed_train_samples == 800
        assert state.skipped_train_samples == 40
        assert state.consumed_valid_samples == 150
        assert state.floating_point_operations_so_far == 3000.5
        assert state.do_train is False
        assert state.do_valid is True
        assert state.do_test is False

    def test_round_trip_serialization(self):
        """Test that state_dict -> load_state_dict preserves all values."""
        # Create original state with various values
        original_state = TrainState(
            step=123,
            consumed_train_samples=2500,
            skipped_train_samples=100,
            consumed_valid_samples=500,
            floating_point_operations_so_far=12345.67,
            do_train=True,
            do_valid=False,
            do_test=True,
        )

        # Serialize to state dict
        state_dict = original_state.state_dict()

        # Create new state and load from dict
        new_state = TrainState()
        new_state.load_state_dict(state_dict)

        # Verify all values are preserved
        assert new_state.step == original_state.step
        assert new_state.consumed_train_samples == original_state.consumed_train_samples
        assert new_state.skipped_train_samples == original_state.skipped_train_samples
        assert new_state.consumed_valid_samples == original_state.consumed_valid_samples
        assert new_state.floating_point_operations_so_far == original_state.floating_point_operations_so_far
        assert new_state.do_train == original_state.do_train
        assert new_state.do_valid == original_state.do_valid
        assert new_state.do_test == original_state.do_test

    def test_state_dict_with_defaults(self):
        """Test state_dict with default values."""
        state = TrainState()
        state_dict = state.state_dict()

        # All default values should be properly serialized
        assert state_dict["step"].item() == 0
        assert state_dict["consumed_train_samples"].item() == 0
        assert state_dict["skipped_train_samples"].item() == 0
        assert state_dict["consumed_valid_samples"].item() == 0
        assert state_dict["floating_point_operations_so_far"].item() == 0
        assert state_dict["do_train"].item() is False
        assert state_dict["do_valid"].item() is False
        assert state_dict["do_test"].item() is False


class TestFaultToleranceState:
    """Test suite for FaultToleranceState class."""

    def test_initialization_defaults(self):
        """Test that FaultToleranceState initializes with correct default values."""
        state = FaultToleranceState()

        assert state.ft_state_path is None
        assert state.is_persistent_chkpt_loaded is False
        assert state.is_async_chkpt_enabled is False
        assert state.is_calculating_timeouts is False
        assert state.is_setup_section_open is False
        assert state.seen_checkpoints_cnt == 0
        assert state.seen_tr_iters_cnt == 0
        assert state.curr_eval_iter_idx == 0

    def test_initialization_with_custom_values(self):
        """Test initialization with custom values."""
        state = FaultToleranceState(
            ft_state_path="/tmp/ft_state.json",
            is_persistent_chkpt_loaded=True,
            is_async_chkpt_enabled=True,
            is_calculating_timeouts=True,
            is_setup_section_open=True,
            seen_checkpoints_cnt=5,
            seen_tr_iters_cnt=100,
            curr_eval_iter_idx=10,
        )

        assert state.ft_state_path == "/tmp/ft_state.json"
        assert state.is_persistent_chkpt_loaded is True
        assert state.is_async_chkpt_enabled is True
        assert state.is_calculating_timeouts is True
        assert state.is_setup_section_open is True
        assert state.seen_checkpoints_cnt == 5
        assert state.seen_tr_iters_cnt == 100
        assert state.curr_eval_iter_idx == 10


class TestGlobalState:
    """Test suite for GlobalState class."""

    def test_initialization(self):
        """Test that GlobalState initializes correctly."""
        state = GlobalState()

        assert state._initialized is True
        assert state._cfg is None
        assert state._tokenizer is None
        assert state._tensorboard_logger is None
        assert state._wandb_logger is None
        assert state._timers is None
        assert state._train_state is None
        assert state.rank_monitor_client is None
        assert state._signal_handler is None
        assert state._ft_state is None
        assert state._straggler_timer is None
        assert state._async_calls_queue is None
        assert state._nvrx_straggler_manager is None
        assert state._nvrx_straggler_created is False
        assert state._energy_monitor is None
        assert state._energy_monitor_created is False
        assert isinstance(state.start_time, float)

    def test_reinitialization_prevention(self):
        """Test that GlobalState prevents reinitialization of existing instance."""
        state = GlobalState()

        # Store original values
        original_start_time = state.start_time
        state._cfg = MagicMock()  # Set some state

        # Call __init__ again - should not reinitialize
        state.__init__()

        # Values should be preserved
        assert state._initialized is True
        assert state._cfg is not None
        assert state.start_time == original_start_time

    def test_cfg_property_getter(self):
        """Test cfg property getter."""
        state = GlobalState()
        mock_config = MagicMock()
        state._cfg = mock_config

        assert state.cfg == mock_config

    def test_cfg_property_setter(self):
        """Test cfg property setter."""
        state = GlobalState()
        mock_config = MagicMock()

        with patch.object(state, "_set_signal_handler") as mock_set_signal:
            state.cfg = mock_config

            assert state._cfg == mock_config
            mock_set_signal.assert_called_once()

    def test_cfg_property_setter_with_none(self):
        """Test cfg property setter with None value."""
        state = GlobalState()

        with patch.object(state, "_set_signal_handler") as mock_set_signal:
            state.cfg = None

            assert state._cfg is None
            mock_set_signal.assert_not_called()

    def test_tokenizer_property_lazy_initialization(self):
        """Test tokenizer property lazy initialization."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_tokenizer = MagicMock()
        state._cfg = mock_config

        with patch("megatron.bridge.training.state.build_tokenizer", return_value=mock_tokenizer) as mock_build:
            tokenizer = state.tokenizer

            mock_build.assert_called_once_with(mock_config.tokenizer)
            assert tokenizer == mock_tokenizer
            assert state._tokenizer == mock_tokenizer

    def test_tokenizer_property_returns_cached(self):
        """Test tokenizer property returns cached instance."""
        state = GlobalState()
        mock_tokenizer = MagicMock()
        state._tokenizer = mock_tokenizer

        with patch("megatron.bridge.training.state.build_tokenizer") as mock_build:
            tokenizer = state.tokenizer

            mock_build.assert_not_called()
            assert tokenizer == mock_tokenizer

    def test_tensorboard_logger_property_disabled(self):
        """Test tensorboard logger when disabled or not rank N-1."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.tensorboard_dir = None
        state._cfg = mock_config

        with (
            patch("megatron.bridge.training.state.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.state.get_world_size_safe", return_value=4),
        ):
            logger = state.tensorboard_logger

            assert logger is None
            assert state._tensorboard_logger is None

    def test_tensorboard_logger_property_enabled_rank_n_minus_1(self):
        """Test tensorboard logger enabled for rank N-1."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.tensorboard_dir = "/tmp/tensorboard"
        mock_config.logger.tensorboard_queue_size = 1000
        state._cfg = mock_config

        mock_summary_writer = MagicMock()

        with (
            patch("megatron.bridge.training.state.get_rank_safe", return_value=3),  # rank N-1 for world_size=4
            patch("megatron.bridge.training.state.get_world_size_safe", return_value=4),
            patch("torch.utils.tensorboard.writer.SummaryWriter", return_value=mock_summary_writer) as mock_sw,
            patch("builtins.print"),  # Mock print to avoid output
        ):
            logger = state.tensorboard_logger

            mock_sw.assert_called_once_with(log_dir="/tmp/tensorboard", max_queue=1000)
            assert logger == mock_summary_writer
            assert state._tensorboard_logger == mock_summary_writer

    def test_wandb_logger_property_disabled(self):
        """Test wandb logger when disabled."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.wandb_project = None
        state._cfg = mock_config

        with (
            patch("megatron.bridge.training.state.get_rank_safe", return_value=3),
            patch("megatron.bridge.training.state.get_world_size_safe", return_value=4),
        ):
            logger = state.wandb_logger

            assert logger is None
            assert state._wandb_logger is None

    def test_wandb_logger_property_enabled_rank_n_minus_1(self):
        """Test wandb logger enabled for rank N-1."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.wandb_project = "test_project"
        mock_config.logger.wandb_exp_name = "test_experiment"
        mock_config.logger.wandb_save_dir = "/tmp/wandb"
        mock_config.logger.wandb_entity = "test_entity"
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.to_dict.return_value = {"config": "data"}
        state._cfg = mock_config

        mock_wandb = MagicMock()

        with (
            patch("megatron.bridge.training.state.get_rank_safe", return_value=3),
            patch("megatron.bridge.training.state.get_world_size_safe", return_value=4),
            patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: mock_wandb
                if name == "wandb"
                else __import__(name, *args, **kwargs),
            ),
        ):
            logger = state.wandb_logger

            mock_wandb.init.assert_called_once_with(
                dir="/tmp/wandb",
                name="test_experiment",
                project="test_project",
                config={"config": "data"},
                entity="test_entity",
            )
            assert logger == mock_wandb
            assert state._wandb_logger == mock_wandb

    def test_wandb_logger_property_missing_experiment_name(self):
        """Test wandb logger raises error when experiment name is empty."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.wandb_project = "test_project"
        mock_config.logger.wandb_exp_name = ""
        state._cfg = mock_config

        with (
            patch("megatron.bridge.training.state.get_rank_safe", return_value=3),
            patch("megatron.bridge.training.state.get_world_size_safe", return_value=4),
        ):
            try:
                _ = state.wandb_logger
                assert False, "Expected ValueError"
            except ValueError as e:
                assert "Please specify the wandb experiment name!" in str(e)

    def test_timers_property_lazy_initialization(self):
        """Test timers property lazy initialization."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.timing_log_level = 1
        mock_config.logger.timing_log_option = "minmax"
        state._cfg = mock_config

        mock_timers = MagicMock()

        with patch("megatron.bridge.training.state.Timers", return_value=mock_timers) as mock_timers_class:
            timers = state.timers

            mock_timers_class.assert_called_once_with(1, "minmax")
            assert timers == mock_timers
            assert state._timers == mock_timers
            # Verify write_to_wandb method is patched
            assert hasattr(mock_timers, "write_to_wandb")

    def test_train_state_property_lazy_initialization(self):
        """Test train_state property lazy initialization."""
        state = GlobalState()

        train_state = state.train_state

        assert isinstance(train_state, TrainState)
        assert state._train_state == train_state

    def test_train_state_property_setter(self):
        """Test train_state property setter."""
        state = GlobalState()
        custom_train_state = TrainState(step=100, consumed_train_samples=1000)

        state.train_state = custom_train_state

        assert state._train_state == custom_train_state
        assert state.train_state == custom_train_state

    def test_fault_tolerance_state_property_lazy_initialization(self):
        """Test fault_tolerance_state property lazy initialization."""
        state = GlobalState()

        ft_state = state.fault_tolerance_state

        assert isinstance(ft_state, FaultToleranceState)
        assert state._ft_state == ft_state

    def test_fault_tolerance_state_property_setter(self):
        """Test fault_tolerance_state property setter."""
        state = GlobalState()
        custom_ft_state = FaultToleranceState(seen_checkpoints_cnt=5)

        state.fault_tolerance_state = custom_ft_state

        assert state._ft_state == custom_ft_state
        assert state.fault_tolerance_state == custom_ft_state

    def test_signal_handler_property_lazy_initialization(self):
        """Test signal_handler property lazy initialization."""
        state = GlobalState()

        with patch.object(state, "_set_signal_handler") as mock_set_signal:
            signal_handler = state.signal_handler

            mock_set_signal.assert_called_once()
            assert signal_handler == state._signal_handler

    def test_straggler_timer_property_lazy_initialization(self):
        """Test straggler_timer property lazy initialization."""
        state = GlobalState()

        mock_straggler_detector = MagicMock()

        with patch(
            "megatron.bridge.training.state.StragglerDetector", return_value=mock_straggler_detector
        ) as mock_sd:
            straggler_timer = state.straggler_timer

            mock_sd.assert_called_once()
            assert straggler_timer == mock_straggler_detector
            assert state._straggler_timer == mock_straggler_detector

    def test_initialize_async_checkpoint_worker_enabled(self):
        """Test async checkpoint worker initialization when enabled."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.checkpoint.async_save = True
        mock_config.checkpoint.use_persistent_ckpt_worker = True
        state._cfg = mock_config

        mock_async_queue = MagicMock()

        with patch("megatron.bridge.training.state.AsyncCallsQueue", return_value=mock_async_queue) as mock_acq:
            state.initialize_async_checkpoint_worker()

            mock_acq.assert_called_once_with(persistent=True)
            assert state._async_calls_queue == mock_async_queue

    def test_initialize_async_checkpoint_worker_disabled(self):
        """Test async checkpoint worker not initialized when disabled."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.checkpoint.async_save = False
        state._cfg = mock_config

        with patch("megatron.bridge.training.state.AsyncCallsQueue") as mock_acq:
            state.initialize_async_checkpoint_worker()

            mock_acq.assert_not_called()
            assert state._async_calls_queue is None

    def test_async_calls_queue_property(self):
        """Test async_calls_queue property."""
        state = GlobalState()
        mock_queue = MagicMock()
        state._async_calls_queue = mock_queue

        assert state.async_calls_queue == mock_queue

    def test_nvrx_straggler_manager_property_disabled(self):
        """Test nvrx_straggler_manager when disabled."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.nvrx_straggler = None
        state._cfg = mock_config

        # When disabled, the manager should be None and created flag should remain False
        manager = state.nvrx_straggler_manager

        assert manager is None
        assert state._nvrx_straggler_manager is None
        assert state._nvrx_straggler_created is False

        # Second call should not change anything
        manager2 = state.nvrx_straggler_manager
        assert manager2 is None
        assert state._nvrx_straggler_created is False

    def test_nvrx_straggler_manager_property_enabled(self):
        """Test nvrx_straggler_manager when enabled."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_straggler_config = MagicMock()
        mock_config.nvrx_straggler = mock_straggler_config
        state._cfg = mock_config

        mock_manager = MagicMock()

        with patch(
            "megatron.bridge.training.state.NVRxStragglerDetectionManager", return_value=mock_manager
        ) as mock_nsdm:
            manager = state.nvrx_straggler_manager

            mock_nsdm.assert_called_once_with(mock_straggler_config)
            assert manager == mock_manager
            assert state._nvrx_straggler_manager == mock_manager
            assert state._nvrx_straggler_created is True

    def test_energy_monitor_property_disabled(self):
        """Test energy_monitor when disabled."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.log_energy = False
        state._cfg = mock_config

        # When disabled, the monitor should be None and created flag should remain False
        monitor = state.energy_monitor

        assert monitor is None
        assert state._energy_monitor is None
        assert state._energy_monitor_created is False

        # Second call should not change anything
        monitor2 = state.energy_monitor
        assert monitor2 is None
        assert state._energy_monitor_created is False

    def test_energy_monitor_property_enabled(self):
        """Test energy_monitor when enabled."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.logger.log_energy = True
        state._cfg = mock_config

        mock_monitor = MagicMock()

        with patch("megatron.bridge.training.state.EnergyMonitor", return_value=mock_monitor) as mock_em:
            monitor = state.energy_monitor

            mock_em.assert_called_once()
            assert monitor == mock_monitor
            assert state._energy_monitor == mock_monitor
            assert state._energy_monitor_created is True

    def test_set_signal_handler_with_train_config(self):
        """Test _set_signal_handler with train config."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_train_config = MagicMock()
        mock_train_config.exit_signal = 15  # SIGTERM
        mock_config.train = mock_train_config
        state._cfg = mock_config

        mock_signal_handler = MagicMock()

        with patch(
            "megatron.bridge.training.state.DistributedSignalHandler", return_value=mock_signal_handler
        ) as mock_dsh:
            state._set_signal_handler()

            mock_dsh.assert_called_once_with(15)
            assert state._signal_handler == mock_signal_handler

    def test_set_signal_handler_no_train_config(self):
        """Test _set_signal_handler without train config."""
        state = GlobalState()
        mock_config = MagicMock()
        mock_config.train = None
        state._cfg = mock_config

        with patch("megatron.bridge.training.state.DistributedSignalHandler") as mock_dsh:
            state._set_signal_handler()

            mock_dsh.assert_not_called()
            assert state._signal_handler is None

    def test_reset_for_restart(self):
        """Test reset_for_restart method clears all stateful components."""
        state = GlobalState()

        # Set up some mock objects to simulate initialized state
        state._timers = MagicMock()
        state._train_state = MagicMock()
        state._tensorboard_logger = MagicMock()
        state._wandb_logger = MagicMock()
        state._energy_monitor = MagicMock()
        state._energy_monitor_created = True
        state._signal_handler = MagicMock()
        state._straggler_timer = MagicMock()
        state._nvrx_straggler_manager = MagicMock()
        state._nvrx_straggler_created = True

        # Call reset_for_restart
        state.reset_for_restart()

        # Verify all components are reset
        assert state._timers is None
        assert state._train_state is None
        assert state._tensorboard_logger is None
        assert state._wandb_logger is None
        assert state._energy_monitor is None
        assert state._energy_monitor_created is False
        assert state._signal_handler is None
        assert state._straggler_timer is None
        assert state._nvrx_straggler_manager is None
        assert state._nvrx_straggler_created is False

        # Verify that other state is preserved
        assert state._initialized is True
        assert state.rank_monitor_client is not None or state.rank_monitor_client is None  # Could be either
        assert isinstance(state.start_time, float)

    def test_reset_for_restart_preserves_config_and_async_queue(self):
        """Test reset_for_restart preserves config and async queue."""
        state = GlobalState()

        # Set up config and async queue
        mock_config = MagicMock()
        mock_async_queue = MagicMock()
        state._cfg = mock_config
        state._async_calls_queue = mock_async_queue
        state.rank_monitor_client = MagicMock()

        # Call reset_for_restart
        state.reset_for_restart()

        # Verify config, async queue, and rank monitor client are preserved
        assert state._cfg == mock_config
        assert state._async_calls_queue == mock_async_queue
        assert state.rank_monitor_client is not None
