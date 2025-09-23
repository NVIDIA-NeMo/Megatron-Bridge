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

import json
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.state import FaultToleranceState, GlobalState


class TestConstants:
    """Test module constants."""

    def test_constants_values(self):
        """Test that constants have expected values."""
        assert fault_tolerance._NUM_WARMUP_ITERS == 1
        assert fault_tolerance._MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE == 16


class TestSetup:
    """Test setup function."""

    def test_setup_success(self):
        """Test successful fault tolerance setup."""
        # Create mock objects
        mock_config = MagicMock()
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.checkpoint.async_save = True
        mock_config.ft.calc_ft_timeouts = True

        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.fault_tolerance_state = MagicMock(spec=FaultToleranceState)
        mock_global_state.rank_monitor_client = None  # Initially no client

        mock_rank_monitor_client = MagicMock()
        mock_rank_monitor_client.section_timeouts = {"setup": 600, "step": 180}

        with (
            patch("megatron.bridge.training.fault_tolerance.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
            patch("nvidia_resiliency_ext.fault_tolerance.RankMonitorClient", return_value=mock_rank_monitor_client),
            patch("megatron.bridge.training.fault_tolerance._load_state_if_exists") as mock_load_state,
        ):
            fault_tolerance.setup(mock_config, mock_global_state)

            # Verify setup calls
            mock_makedirs.assert_called_once_with("/tmp/checkpoints", exist_ok=True)
            assert mock_global_state.rank_monitor_client == mock_rank_monitor_client
            assert mock_global_state.fault_tolerance_state.ft_state_path == "/tmp/checkpoints/ft_state.json"
            assert mock_global_state.fault_tolerance_state.is_async_chkpt_enabled is True
            assert mock_global_state.fault_tolerance_state.is_calculating_timeouts is True

            mock_rank_monitor_client.init_workload_monitoring.assert_called_once()
            mock_load_state.assert_called_once_with(mock_global_state)
            mock_rank_monitor_client.start_section.assert_called_once_with("setup")
            assert mock_global_state.fault_tolerance_state.is_setup_section_open is True

    def test_setup_no_checkpoint_dir(self):
        """Test setup failure when no checkpoint directory is configured."""
        mock_config = MagicMock()
        mock_config.checkpoint.save = None

        mock_global_state = MagicMock(spec=GlobalState)

        with pytest.raises(ValueError, match="checkpointing save dir must be set"):
            fault_tolerance.setup(mock_config, mock_global_state)

    def test_setup_checkpoint_dir_exists(self):
        """Test setup when checkpoint directory already exists."""
        mock_config = MagicMock()
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.checkpoint.async_save = False
        mock_config.ft.calc_ft_timeouts = False

        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.fault_tolerance_state = MagicMock(spec=FaultToleranceState)
        mock_global_state.rank_monitor_client = None  # Initially no client

        mock_rank_monitor_client = MagicMock()

        with (
            patch("megatron.bridge.training.fault_tolerance.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("os.path.exists", return_value=True),
            patch("os.makedirs") as mock_makedirs,
            patch("nvidia_resiliency_ext.fault_tolerance.RankMonitorClient", return_value=mock_rank_monitor_client),
            patch("megatron.bridge.training.fault_tolerance._load_state_if_exists"),
        ):
            fault_tolerance.setup(mock_config, mock_global_state)

            # Should not create directory if it exists
            mock_makedirs.assert_not_called()

    def test_setup_non_rank_zero(self):
        """Test setup from non-rank-zero process."""
        mock_config = MagicMock()
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.checkpoint.async_save = False
        mock_config.ft.calc_ft_timeouts = False

        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.fault_tolerance_state = MagicMock(spec=FaultToleranceState)
        mock_global_state.rank_monitor_client = None  # Initially no client

        mock_rank_monitor_client = MagicMock()

        with (
            patch("megatron.bridge.training.fault_tolerance.get_rank_safe", return_value=1),
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
            patch("nvidia_resiliency_ext.fault_tolerance.RankMonitorClient", return_value=mock_rank_monitor_client),
            patch("megatron.bridge.training.fault_tolerance._load_state_if_exists"),
        ):
            fault_tolerance.setup(mock_config, mock_global_state)

            # Non-rank-zero should not create directory
            mock_makedirs.assert_not_called()


class TestTrainingStepCallbacks:
    """Test training step callback functions."""

    def test_on_training_step_start_first_call(self):
        """Test on_training_step_start on first call with setup section open."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = True
        mock_ft_state.seen_tr_iters_cnt = 0
        mock_ft_state.curr_eval_iter_idx = 5
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_training_step_start(mock_global_state)

        # Should close setup section and reset eval iter
        mock_rank_monitor_client.end_section.assert_called_once_with("setup")
        assert mock_ft_state.is_setup_section_open is False
        assert mock_ft_state.curr_eval_iter_idx == 0

        # Should not start step section yet (warmup)
        mock_rank_monitor_client.start_section.assert_not_called()

    def test_on_training_step_start_after_warmup(self):
        """Test on_training_step_start after warmup iterations."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = False
        mock_ft_state.seen_tr_iters_cnt = 5  # > _NUM_WARMUP_ITERS
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_training_step_start(mock_global_state)

        # Should start step section
        mock_rank_monitor_client.start_section.assert_called_once_with("step")
        mock_rank_monitor_client.end_section.assert_not_called()

    def test_on_training_step_start_no_client(self):
        """Test on_training_step_start when no rank monitor client."""
        mock_global_state = MagicMock()
        mock_global_state.rank_monitor_client = None

        # Should not raise exception
        fault_tolerance.on_training_step_start(mock_global_state)

    def test_on_training_step_end_warmup_period(self):
        """Test on_training_step_end during warmup period."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.seen_tr_iters_cnt = 0  # < _NUM_WARMUP_ITERS
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_training_step_end(mock_global_state)

        # Should not end step section during warmup
        mock_rank_monitor_client.end_section.assert_not_called()
        # Should increment counter
        assert mock_ft_state.seen_tr_iters_cnt == 1

    def test_on_training_step_end_after_warmup(self):
        """Test on_training_step_end after warmup period."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.seen_tr_iters_cnt = 5  # >= _NUM_WARMUP_ITERS
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_training_step_end(mock_global_state)

        # Should end step section
        mock_rank_monitor_client.end_section.assert_called_once_with("step")
        # Should increment counter
        assert mock_ft_state.seen_tr_iters_cnt == 6


class TestEvaluationStepCallbacks:
    """Test evaluation step callback functions."""

    def test_on_eval_step_start_first_call(self):
        """Test on_eval_step_start on first call with setup section open."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = True
        mock_ft_state.curr_eval_iter_idx = 0
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_eval_step_start(mock_global_state)

        # Should close setup section
        mock_rank_monitor_client.end_section.assert_called_once_with("setup")
        assert mock_ft_state.is_setup_section_open is False

        # Should not start step section yet (warmup)
        mock_rank_monitor_client.start_section.assert_not_called()

    def test_on_eval_step_start_after_warmup(self):
        """Test on_eval_step_start after warmup iterations."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = False
        mock_ft_state.curr_eval_iter_idx = 5  # > _NUM_WARMUP_ITERS
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_eval_step_start(mock_global_state)

        # Should start step section
        mock_rank_monitor_client.start_section.assert_called_once_with("step")
        mock_rank_monitor_client.end_section.assert_not_called()

    def test_on_eval_step_start_setup_section_before_eval(self):
        """Test on_eval_step_start when setup section is open before evaluation."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = True
        mock_ft_state.curr_eval_iter_idx = 5  # > _NUM_WARMUP_ITERS
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_eval_step_start(mock_global_state)

        # Should close setup section and start step section
        mock_rank_monitor_client.end_section.assert_called_once_with("setup")
        mock_rank_monitor_client.start_section.assert_called_once_with("step")
        assert mock_ft_state.is_setup_section_open is False

    def test_on_eval_step_start_no_client(self):
        """Test on_eval_step_start when no rank monitor client."""
        mock_global_state = MagicMock()
        mock_global_state.rank_monitor_client = None

        # Should not raise exception
        fault_tolerance.on_eval_step_start(mock_global_state)

    def test_on_eval_step_end_warmup_period(self):
        """Test on_eval_step_end during warmup period."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.curr_eval_iter_idx = 0  # < _NUM_WARMUP_ITERS
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_eval_step_end(mock_global_state)

        # Should not end step section during warmup
        mock_rank_monitor_client.end_section.assert_not_called()
        # Should increment counter
        assert mock_ft_state.curr_eval_iter_idx == 1

    def test_on_eval_step_end_after_warmup(self):
        """Test on_eval_step_end after warmup period."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.curr_eval_iter_idx = 5  # >= _NUM_WARMUP_ITERS
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_eval_step_end(mock_global_state)

        # Should end step section
        mock_rank_monitor_client.end_section.assert_called_once_with("step")
        # Should increment counter
        assert mock_ft_state.curr_eval_iter_idx == 6

    def test_on_eval_step_end_no_client(self):
        """Test on_eval_step_end when no rank monitor client."""
        mock_global_state = MagicMock()
        mock_global_state.rank_monitor_client = None

        mock_ft_state = MagicMock()
        mock_ft_state.curr_eval_iter_idx = 0
        mock_global_state.fault_tolerance_state = mock_ft_state

        # Should not raise exception
        fault_tolerance.on_eval_step_end(mock_global_state)
        # Should not increment counter when no client
        assert mock_ft_state.curr_eval_iter_idx == 0

    def test_eval_step_workflow(self):
        """Test complete evaluation step workflow."""
        mock_global_state = MagicMock()
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = True
        mock_ft_state.curr_eval_iter_idx = 0
        mock_global_state.fault_tolerance_state = mock_ft_state

        # Simulate several evaluation steps
        for i in range(3):
            fault_tolerance.on_eval_step_start(mock_global_state)
            fault_tolerance.on_eval_step_end(mock_global_state)

        # call should close setup section
        mock_rank_monitor_client.end_section.assert_any_call("setup")
        # After warmup, should start/end step sections
        assert mock_rank_monitor_client.start_section.call_count == 2  # Steps 2 and 3
        assert mock_rank_monitor_client.end_section.call_count == 3  # Setup + Steps 2 and 3
        # Should increment eval iter counter to 3
        assert mock_ft_state.curr_eval_iter_idx == 3


class TestCheckpointingCallbacks:
    """Test checkpointing callback functions."""

    def test_on_checkpointing_start(self):
        """Test on_checkpointing_start callback."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        fault_tolerance.on_checkpointing_start(mock_global_state)

        mock_rank_monitor_client.start_section.assert_called_once_with("checkpointing")

    def test_on_checkpointing_start_no_client(self):
        """Test on_checkpointing_start with no client."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.rank_monitor_client = None

        # Should not raise exception
        fault_tolerance.on_checkpointing_start(mock_global_state)

    def test_on_checkpointing_end_sync(self):
        """Test on_checkpointing_end for synchronous checkpoint."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.seen_checkpoints_cnt = 0
        mock_global_state.fault_tolerance_state = mock_ft_state

        with patch("megatron.bridge.training.fault_tolerance._maybe_update_timeouts") as mock_update:
            fault_tolerance.on_checkpointing_end(is_async_finalization=False, global_state=mock_global_state)

            mock_rank_monitor_client.end_section.assert_called_once_with("checkpointing")
            assert mock_ft_state.seen_checkpoints_cnt == 1
            mock_update.assert_called_once_with(mock_global_state)

    def test_on_checkpointing_end_async(self):
        """Test on_checkpointing_end for async checkpoint finalization."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.seen_checkpoints_cnt = 0
        mock_global_state.fault_tolerance_state = mock_ft_state

        with patch("megatron.bridge.training.fault_tolerance._maybe_update_timeouts") as mock_update:
            fault_tolerance.on_checkpointing_end(is_async_finalization=True, global_state=mock_global_state)

            mock_rank_monitor_client.end_section.assert_called_once_with("checkpointing")
            # Should not increment counter or update timeouts for async finalization
            assert mock_ft_state.seen_checkpoints_cnt == 0
            mock_update.assert_not_called()

    def test_on_checkpoint_loaded_persistent(self):
        """Test on_checkpoint_loaded with persistent checkpoint."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_checkpoint_loaded(is_local_chkpt=False, global_state=mock_global_state)

        assert mock_ft_state.is_persistent_chkpt_loaded is True

    def test_on_checkpoint_loaded_local(self):
        """Test on_checkpoint_loaded with local checkpoint."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_global_state.fault_tolerance_state = mock_ft_state

        fault_tolerance.on_checkpoint_loaded(is_local_chkpt=True, global_state=mock_global_state)

        assert mock_ft_state.is_persistent_chkpt_loaded is False


class TestShutdown:
    """Test shutdown function."""

    def test_shutdown_with_client(self):
        """Test shutdown when rank monitor client exists."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("megatron.bridge.training.fault_tolerance._maybe_update_timeouts") as mock_update,
        ):
            fault_tolerance.shutdown(mock_global_state)

            mock_update.assert_called_once_with(mock_global_state, is_closing_ft=True)
            mock_rank_monitor_client.shutdown_workload_monitoring.assert_called_once()
            assert mock_global_state.rank_monitor_client is None

    def test_shutdown_no_client(self):
        """Test shutdown when no rank monitor client."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.rank_monitor_client = None

        # Should not raise exception
        fault_tolerance.shutdown(mock_global_state)
        assert mock_global_state.rank_monitor_client is None


class TestSimulatedFault:
    """Test simulated fault functionality."""

    def test_maybe_setup_simulated_fault_disabled(self):
        """Test when simulated fault is disabled."""
        mock_config = MagicMock()
        mock_config.simulate_fault = False

        with patch("megatron.bridge.training.fault_tolerance.print_rank_0"):
            fault_tolerance.maybe_setup_simulated_fault(mock_config)
            # Should return early, no other operations

    def test_maybe_setup_simulated_fault_rank_killed(self):
        """Test simulated fault with rank_killed type."""
        mock_config = MagicMock()
        mock_config.simulate_fault = True
        mock_config.simulated_fault_type = "rank_killed"
        mock_config.simulated_fault_rank = 1
        mock_config.simulated_fault_base_delay = 10.0

        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 1

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_world_size", return_value=4),
            patch("torch.tensor", return_value=mock_tensor),
            patch("torch.distributed.broadcast"),
            patch("torch.cuda.current_device", return_value=0),
            patch("threading.Thread") as mock_thread_class,
            patch("os.getpid", return_value=12345),
        ):
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread

            fault_tolerance.maybe_setup_simulated_fault(mock_config)

            # Should create and start thread
            mock_thread_class.assert_called_once()
            assert mock_thread.daemon is True
            mock_thread.start.assert_called_once()

    def test_maybe_setup_simulated_fault_not_target_rank(self):
        """Test simulated fault when current rank is not the target rank."""
        mock_config = MagicMock()
        mock_config.simulate_fault = True
        mock_config.simulated_fault_type = "rank_killed"
        mock_config.simulated_fault_rank = 1
        mock_config.simulated_fault_base_delay = 10.0

        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 1

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=4),
            patch("torch.tensor", return_value=mock_tensor),
            patch("torch.distributed.broadcast"),
            patch("torch.cuda.current_device", return_value=0),
            patch("threading.Thread") as mock_thread_class,
        ):
            fault_tolerance.maybe_setup_simulated_fault(mock_config)

            # Should not create thread if not target rank
            mock_thread_class.assert_not_called()

    def test_maybe_setup_simulated_fault_invalid_type(self):
        """Test simulated fault with invalid fault type."""
        mock_config = MagicMock()
        mock_config.simulate_fault = True
        mock_config.simulated_fault_type = "invalid_type"
        mock_config.simulated_fault_rank = 0
        mock_config.simulated_fault_base_delay = 10.0

        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 0

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=4),
            patch("torch.tensor", return_value=mock_tensor),
            patch("torch.distributed.broadcast"),
            patch("torch.cuda.current_device", return_value=0),
        ):
            with pytest.raises(Exception, match="Unknown fault type invalid_type"):
                fault_tolerance.maybe_setup_simulated_fault(mock_config)


class TestPrivateFunctions:
    """Test private helper functions."""

    def test_load_state_if_exists_file_exists(self):
        """Test _load_state_if_exists when file exists."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_rank_monitor_client = MagicMock()
        mock_rank_monitor_client.section_timeouts = {"setup": 600}
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.ft_state_path = "/tmp/ft_state.json"
        mock_global_state.fault_tolerance_state = mock_ft_state

        mock_state_data = {"section_timeouts": {"setup": 600, "step": 180}}

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_state_data))),
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
        ):
            fault_tolerance._load_state_if_exists(mock_global_state)

            mock_rank_monitor_client.load_state_dict.assert_called_once_with(mock_state_data)

    def test_load_state_if_exists_file_not_exists(self):
        """Test _load_state_if_exists when file does not exist."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_rank_monitor_client = MagicMock()
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.ft_state_path = "/tmp/ft_state.json"
        mock_global_state.fault_tolerance_state = mock_ft_state

        with patch("os.path.exists", return_value=False):
            fault_tolerance._load_state_if_exists(mock_global_state)

            # Should not attempt to load
            mock_rank_monitor_client.load_state_dict.assert_not_called()

    def test_update_timeouts(self):
        """Test _update_timeouts function."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_rank_monitor_client = MagicMock()
        mock_rank_monitor_client.state_dict.return_value = {"timeouts": "data"}
        mock_rank_monitor_client.section_timeouts = {"setup": 600}
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.ft_state_path = "/tmp/ft_state.json"
        mock_global_state.fault_tolerance_state = mock_ft_state

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("megatron.bridge.training.fault_tolerance.get_rank_safe", return_value=0),
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
        ):
            fault_tolerance._update_timeouts(
                selected_sections=["setup", "step"], calc_out_of_section=True, global_state=mock_global_state
            )

            mock_rank_monitor_client.calculate_and_set_section_timeouts.assert_called_once_with(
                selected_sections=["setup", "step"], calc_out_of_section=True
            )
            mock_json_dump.assert_called_once_with({"timeouts": "data"}, mock_file())


class TestMaybeUpdateTimeouts:
    """Test _maybe_update_timeouts function."""

    def test_maybe_update_timeouts_no_client(self):
        """Test _maybe_update_timeouts when no client."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.rank_monitor_client = None

        with patch("megatron.bridge.training.fault_tolerance._update_timeouts") as mock_update:
            fault_tolerance._maybe_update_timeouts(mock_global_state)
            mock_update.assert_not_called()

    def test_maybe_update_timeouts_not_calculating(self):
        """Test _maybe_update_timeouts when not calculating timeouts."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.rank_monitor_client = MagicMock()

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.is_calculating_timeouts = False
        mock_global_state.fault_tolerance_state = mock_ft_state

        with patch("megatron.bridge.training.fault_tolerance._update_timeouts") as mock_update:
            fault_tolerance._maybe_update_timeouts(mock_global_state)
            mock_update.assert_not_called()

    def test_maybe_update_timeouts_all_conditions_met(self):
        """Test _maybe_update_timeouts when all conditions are met."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.rank_monitor_client = MagicMock()

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.is_calculating_timeouts = True
        mock_ft_state.is_persistent_chkpt_loaded = True
        mock_ft_state.seen_tr_iters_cnt = 20  # >= _MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE
        mock_ft_state.seen_checkpoints_cnt = 3
        mock_ft_state.is_async_chkpt_enabled = False
        mock_global_state.fault_tolerance_state = mock_ft_state

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("megatron.bridge.training.fault_tolerance._update_timeouts") as mock_update,
        ):
            fault_tolerance._maybe_update_timeouts(mock_global_state)

            mock_update.assert_called_once_with(
                selected_sections=["setup", "step", "checkpointing"],
                calc_out_of_section=False,
                global_state=mock_global_state,
            )

    def test_maybe_update_timeouts_async_checkpointing(self):
        """Test _maybe_update_timeouts with async checkpointing enabled."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.rank_monitor_client = MagicMock()

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.is_calculating_timeouts = True
        mock_ft_state.is_persistent_chkpt_loaded = True
        mock_ft_state.seen_tr_iters_cnt = 20
        mock_ft_state.seen_checkpoints_cnt = 3
        mock_ft_state.is_async_chkpt_enabled = True  # Async enabled
        mock_global_state.fault_tolerance_state = mock_ft_state

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("megatron.bridge.training.fault_tolerance._update_timeouts") as mock_update,
        ):
            fault_tolerance._maybe_update_timeouts(mock_global_state)

            # Should not include checkpointing section with async enabled
            mock_update.assert_called_once_with(
                selected_sections=["setup", "step"], calc_out_of_section=False, global_state=mock_global_state
            )

    def test_maybe_update_timeouts_closing_ft(self):
        """Test _maybe_update_timeouts when closing FT."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.rank_monitor_client = MagicMock()

        mock_ft_state = MagicMock(spec=FaultToleranceState)
        mock_ft_state.is_calculating_timeouts = True
        mock_ft_state.is_persistent_chkpt_loaded = True
        mock_ft_state.seen_tr_iters_cnt = 20
        mock_ft_state.seen_checkpoints_cnt = 3
        mock_ft_state.is_async_chkpt_enabled = True
        mock_global_state.fault_tolerance_state = mock_ft_state

        with (
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("megatron.bridge.training.fault_tolerance._update_timeouts") as mock_update,
        ):
            fault_tolerance._maybe_update_timeouts(mock_global_state, is_closing_ft=True)

            # Should update out-of-section when closing
            mock_update.assert_called_once_with(
                selected_sections=["setup", "step"], calc_out_of_section=True, global_state=mock_global_state
            )


class TestTrainingIntegration:
    """Test training integration."""

    def test_complete_training_workflow(self):
        """Test a complete training workflow with fault tolerance."""
        # Setup
        mock_config = MagicMock()
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.checkpoint.async_save = False
        mock_config.ft.calc_ft_timeouts = True

        mock_global_state = MagicMock()
        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = True
        mock_ft_state.seen_tr_iters_cnt = 0
        mock_ft_state.curr_eval_iter_idx = 0
        mock_ft_state.seen_checkpoints_cnt = 0
        mock_global_state.fault_tolerance_state = mock_ft_state

        # Create separate mock clients for existing and new clients
        mock_existing_client = MagicMock()
        mock_existing_client.is_initialized = True
        mock_global_state.rank_monitor_client = mock_existing_client

        mock_new_rank_monitor_client = MagicMock()
        mock_new_rank_monitor_client.section_timeouts = {"setup": 600, "step": 180}

        with (
            patch("megatron.bridge.training.fault_tolerance.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("os.path.exists", return_value=True),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.RankMonitorClient", return_value=mock_new_rank_monitor_client
            ),
            patch("megatron.bridge.training.fault_tolerance._load_state_if_exists"),
            patch("megatron.bridge.training.fault_tolerance._maybe_update_timeouts"),
        ):
            # Initialize fault tolerance
            fault_tolerance.setup(mock_config, mock_global_state)

            # Simulate training steps
            for i in range(3):
                fault_tolerance.on_training_step_start(mock_global_state)
                fault_tolerance.on_training_step_end(mock_global_state)

            # Simulate checkpointing
            fault_tolerance.on_checkpointing_start(mock_global_state)
            fault_tolerance.on_checkpointing_end(is_async_finalization=False, global_state=mock_global_state)

            # Shutdown
            fault_tolerance.shutdown(mock_global_state)

            # Verify the sequence of calls
            expected_start_calls = [
                call("setup"),  # Initial setup
                call("step"),  # Training steps after warmup
                call("step"),
                call("checkpointing"),
            ]

            mock_new_rank_monitor_client.start_section.assert_has_calls(expected_start_calls, any_order=False)

            # Verify shutdown calls - existing client shutdown during setup, new client shutdown during shutdown
            mock_existing_client.shutdown_workload_monitoring.assert_called_once()
            mock_new_rank_monitor_client.shutdown_workload_monitoring.assert_called_once()

    def test_complete_training_with_evaluation_workflow(self):
        """Test a complete training and evaluation workflow with fault tolerance."""
        # Setup
        mock_config = MagicMock()
        mock_config.checkpoint.save = "/tmp/checkpoints"
        mock_config.checkpoint.async_save = False
        mock_config.ft.calc_ft_timeouts = True

        mock_global_state = MagicMock()
        mock_ft_state = MagicMock()
        mock_ft_state.is_setup_section_open = True
        mock_ft_state.seen_tr_iters_cnt = 0
        mock_ft_state.curr_eval_iter_idx = 0
        mock_ft_state.seen_checkpoints_cnt = 0
        mock_global_state.fault_tolerance_state = mock_ft_state

        mock_rank_monitor_client = MagicMock()
        mock_rank_monitor_client.section_timeouts = {"setup": 600, "step": 180}
        mock_global_state.rank_monitor_client = mock_rank_monitor_client

        with (
            patch("megatron.bridge.training.fault_tolerance.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.fault_tolerance.print_rank_0"),
            patch("os.path.exists", return_value=True),
            patch("nvidia_resiliency_ext.fault_tolerance.RankMonitorClient", return_value=mock_rank_monitor_client),
            patch("megatron.bridge.training.fault_tolerance._load_state_if_exists"),
            patch("megatron.bridge.training.fault_tolerance._maybe_update_timeouts"),
        ):
            # Initialize fault tolerance
            fault_tolerance.setup(mock_config, mock_global_state)

            # Simulate training steps
            fault_tolerance.on_training_step_start(mock_global_state)
            fault_tolerance.on_training_step_end(mock_global_state)

            # Simulate evaluation steps (should reset curr_eval_iter_idx)
            for i in range(3):
                fault_tolerance.on_eval_step_start(mock_global_state)
                fault_tolerance.on_eval_step_end(mock_global_state)

            # More training steps
            fault_tolerance.on_training_step_start(mock_global_state)
            fault_tolerance.on_training_step_end(mock_global_state)

            # Simulate checkpointing
            fault_tolerance.on_checkpointing_start(mock_global_state)
            fault_tolerance.on_checkpointing_end(is_async_finalization=False, global_state=mock_global_state)

            # Shutdown
            fault_tolerance.shutdown(mock_global_state)

            # Verify eval counter was reset when training started again
            assert mock_ft_state.curr_eval_iter_idx == 0

            # Verify the sequence of calls includes both training and eval steps
            expected_start_calls = [
                call("setup"),  # Initial setup
                call("step"),  # Eval steps after warmup (1st eval)
                call("step"),  # 2nd eval step
                call("step"),  # 2nd training step (after eval)
                call("checkpointing"),
            ]

            mock_rank_monitor_client.start_section.assert_has_calls(expected_start_calls, any_order=False)
