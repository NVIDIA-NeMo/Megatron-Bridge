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

"""Tests for train module utility functions."""

import time
from unittest.mock import Mock, patch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from megatron.bridge.training.train import (
    _handle_mxfp8_param_buffer_copy,
    checkpoint_and_decide_exit,
    should_disable_forward_pre_hook,
)
from megatron.bridge.training.utils.train_utils import maybe_inject_state


class TestMxfp8ParamBufferCopy:
    """Unit tests for mxfp8 parameter buffer copying functionality."""

    def test_copy_main_params_called_when_both_flags_true(self):
        """Test that _copy_main_params_to_param_buffer is called when both config flags are True."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_other_optimizer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_other_optimizer,
            mock_distributed_optimizer,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_called_once()
        assert (
            not hasattr(mock_other_optimizer, "_copy_main_params_to_param_buffer")
            or not mock_other_optimizer._copy_main_params_to_param_buffer.called
        )

    def test_no_copy_when_reuse_grad_buf_false(self):
        """Test that no copying occurs when reuse_grad_buf_for_mxfp8_param_ag is False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=False, overlap_param_gather=True
        )
        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_no_copy_when_overlap_param_gather_false(self):
        """Test that no copying occurs when overlap_param_gather is False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]
        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=False
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_no_copy_when_both_flags_false(self):
        """Test that no copying occurs when both flags are False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=False, overlap_param_gather=False
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_handles_multiple_distributed_optimizers(self):
        """Test that function calls copy on multiple DistributedOptimizers."""
        mock_distributed_optimizer_1 = Mock(spec=DistributedOptimizer)
        mock_distributed_optimizer_2 = Mock(spec=DistributedOptimizer)
        mock_other_optimizer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_other_optimizer,
            mock_distributed_optimizer_1,
            mock_distributed_optimizer_2,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer_1._copy_main_params_to_param_buffer.assert_called_once()
        mock_distributed_optimizer_2._copy_main_params_to_param_buffer.assert_called_once()

    def test_only_calls_on_distributed_optimizers(self):
        """Test that only DistributedOptimizer instances get the copy call."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_regular_optimizer = Mock()  # Regular optimizer without _copy_main_params_to_param_buffer
        mock_different_optimizer = Mock()

        # Add the method to one non-DistributedOptimizer to ensure it's not called
        mock_different_optimizer._copy_main_params_to_param_buffer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_regular_optimizer,
            mock_different_optimizer,
            mock_distributed_optimizer,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_called_once()
        mock_different_optimizer._copy_main_params_to_param_buffer.assert_not_called()

        assert (
            not hasattr(mock_regular_optimizer, "_copy_main_params_to_param_buffer")
            or not mock_regular_optimizer._copy_main_params_to_param_buffer.called
        )


class TestShouldDisableForwardPreHook:
    """Unit tests for should_disable_forward_pre_hook function."""

    def test_disable_with_distributed_optimizer_and_overlap_no_fsdp(self):
        """Test that pre-hook is disabled when using distributed optimizer + overlap without FSDP."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=True, overlap_param_gather=True
        )
        assert result is True

    def test_keep_enabled_with_megatron_fsdp(self):
        """Test that pre-hook stays enabled when using Megatron FSDP."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=True, use_distributed_optimizer=True, overlap_param_gather=True
        )
        assert result is False

    def test_callable_class_state_injection_integration(self):
        """Integration test ensuring state injection works with functors in training context."""

        class ForwardFunctor:
            def __init__(self):
                self.state_seen = None

            def __call__(self, state, data_iterator, model, return_schedule_plan=False):
                self.state_seen = state
                return "ok"

        mock_state = Mock()
        functor = ForwardFunctor()

        wrapped = maybe_inject_state(functor, mock_state)
        assert callable(wrapped)

        data_iterator = Mock()
        model = Mock()
        result = wrapped(data_iterator, model, return_schedule_plan=True)

        assert result == "ok"
        assert functor.state_seen is mock_state

    def test_keep_enabled_without_distributed_optimizer(self):
        """Test that pre-hook stays enabled when not using distributed optimizer."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=False, overlap_param_gather=True
        )
        assert result is False

    def test_keep_enabled_without_overlap_param_gather(self):
        """Test that pre-hook stays enabled when not overlapping parameter gathering."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=True, overlap_param_gather=False
        )
        assert result is False

    def test_keep_enabled_all_false(self):
        """Test that pre-hook stays enabled when all conditions are false."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=False, use_distributed_optimizer=False, overlap_param_gather=False
        )
        assert result is False

    def test_keep_enabled_all_true_with_fsdp(self):
        """Test that pre-hook stays enabled when FSDP is used (even with other conditions true)."""
        result = should_disable_forward_pre_hook(
            use_megatron_fsdp=True, use_distributed_optimizer=True, overlap_param_gather=True
        )
        assert result is False


class TestCheckpointAndDecideExit:
    """Unit tests for checkpoint_and_decide_exit function"""

    def _create_mock_state(
        self,
        exit_duration_in_mins=None,
        start_time=None,
        checkpoint_save=False,
        checkpoint_save_interval=None,
        exit_signal_handler=False,
        exit_interval=None,
        step=0,
    ):
        """Helper method to create a mock state object with specified configuration."""
        mock_state = Mock()

        # Mock the configuration structure
        mock_state.cfg.train.exit_duration_in_mins = exit_duration_in_mins
        mock_state.cfg.train.exit_signal_handler = exit_signal_handler
        mock_state.cfg.train.exit_interval = exit_interval
        mock_state.cfg.checkpoint.save = checkpoint_save
        mock_state.cfg.checkpoint.save_interval = checkpoint_save_interval
        mock_state.cfg.checkpoint.non_persistent_save_interval = None

        # Mock train state
        mock_state.train_state.step = step

        # Mock start_time
        mock_state.start_time = start_time if start_time is not None else time.time()

        # Mock other required attributes
        mock_state.signal_handler = Mock()
        mock_state.signal_handler.signals_received.return_value = []
        mock_state.nvrx_straggler_manager = None

        return mock_state

    def _create_mock_args(self):
        """Helper method to create mock arguments for checkpoint_and_decide_exit."""
        return {
            "model": [Mock()],
            "optimizer": Mock(),
            "opt_param_scheduler": Mock(),
            "num_floating_point_operations_so_far": 1000.0,
            "checkpointing_context": {},
            "train_data_iterator": None,
        }

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    @patch("torch.distributed.all_reduce")
    @patch("time.time")
    def test_duration_exit_uses_correct_start_time(
        self, mock_time, mock_all_reduce, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint
    ):
        """Test that duration exit logic uses state.start_time, not state.train_state.start_time."""
        # Setup
        current_time = 1000.0
        start_time = 900.0  # 100 seconds ago
        exit_duration_mins = 1.0  # 1 minute threshold

        mock_time.return_value = current_time
        mock_check_nvrx.return_value = False

        # Create state with start_time set to a specific value
        state = self._create_mock_state(
            exit_duration_in_mins=exit_duration_mins, start_time=start_time, checkpoint_save=False
        )

        # Mock torch tensor operations
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1  # Simulate duration exceeded

        with patch("torch.tensor", return_value=mock_tensor):
            args = self._create_mock_args()
            result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify that the correct start_time was used in calculation
        # train_time should be (1000.0 - 900.0) / 60.0 = 1.67 minutes
        expected_train_time = (current_time - start_time) / 60.0
        assert expected_train_time > exit_duration_mins

        # Verify torch operations were called correctly
        mock_all_reduce.assert_called_once()
        mock_barrier_log.assert_called_once()

        # Verify the log message contains the correct calculated time
        log_call_args = mock_barrier_log.call_args[0][0]
        assert f"exiting program after {expected_train_time}" in log_call_args

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    @patch("torch.distributed.all_reduce")
    @patch("time.time")
    def test_duration_exit_under_threshold_continues_training(
        self, mock_time, mock_all_reduce, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint
    ):
        """Test that training continues when duration is under the threshold."""
        # Setup
        current_time = 1000.0
        start_time = 980.0  # 20 seconds ago
        exit_duration_mins = 1.0  # 1 minute threshold

        mock_time.return_value = current_time
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_duration_in_mins=exit_duration_mins, start_time=start_time, checkpoint_save=False
        )

        # Mock torch tensor operations - duration NOT exceeded
        mock_tensor = Mock()
        mock_tensor.item.return_value = 0  # Simulate duration not exceeded

        with patch("torch.tensor", return_value=mock_tensor):
            args = self._create_mock_args()
            result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue training)
        assert result is False

        # Verify torch operations were called
        mock_all_reduce.assert_called_once()

        # Verify no exit log message was called
        mock_barrier_log.assert_not_called()
        mock_save_checkpoint.assert_not_called()

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    @patch("torch.distributed.all_reduce")
    @patch("time.time")
    def test_duration_exit_with_checkpoint_saving(
        self, mock_time, mock_all_reduce, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint
    ):
        """Test that checkpoint is saved when exiting due to duration and checkpointing is enabled."""
        # Setup
        current_time = 1000.0
        start_time = 900.0  # 100 seconds ago
        exit_duration_mins = 1.0  # 1 minute threshold

        mock_time.return_value = current_time
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_duration_in_mins=exit_duration_mins,
            start_time=start_time,
            checkpoint_save=True,  # Enable checkpoint saving
        )

        # Mock torch tensor operations
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1  # Simulate duration exceeded

        with patch("torch.tensor", return_value=mock_tensor):
            args = self._create_mock_args()
            result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify checkpoint was saved
        mock_save_checkpoint.assert_called_once()

        # Verify the correct arguments were passed to save_checkpoint_and_time
        save_call_args = mock_save_checkpoint.call_args
        assert save_call_args[0][0] == state  # state argument
        assert save_call_args[0][1] == args["model"]  # model argument

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    @patch("torch.distributed.all_reduce")
    @patch("time.time")
    def test_duration_exit_no_checkpoint_when_already_saved(
        self, mock_time, mock_all_reduce, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint
    ):
        """Test that no additional checkpoint is saved if one was already saved in the same iteration."""
        # Setup
        current_time = 1000.0
        start_time = 900.0  # 100 seconds ago
        exit_duration_mins = 1.0  # 1 minute threshold

        mock_time.return_value = current_time
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_duration_in_mins=exit_duration_mins,
            start_time=start_time,
            checkpoint_save=True,
            checkpoint_save_interval=1,  # This will cause a checkpoint to be saved first
        )

        # Mock torch tensor operations
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1  # Simulate duration exceeded

        with patch("torch.tensor", return_value=mock_tensor):
            args = self._create_mock_args()
            result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify checkpoint was saved only once (for the regular interval, not for exit)
        assert mock_save_checkpoint.call_count == 1

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    @patch("torch.distributed.all_reduce")
    @patch("time.time")
    def test_no_duration_exit_when_disabled(
        self, mock_time, mock_all_reduce, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint
    ):
        """Test that duration-based exit is skipped when exit_duration_in_mins is not set."""
        # Setup
        current_time = 1000.0
        start_time = 900.0  # 100 seconds ago

        mock_time.return_value = current_time
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_duration_in_mins=None,  # Disabled
            start_time=start_time,
            checkpoint_save=False,
        )

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue training)
        assert result is False

        # Verify no torch operations were called for duration checking
        mock_all_reduce.assert_not_called()
        mock_barrier_log.assert_not_called()
        mock_save_checkpoint.assert_not_called()

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    @patch("torch.distributed.all_reduce")
    @patch("time.time")
    def test_duration_calculation_precision(
        self, mock_time, mock_all_reduce, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint
    ):
        """Test that duration calculation handles edge cases and precision correctly."""
        # Setup - test with fractional minutes
        current_time = 1090.5  # Fractional seconds
        start_time = 1000.0
        exit_duration_mins = 1.5  # 1.5 minutes threshold

        mock_time.return_value = current_time
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_duration_in_mins=exit_duration_mins, start_time=start_time, checkpoint_save=False
        )

        # Mock torch tensor operations - should NOT exceed threshold
        # (1090.5 - 1000.0) / 60.0 = 1.508 minutes, which is > 1.5
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1  # Simulate duration exceeded

        with patch("torch.tensor", return_value=mock_tensor) as mock_torch_tensor:
            args = self._create_mock_args()
            result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify the tensor was created with the correct comparison
        # The tensor should contain [True] since 1.508 > 1.5
        tensor_call_args = mock_torch_tensor.call_args[0][0]  # First positional argument
        expected_train_time = (current_time - start_time) / 60.0
        assert tensor_call_args == [expected_train_time > exit_duration_mins]

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_signal_handler_exit_with_signals(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test exit when signal handler is enabled and signals are received."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(exit_signal_handler=True, checkpoint_save=True)

        # Mock signal handler to return received signals
        state.signal_handler.signals_received.return_value = ["SIGTERM"]

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify checkpoint was saved and correct log message
        mock_save_checkpoint.assert_called_once()
        mock_barrier_log.assert_called_once_with("exiting program after receiving SIGTERM.")

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_signal_handler_exit_no_signals(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test that training continues when signal handler is enabled but no signals received."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(exit_signal_handler=True, checkpoint_save=True)

        # Mock signal handler to return no signals
        state.signal_handler.signals_received.return_value = []

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue)
        assert result is False

        # Verify no exit-related actions were taken
        mock_barrier_log.assert_not_called()

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_iteration_interval_exit(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test exit when iteration interval is reached."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_interval=10,
            step=20,  # 20 % 10 == 0, should trigger exit
            checkpoint_save=True,
        )

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify checkpoint was saved and correct log message
        mock_save_checkpoint.assert_called_once()
        mock_barrier_log.assert_called_once_with("exiting program at iteration 20")

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_iteration_interval_not_reached(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test that training continues when iteration interval is not reached."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_interval=10,
            step=15,  # 15 % 10 != 0, should not trigger exit
            checkpoint_save=True,
        )

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue)
        assert result is False

        # Verify no exit-related actions were taken
        mock_barrier_log.assert_not_called()

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_nvrx_straggler_detection_exit(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test exit when NVRx straggler detection triggers."""
        mock_check_nvrx.return_value = True

        state = self._create_mock_state(checkpoint_save=True)

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify checkpoint was saved and correct log message
        mock_save_checkpoint.assert_called_once()
        mock_barrier_log.assert_called_once_with("Exiting program due to straggler detection.")

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_regular_persistent_checkpoint_save(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test regular persistent checkpoint saving."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            checkpoint_save=True,
            checkpoint_save_interval=5,
            step=10,  # 10 % 5 == 0, should save checkpoint
        )

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue training)
        assert result is False

        # Verify checkpoint was saved
        mock_save_checkpoint.assert_called_once()

        # Verify train_data_iterator was passed correctly
        save_call_args = mock_save_checkpoint.call_args
        assert "train_data_iterator" in save_call_args[1]
        assert "non_persistent_ckpt" not in save_call_args[1]  # Should not be non-persistent

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_regular_non_persistent_checkpoint_save(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test regular non-persistent checkpoint saving."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            checkpoint_save=True,
            checkpoint_save_interval=None,  # No persistent saves
            step=15,
        )

        # Set non-persistent save interval
        state.cfg.checkpoint.non_persistent_save_interval = 3
        # 15 % 3 == 0, should save non-persistent checkpoint

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue training)
        assert result is False

        # Verify checkpoint was saved with non_persistent_ckpt=True
        mock_save_checkpoint.assert_called_once()
        save_call_args = mock_save_checkpoint.call_args
        assert save_call_args[1]["non_persistent_ckpt"] is True

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_no_checkpoint_when_disabled(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test that no checkpoint is saved when checkpointing is disabled."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            checkpoint_save=False,  # Disabled
            checkpoint_save_interval=5,
            step=10,  # Would normally trigger save
        )

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue training)
        assert result is False

        # Verify no checkpoint was saved
        mock_save_checkpoint.assert_not_called()

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_multiple_exit_conditions_signal_wins(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test that signal handler exit takes precedence over other exit conditions."""
        mock_check_nvrx.return_value = True  # Straggler detection would also trigger

        state = self._create_mock_state(
            exit_signal_handler=True,
            exit_interval=10,
            step=10,  # Would also trigger iteration exit
            checkpoint_save=True,
        )

        # Mock signal handler to return received signals
        state.signal_handler.signals_received.return_value = ["SIGTERM"]

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # Verify signal handler message was logged, not straggler message
        mock_barrier_log.assert_called_once_with("exiting program after receiving SIGTERM.")

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_checkpoint_save_none_vs_false(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test the difference between checkpoint.save being None vs False for straggler detection."""
        mock_check_nvrx.return_value = True

        state = self._create_mock_state()
        state.cfg.checkpoint.save = None  # Explicitly set to None

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns True (should exit)
        assert result is True

        # For straggler detection, checkpoint should NOT be saved when save is None
        # (note the condition: state.cfg.checkpoint.save is not None)
        mock_save_checkpoint.assert_not_called()
        mock_barrier_log.assert_called_once_with("Exiting program due to straggler detection.")

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_no_exit_conditions_met(self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint):
        """Test that function returns False when no exit conditions are met."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            exit_signal_handler=False,
            exit_duration_in_mins=None,
            exit_interval=None,
            checkpoint_save=False,
            step=7,  # Doesn't match any interval
        )

        # Ensure no signals are received
        state.signal_handler.signals_received.return_value = []

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue training)
        assert result is False

        # Verify no actions were taken
        mock_save_checkpoint.assert_not_called()
        mock_barrier_log.assert_not_called()

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection")
    def test_persistent_checkpoint_priority_over_non_persistent(
        self, mock_check_nvrx, mock_barrier_log, mock_save_checkpoint
    ):
        """Test that persistent checkpoint takes priority over non-persistent when both intervals match."""
        mock_check_nvrx.return_value = False

        state = self._create_mock_state(
            checkpoint_save=True,
            checkpoint_save_interval=5,  # Persistent every 5 steps
            step=10,  # 10 % 5 == 0, would trigger persistent
        )

        # Set non-persistent save interval that would also trigger
        state.cfg.checkpoint.non_persistent_save_interval = 2  # 10 % 2 == 0, would also trigger

        args = self._create_mock_args()
        result = checkpoint_and_decide_exit(state, **args)

        # Verify the function returns False (should continue training)
        assert result is False

        # Verify checkpoint was saved exactly once
        mock_save_checkpoint.assert_called_once()

        # Verify it was a PERSISTENT checkpoint (no non_persistent_ckpt flag)
        save_call_args = mock_save_checkpoint.call_args
        assert (
            "non_persistent_ckpt" not in save_call_args[1] or save_call_args[1].get("non_persistent_ckpt") is not True
        )

        # Verify the persistent checkpoint arguments were used
        assert save_call_args[0][0] == state  # state argument
        assert save_call_args[0][1] == args["model"]  # model argument
        assert "train_data_iterator" in save_call_args[1]
