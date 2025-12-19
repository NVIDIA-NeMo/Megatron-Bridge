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

"""Tests for eval module functions."""

from unittest.mock import Mock, patch


class TestEvaluateCudaGraphSync:
    """Unit tests for CUDA graph synchronization in evaluate function."""

    def _create_mock_state(
        self,
        cuda_graph_impl="none",
        cuda_graph_scope=None,
        cuda_graph_warmup_steps=0,
        eval_iters=1,
    ):
        """Create a mock GlobalState for testing."""
        mock_state = Mock()

        # Model config
        mock_state.cfg.model.cuda_graph_impl = cuda_graph_impl
        mock_state.cfg.model.cuda_graph_scope = cuda_graph_scope if cuda_graph_scope is not None else []
        mock_state.cfg.model.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        mock_state.cfg.model.seq_length = 512

        # Train config
        mock_state.cfg.train.global_batch_size = 8
        mock_state.cfg.train.micro_batch_size = 2
        mock_state.cfg.train.eval_iters = eval_iters
        mock_state.cfg.train.exit_duration_in_mins = None
        mock_state.cfg.train.empty_unused_memory_level = 0

        # Dataset config
        mock_state.cfg.dataset.dataloader_type = "cyclic"

        # Data parallel size
        mock_state.cfg.data_parallel_size = 1

        # Timers
        mock_timers = Mock()
        mock_timer_context = Mock()
        mock_timers.return_value = mock_timer_context
        mock_state.timers = mock_timers

        # Train state
        mock_state.train_state.consumed_valid_samples = 0

        # Start time
        mock_state.start_time = 0

        return mock_state

    @patch("megatron.bridge.training.eval.get_pg_collection")
    @patch("megatron.bridge.training.eval.get_rerun_state_machine")
    @patch("megatron.bridge.training.eval.get_forward_backward_func")
    @patch("megatron.bridge.training.eval.FullCudaGraphWrapper")
    @patch("megatron.bridge.training.eval.is_pp_last_stage", return_value=False)
    @patch("megatron.bridge.training.eval.fault_tolerance")
    @patch("megatron.bridge.training.eval.prepare_forward_step_func")
    @patch("megatron.bridge.training.eval.torch.cuda.synchronize")
    def test_cuda_graph_synchronize_called_with_local_full_iteration(
        self,
        mock_cuda_sync,
        mock_prepare_forward_step,
        mock_ft,
        mock_is_pp_last,
        mock_cuda_graph_wrapper,
        mock_get_fwd_bwd_func,
        mock_get_rerun,
        mock_get_pg,
    ):
        """Test that torch.cuda.synchronize is called when cuda_graph_impl is 'local' and 'full_iteration' in scope."""
        from megatron.bridge.training.eval import evaluate

        # Setup mocks
        mock_rerun_machine = Mock()
        mock_rerun_machine.get_mode.return_value = Mock()
        mock_get_rerun.return_value = mock_rerun_machine

        mock_pg = Mock()
        mock_pg.pp = Mock()
        mock_pg.dp_cp = Mock()
        mock_get_pg.return_value = mock_pg

        mock_fwd_bwd = Mock(return_value=[{}])
        mock_get_fwd_bwd_func.return_value = mock_fwd_bwd

        mock_wrapper_instance = Mock(return_value=[{}])
        mock_cuda_graph_wrapper.return_value = mock_wrapper_instance

        mock_prepare_forward_step.return_value = Mock()

        # Create state with CUDA graph config
        state = self._create_mock_state(
            cuda_graph_impl="local",
            cuda_graph_scope=["full_iteration"],
            cuda_graph_warmup_steps=0,
            eval_iters=1,
        )

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].eval = Mock()
        mock_model[0].train = Mock()

        # Create mock config
        mock_config = Mock()

        # Call evaluate
        evaluate(
            state=state,
            forward_step_func=Mock(),
            data_iterator=Mock(),
            model=mock_model,
            process_non_loss_data_func=None,
            config=mock_config,
            verbose=False,
        )

        # Verify FullCudaGraphWrapper was used
        mock_cuda_graph_wrapper.assert_called_once()

        # Verify cuda synchronize was called
        mock_cuda_sync.assert_called()

    @patch("megatron.bridge.training.eval.get_pg_collection")
    @patch("megatron.bridge.training.eval.get_rerun_state_machine")
    @patch("megatron.bridge.training.eval.get_forward_backward_func")
    @patch("megatron.bridge.training.eval.FullCudaGraphWrapper")
    @patch("megatron.bridge.training.eval.is_pp_last_stage", return_value=False)
    @patch("megatron.bridge.training.eval.fault_tolerance")
    @patch("megatron.bridge.training.eval.prepare_forward_step_func")
    @patch("megatron.bridge.training.eval.torch.cuda.synchronize")
    def test_cuda_graph_synchronize_not_called_without_local_impl(
        self,
        mock_cuda_sync,
        mock_prepare_forward_step,
        mock_ft,
        mock_is_pp_last,
        mock_cuda_graph_wrapper,
        mock_get_fwd_bwd_func,
        mock_get_rerun,
        mock_get_pg,
    ):
        """Test that torch.cuda.synchronize is NOT called when cuda_graph_impl is not 'local'."""
        from megatron.bridge.training.eval import evaluate

        # Setup mocks
        mock_rerun_machine = Mock()
        mock_rerun_machine.get_mode.return_value = Mock()
        mock_get_rerun.return_value = mock_rerun_machine

        mock_pg = Mock()
        mock_pg.pp = Mock()
        mock_pg.dp_cp = Mock()
        mock_get_pg.return_value = mock_pg

        mock_fwd_bwd = Mock(return_value=[{}])
        mock_get_fwd_bwd_func.return_value = mock_fwd_bwd

        mock_prepare_forward_step.return_value = Mock()

        # Create state WITHOUT cuda graph enabled
        state = self._create_mock_state(
            cuda_graph_impl="none",
            cuda_graph_scope=["full_iteration"],
            eval_iters=1,
        )

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].eval = Mock()
        mock_model[0].train = Mock()

        # Create mock config
        mock_config = Mock()

        # Call evaluate
        evaluate(
            state=state,
            forward_step_func=Mock(),
            data_iterator=Mock(),
            model=mock_model,
            process_non_loss_data_func=None,
            config=mock_config,
            verbose=False,
        )

        # Verify FullCudaGraphWrapper was NOT used
        mock_cuda_graph_wrapper.assert_not_called()

        # Verify cuda synchronize was NOT called
        mock_cuda_sync.assert_not_called()

    @patch("megatron.bridge.training.eval.get_pg_collection")
    @patch("megatron.bridge.training.eval.get_rerun_state_machine")
    @patch("megatron.bridge.training.eval.get_forward_backward_func")
    @patch("megatron.bridge.training.eval.FullCudaGraphWrapper")
    @patch("megatron.bridge.training.eval.is_pp_last_stage", return_value=False)
    @patch("megatron.bridge.training.eval.fault_tolerance")
    @patch("megatron.bridge.training.eval.prepare_forward_step_func")
    @patch("megatron.bridge.training.eval.torch.cuda.synchronize")
    def test_cuda_graph_synchronize_not_called_without_full_iteration_scope(
        self,
        mock_cuda_sync,
        mock_prepare_forward_step,
        mock_ft,
        mock_is_pp_last,
        mock_cuda_graph_wrapper,
        mock_get_fwd_bwd_func,
        mock_get_rerun,
        mock_get_pg,
    ):
        """Test that torch.cuda.synchronize is NOT called when 'full_iteration' is not in scope."""
        from megatron.bridge.training.eval import evaluate

        # Setup mocks
        mock_rerun_machine = Mock()
        mock_rerun_machine.get_mode.return_value = Mock()
        mock_get_rerun.return_value = mock_rerun_machine

        mock_pg = Mock()
        mock_pg.pp = Mock()
        mock_pg.dp_cp = Mock()
        mock_get_pg.return_value = mock_pg

        mock_fwd_bwd = Mock(return_value=[{}])
        mock_get_fwd_bwd_func.return_value = mock_fwd_bwd

        mock_prepare_forward_step.return_value = Mock()

        # Create state with local impl but NOT full_iteration scope
        state = self._create_mock_state(
            cuda_graph_impl="local",
            cuda_graph_scope=["attn"],  # Not full_iteration
            eval_iters=1,
        )

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].eval = Mock()
        mock_model[0].train = Mock()

        # Create mock config
        mock_config = Mock()

        # Call evaluate
        evaluate(
            state=state,
            forward_step_func=Mock(),
            data_iterator=Mock(),
            model=mock_model,
            process_non_loss_data_func=None,
            config=mock_config,
            verbose=False,
        )

        # Verify FullCudaGraphWrapper was NOT used
        mock_cuda_graph_wrapper.assert_not_called()

        # Verify cuda synchronize was NOT called
        mock_cuda_sync.assert_not_called()
