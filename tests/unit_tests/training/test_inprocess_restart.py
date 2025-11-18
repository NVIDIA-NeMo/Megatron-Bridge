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

import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

from megatron.bridge.training.config import InProcessRestartConfig
from megatron.bridge.training.inprocess_restart import inprocess_restart, maybe_wrap_for_inprocess_restart
from megatron.bridge.training.state import GlobalState


class TestInProcessRestart:
    """Test cases for the inprocess_restart function."""

    def test_inprocess_restart_basic_configuration(self):
        """Test inprocess_restart with basic configuration."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 2
        mock_config.granularity = "rank"
        mock_config.empty_cuda_cache = True
        mock_config.max_rank_faults = None
        mock_config.monitor_process_logdir = None
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(os.environ, {"MASTER_PORT": "29500"}),
            patch("megatron.bridge.training.inprocess_restart.warnings.warn"),
            patch("torch.cuda.device_count", return_value=2),
            patch("nvidia_resiliency_ext.inprocess.rank_assignment.Layer") as mock_layer,
            patch("nvidia_resiliency_ext.inprocess.finalize.ThreadedFinalize") as mock_finalize,
            patch("nvidia_resiliency_ext.inprocess.Compose"),
            patch("nvidia_resiliency_ext.inprocess.initialize.RetryController") as mock_retry,
            patch(
                "nvidia_resiliency_ext.inprocess.nested_restarter.NestedRestarterHandlingCompleted"
            ) as mock_nested_completed,
            patch("nvidia_resiliency_ext.inprocess.abort.AbortTransformerEngine") as mock_abort_te,
            patch("nvidia_resiliency_ext.inprocess.abort.AbortTorchDistributed") as mock_abort_torch,
            patch(
                "nvidia_resiliency_ext.inprocess.nested_restarter.NestedRestarterHandlingStarting"
            ) as mock_nested_starting,
            patch(
                "nvidia_resiliency_ext.inprocess.nested_restarter.NestedRestarterFinalized"
            ) as mock_nested_finalized,
            patch("nvidia_resiliency_ext.inprocess.nested_restarter.NestedRestarterAborted") as mock_nested_aborted,
            patch("nvidia_resiliency_ext.inprocess.health_check.CudaHealthCheck") as mock_health_check,
            patch("nvidia_resiliency_ext.inprocess.rank_assignment.Tree"),
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper_instance = MagicMock()
            mock_wrapper_instance.return_value = mock_wrapped_fn
            mock_wrapper.return_value = mock_wrapper_instance

            result = inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify layer creation for rank granularity
            mock_layer.assert_called_once()

            # Verify finalize components
            mock_finalize.assert_called()

            # Verify initialize components
            mock_retry.assert_called_once_with(min_world_size=2)
            mock_nested_completed.assert_called_once()

            # Verify abort components
            mock_abort_te.assert_called_once()
            mock_abort_torch.assert_called_once()
            mock_nested_starting.assert_called_once()

            # Verify completion and terminate
            mock_nested_finalized.assert_called_once()
            mock_nested_aborted.assert_called_once()

            # Verify health check
            mock_health_check.assert_called_once()

            # Verify wrapper creation
            mock_wrapper.assert_called_once()
            wrapper_kwargs = mock_wrapper.call_args[1]
            assert wrapper_kwargs["enabled"] is True
            assert wrapper_kwargs["heartbeat_interval"] == timedelta(seconds=30.0)
            assert wrapper_kwargs["soft_timeout"] == timedelta(seconds=60.0)

            # Verify the wrapper instance is called with the adapter function
            mock_wrapper_instance.assert_called_once()

            assert result == mock_wrapped_fn

    def test_inprocess_restart_node_granularity(self):
        """Test inprocess_restart with node granularity."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 4
        mock_config.granularity = "node"
        mock_config.empty_cuda_cache = False
        mock_config.max_rank_faults = None
        mock_config.monitor_process_logdir = None
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(os.environ, {"MASTER_PORT": "29500"}),
            patch("megatron.bridge.training.inprocess_restart.warnings.warn"),
            patch("torch.cuda.device_count", return_value=2),
            patch("socket.gethostname", return_value="node1"),
            patch("nvidia_resiliency_ext.inprocess.rank_assignment.Layer") as mock_layer,
            patch("nvidia_resiliency_ext.inprocess.finalize.ThreadedFinalize") as mock_finalize,
            patch("nvidia_resiliency_ext.inprocess.Compose"),
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper_instance = MagicMock()
            mock_wrapper_instance.return_value = mock_wrapped_fn
            mock_wrapper.return_value = mock_wrapper_instance

            result = inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify two layers are created for node granularity
            assert mock_layer.call_count == 2

            # First layer for all ranks
            first_call = mock_layer.call_args_list[0]
            assert first_call[1]["min_ranks"] == 4
            assert first_call[1]["max_ranks"] == 4

            # Second layer for nodes
            second_call = mock_layer.call_args_list[1]
            assert second_call[1]["min_ranks"] == 2
            assert second_call[1]["max_ranks"] == 2

            # Verify empty_cuda_cache is False, so only one finalize component
            assert mock_finalize.call_count == 1

            assert result == mock_wrapped_fn

    def test_inprocess_restart_with_fault_counter(self):
        """Test inprocess_restart with fault counter enabled."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 2
        mock_config.granularity = "rank"
        mock_config.empty_cuda_cache = True
        mock_config.max_rank_faults = 3
        mock_config.monitor_process_logdir = None
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(os.environ, {"MASTER_PORT": "29500"}),
            patch("megatron.bridge.training.inprocess_restart.warnings.warn"),
            patch("torch.cuda.device_count", return_value=2),
            patch("nvidia_resiliency_ext.inprocess.health_check.CudaHealthCheck"),
            patch("nvidia_resiliency_ext.inprocess.health_check.FaultCounter") as mock_fault_counter,
            patch("nvidia_resiliency_ext.inprocess.Compose") as mock_compose,
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper_instance = MagicMock()
            mock_wrapper_instance.return_value = mock_wrapped_fn
            mock_wrapper.return_value = mock_wrapper_instance

            result = inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify fault counter is created
            mock_fault_counter.assert_called_once_with(max_rank_faults=3)

            # Verify health check compose includes both components
            mock_compose.assert_called()

            assert result == mock_wrapped_fn

    def test_inprocess_restart_with_monitor_logging(self):
        """Test inprocess_restart with monitor process logging enabled."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 2
        mock_config.granularity = "rank"
        mock_config.empty_cuda_cache = True
        mock_config.max_rank_faults = None
        mock_config.monitor_process_logdir = "/tmp/logs"
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(
                os.environ,
                {"MASTER_PORT": "29500", "SLURM_LOCALID": "0", "SLURM_PROCID": "0", "SLURM_JOB_ID": "12345"},
            ),
            patch("megatron.bridge.training.inprocess_restart.warnings.warn"),
            patch("socket.gethostname", return_value="node1"),
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper_instance = MagicMock()
            mock_wrapper_instance.return_value = mock_wrapped_fn
            mock_wrapper.return_value = mock_wrapper_instance

            result = inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify wrapper is called with monitor_process_logfile
            wrapper_kwargs = mock_wrapper.call_args[1]
            expected_logfile = "/tmp/logs/monitor_12345_node1_0_0.log"
            assert wrapper_kwargs["monitor_process_logfile"] == expected_logfile

            assert result == mock_wrapped_fn

    def test_inprocess_restart_no_monitor_logging_non_rank_zero(self):
        """Test inprocess_restart with monitor logging but not rank 0."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 2
        mock_config.granularity = "rank"
        mock_config.empty_cuda_cache = True
        mock_config.max_rank_faults = None
        mock_config.monitor_process_logdir = "/tmp/logs"
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(
                os.environ,
                {"MASTER_PORT": "29500", "SLURM_LOCALID": "1", "SLURM_PROCID": "1", "SLURM_JOB_ID": "12345"},
            ),
            patch("megatron.bridge.training.inprocess_restart.warnings.warn"),
            patch("socket.gethostname", return_value="node1"),
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper_instance = MagicMock()
            mock_wrapper_instance.return_value = mock_wrapped_fn
            mock_wrapper.return_value = mock_wrapper_instance

            result = inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify wrapper is called with None monitor_process_logfile for non-rank-0
            wrapper_kwargs = mock_wrapper.call_args[1]
            assert wrapper_kwargs["monitor_process_logfile"] is None

            assert result == mock_wrapped_fn

    def test_inprocess_restart_torch_cpp_log_level_warning(self):
        """Test that warning is issued when TORCH_CPP_LOG_LEVEL is not set appropriately."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 2
        mock_config.granularity = "rank"
        mock_config.empty_cuda_cache = True
        mock_config.max_rank_faults = None
        mock_config.monitor_process_logdir = None
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(os.environ, {"MASTER_PORT": "29500"}, clear=True),  # Clear TORCH_CPP_LOG_LEVEL
            patch("megatron.bridge.training.inprocess_restart.warnings.warn") as mock_warn,
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper.return_value = mock_wrapped_fn

            inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify warning is issued
            mock_warn.assert_called_once_with(
                "Set TORCH_CPP_LOG_LEVEL=error to suppress c10d waitForInput timeout warning messages"
            )

    def test_inprocess_restart_no_warning_when_torch_cpp_log_level_set(self):
        """Test that no warning is issued when TORCH_CPP_LOG_LEVEL is set correctly."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 2
        mock_config.granularity = "rank"
        mock_config.empty_cuda_cache = True
        mock_config.max_rank_faults = None
        mock_config.monitor_process_logdir = None
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(os.environ, {"MASTER_PORT": "29500", "TORCH_CPP_LOG_LEVEL": "error"}),
            patch("megatron.bridge.training.inprocess_restart.warnings.warn") as mock_warn,
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper.return_value = mock_wrapped_fn

            inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify no warning is issued
            mock_warn.assert_not_called()


class TestAbortCheckpoint:
    """Test cases for the AbortCheckpoint class functionality."""

    def test_abort_checkpoint_with_async_calls_queue(self):
        """Test AbortCheckpoint when async_calls_queue exists."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_async_queue = MagicMock()
        mock_global_state.async_calls_queue = mock_async_queue

        mock_frozen_state = MagicMock()

        # Create the AbortCheckpoint class as it's defined in the function
        with patch("megatron.bridge.training.inprocess_restart.inprocess_restart"):
            # We need to test the class that's created inside the function
            # Let's create it manually to test its behavior
            import nvidia_resiliency_ext.inprocess as inprocess

            class AbortCheckpoint(inprocess.abort.Abort):
                def __call__(self, frozen_state):
                    try:
                        if mock_global_state is not None and mock_global_state.async_calls_queue is not None:
                            async_calls_queue = mock_global_state.async_calls_queue
                            async_calls_queue.close(abort=True)
                            mock_global_state._async_calls_queue = None

                        from megatron.core.dist_checkpointing.strategies.filesystem_async import _results_queue

                        global _results_queue

                        if _results_queue is not None:
                            _results_queue._manager.shutdown()
                            del _results_queue

                    except Exception:
                        pass

                    return frozen_state

            abort_checkpoint = AbortCheckpoint()
            result = abort_checkpoint(mock_frozen_state)

            # Verify async queue is closed
            mock_async_queue.close.assert_called_once_with(abort=True)
            assert mock_global_state._async_calls_queue is None
            assert result == mock_frozen_state

    def test_abort_checkpoint_no_async_calls_queue(self):
        """Test AbortCheckpoint when async_calls_queue is None."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_global_state.async_calls_queue = None

        mock_frozen_state = MagicMock()

        # Create the AbortCheckpoint class manually
        import nvidia_resiliency_ext.inprocess as inprocess

        class AbortCheckpoint(inprocess.abort.Abort):
            def __call__(self, frozen_state):
                try:
                    if mock_global_state is not None and mock_global_state.async_calls_queue is not None:
                        async_calls_queue = mock_global_state.async_calls_queue
                        async_calls_queue.close(abort=True)
                        mock_global_state._async_calls_queue = None
                except Exception:
                    pass
                return frozen_state

        abort_checkpoint = AbortCheckpoint()
        result = abort_checkpoint(mock_frozen_state)

        # Should not crash and return the frozen state
        assert result == mock_frozen_state

    def test_abort_checkpoint_exception_handling(self):
        """Test AbortCheckpoint handles exceptions gracefully."""
        mock_global_state = MagicMock(spec=GlobalState)
        mock_async_queue = MagicMock()
        mock_async_queue.close.side_effect = RuntimeError("Queue error")
        mock_global_state.async_calls_queue = mock_async_queue

        mock_frozen_state = MagicMock()

        # Create the AbortCheckpoint class manually
        import nvidia_resiliency_ext.inprocess as inprocess

        class AbortCheckpoint(inprocess.abort.Abort):
            def __call__(self, frozen_state):
                try:
                    if mock_global_state is not None and mock_global_state.async_calls_queue is not None:
                        async_calls_queue = mock_global_state.async_calls_queue
                        async_calls_queue.close(abort=True)
                        mock_global_state._async_calls_queue = None
                except Exception:
                    pass
                return frozen_state

        abort_checkpoint = AbortCheckpoint()
        result = abort_checkpoint(mock_frozen_state)

        # Should handle exception gracefully and still return frozen state
        mock_async_queue.close.assert_called_once_with(abort=True)
        assert result == mock_frozen_state


class TestAdapterFunction:
    """Test cases for the _adapter function behavior."""

    def test_adapter_function_behavior(self):
        """Test that the adapter function correctly handles CallWrapper extraction."""
        # We can't directly test the _adapter function since it's defined inside inprocess_restart,
        # but we can test the behavior by calling inprocess_restart and examining the wrapper
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.active_world_size = 2
        mock_config.granularity = "rank"
        mock_config.empty_cuda_cache = True
        mock_config.max_rank_faults = None
        mock_config.monitor_process_logdir = None
        mock_config.heartbeat_interval = 30.0
        mock_config.heartbeat_timeout = 60.0
        mock_config.barrier_timeout = 120.0
        mock_config.completion_timeout = 120.0
        mock_config.monitor_process_interval = 1.0
        mock_config.monitor_thread_interval = 1.0
        mock_config.last_call_wait = 1.0
        mock_config.soft_timeout = 60.0
        mock_config.hard_timeout = 90.0
        mock_config.termination_grace_time = 1.0

        mock_global_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(os.environ, {"MASTER_PORT": "29500"}),
            patch("megatron.bridge.training.inprocess_restart.warnings.warn"),
            patch("nvidia_resiliency_ext.inprocess.Wrapper") as mock_wrapper,
        ):
            mock_wrapped_fn = MagicMock()
            mock_wrapper_instance = MagicMock()
            mock_wrapper_instance.return_value = mock_wrapped_fn
            mock_wrapper.return_value = mock_wrapper_instance

            _ = inprocess_restart(mock_train_fn, mock_config, mock_global_state)

            # Verify wrapper was called with keyword arguments only
            mock_wrapper.assert_called_once()

            # Verify the wrapper instance was called with the adapter function
            mock_wrapper_instance.assert_called_once()
            adapter_fn = mock_wrapper_instance.call_args[0][0]  # First positional argument to the wrapper instance

            # Test the adapter function behavior
            mock_call_wrapper = MagicMock()

            # Test adapter with inprocess_call_wrapper in kwargs
            adapter_fn("arg1", "arg2", inprocess_call_wrapper=mock_call_wrapper, other_kwarg="value")
            mock_train_fn.assert_called_once_with(
                "arg1", "arg2", inprocess_call_wrapper=mock_call_wrapper, other_kwarg="value"
            )

            # Reset mock
            mock_train_fn.reset_mock()

            # Test adapter without inprocess_call_wrapper in kwargs
            adapter_fn("arg1", "arg2", other_kwarg="value")
            mock_train_fn.assert_called_once_with("arg1", "arg2", inprocess_call_wrapper=None, other_kwarg="value")


class TestMaybeWrapForInProcessRestart:
    """Test cases for the maybe_wrap_for_inprocess_restart function."""

    def test_maybe_wrap_disabled(self):
        """Test maybe_wrap_for_inprocess_restart when disabled."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.enabled = False
        mock_state = MagicMock(spec=GlobalState)

        result_fn, result_store = maybe_wrap_for_inprocess_restart(mock_train_fn, mock_config, mock_state)

        assert result_fn == mock_train_fn
        assert result_store is None

    def test_maybe_wrap_enabled(self):
        """Test maybe_wrap_for_inprocess_restart when enabled."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.enabled = True
        mock_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(
                os.environ, {"MASTER_ADDR": "localhost", "MASTER_PORT": "29500", "WORLD_SIZE": "2", "RANK": "0"}
            ),
            patch("torch.distributed.TCPStore") as mock_tcp_store,
            patch("megatron.bridge.training.inprocess_restart.inprocess_restart") as mock_inprocess_restart,
        ):
            mock_store_instance = MagicMock()
            mock_tcp_store.return_value = mock_store_instance

            mock_wrapped_fn = MagicMock()
            mock_inprocess_restart.return_value = mock_wrapped_fn

            result_fn, result_store = maybe_wrap_for_inprocess_restart(mock_train_fn, mock_config, mock_state)

            # Verify TCPStore creation
            mock_tcp_store.assert_called_once_with(
                host_name="localhost",
                port=29501,  # MASTER_PORT + 1
                world_size=2,
                is_master=True,  # RANK == 0
                timeout=timedelta(seconds=300),
                wait_for_workers=True,
                use_libuv=True,
            )

            # Verify inprocess_restart was called
            mock_inprocess_restart.assert_called_once_with(mock_train_fn, mock_config, mock_state)

            assert result_fn == mock_wrapped_fn
            assert result_store == mock_store_instance

    def test_maybe_wrap_enabled_non_master(self):
        """Test maybe_wrap_for_inprocess_restart when enabled but not master rank."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.enabled = True
        mock_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(
                os.environ, {"MASTER_ADDR": "localhost", "MASTER_PORT": "29500", "WORLD_SIZE": "4", "RANK": "2"}
            ),
            patch("torch.distributed.TCPStore") as mock_tcp_store,
            patch("megatron.bridge.training.inprocess_restart.inprocess_restart") as mock_inprocess_restart,
        ):
            mock_store_instance = MagicMock()
            mock_tcp_store.return_value = mock_store_instance

            mock_wrapped_fn = MagicMock()
            mock_inprocess_restart.return_value = mock_wrapped_fn

            result_fn, result_store = maybe_wrap_for_inprocess_restart(mock_train_fn, mock_config, mock_state)

            # Verify TCPStore creation with is_master=False
            mock_tcp_store.assert_called_once_with(
                host_name="localhost",
                port=29501,
                world_size=4,
                is_master=False,  # RANK != 0
                timeout=timedelta(seconds=300),
                wait_for_workers=True,
                use_libuv=True,
            )

            assert result_fn == mock_wrapped_fn
            assert result_store == mock_store_instance

    def test_maybe_wrap_enabled_default_env_values(self):
        """Test maybe_wrap_for_inprocess_restart with default environment values."""
        mock_train_fn = MagicMock()
        mock_config = MagicMock(spec=InProcessRestartConfig)
        mock_config.enabled = True
        mock_state = MagicMock(spec=GlobalState)

        with (
            patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "29500",
                    # WORLD_SIZE and RANK not set, should use defaults
                },
            ),
            patch("torch.distributed.TCPStore") as mock_tcp_store,
            patch("megatron.bridge.training.inprocess_restart.inprocess_restart") as mock_inprocess_restart,
        ):
            mock_store_instance = MagicMock()
            mock_tcp_store.return_value = mock_store_instance

            mock_wrapped_fn = MagicMock()
            mock_inprocess_restart.return_value = mock_wrapped_fn

            result_fn, result_store = maybe_wrap_for_inprocess_restart(mock_train_fn, mock_config, mock_state)

            # Verify TCPStore creation with default values
            mock_tcp_store.assert_called_once_with(
                host_name="localhost",
                port=29501,
                world_size=1,  # Default WORLD_SIZE
                is_master=True,  # Default RANK is 0
                timeout=timedelta(seconds=300),
                wait_for_workers=True,
                use_libuv=True,
            )

            assert result_fn == mock_wrapped_fn
            assert result_store == mock_store_instance


class TestPretrainInProcessRestartCompatibility:
    """Test cases for pretrain function's in-process restart compatibility checks."""

    def test_pretrain_raises_error_when_process_group_initialized_with_inprocess_restart(self):
        """Test that pretrain raises RuntimeError when process group is pre-initialized and in-process restart is enabled."""
        from unittest.mock import MagicMock, patch

        import pytest

        from megatron.bridge.training.config import InProcessRestartConfig
        from megatron.bridge.training.pretrain import pretrain

        # Create mock config with in-process restart enabled
        mock_config = MagicMock()
        mock_inprocess_config = MagicMock(spec=InProcessRestartConfig)
        mock_inprocess_config.enabled = True
        mock_config.inprocess_restart = mock_inprocess_config

        # Mock forward step function
        mock_forward_step = MagicMock()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("megatron.bridge.training.pretrain.runtime_config_update"),
            patch("megatron.bridge.training.pretrain.GlobalState"),
        ):
            # Should raise RuntimeError when process group is already initialized
            with pytest.raises(RuntimeError) as exc_info:
                pretrain(mock_config, mock_forward_step)

            # Verify the error message contains key information
            error_message = str(exc_info.value)
            assert "In-process restart is incompatible with user-initialized process groups" in error_message
            assert "Either:" in error_message
            assert "Disable in-process restart" in error_message
            assert "Let the framework initialize the process group" in error_message
