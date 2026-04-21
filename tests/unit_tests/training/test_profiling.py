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

"""Unit tests for profiling utility functions."""

from unittest.mock import MagicMock, Mock, patch

from megatron.bridge.training.config import ProfilingConfig
from megatron.bridge.training.profiling import (
    handle_profiling_step,
    handle_profiling_stop,
    initialize_pytorch_profiler,
    should_profile_rank,
    start_nsys_profiler,
    stop_nsys_profiler,
)


class TestShouldProfileRank:
    """Tests for should_profile_rank function."""

    def test_should_profile_rank_with_no_config(self):
        """Test that profiling is disabled when config is None."""
        assert should_profile_rank(None, 0) is False
        assert should_profile_rank(None, 1) is False

    def test_should_profile_rank_with_matching_rank(self):
        """Test that profiling is enabled for ranks in profile_ranks."""
        config = ProfilingConfig(use_pytorch_profiler=True, profile_ranks=[0, 2])
        assert should_profile_rank(config, 0) is True
        assert should_profile_rank(config, 2) is True

    def test_should_profile_rank_with_non_matching_rank(self):
        """Test that profiling is disabled for ranks not in profile_ranks."""
        config = ProfilingConfig(use_pytorch_profiler=True, profile_ranks=[0, 2])
        assert should_profile_rank(config, 1) is False
        assert should_profile_rank(config, 3) is False

    def test_should_profile_rank_empty_list(self):
        """Test that profiling is disabled when profile_ranks is empty."""
        config = ProfilingConfig(use_pytorch_profiler=True, profile_ranks=[])
        assert should_profile_rank(config, 0) is False


class TestInitializePytorchProfiler:
    """Tests for initialize_pytorch_profiler function."""

    @patch("torch.profiler.profile")
    @patch("torch.profiler.tensorboard_trace_handler")
    @patch("torch.profiler.schedule")
    def test_initialize_pytorch_profiler_basic(self, mock_schedule, mock_handler, mock_profile):
        """Test PyTorch profiler initialization with basic parameters."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_start=5,
            profile_step_end=10,
            record_shapes=False,
        )

        mock_schedule_instance = Mock()
        mock_schedule.return_value = mock_schedule_instance
        mock_handler_instance = Mock()
        mock_handler.return_value = mock_handler_instance
        mock_profiler = Mock()
        mock_profile.return_value = mock_profiler

        prof = initialize_pytorch_profiler(config, "/tmp/tensorboard")

        # Verify schedule was created with correct parameters
        mock_schedule.assert_called_once_with(
            wait=4,  # max(5-1, 0)
            warmup=1,  # 1 if start > 0
            active=5,  # end - start
            repeat=1,
        )

        # Verify handler was called with correct directory
        mock_handler.assert_called_once_with("/tmp/tensorboard")

        # Verify profiler was created with correct kwargs
        mock_profile.assert_called_once()
        call_kwargs = mock_profile.call_args.kwargs
        assert call_kwargs["schedule"] == mock_schedule_instance
        assert call_kwargs["on_trace_ready"] == mock_handler_instance
        assert call_kwargs["record_shapes"] is False
        assert call_kwargs["with_stack"] is True

        # Verify returned profiler
        assert prof == mock_profiler

    @patch("torch.profiler.profile")
    @patch("torch.profiler.tensorboard_trace_handler")
    @patch("torch.profiler.schedule")
    def test_initialize_pytorch_profiler_with_shapes(self, mock_schedule, mock_handler, mock_profile):
        """Test profiler initialization with shape recording enabled."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_start=3,
            profile_step_end=8,
            record_shapes=True,
        )

        initialize_pytorch_profiler(config, "/tmp/tb")

        # Verify record_shapes is True
        call_kwargs = mock_profile.call_args.kwargs
        assert call_kwargs["record_shapes"] is True

    @patch("torch.profiler.profile")
    @patch("torch.profiler.tensorboard_trace_handler")
    @patch("torch.profiler.schedule")
    def test_initialize_pytorch_profiler_start_at_zero(self, mock_schedule, mock_handler, mock_profile):
        """Test profiler initialization when starting at iteration 0."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_start=0,
            profile_step_end=3,
        )

        initialize_pytorch_profiler(config, "/tmp/tb")

        # When start=0, wait should be 0 and warmup should be 0
        mock_schedule.assert_called_once_with(
            wait=0,  # max(0-1, 0) = 0
            warmup=0,  # 0 if start == 0
            active=3,
            repeat=1,
        )

    @patch("torch.profiler.profile")
    @patch("torch.profiler.tensorboard_trace_handler")
    @patch("torch.profiler.schedule")
    def test_initialize_pytorch_profiler_start_at_one(self, mock_schedule, mock_handler, mock_profile):
        """Test profiler initialization when starting at iteration 1."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_start=1,
            profile_step_end=4,
        )

        initialize_pytorch_profiler(config, "/tmp/tb")

        # When start=1, wait should be 0, warmup should be 1
        mock_schedule.assert_called_once_with(
            wait=0,  # max(1-1, 0) = 0
            warmup=1,  # 1 if start > 0
            active=3,
            repeat=1,
        )


class TestStartNsysProfiler:
    """Tests for start_nsys_profiler function."""

    @patch("torch.cuda.cudart")
    @patch("torch.autograd.profiler.emit_nvtx")
    @patch("torch.cuda.check_error")
    def test_start_nsys_profiler_without_shapes(self, mock_check_error, mock_nvtx, mock_cudart):
        """Test nsys profiler start without shape recording."""
        mock_cudart_instance = Mock()
        mock_cudart_instance.cudaProfilerStart.return_value = (0,)
        mock_cudart.return_value = mock_cudart_instance

        mock_nvtx_context = MagicMock()
        mock_nvtx.return_value = mock_nvtx_context

        config = ProfilingConfig(
            use_nsys_profiler=True,
            record_shapes=False,
        )

        result = start_nsys_profiler(config)

        # Verify CUDA profiler was started
        mock_cudart_instance.cudaProfilerStart.assert_called_once()
        mock_check_error.assert_called_once_with((0,))

        # Verify NVTX was called without record_shapes
        mock_nvtx.assert_called_once_with()
        mock_nvtx_context.__enter__.assert_called_once()

        # Verify context is returned
        assert result == mock_nvtx_context

    @patch("torch.cuda.cudart")
    @patch("torch.autograd.profiler.emit_nvtx")
    @patch("torch.cuda.check_error")
    def test_start_nsys_profiler_with_shapes(self, mock_check_error, mock_nvtx, mock_cudart):
        """Test nsys profiler start with shape recording."""
        mock_cudart_instance = Mock()
        mock_cudart_instance.cudaProfilerStart.return_value = (0,)
        mock_cudart.return_value = mock_cudart_instance

        mock_nvtx_context = MagicMock()
        mock_nvtx.return_value = mock_nvtx_context

        config = ProfilingConfig(
            use_nsys_profiler=True,
            record_shapes=True,
        )

        result = start_nsys_profiler(config)

        # Verify NVTX was called WITH record_shapes
        mock_nvtx.assert_called_once_with(record_shapes=True)
        mock_nvtx_context.__enter__.assert_called_once()

        # Verify context is returned
        assert result == mock_nvtx_context


class TestStopNsysProfiler:
    """Tests for stop_nsys_profiler function."""

    @patch("torch.cuda.cudart")
    @patch("torch.cuda.check_error")
    def test_stop_nsys_profiler(self, mock_check_error, mock_cudart):
        """Test nsys profiler stop."""
        mock_cudart_instance = Mock()
        mock_cudart_instance.cudaProfilerStop.return_value = (0,)
        mock_cudart.return_value = mock_cudart_instance

        mock_nvtx_context = MagicMock()

        stop_nsys_profiler(mock_nvtx_context)

        # Verify CUDA profiler was stopped
        mock_cudart_instance.cudaProfilerStop.assert_called_once()
        mock_check_error.assert_called_once_with((0,))

        # Verify NVTX context was exited
        mock_nvtx_context.__exit__.assert_called_once_with(None, None, None)

    @patch("torch.cuda.cudart")
    @patch("torch.cuda.check_error")
    def test_stop_nsys_profiler_with_none_context(self, mock_check_error, mock_cudart):
        """Test nsys profiler stop handles None context gracefully."""
        mock_cudart_instance = Mock()
        mock_cudart_instance.cudaProfilerStop.return_value = (0,)
        mock_cudart.return_value = mock_cudart_instance

        # Should not raise exception
        stop_nsys_profiler(None)

        # Verify CUDA profiler was still stopped
        mock_cudart_instance.cudaProfilerStop.assert_called_once()


class TestHandleProfilingStep:
    """Tests for handle_profiling_step function."""

    def test_handle_profiling_step_with_no_config(self):
        """Test that profiling step does nothing when config is None."""
        mock_prof = Mock()

        handle_profiling_step(None, iteration=5, rank=0, pytorch_prof=mock_prof)

        # Profiler should not be called
        mock_prof.step.assert_not_called()

    def test_handle_profiling_step_skips_non_profiled_rank(self):
        """Test that profiling step is skipped for non-profiled ranks."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_ranks=[0],
        )
        mock_prof = Mock()

        # Rank 1 should not profile
        handle_profiling_step(config, iteration=5, rank=1, pytorch_prof=mock_prof)

        # PyTorch profiler step should NOT be called
        mock_prof.step.assert_not_called()

    def test_handle_profiling_step_pytorch_profiler(self):
        """Test profiling step calls PyTorch profiler.step()."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_ranks=[0],
        )
        mock_prof = Mock()

        handle_profiling_step(config, iteration=5, rank=0, pytorch_prof=mock_prof)

        # PyTorch profiler step should be called
        mock_prof.step.assert_called_once()

    def test_handle_profiling_step_pytorch_profiler_none(self):
        """Test profiling step handles None profiler gracefully."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_ranks=[0],
        )

        # Should not raise exception
        handle_profiling_step(config, iteration=5, rank=0, pytorch_prof=None)

    @patch("megatron.bridge.training.profiling.start_nsys_profiler")
    def test_handle_profiling_step_nsys_before_start(self, mock_start_nsys):
        """Test nsys profiler does not start before profile_step_start."""
        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_start=10,
            profile_step_end=15,
            profile_ranks=[0],
        )

        # Before start iteration - should not start
        handle_profiling_step(config, iteration=9, rank=0, pytorch_prof=None)
        mock_start_nsys.assert_not_called()

    @patch("megatron.bridge.training.profiling.start_nsys_profiler")
    def test_handle_profiling_step_nsys_at_start_iteration(self, mock_start_nsys):
        """Test nsys profiler starts at profile_step_start."""
        mock_nvtx_context = Mock()
        mock_start_nsys.return_value = mock_nvtx_context

        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_start=10,
            profile_step_end=15,
            profile_ranks=[0],
        )

        # At start iteration - should start and return context
        result = handle_profiling_step(config, iteration=10, rank=0, pytorch_prof=None)
        mock_start_nsys.assert_called_once_with(config)
        assert result == mock_nvtx_context

    @patch("megatron.bridge.training.profiling.start_nsys_profiler")
    def test_handle_profiling_step_nsys_after_start(self, mock_start_nsys):
        """Test nsys profiler does not restart after profile_step_start."""
        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_start=10,
            profile_step_end=15,
            profile_ranks=[0],
        )

        # After start iteration - should not start again
        handle_profiling_step(config, iteration=11, rank=0, pytorch_prof=None)
        mock_start_nsys.assert_not_called()

    @patch("megatron.bridge.training.profiling.start_nsys_profiler")
    def test_handle_profiling_step_nsys_rank_filtering(self, mock_start_nsys):
        """Test nsys profiler respects rank filtering."""
        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_start=10,
            profile_step_end=15,
            profile_ranks=[0, 2],
        )

        # Rank 1 should not start profiler
        handle_profiling_step(config, iteration=10, rank=1, pytorch_prof=None)
        mock_start_nsys.assert_not_called()

        # Rank 0 should start profiler
        handle_profiling_step(config, iteration=10, rank=0, pytorch_prof=None)
        mock_start_nsys.assert_called_once_with(config)


class TestHandleProfilingStop:
    """Tests for handle_profiling_stop function."""

    def test_handle_profiling_stop_with_no_config(self):
        """Test that profiling stop does nothing when config is None."""
        mock_prof = Mock()

        handle_profiling_stop(None, iteration=10, rank=0, pytorch_prof=mock_prof)

        # Profiler should not be stopped
        mock_prof.stop.assert_not_called()

    def test_handle_profiling_stop_skips_non_profiled_rank(self):
        """Test that profiling stop is skipped for non-profiled ranks."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_end=10,
            profile_ranks=[0],
        )
        mock_prof = Mock()

        # Rank 1 should not stop profiler
        handle_profiling_stop(config, iteration=10, rank=1, pytorch_prof=mock_prof)

        # PyTorch profiler stop should NOT be called
        mock_prof.stop.assert_not_called()

    def test_handle_profiling_stop_skips_wrong_iteration(self):
        """Test that profiling stop is skipped for iterations other than profile_step_end."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_end=10,
            profile_ranks=[0],
        )
        mock_prof = Mock()

        # Wrong iteration - should not stop
        handle_profiling_stop(config, iteration=9, rank=0, pytorch_prof=mock_prof)
        mock_prof.stop.assert_not_called()

        # Also test after end iteration
        handle_profiling_stop(config, iteration=11, rank=0, pytorch_prof=mock_prof)
        mock_prof.stop.assert_not_called()

    def test_handle_profiling_stop_pytorch_profiler(self):
        """Test profiling stop calls PyTorch profiler.stop()."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_end=10,
            profile_ranks=[0],
        )
        mock_prof = Mock()

        handle_profiling_stop(config, iteration=10, rank=0, pytorch_prof=mock_prof)

        # PyTorch profiler stop should be called
        mock_prof.stop.assert_called_once()

    def test_handle_profiling_stop_pytorch_profiler_none(self):
        """Test profiling stop handles None profiler gracefully."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_step_end=10,
            profile_ranks=[0],
        )

        # Should not raise exception
        handle_profiling_stop(config, iteration=10, rank=0, pytorch_prof=None)

    @patch("megatron.bridge.training.profiling.stop_nsys_profiler")
    def test_handle_profiling_stop_nsys_at_end_iteration(self, mock_stop_nsys):
        """Test nsys profiler stops at profile_step_end."""
        mock_nvtx_context = Mock()

        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_end=10,
            profile_ranks=[0],
        )

        handle_profiling_stop(config, iteration=10, rank=0, pytorch_prof=None, nsys_nvtx_context=mock_nvtx_context)

        # Nsys stop should be called with the context
        mock_stop_nsys.assert_called_once_with(mock_nvtx_context)

    @patch("megatron.bridge.training.profiling.stop_nsys_profiler")
    def test_handle_profiling_stop_nsys_wrong_iteration(self, mock_stop_nsys):
        """Test nsys profiler does not stop at wrong iteration."""
        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_end=10,
            profile_ranks=[0],
        )

        # Wrong iteration - should not stop
        handle_profiling_stop(config, iteration=9, rank=0, pytorch_prof=None)
        mock_stop_nsys.assert_not_called()

    @patch("megatron.bridge.training.profiling.stop_nsys_profiler")
    def test_handle_profiling_stop_nsys_rank_filtering(self, mock_stop_nsys):
        """Test nsys profiler stop respects rank filtering."""
        mock_nvtx_context = Mock()

        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_end=10,
            profile_ranks=[0, 2],
        )

        # Rank 1 should not stop profiler
        handle_profiling_stop(config, iteration=10, rank=1, pytorch_prof=None, nsys_nvtx_context=mock_nvtx_context)
        mock_stop_nsys.assert_not_called()

        # Rank 0 should stop profiler
        handle_profiling_stop(config, iteration=10, rank=0, pytorch_prof=None, nsys_nvtx_context=mock_nvtx_context)
        mock_stop_nsys.assert_called_once_with(mock_nvtx_context)


class TestProfilingEdgeCases:
    """Tests for edge cases and combinations."""

    def test_handle_profiling_step_both_profilers_disabled(self):
        """Test that nothing happens when both profilers are disabled."""
        config = ProfilingConfig(
            use_pytorch_profiler=False,
            use_nsys_profiler=False,
            profile_ranks=[0],
        )
        mock_prof = Mock()

        handle_profiling_step(config, iteration=5, rank=0, pytorch_prof=mock_prof)

        # Nothing should be called
        mock_prof.step.assert_not_called()

    def test_multiple_ranks_profiling(self):
        """Test that multiple ranks can be profiled."""
        config = ProfilingConfig(
            use_pytorch_profiler=True,
            profile_ranks=[0, 1, 3],
        )

        assert should_profile_rank(config, 0) is True
        assert should_profile_rank(config, 1) is True
        assert should_profile_rank(config, 2) is False
        assert should_profile_rank(config, 3) is True

    @patch("megatron.bridge.training.profiling.start_nsys_profiler")
    def test_handle_profiling_step_nsys_at_iteration_zero(self, mock_start_nsys):
        """Test nsys profiler can start at iteration 0."""
        config = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_start=0,
            profile_step_end=5,
            profile_ranks=[0],
        )

        handle_profiling_step(config, iteration=0, rank=0, pytorch_prof=None)
        mock_start_nsys.assert_called_once_with(config)
