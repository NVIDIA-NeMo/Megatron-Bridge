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

"""Profiling utilities for training loop."""

import torch
import torch.profiler

from megatron.bridge.training.config import ProfilingConfig


# Type alias for NVTX context manager
TNvtxContext = torch.autograd.profiler.emit_nvtx


def should_profile_rank(config: ProfilingConfig | None, rank: int) -> bool:
    """Check if current rank should be profiled.

    Args:
        config: Profiling configuration
        rank: Current process rank

    Returns:
        True if this rank should be profiled
    """
    if config is None:
        return False
    return rank in config.profile_ranks


def handle_profiling_step(
    config: ProfilingConfig | None,
    iteration: int,
    rank: int,
    pytorch_prof: torch.profiler.profile | None,
) -> TNvtxContext | None:
    """Handle profiling logic for a single training step.

    Args:
        config: Profiling configuration
        iteration: Current training iteration
        rank: Current process rank
        pytorch_prof: PyTorch profiler instance (if using PyTorch profiler)

    Returns:
        NVTX context if nsys profiling was started at this step, None otherwise
    """
    if not should_profile_rank(config, rank):
        return None

    if config.use_pytorch_profiler and pytorch_prof is not None:
        pytorch_prof.step()

    if config.use_nsys_profiler:
        if iteration == config.profile_step_start:
            return start_nsys_profiler(config)

    return None


def handle_profiling_stop(
    config: ProfilingConfig | None,
    iteration: int,
    rank: int,
    pytorch_prof: torch.profiler.profile | None,
    nsys_nvtx_context: TNvtxContext | None = None,
) -> None:
    """Handle profiling cleanup at designated stop iteration.

    Args:
        config: Profiling configuration
        iteration: Current training iteration
        rank: Current process rank
        pytorch_prof: PyTorch profiler instance (if using PyTorch profiler)
        nsys_nvtx_context: NVTX context from handle_profiling_step (if using nsys profiler)
    """
    if not should_profile_rank(config, rank):
        return

    if iteration != config.profile_step_end:
        return

    if config.use_pytorch_profiler and pytorch_prof is not None:
        pytorch_prof.stop()

    if config.use_nsys_profiler:
        stop_nsys_profiler(nsys_nvtx_context)


def initialize_pytorch_profiler(
    config: ProfilingConfig,
    tensorboard_dir: str,
) -> torch.profiler.profile:
    """Initialize PyTorch profiler with config settings.

    Args:
        config: Profiling configuration
        tensorboard_dir: Directory for tensorboard outputs

    Returns:
        Initialized (but not started) PyTorch profiler
    """
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(config.profile_step_start - 1, 0),
            warmup=1 if config.profile_step_start > 0 else 0,
            active=config.profile_step_end - config.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir),
        record_shapes=config.record_shapes,
        with_stack=True,
    )
    return prof


def start_nsys_profiler(config: ProfilingConfig) -> TNvtxContext:
    """Start CUDA profiler for nsys profiling.

    Args:
        config: Profiling configuration

    Returns:
        NVTX context manager that must be passed to stop_nsys_profiler
    """
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
    if config.record_shapes:
        nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=True)
    else:
        nvtx_context = torch.autograd.profiler.emit_nvtx()
    nvtx_context.__enter__()
    return nvtx_context


def stop_nsys_profiler(nvtx_context: TNvtxContext | None) -> None:
    """Stop CUDA profiler for nsys profiling.

    Args:
        nvtx_context: NVTX context manager returned from start_nsys_profiler
    """
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
    if nvtx_context is not None:
        nvtx_context.__exit__(None, None, None)
