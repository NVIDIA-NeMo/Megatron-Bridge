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

from typing import Optional

import torch
import torch.profiler

from megatron.bridge.training.config import ProfilingConfig


def should_profile_rank(config: Optional[ProfilingConfig], rank: int) -> bool:
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


def start_nsys_profiler(config: ProfilingConfig) -> None:
    """Start CUDA profiler for nsys profiling.

    Args:
        config: Profiling configuration
    """
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
    if config.record_shapes:
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
    else:
        torch.autograd.profiler.emit_nvtx().__enter__()


def stop_nsys_profiler() -> None:
    """Stop CUDA profiler for nsys profiling."""
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
    torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)


def handle_profiling_step(
    config: Optional[ProfilingConfig],
    iteration: int,
    rank: int,
    pytorch_prof: Optional[torch.profiler.profile],
) -> None:
    """Handle profiling logic for a single training step.

    Args:
        config: Profiling configuration
        iteration: Current training iteration
        rank: Current process rank
        pytorch_prof: PyTorch profiler instance (if using PyTorch profiler)
    """
    if not should_profile_rank(config, rank):
        return

    if config.use_pytorch_profiler and pytorch_prof is not None:
        pytorch_prof.step()

    if config.use_nsys_profiler:
        if iteration == config.profile_step_start:
            start_nsys_profiler(config)


def handle_profiling_stop(
    config: Optional[ProfilingConfig],
    iteration: int,
    rank: int,
    pytorch_prof: Optional[torch.profiler.profile],
) -> None:
    """Handle profiling cleanup at designated stop iteration.

    Args:
        config: Profiling configuration
        iteration: Current training iteration
        rank: Current process rank
        pytorch_prof: PyTorch profiler instance (if using PyTorch profiler)
    """
    if not should_profile_rank(config, rank):
        return

    if iteration != config.profile_step_end:
        return

    if config.use_pytorch_profiler and pytorch_prof is not None:
        pytorch_prof.stop()

    if config.use_nsys_profiler:
        stop_nsys_profiler()
