# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Literal, Optional


try:
    import nvidia_resiliency_ext.inprocess as inprocess
except ImportError:
    inprocess = None

import warnings

import torch

from megatron.bridge.training.initialize import destroy_global_state


@dataclass
class InProcessRestartConfig:
    """Configuration settings for NVIDIA Resiliency Extension in-process restart functionality."""

    enabled: bool = False
    """Enable in-process restart mechanism from nvidia-resiliency-ext."""

    max_iterations: Optional[int] = None
    """Maximum number of in-process restart iterations."""

    monitor_thread_interval: float = 1.0
    """Monitoring interval (in seconds) for the monitoring thread."""

    monitor_process_interval: float = 1.0
    """Monitoring interval (in seconds) for the monitoring process."""

    progress_watchdog_interval: float = 1.0
    """Interval (in seconds) for automatic progress watchdog timestamp updates."""

    heartbeat_interval: float = 30.0
    """Monitoring interval (in seconds) for detecting unresponsive ranks."""

    soft_timeout: float = 60.0
    """Soft progress timeout (in seconds)."""

    hard_timeout: float = 90.0
    """Hard progress timeout (in seconds)."""

    heartbeat_timeout: float = 60.0
    """Timeout (in seconds) for a missing rank detection heartbeat."""

    barrier_timeout: float = 120.0
    """Timeout (in seconds) for internal distributed barrier."""

    completion_timeout: float = 120.0
    """Timeout (in seconds) for barrier on completion on all ranks."""

    last_call_wait: float = 1.0
    """Time interval (in seconds) for other ranks to report concurrent terminal failures."""

    termination_grace_time: float = 1.0
    """Interval (in seconds) between SIGTERM and SIGKILL issued on hard timeout."""

    granularity: Literal["node", "rank"] = "node"
    """Granularity for in-process restart."""

    active_world_size: Optional[int] = None
    """The number of ranks initially executing the workload.
    The remaining ranks from the allocation are set aside as warm reserve.
    If None, defaults to WORLD_SIZE environment variable."""

    empty_cuda_cache: bool = True
    """Empty CUDA cache during restart finalization."""


def inprocess_restart(train_fn, config: InProcessRestartConfig, *, state=None) -> Callable:
    """
    Wraps the train_fn with in-process restart functionality.

    Args:
        train_fn: The training function to wrap.
        config: Configuration settings for in-process restart.
        state: Optional state object for the training function.

    Returns:
        The wrapped training function.
    """

    if "TORCH_CPP_LOG_LEVEL" not in os.environ or os.environ["TORCH_CPP_LOG_LEVEL"] not in (
        "error",
        "fatal",
    ):
        warnings.warn("Set TORCH_CPP_LOG_LEVEL=error to suppress c10d waitForInput timeout warning messages")

    # Layers represents a configuration for a layer of branches at a certain
    # depth in a topology tree constructed by inprocess.rank_assignment.Tree.
    # First layer contains all ranks and it's the root of the topology tree,
    # the second optional layer groups ranks by nodes.
    layers = [
        inprocess.rank_assignment.Layer(
            min_ranks=config.active_world_size,
            max_ranks=config.active_world_size,
            flag=inprocess.rank_assignment.LayerFlag.RESERVE,
        )
    ]
    if config.granularity == "node":
        device_count = torch.cuda.device_count()

        layers.append(
            inprocess.rank_assignment.Layer(
                min_ranks=device_count,
                max_ranks=device_count,
                key_or_fn=lambda _: socket.gethostname(),
                flag=inprocess.rank_assignment.LayerFlag.RESERVE,
            )
        )

    finalize = [inprocess.finalize.ThreadedFinalize(timeout=timedelta(seconds=10), fn=destroy_global_state)]

    if config.empty_cuda_cache:
        finalize.append(inprocess.finalize.ThreadedFinalize(timeout=timedelta(seconds=10), fn=torch.cuda.empty_cache))

    initialize = inprocess.Compose(
        inprocess.initialize.RetryController(min_world_size=config.active_world_size),
        inprocess.nested_restarter.NestedRestarterHandlingCompleted(),
    )

    class AbortCheckpoint(inprocess.abort.Abort):
        def __call__(self, frozen_state: inprocess.state.FrozenState) -> inprocess.state.FrozenState:
            # Abort persistent async worker processes if present
            try:
                if state is not None and getattr(state, "async_calls_queue", None) is not None:
                    queue = state.async_calls_queue
                    if queue is not None:
                        queue.close(abort=True)
                # Attempt to shutdown filesystem async results queue manager if present
                try:
                    from megatron.core.dist_checkpointing.strategies.filesystem_async import _results_queue

                    if _results_queue is not None:
                        _results_queue._manager.shutdown()
                except Exception:
                    pass
            except Exception:
                pass
            return frozen_state

    abort = inprocess.Compose(
        inprocess.abort.AbortTransformerEngine(),
        inprocess.abort.AbortTorchDistributed(),
        AbortCheckpoint(),
        inprocess.nested_restarter.NestedRestarterHandlingStarting(),
    )
    completion = inprocess.nested_restarter.NestedRestarterFinalized()
    terminate = inprocess.nested_restarter.NestedRestarterAborted()

    new_train_fn = inprocess.Wrapper(
        store_kwargs={
            "timeout": timedelta(seconds=300),
            "port": int(os.environ["MASTER_PORT"]) + 2,
        },
        initialize=initialize,
        abort=abort,
        completion=completion,
        terminate=terminate,
        health_check=inprocess.health_check.CudaHealthCheck(timeout=timedelta(seconds=10)),
        rank_assignment=inprocess.rank_assignment.Tree(layers=layers),
        finalize=inprocess.Compose(*finalize),
        heartbeat_interval=timedelta(seconds=config.heartbeat_interval),
        heartbeat_timeout=timedelta(seconds=config.heartbeat_timeout),
        barrier_timeout=timedelta(seconds=config.barrier_timeout),
        completion_timeout=timedelta(seconds=config.completion_timeout),
        monitor_process_interval=timedelta(seconds=config.monitor_process_interval),
        monitor_thread_interval=timedelta(seconds=config.monitor_thread_interval),
        last_call_wait=timedelta(seconds=config.last_call_wait),
        soft_timeout=timedelta(seconds=config.soft_timeout),
        hard_timeout=timedelta(seconds=config.hard_timeout),
        termination_grace_time=timedelta(seconds=config.termination_grace_time),
        enabled=True,
    )(train_fn)

    return new_train_fn


def maybe_wrap_for_inprocess_restart(
    train_fn, config: InProcessRestartConfig, state=None
) -> tuple[Callable, Optional[torch.distributed.Store]]:
    """Conditionally wrap function for in-process restart."""

    if not config.enabled:
        return train_fn, None

    # Apply inprocess restart wrapper
    wrapped_train_fn = inprocess_restart(train_fn, config, state=state)

    # Create the TCPStore
    store = torch.distributed.TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]) + 1,
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        is_master=(int(os.getenv("RANK", "0")) == 0),
        timeout=timedelta(seconds=300),
        wait_for_workers=True,
        use_libuv=True,
    )

    return wrapped_train_fn, store
