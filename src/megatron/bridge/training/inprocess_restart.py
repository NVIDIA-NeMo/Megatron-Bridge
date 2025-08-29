# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import socket
from datetime import timedelta
from typing import Callable, Optional


try:
    import nvidia_resiliency_ext.inprocess as inprocess
except ImportError:
    inprocess = None

import warnings

import torch

from megatron.bridge.training.config import InProcessRestartConfig
from megatron.bridge.training.initialize import destroy_global_state
from megatron.bridge.training.state import GlobalState


def inprocess_restart(train_fn: Callable, config: InProcessRestartConfig, global_state: GlobalState) -> Callable:
    """
    Wraps the train_fn with in-process restart functionality.

    Args:
        train_fn: The training function to wrap.
        config: Configuration settings for in-process restart.
        global_state: State object for the training function.

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
                if global_state is not None and global_state.async_calls_queue is not None:
                    async_calls_queue = global_state.async_calls_queue
                    async_calls_queue.close(abort=True)
                    global_state._async_calls_queue = None

                from megatron.core.dist_checkpointing.strategies.filesystem_async import _results_queue

                global _results_queue

                if _results_queue is not None:
                    _results_queue._manager.shutdown()
                    del _results_queue

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
    train_fn: Callable, config: InProcessRestartConfig, state: GlobalState
) -> tuple[Callable, Optional[torch.distributed.Store]]:
    """Conditionally wrap function for in-process restart."""

    if not config.enabled:
        return train_fn, None

    # Apply inprocess restart wrapper
    wrapped_train_fn = inprocess_restart(train_fn, config, state)

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
