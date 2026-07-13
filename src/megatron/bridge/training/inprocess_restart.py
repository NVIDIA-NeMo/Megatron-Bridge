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

import logging
import os
import socket
import warnings
from datetime import timedelta
from typing import Callable, Optional

import nvidia_resiliency_ext.inprocess as inprocess
import torch

from megatron.bridge.training.config import InProcessRestartConfig
from megatron.bridge.training.state import GlobalState


logger: logging.Logger = logging.getLogger(__name__)


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

    def destroy_state():
        """Comprehensive state cleanup for in-process restart."""

        from megatron.bridge.training.initialize import destroy_global_state

        try:
            # Clean up Megatron global state
            destroy_global_state()
            global_state.reset_for_restart()
        except Exception as e:
            logger.error(f"destroy_state failed: {type(e).__name__}: {e}", exc_info=True)

    finalize = [inprocess.finalize.ThreadedFinalize(timeout=timedelta(seconds=10), fn=destroy_state)]

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

    # Configure health check with optional fault counter
    health_check_components = [inprocess.health_check.CudaHealthCheck(timeout=timedelta(seconds=10))]
    if config.max_rank_faults is not None:
        health_check_components.append(inprocess.health_check.FaultCounter(max_rank_faults=config.max_rank_faults))
    health_check = inprocess.Compose(*health_check_components)

    # Configure monitor process logging if enabled
    monitor_process_logfile = None
    if config.monitor_process_logdir is not None:
        slurm_local_id = os.environ.get("SLURM_LOCALID")
        slurm_global_rank = os.environ.get("SLURM_PROCID")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        hostname = socket.gethostname()

        if slurm_global_rank is not None and int(slurm_global_rank) == 0:
            monitor_process_logfile = os.path.join(
                config.monitor_process_logdir,
                f"monitor_{slurm_job_id}_{hostname}_{slurm_global_rank}_{slurm_local_id}.log",
            )

    # Adapter function to bridge nvidia-resiliency-ext 0.4.1 calling convention
    # with Megatron-Bridge function signatures.
    #
    # Why this is needed:
    # - NVRx 0.4.1 calls wrapped functions via CallWrapper.__call__(fn, args, kwargs)
    # - NVRx injects the active CallWrapper instance for parameters annotated with
    #   Optional[CallWrapper] or CallWrapper after binding arguments
    # - Our _pretrain function expects the CallWrapper as a keyword argument named
    #   'inprocess_call_wrapper', but NVRx may pass it differently
    # - This adapter ensures compatibility by extracting the CallWrapper and passing
    #   it correctly to the actual training function
    def _adapter(*args, **kwargs):
        # Extract the injected CallWrapper from kwargs if NVRx placed it there
        call_wrapper = kwargs.pop("inprocess_call_wrapper", None)

        # Call the actual training function with the CallWrapper as expected keyword arg
        result = train_fn(*args, inprocess_call_wrapper=call_wrapper, **kwargs)
        return result

    new_train_fn = inprocess.Wrapper(
        store_kwargs={
            "timeout": timedelta(seconds=300),
            "port": int(os.environ["MASTER_PORT"]) + 2,
        },
        initialize=initialize,
        abort=abort,
        completion=completion,
        terminate=terminate,
        health_check=health_check,
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
        monitor_process_logfile=monitor_process_logfile,
        enabled=True,
    )(_adapter)

    return new_train_fn


def maybe_wrap_for_inprocess_restart(
    train_fn: Callable, config: InProcessRestartConfig, state: GlobalState
) -> tuple[Callable, Optional[torch.distributed.Store]]:
    """Conditionally wrap function for in-process restart."""

    if not config.enabled:
        return train_fn, None

    # Create the coordination TCPStore first
    store = torch.distributed.TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]) + 1,
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        is_master=(int(os.getenv("RANK", "0")) == 0),
        timeout=timedelta(seconds=300),
        wait_for_workers=True,
        use_libuv=True,
    )

    # Apply inprocess restart wrapper
    wrapped_train_fn = inprocess_restart(train_fn, config, state)

    return wrapped_train_fn, store
