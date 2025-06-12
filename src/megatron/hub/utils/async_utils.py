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

"""
This module provides utilities for managing asynchronous checkpoint save calls.
"""

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest

from megatron.hub.training.config import CheckpointConfig
from megatron.hub.utils.common_utils import print_rank_0


def schedule_async_save(global_state, async_request: AsyncRequest) -> None:
    """Schedule the async save request.

    Args:
        global_state: The global training state containing the async calls queue.
        async_request (AsyncRequest): the async save request.
    """
    async_queue = global_state.async_calls_queue
    if async_queue is not None:
        async_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(
    global_state, ckpt_cfg: CheckpointConfig, blocking: bool = False, terminate: bool = False
) -> None:
    """Finalizes active async save calls.

    Args:
        global_state: The global training state containing the async calls queue.
        ckpt_cfg (CheckpointConfig): The checkpoint configuration.
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    if not ckpt_cfg.async_save:
        return

    async_queue = global_state.async_calls_queue
    if async_queue is None:
        return

    if blocking and not is_empty_async_queue(global_state):
        print_rank_0("Unfinalized async checkpoint saves. Finalizing them synchronously now.")

    async_queue.maybe_finalize_async_calls(blocking)

    if terminate:
        async_queue.close()


def is_empty_async_queue(global_state) -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Args:
        global_state: The global training state containing the async calls queue.

    Returns:
        bool: True if there is any ongoing async call.
    """
    async_queue = global_state.async_calls_queue
    if async_queue is None:
        return True
    return async_queue.get_num_unfinalized_calls() == 0
