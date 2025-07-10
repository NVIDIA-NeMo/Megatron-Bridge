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

from pathlib import Path

import torch

from megatron.hub.core.utils.common_utils import get_local_rank_preinit, get_rank_safe, get_world_size_safe


def initialize_distributed() -> None:
    """Initialize global process group for distributed execution."""
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return
    device_count = torch.cuda.device_count()
    if device_count > 0:
        torch.cuda.set_device(get_local_rank_preinit())

    # Call the init process
    init_process_group_kwargs = {
        "backend": "nccl",
        "world_size": get_world_size_safe(),
        "rank": get_rank_safe(),
    }
    torch.distributed.init_process_group(**init_process_group_kwargs)
    torch.distributed.barrier(device_ids=[get_local_rank_preinit()])


def broadcast_path(path: str | Path) -> str:
    """
    Broadcast a path from rank 0 to all ranks.
    """
    if get_world_size_safe() == 1:
        return path

    # Create a shared directory path - rank 0 creates it, then broadcasts to all ranks
    if get_rank_safe() == 0:
        ret_path = str(path)
    else:
        ret_path = None

    assert torch.distributed.is_initialized(), "Distributed is not initialized"
    shared_dir_list = [ret_path]
    torch.distributed.broadcast_object_list(shared_dir_list, src=0)
    shared_path = shared_dir_list[0]
    return shared_path
