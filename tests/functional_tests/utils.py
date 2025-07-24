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
from pathlib import Path

import torch


def initialize_distributed() -> None:
    """Initialize global process group for distributed execution."""
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    device_count = torch.cuda.device_count()
    if device_count > 0:
        torch.cuda.set_device(local_rank)

    # Call the init process
    init_process_group_kwargs = {
        "backend": "nccl",
        "world_size": world_size,
        "rank": rank,
    }
    torch.distributed.init_process_group(**init_process_group_kwargs)
    torch.distributed.barrier(device_ids=[local_rank])


def broadcast_path(path: str | Path) -> str:
    """
    Broadcast a path from rank 0 to all ranks. This function assumes that torch.distributed is already initialized.

    Args:
        path: Path to broadcast

    Returns:
        str: Broadcasted path
    """
    assert torch.distributed.is_initialized(), "Distributed is not initialized"

    if torch.distributed.get_world_size() == 1:
        return path

    # Create a shared directory path - rank 0 creates it, then broadcasts to all ranks
    if torch.distributed.get_rank() == 0:
        ret_path = str(path)
    else:
        ret_path = None

    shared_dir_list = [ret_path]
    torch.distributed.broadcast_object_list(shared_dir_list, src=0)
    shared_path = shared_dir_list[0]
    return shared_path
