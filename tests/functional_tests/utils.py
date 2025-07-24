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
import shutil
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


def get_directory_size(path: str) -> int:
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def clear_directories(path: str) -> None:
    """Delete a directory on rank 0."""
    if not torch.distributed.is_initialized():
        if os.path.exists(path):
            shutil.rmtree(path)
        return

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            if os.path.exists(path):
                shutil.rmtree(path)
        torch.distributed.barrier()


def verify_checkpoint_files(checkpoint_dir: str, iteration_count: int) -> None:
    """Verify that checkpoint files were created correctly."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        latest_tracker_file = os.path.join(checkpoint_dir, "latest_train_state.pt")
        assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

        final_iter_dir = os.path.join(checkpoint_dir, f"iter_{iteration_count:07d}")
        assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"

        metadata_file = os.path.join(final_iter_dir, ".metadata")
        assert os.path.exists(metadata_file), "Checkpoint metadata file not found"

        distcp_files = [f for f in os.listdir(final_iter_dir) if f.endswith(".distcp")]
        num_expected_files = 2 * torch.distributed.get_world_size()
        assert len(distcp_files) == num_expected_files, (
            f"Expected {num_expected_files} .distcp files, found {len(distcp_files)}: {distcp_files}"
        )


def verify_peft_checkpoint_smaller(pretrain_checkpoint_dir, peft_checkpoint_dir, pretrain_iters, peft_iters) -> None:
    """Verify that PEFT checkpoint is smaller than pretrained checkpoint (adapter weights only)."""
    if torch.distributed.get_rank() == 0:
        pretrain_iter_dir = os.path.join(pretrain_checkpoint_dir, f"iter_{pretrain_iters:07d}")
        peft_iter_dir = os.path.join(peft_checkpoint_dir, f"iter_{peft_iters:07d}")

        assert os.path.exists(pretrain_iter_dir), f"Pretrain checkpoint directory not found at {pretrain_iter_dir}"
        assert os.path.exists(peft_iter_dir), f"PEFT checkpoint directory not found at {peft_iter_dir}"

        pretrain_size = get_directory_size(pretrain_iter_dir)
        peft_size = get_directory_size(peft_iter_dir)

        # PEFT checkpoint should be significantly smaller (only adapter weights)
        assert peft_size < pretrain_size * 0.6, (
            f"PEFT checkpoint ({peft_size}) should be smaller than 60% of pretrain checkpoint ({pretrain_size})"
        )
