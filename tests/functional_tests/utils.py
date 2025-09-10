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


def verify_checkpoint_files(checkpoint_dir: str, iteration_count: int, ckpt_format: str = "torch_dist") -> None:
    """Verify that checkpoint files were created correctly for different checkpoint formats.

    Args:
        checkpoint_dir: Directory containing checkpoints
        iteration_count: Expected iteration number for the checkpoint
        ckpt_format: Checkpoint format ("torch_dist", "fsdp_dtensor", etc.)
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        latest_tracker_file = os.path.join(checkpoint_dir, "latest_train_state.pt")
        assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

        final_iter_dir = os.path.join(checkpoint_dir, f"iter_{iteration_count:07d}")
        assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"

        metadata_file = os.path.join(final_iter_dir, ".metadata")
        assert os.path.exists(metadata_file), "Checkpoint metadata file not found"

        # Both formats use torch.distributed.checkpoint but may create different numbers of .distcp files
        distcp_files = [f for f in os.listdir(final_iter_dir) if f.endswith(".distcp")]

        if ckpt_format == "torch_dist":
            num_expected_files = 2 * torch.distributed.get_world_size()
        elif ckpt_format == "fsdp_dtensor":
            # fsdp_dtensor format creates .distcp files (one per rank)
            num_expected_files = torch.distributed.get_world_size()
        else:
            raise ValueError(f"Unsupported checkpoint format for verification: {ckpt_format}")

        assert len(distcp_files) == num_expected_files, (
            f"Expected {num_expected_files} .distcp files for fsdp_dtensor, found {len(distcp_files)}: {distcp_files}"
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


def compare_provider_configs(converted_provider, predefined_provider, model_id):
    """Compare ALL configuration attributes between converted and predefined providers."""

    # Get all attributes from both providers
    converted_attrs = vars(converted_provider)
    predefined_attrs = vars(predefined_provider)

    # First check that both providers have the same set of attributes
    converted_keys = set(converted_attrs.keys())
    predefined_keys = set(predefined_attrs.keys())

    missing_in_converted = predefined_keys - converted_keys
    missing_in_predefined = converted_keys - predefined_keys

    if missing_in_converted:
        raise AssertionError(f"Converted provider for {model_id} is missing attributes: {missing_in_converted}")

    if missing_in_predefined:
        raise AssertionError(f"Predefined provider for {model_id} is missing attributes: {missing_in_predefined}")

    # Compare all attribute values
    mismatched_attrs = []
    excluded_attrs = set()

    for attr_name in sorted(converted_keys):
        # Skip excluded attributes
        if "init_method" in attr_name or attr_name == "generation_config" or attr_name == "vocab_size":
            excluded_attrs.add(attr_name)
            continue

        converted_value = converted_attrs[attr_name]
        predefined_value = predefined_attrs[attr_name]

        # Handle special comparison cases for different types
        if converted_value != predefined_value:
            # For functions, compare by name/identity since they might be the same function
            # but not pass == comparison
            if callable(converted_value) and callable(predefined_value):
                if (
                    hasattr(converted_value, "__name__")
                    and hasattr(predefined_value, "__name__")
                    and converted_value.__name__ == predefined_value.__name__
                ):
                    continue
                elif converted_value is predefined_value:
                    continue

            mismatched_attrs.append(f"  {attr_name}: converted={converted_value} vs predefined={predefined_value}")

    if mismatched_attrs:
        raise AssertionError(f"Configuration mismatch for {model_id}:\n" + "\n".join(mismatched_attrs))
