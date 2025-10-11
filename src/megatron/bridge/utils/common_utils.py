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
import types
import warnings

import torch
import torch.distributed
from megatron.core import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module


try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def get_rank_safe() -> int:
    """Get the distributed rank safely, even if torch.distributed is not initialized.

    Returns:
        The current process rank.
    """
    # In megatron init, args.rank comes from the torchrun env var.
    # Once init has been done, args.rank is updated to value of torch get_rank()
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return int(os.getenv("RANK", "0"))


def get_world_size_safe() -> int:
    """Get the distributed world size safely, even if torch.distributed is not initialized.

    Returns:
        The total number of processes in the distributed job.
    """
    # In megatron init, args.world_size comes from the torchrun env var.
    # Once init has been done, args.world_size is updated to value of torch get_world_size()
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", "1"))


def get_last_rank() -> int:
    """Get the last rank in the distributed group"""
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_world_size() - 1


def get_local_rank_preinit() -> int:
    """Get the local rank from the environment variable, intended for use before full init.

    Returns:
        The local rank of the current process.
    """
    return int(os.getenv("LOCAL_RANK", "0"))


def print_rank_0(message: str) -> None:
    """Print a message only on global rank 0.

    Args:
        message: The message string to print.
    """
    rank = get_rank_safe()
    if rank == 0:
        print(message, flush=True)


def warn_rank_0(message):
    """Warn only on rank 0."""
    rank = get_rank_safe()
    if rank == 0:
        warnings.warn(message)


def is_last_rank() -> bool:
    """Check if the current rank is the last rank in the default process group.

    Returns:
        True if the current rank is the last one, False otherwise.
    """
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message: str) -> None:
    """Print a message only on the last rank of the default process group.

    Args:
        message: The message string to print.
    """
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def hook_hf_module_setattr_for_tp_grad_sync(module: torch.nn.Module) -> torch.nn.Module:
    """Mark params for TP grad sync and hook __setattr__ on a module and its children.

    This ensures that all existing parameters under the provided module have the
    attribute ``average_gradients_across_tp_domain=True`` and that any future
    submodules assigned onto this module (or any of its current children) will
    also have their parameters marked automatically.

    Args:
        module: The root module (typically a Hugging Face module instance).

    Returns:
        The same module instance for convenience.
    """
    if module is None:
        return module

    # Mark all existing parameters recursively
    for param in module.parameters(recurse=True):
        setattr(param, "average_gradients_across_tp_domain", True)

    def _wrap_setattr(original_setattr):
        def _wrapped(self, name, value):
            original_setattr(name, value)
            if isinstance(value, torch.nn.Module):
                for p in value.parameters(recurse=True):
                    setattr(p, "average_gradients_across_tp_domain", True)

        return _wrapped

    # Hook __setattr__ on the module and all existing submodules to catch
    # future dynamic assignments anywhere in the hierarchy.
    for submodule in module.modules():
        if getattr(submodule, "_tp_grad_sync_setattr_wrapped", False):
            continue
        original_setattr = submodule.__setattr__
        wrapped = _wrap_setattr(original_setattr)
        submodule.__setattr__ = types.MethodType(wrapped, submodule)
        setattr(submodule, "_tp_grad_sync_setattr_wrapped", True)

    return module
