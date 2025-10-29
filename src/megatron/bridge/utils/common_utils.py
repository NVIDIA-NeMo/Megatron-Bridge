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
import re
import types
import warnings

import torch
import torch.distributed
from megatron.core import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module

from megatron.bridge.utils.slurm_utils import (
    resolve_slurm_local_rank,
    resolve_slurm_master_addr,
    resolve_slurm_master_port,
    resolve_slurm_rank,
    resolve_slurm_world_size,
)


try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def get_rank_safe() -> int:
    """Get the distributed rank safely, even if torch.distributed is not initialized.

    Fallback order:
    1. torch.distributed.get_rank() (if initialized)
    2. RANK environment variable (torchrun/torchelastic)
    3. SLURM_PROCID environment variable (SLURM)
    4. Default: 0 (with warning)

    Returns:
        The current process rank.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    if "RANK" in os.environ:
        return int(os.environ["RANK"])

    slurm_rank = resolve_slurm_rank()
    if slurm_rank is not None:
        return slurm_rank

    warnings.warn("Could not determine rank from torch.distributed, RANK, or SLURM_PROCID. Defaulting to rank 0.")
    return 0


def get_world_size_safe() -> int:
    """Get the distributed world size safely, even if torch.distributed is not initialized.

    Fallback order:
    1. torch.distributed.get_world_size() (if initialized)
    2. WORLD_SIZE environment variable (torchrun/torchelastic)
    3. SLURM_NTASKS environment variable (SLURM)
    4. Default: 1 (with warning)

    Returns:
        The total number of processes in the distributed job.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    slurm_world_size = resolve_slurm_world_size()
    if slurm_world_size is not None:
        return slurm_world_size

    warnings.warn(
        "Could not determine world size from torch.distributed, WORLD_SIZE, or SLURM_NTASKS. "
        "Defaulting to world size 1."
    )
    return 1


def get_last_rank() -> int:
    """Get the last rank in the distributed group"""
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_world_size() - 1


def get_local_rank_preinit() -> int:
    """Get the local rank from the environment variable, intended for use before full init.

    Fallback order:
    1. LOCAL_RANK environment variable (torchrun/torchelastic)
    2. SLURM_LOCALID environment variable (SLURM)
    3. Default: 0 (with warning)

    Returns:
        The local rank of the current process.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    slurm_local_rank = resolve_slurm_local_rank()
    if slurm_local_rank is not None:
        return slurm_local_rank

    warnings.warn("Could not determine local rank from LOCAL_RANK or SLURM_LOCALID. Defaulting to local rank 0.")
    return 0


def get_master_addr_safe() -> str:
    """Get the master address for distributed initialization.

    Fallback order:
    1. MASTER_ADDR environment variable (torchrun/torchelastic)
    2. SLURM_NODELIST parsed (SLURM)
    3. Default: localhost (with warning)

    Returns:
        The master node address.
    """
    if "MASTER_ADDR" in os.environ:
        return os.environ["MASTER_ADDR"]

    slurm_addr = resolve_slurm_master_addr()
    if slurm_addr is not None:
        return slurm_addr

    warnings.warn("Could not determine master address from MASTER_ADDR or SLURM_NODELIST. Defaulting to 'localhost'.")
    return "localhost"


def get_master_port_safe() -> int:
    """Get the master port for distributed initialization.

    Fallback order:
    1. MASTER_PORT environment variable (torchrun/torchelastic)
    2. SLURM job-based port (SLURM_JOB_ID derived)
    3. Default: 29500 (with warning)

    Returns:
        The master port.
    """
    if "MASTER_PORT" in os.environ:
        return int(os.environ["MASTER_PORT"])

    slurm_port = resolve_slurm_master_port()
    if slurm_port is not None:
        return slurm_port

    warnings.warn("Could not determine master port from MASTER_PORT or SLURM environment. Defaulting to 29500.")
    return 29500


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


def extract_expert_number_from_param(param_name: str) -> int:
    """Extract the expert number from a parameter name.
    Args:
        param_name: The parameter name to extract the expert number from.
    Returns:
        The expert number.
    """
    pattern = r"(?:experts\.|weight|bias)(\d+)"
    match = re.search(pattern, param_name)
    if not match:
        raise ValueError(
            f"No expert number found in parameter name: {param_name}. Please update the regex {pattern} if necessary."
        )
    return int(match.group(1))
