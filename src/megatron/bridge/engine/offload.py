# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""cpu() / cuda() movement of model, optimizer and live accumulation state.

Parameters that live in MCore DDP buffers move through the buffers' own ``offload_to_cpu`` /
``reload_from_cpu``: one pinned host copy per buffer, then the GPU storage is resized to zero, so the
module parameters (views into the buffer) stay valid. Everything else that is a standalone CUDA tensor
(fp32 master shards, Adam state, parameters outside DDP buffers such as a frozen engine's) is moved
with a ``.data`` swap into a pinned host copy. The tensor objects keep their identity, which is what the
optimizer's param groups and state dict are keyed by. Host copies are allocated once per engine and
reused on every later move.

Gradients are copied to the host only when an accumulation window is open and copied back after the
reload. With a closed window they are freed and come back zeroed, which is the state a new window
expects anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from megatron.core.distributed import DistributedDataParallel
from megatron.core.transformer import MegatronModule


logger = logging.getLogger(__name__)


@dataclass
class HostState:
    """Pinned host copies and bookkeeping that survive across cpu()/cuda() cycles of one engine."""

    # id(buffer) -> pinned copy of that buffer's live gradients while the engine is on the host.
    grads: dict[int, torch.Tensor] = field(default_factory=dict)
    # id(tensor) -> pinned host copy for the tensors moved with a .data swap (allocated once, reused).
    hosts: dict[int, torch.Tensor] = field(default_factory=dict)
    # (tensor, original device) for every tensor currently swapped to the host.
    swapped: list[tuple[torch.Tensor, torch.device]] = field(default_factory=list)


def check_movable(recipe: Any, optimizer: Any) -> None:
    """Raise NotImplementedError for configurations whose buffers or optimizer cannot be moved yet."""
    ddp, optim_cfg = recipe.ddp, recipe.optimizer
    if getattr(ddp, "overlap_param_gather", False) or getattr(optim_cfg, "overlap_param_gather", False):
        # TODO(overlap): force start_param_sync(force_sync=True) before moving so the buffers hold the
        # updated weights and no all-gather is in flight.
        raise NotImplementedError("cpu()/cuda() with overlap_param_gather=True is not supported yet.")
    if getattr(ddp, "reuse_grad_buf_for_mxfp8_param_ag", False):
        # TODO(mxfp8): the all-gather buffer aliases gradient storage and cannot be resized.
        raise NotImplementedError("cpu()/cuda() with reuse_grad_buf_for_mxfp8_param_ag=True is not supported yet.")
    if getattr(ddp, "nccl_ub", False):
        # TODO(nccl-ub): buffers registered in an NCCL memory pool need their own offload path.
        raise NotImplementedError("cpu()/cuda() with nccl_ub=True is not supported yet.")
    if getattr(optim_cfg, "optimizer_cpu_offload", False):
        # TODO(hdo): HybridDeviceOptimizer places its own state; moving its GPU half corrupts the next step.
        raise NotImplementedError("cpu()/cuda() with optimizer_cpu_offload=True is not supported yet.")
    if getattr(optim_cfg, "use_precision_aware_optimizer", False):
        # TODO(precision-aware): master weights live inside FusedAdam's state under other keys.
        raise NotImplementedError("cpu()/cuda() with use_precision_aware_optimizer=True is not supported yet.")


def offload_to_cpu(
    model: list[MegatronModule], optimizer: Any | None, host: HostState, *, keep_gradients: bool
) -> int:
    """Move parameters, optimizer state and (with ``keep_gradients``) live gradients to host memory.

    Returns the number of GPU bytes released (parameter and gradient buffers, swapped tensors), which
    callers can compare with the drop in ``torch.cuda.memory_allocated``.
    """
    torch.cuda.synchronize()
    released = 0
    for chunk in model:
        if isinstance(chunk, DistributedDataParallel):
            for buffer in chunk.buffers + chunk.expert_parallel_buffers:
                if keep_gradients and buffer.grad_data is not None and buffer.grad_data.untyped_storage().size() > 0:
                    grads = host.grads.get(id(buffer))
                    if grads is None:
                        grads = torch.empty_like(buffer.grad_data, device="cpu", pin_memory=True)
                        host.grads[id(buffer)] = grads
                    grads.copy_(buffer.grad_data, non_blocking=True)
                released += _buffer_bytes(buffer)
                buffer.offload_to_cpu(move_params=True, move_grads=True)
        # Parameters outside DDP buffers (frozen engines, frozen parameters). Buffer views now have an
        # empty storage and are skipped.
        for parameter in chunk.parameters():
            if parameter.data.device.type == "cuda" and parameter.data.untyped_storage().size() > 0:
                released += _swap_to_host(parameter, host)
    if optimizer is not None:
        for tensor in _optimizer_tensors(optimizer):
            if tensor.device.type == "cuda":
                released += _swap_to_host(tensor, host)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return released


def reload_to_cuda(model: list[MegatronModule], host: HostState) -> int:
    """Restore everything moved by ``offload_to_cpu`` (optimizer tensors included, through ``host.swapped``);
    buffer gradients come back as saved, or zeroed. Returns the number of GPU bytes allocated again."""
    restored = 0
    for chunk in model:
        if isinstance(chunk, DistributedDataParallel):
            for buffer in chunk.buffers + chunk.expert_parallel_buffers:
                buffer.reload_from_cpu(move_params=True, move_grads=True)
                grads = host.grads.pop(id(buffer), None)
                if grads is not None:
                    buffer.grad_data.copy_(grads, non_blocking=True)
                restored += _buffer_bytes(buffer)
    for tensor, device in host.swapped:
        tensor.data = tensor.data.to(device, non_blocking=True)
        restored += tensor.data.numel() * tensor.data.element_size()
    host.swapped.clear()
    torch.cuda.synchronize()
    return restored


def _buffer_bytes(buffer: Any) -> int:
    """Bytes currently held on the GPU by a DDP buffer's parameter and gradient storage."""
    total = 0
    for data in (buffer.param_data, buffer.grad_data):
        if data is not None and data.untyped_storage().size() > 0:
            total += data.numel() * data.element_size()
    return total


def _swap_to_host(tensor: torch.Tensor, host: HostState) -> int:
    pinned = host.hosts.get(id(tensor))
    if pinned is None or pinned.shape != tensor.data.shape or pinned.dtype != tensor.data.dtype:
        pinned = torch.empty_like(tensor.data, device="cpu", pin_memory=True)
        host.hosts[id(tensor)] = pinned
    pinned.copy_(tensor.data, non_blocking=True)
    host.swapped.append((tensor, tensor.data.device))
    nbytes = tensor.data.numel() * tensor.data.element_size()
    tensor.data = pinned
    return nbytes


def _optimizer_tensors(optimizer: Any) -> list[torch.Tensor]:
    """fp32 master shards and per-parameter optimizer state tensors of an MCore (possibly chained) optimizer."""
    tensors = []
    for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
        for group in getattr(opt, "shard_fp32_from_float16_groups", []):
            tensors.extend(t for t in group if t is not None)
        inner = getattr(opt, "optimizer", None)
        state = getattr(inner, "state", None) or {}
        for entry in state.values():
            tensors.extend(v for v in entry.values() if isinstance(v, torch.Tensor))
    return tensors
