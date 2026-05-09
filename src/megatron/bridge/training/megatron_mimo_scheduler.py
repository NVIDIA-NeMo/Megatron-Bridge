# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO learning-rate scheduler helpers."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler


def optimizer_has_params(optimizer: object) -> bool:
    """Return whether an optimizer owns any rank-local trainable parameters."""
    return any(group.get("params") for group in getattr(optimizer, "param_groups", ()))


def first_scheduler_with_param_groups(
    schedulers: Dict[str, "OptimizerParamScheduler"],
) -> Optional["OptimizerParamScheduler"]:
    """Return the first scheduler backed by a non-empty optimizer."""
    for scheduler in schedulers.values():
        if scheduler is not None and optimizer_has_params(scheduler.optimizer):
            return scheduler
    return None


class MegatronMIMOSchedulerCheckpointProxy:
    """Rank-placement independent checkpoint adapter for shared MIMO LR schedules.

    Non-colocated MegatronMIMO can place the only trainable module on a nonzero
    rank. Generic checkpointing writes common scheduler state from rank 0, so
    this proxy broadcasts the scheduler state from the first rank that owns a
    real scheduler and exposes it as a normal ``opt_param_scheduler`` object.
    """

    def __init__(self, schedulers: Dict[str, "OptimizerParamScheduler"], source_rank: int) -> None:
        self.schedulers = schedulers
        self.source_rank = source_rank
        self._loaded_state_dict: dict | None = None

    def state_dict(self) -> dict:
        """Return the globally canonical scheduler state."""
        local_scheduler = first_scheduler_with_param_groups(self.schedulers)
        state = copy.deepcopy(local_scheduler.state_dict()) if local_scheduler is not None else None

        if dist.is_available() and dist.is_initialized():
            obj_list = [state if dist.get_rank() == self.source_rank else None]
            dist.broadcast_object_list(obj_list, src=self.source_rank)
            state = obj_list[0]

        return copy.deepcopy(state) if state is not None else {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state into every rank-local active scheduler."""
        self._loaded_state_dict = copy.deepcopy(state_dict)
        for scheduler in self.schedulers.values():
            if scheduler is not None and optimizer_has_params(scheduler.optimizer):
                scheduler.load_state_dict(copy.deepcopy(state_dict))

    @property
    def loaded_state_dict(self) -> dict | None:
        """Return the state loaded by checkpointing, if any."""
        return copy.deepcopy(self._loaded_state_dict)


def _distributed_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def make_scheduler_checkpoint_proxy(
    schedulers: Dict[str, "OptimizerParamScheduler"],
) -> MegatronMIMOSchedulerCheckpointProxy | None:
    """Return a scheduler checkpoint proxy when any rank owns a scheduler."""
    local_scheduler = first_scheduler_with_param_groups(schedulers)

    if not dist.is_available() or not dist.is_initialized():
        return MegatronMIMOSchedulerCheckpointProxy(schedulers, source_rank=0) if local_scheduler is not None else None

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_source = rank if local_scheduler is not None else world_size
    source_tensor = torch.tensor([local_source], dtype=torch.int64, device=_distributed_device())
    dist.all_reduce(source_tensor, op=dist.ReduceOp.MIN)
    source_rank = int(source_tensor.item())

    if source_rank == world_size:
        return None
    return MegatronMIMOSchedulerCheckpointProxy(schedulers, source_rank=source_rank)
