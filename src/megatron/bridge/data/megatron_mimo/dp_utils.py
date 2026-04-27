# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data parallel utilities for MegatronMIMO data loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid

    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


def _find_rank_module(
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple["HyperCommGrid | None", "str | None"]:
    """Find which module grid the current rank belongs to."""
    current_rank = dist.get_rank()
    for module_name, grid in grids.items():
        if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
            return grid, module_name
    return None, None


def _needs_data_for_module(grid: "HyperCommGrid", module_name: str) -> bool:
    """Determine if the current rank needs to load data for the given module.

    LLM: first and last PP stage need data (input_ids and labels respectively).
    Encoders: only the first PP stage needs raw modality inputs.
    """
    pp_group = grid.get_pg(["pp"])
    pp_rank = pp_group.rank()
    pp_size = pp_group.size()
    if module_name == MIMO_LANGUAGE_MODULE_KEY:
        return (pp_rank == 0) or (pp_rank == pp_size - 1)
    return pp_rank == 0


def get_megatron_mimo_dp_info(
    megatron_mimo_cfg: "MegatronMIMOParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple[int, int, bool, str]:
    """Get **module-local** DP rank, size, data-loading flag, and module name.

    Returns the DP settings for the module that the current rank participates
    in.  These are used by :func:`slice_batch_for_megatron_mimo` to sub-shard a global
    micro-batch into per-module DP shards.

    .. note::
        Do **not** use these values to construct a ``DistributedSampler``.
        For sampler construction use :func:`get_megatron_mimo_sampling_info` instead,
        which returns settings that keep all data-loading ranks synchronised
        on the same sample order.

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping from build_hypercomm_grids().

    Returns:
        Tuple of (dp_rank, dp_size, needs_data, loader_module).
    """
    my_grid, my_module = _find_rank_module(grids)
    if my_grid is None or my_module is None:
        return 0, 1, False, MIMO_LANGUAGE_MODULE_KEY

    dp_rank = my_grid.get_pg(["dp"]).rank()
    dp_size = my_grid.get_pg(["dp"]).size()
    needs_data = _needs_data_for_module(my_grid, my_module)
    return dp_rank, dp_size, needs_data, my_module


def get_megatron_mimo_sampling_info(
    megatron_mimo_cfg: "MegatronMIMOParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple[int, int, bool]:
    """Get sampler DP rank, size, and data-loading flag for MegatronMIMO.

    In heterogeneous MegatronMIMO, modules may have different DP sizes.  The data
    loader must give every data-loading rank the **same global micro-batch**
    so that :func:`slice_batch_for_megatron_mimo` (called in the forward step) can
    sub-shard it consistently with the :class:`BridgeCommunicator` fan-in /
    fan-out routing.

    This function therefore returns ``dp_size=1, dp_rank=0`` for all ranks,
    disabling DP sharding at the sampler level.  Per-module DP sharding is
    deferred to :func:`slice_batch_for_megatron_mimo`.

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping.

    Returns:
        Tuple of (sampler_dp_rank, sampler_dp_size, needs_data).
    """
    my_grid, my_module = _find_rank_module(grids)
    if my_grid is None or my_module is None:
        return 0, 1, False

    needs_data = _needs_data_for_module(my_grid, my_module)
    # All data-loading ranks use the same sampler settings so they load
    # identical global micro-batches.  Module-local DP slicing happens later
    # in forward_step via slice_batch_for_megatron_mimo.
    return 0, 1, needs_data


def slice_batch_for_megatron_mimo(
    batch: Dict[str, Any],
    dp_rank: int,
    dp_size: int,
) -> Dict[str, Any]:
    """Slice a global micro-batch for this rank's module-local DP shard.

    All data-loading ranks receive the same global micro-batch (the sampler
    uses ``dp_size=1``).  This function contiguously slices it so that each
    module-local DP replica processes the correct subset.  The slicing is
    contiguous to match the :class:`BridgeCommunicator`'s batch-dimension
    split / concatenate logic for fan-out and fan-in routing.

    Handles nested dicts (e.g. ``modality_inputs``) by recursing.

    Args:
        batch: Global batch dictionary with tensors of shape [global_batch, ...].
            May contain nested dicts (e.g. modality_inputs → encoder → kwargs).
        dp_rank: This rank's position in its **module-local** DP group.
        dp_size: Size of the module-local DP group.

    Returns:
        Dict with tensors sliced to shape [global_batch // dp_size, ...].

    Example:
        >>> global_batch = {'tokens': torch.randn(12, 2048)}
        >>> local_batch = slice_batch_for_megatron_mimo(global_batch, dp_rank=1, dp_size=3)
        >>> local_batch['tokens'].shape  # torch.Size([4, 2048])
    """
    if dp_size == 1:
        return batch

    sliced = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Slice along batch dimension (dim=0)
            batch_size = value.size(0)
            if batch_size % dp_size != 0:
                raise ValueError(
                    f"Batch size {batch_size} for key '{key}' is not divisible "
                    f"by DP size {dp_size}. Ensure micro_batch_size is divisible "
                    f"by every module's data_parallel_size."
                )
            local_batch_size = batch_size // dp_size
            start_idx = dp_rank * local_batch_size
            end_idx = start_idx + local_batch_size
            sliced[key] = value[start_idx:end_idx]
        elif isinstance(value, dict):
            # Recurse into nested dicts (e.g. modality_inputs)
            sliced[key] = slice_batch_for_megatron_mimo(value, dp_rank, dp_size)
        elif isinstance(value, list) and len(value) > 0:
            list_len = len(value)
            if list_len % dp_size == 0:
                local_len = list_len // dp_size
                start_idx = dp_rank * local_len
                end_idx = start_idx + local_len
                sliced[key] = value[start_idx:end_idx]
            else:
                # Keep as-is if not evenly divisible (global metadata)
                sliced[key] = value
        else:
            # Keep non-tensor, non-list values as-is
            sliced[key] = value

    return sliced
