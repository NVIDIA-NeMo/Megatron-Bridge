# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data parallel utilities for MegatronMIMO data loading."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid

    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


logger = logging.getLogger(__name__)


def _find_rank_modules(
    grids: Dict[str, "HyperCommGrid"],
) -> Dict[str, "HyperCommGrid"]:
    """Return every module grid the current rank belongs to.

    In non-colocated deployments (disjoint rank ranges) this returns at most
    one entry. In colocated deployments (all grids share the same
    ``(rank_offset, size)``) every module grid is returned.
    """
    current_rank = dist.get_rank()
    return {
        name: grid for name, grid in grids.items() if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)
    }


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
    module_name: Optional[str] = None,
) -> Tuple[int, int, bool, str]:
    """Get **module-local** DP rank, size, data-loading flag, and module name.

    These values feed :func:`slice_batch_for_megatron_mimo`, which sub-shards a
    global micro-batch into per-module DP shards.

    .. note::
        Do **not** use these values to construct a ``DistributedSampler``.
        For sampler construction use :func:`get_megatron_mimo_sampling_info` instead,
        which returns settings that keep all data-loading ranks synchronised
        on the same sample order.

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping from build_hypercomm_grids().
        module_name: Explicit module to query. Required when the current rank
            participates in multiple modules (colocated mode) because a rank's
            encoder-DP and LLM-DP generally differ. If omitted and the rank
            serves multiple modules, defaults to the language module and emits
            a warning.

    Returns:
        Tuple of (dp_rank, dp_size, needs_data, loader_module).
    """
    rank_modules = _find_rank_modules(grids)
    if not rank_modules:
        return 0, 1, False, MIMO_LANGUAGE_MODULE_KEY

    if module_name is not None:
        grid = rank_modules.get(module_name)
        if grid is None:
            return 0, 1, False, module_name
        selected_name = module_name
    elif len(rank_modules) == 1:
        selected_name, grid = next(iter(rank_modules.items()))
    else:
        selected_name = (
            MIMO_LANGUAGE_MODULE_KEY if MIMO_LANGUAGE_MODULE_KEY in rank_modules else next(iter(rank_modules))
        )
        grid = rank_modules[selected_name]
        logger.warning(
            "get_megatron_mimo_dp_info called without module_name on a rank serving "
            "multiple modules (%s). Defaulting to '%s'. Pass module_name explicitly "
            "to disambiguate in colocated mode.",
            sorted(rank_modules),
            selected_name,
        )

    dp_rank = grid.get_pg(["dp"]).rank()
    dp_size = grid.get_pg(["dp"]).size()
    needs_data = _needs_data_for_module(grid, selected_name)
    return dp_rank, dp_size, needs_data, selected_name


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

    In colocated deployments a rank belongs to multiple module grids. We say
    the rank needs data if **any** of its modules needs data — missing a
    needed-by-LLM batch because the rank was only checked against the encoder
    grid would cause silent no-op training.

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping.

    Returns:
        Tuple of (sampler_dp_rank, sampler_dp_size, needs_data).
    """
    rank_modules = _find_rank_modules(grids)
    if not rank_modules:
        return 0, 1, False

    needs_data = any(_needs_data_for_module(grid, name) for name, grid in rank_modules.items())
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
