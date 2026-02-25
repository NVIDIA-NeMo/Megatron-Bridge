# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data parallel utilities for MIMO data loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch.distributed as dist


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid

    from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig


def get_mimo_dp_info(
    mimo_cfg: "MimoParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple[int, int, bool, str]:
    """Get DP rank, size, data-loading responsibility, and loader module for MIMO.

    Determines which module's DP settings to use for data loading based on
    current rank's participation in heterogeneous deployment.

    In heterogeneous mode, each rank uses its own module's DP settings.

    Args:
        mimo_cfg: MIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping from build_hypercomm_grids().

    Returns:
        Tuple of (dp_rank, dp_size, needs_data, loader_module):
        - dp_rank: This rank's position in DP group.
        - dp_size: Size of DP group for data sharding.
        - needs_data: Whether this rank needs to load data (first/last PP stage).
        - loader_module: Which module's DP settings are being used.

    Example:
        >>> from megatron.bridge.models.mimo.mimo_builder import build_hypercomm_grids
        >>> grids = build_hypercomm_grids(mimo_cfg)
        >>> dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
        >>> if needs_data:
        ...     # Build data loader with dp_rank and dp_size
        ...     sampler = DistributedSampler(dataset, num_replicas=dp_size, rank=dp_rank)
    """
    current_rank = dist.get_rank()

    # Heterogeneous: find which module this rank belongs to
    my_grid = None
    my_module = None
    for module_name, grid in grids.items():
        if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
            my_grid = grid
            my_module = module_name
            break

    if my_grid is None or my_module is None:
        # Rank doesn't participate in any module
        return 0, 1, False, "llm"

    dp_rank = my_grid.get_pg(["dp"]).rank()
    dp_size = my_grid.get_pg(["dp"]).size()

    pp_group = my_grid.get_pg(["pp"])
    pp_rank = pp_group.rank()
    pp_size = pp_group.size()

    if my_module == "llm":
        needs_data = (pp_rank == 0) or (pp_rank == pp_size - 1)
    else:
        needs_data = pp_rank == 0

    return dp_rank, dp_size, needs_data, my_module
