# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data parallel utilities for MIMO data loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid

    from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig


def get_mimo_dp_info(
    mimo_cfg: "MimoParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple[int, int, bool, str]:
    """
    Return the DP rank and size, whether the current rank must load data, and which module's DP settings to use for the current distributed rank.
    
    Determines which module (from `grids`) contains the current global rank and computes the DP process-group rank and size. If the rank does not belong to any provided grid, returns defaults for a non-participating rank: (0, 1, False, MIMO_LANGUAGE_MODULE_KEY). `needs_data` is true for the language module when the rank is on the first or last PP stage; for other modules it is true only on the first PP stage.
    
    Parameters:
        mimo_cfg (MimoParallelismConfig): MIMO parallelism configuration (kept for API compatibility).
        grids (Dict[str, HyperCommGrid]): Mapping from module name to its HyperCommGrid produced by build_hypercomm_grids().
    
    Returns:
        Tuple[int, int, bool, str]: `(dp_rank, dp_size, needs_data, loader_module)` where
            - `dp_rank`: this rank's index within its DP process group,
            - `dp_size`: size of the DP process group used for data sharding,
            - `needs_data`: `True` if the rank should load data, `False` otherwise,
            - `loader_module`: name of the module whose DP settings should be used (or `MIMO_LANGUAGE_MODULE_KEY` for non-participating ranks).
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
        return 0, 1, False, MIMO_LANGUAGE_MODULE_KEY

    dp_rank = my_grid.get_pg(["dp"]).rank()
    dp_size = my_grid.get_pg(["dp"]).size()

    pp_group = my_grid.get_pg(["pp"])
    pp_rank = pp_group.rank()
    pp_size = pp_group.size()

    if my_module == MIMO_LANGUAGE_MODULE_KEY:
        needs_data = (pp_rank == 0) or (pp_rank == pp_size - 1)
    else:
        needs_data = pp_rank == 0

    return dp_rank, dp_size, needs_data, my_module
