# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch.distributed as dist

from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


def build_hypercomm_grids(
    mimo_parallelism_config: MimoParallelismConfig,
) -> Dict[str, "HyperCommGrid"]:
    """
    Constructs a HyperCommGrid for each module in the given MIMO parallelism configuration.
    
    Grids are created on all ranks (so collective operations are consistent); only ranks within a grid's range participate in that grid's operations.
    
    Parameters:
        mimo_parallelism_config (MimoParallelismConfig): Configuration mapping module names to their parallelism specifications.
    
    Returns:
        Dict[str, "HyperCommGrid"]: Mapping from module name to its HyperCommGrid.
    """
    from megatron.core.hyper_comm_grid import HyperCommGrid

    grids: Dict[str, HyperCommGrid] = {}
    for module_name, parallelism in mimo_parallelism_config.module_parallelisms.items():
        shape = [
            parallelism.tensor_model_parallel_size,
            parallelism.context_parallel_size,
            parallelism.expert_tensor_parallel_size,
            parallelism.pipeline_model_parallel_size,
            parallelism.data_parallel_size,
        ]
        grid = HyperCommGrid(
            shape=shape,
            dim_names=["tp", "cp", "ep", "pp", "dp"],
            rank_offset=parallelism.rank_offset,
            backend="nccl",
        )
        # Create all standard process groups
        for dim in ("tp", "cp", "ep", "pp", "dp"):
            _ = grid.create_pg([dim])
        _ = grid.create_pg(["dp", "cp"])
        _ = grid.create_pg(["tp", "pp"])
        _ = grid.create_pg(["tp", "ep", "pp"])
        _ = grid.create_pg(["dp", "ep"])

        grids[module_name] = grid

    return grids


def populate_embedding_and_position_groups(
    pp_group: dist.ProcessGroup,
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    """
    Create process groups for position embeddings and tied word embeddings based on pipeline-parallel ranks.
    
    Position-embedding group contains only the first PP stage rank; embedding group contains the first and, if different, the last PP stage ranks. This operation calls `dist.new_group`, which is a collective and must be invoked on all ranks that could participate.
    
    Args:
        pp_group (dist.ProcessGroup): The pipeline-parallel process group or `None`.
    
    Returns:
        Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]: `(pos_embd_pg, embd_pg)` where `pos_embd_pg` is the group for position embeddings and `embd_pg` is the group for tied word embeddings; returns `(None, None)` if `pp_group` is `None`.
    """
    if pp_group is None:
        return None, None

    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))

    # Position embeddings only on first PP stage
    pos_embd_ranks = [pp_ranks[0]]
    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)

    # Word embeddings on first and last PP stages (for tied embeddings)
    embd_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1 and pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])
    embd_pg = dist.new_group(ranks=embd_ranks)

    return pos_embd_pg, embd_pg


def is_pp_first_stage(pp_group: Optional[dist.ProcessGroup]) -> bool:
    """
    Determine whether the current process is the first stage in the given pipeline-parallel process group.
    
    Parameters:
        pp_group (Optional[dist.ProcessGroup]): The pipeline-parallel process group to inspect. If `None`, the current process is treated as first stage.
    
    Returns:
        bool: `true` if the current rank equals the smallest rank in `pp_group` or if `pp_group` is `None`, `false` otherwise.
    """
    if pp_group is None:
        return True
    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
    return dist.get_rank() == pp_ranks[0]


def is_pp_last_stage(pp_group: Optional[dist.ProcessGroup]) -> bool:
    """
    Determine whether the current process is the last stage of the given pipeline-parallel process group.
    
    Parameters:
        pp_group (Optional[dist.ProcessGroup]): The pipeline-parallel process group to inspect. If `None`, the pipeline is treated as a single-stage group.
    
    Returns:
        `true` if the current process rank is the last rank in `pp_group`, `false` otherwise.
    """
    if pp_group is None:
        return True
    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
    return dist.get_rank() == pp_ranks[-1]
