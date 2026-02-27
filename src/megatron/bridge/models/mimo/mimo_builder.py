from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch.distributed as dist

from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


def build_hypercomm_grids(
    mimo_parallelism_config: MimoParallelismConfig,
) -> Dict[str, "HyperCommGrid"]:
    """Create HyperCommGrid objects per module from MIMO parallelism config.

    Creates grids on ALL ranks (required for consistent collective calls),
    but only ranks in each grid's range will participate in its operations.

    Args:
        mimo_parallelism_config: MimoParallelismConfig specifying parallelism per module.

    Returns:
        Dict mapping module names to their HyperCommGrids.
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


def _default_topology(mimo_parallelism_config: MimoParallelismConfig) -> Dict[str, List[str]]:
    """Infer a default multi-encoder -> LLM topology."""
    return {name: ["llm"] for name in mimo_parallelism_config.module_names if name != "llm"} | {"llm": []}


def populate_embedding_and_position_groups(
    pp_group: dist.ProcessGroup,
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    """Create embedding-related process groups from PP group ranks.

    Following MCore semantics:
    - pos_embd_pg: Only rank 0 of PP (first stage) - for position embeddings
    - embd_pg: Ranks 0 and -1 of PP (first and last stages) - for tied word embeddings

    IMPORTANT: This calls dist.new_group which is a collective operation.
    Must be called on all ranks that could participate.

    Args:
        pp_group: The pipeline parallel process group.

    Returns:
        Tuple of (pos_embd_pg, embd_pg). Returns (None, None) if pp_group is None.
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
    """Check if current rank is first stage in pipeline."""
    if pp_group is None:
        return True
    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
    return dist.get_rank() == pp_ranks[0]


def is_pp_last_stage(pp_group: Optional[dist.ProcessGroup]) -> bool:
    """Check if current rank is last stage in pipeline."""
    if pp_group is None:
        return True
    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
    return dist.get_rank() == pp_ranks[-1]


def is_current_rank_in_grid(grid: "HyperCommGrid") -> bool:
    """Check if the current rank participates in this grid.

    Args:
        grid: A HyperCommGrid instance.

    Returns:
        True if dist.get_rank() is within the grid's rank range.
    """
    return grid.rank_offset <= dist.get_rank() < (grid.rank_offset + grid.size)
