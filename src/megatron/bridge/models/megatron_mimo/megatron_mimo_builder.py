# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch.distributed as dist

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


EXPERT_VIEW_NAME = "expert"


def build_hypercomm_grids(
    megatron_mimo_parallelism_config: MegatronMIMOParallelismConfig,
) -> Dict[str, "HyperCommGrid"]:
    """Create HyperCommGrid objects per module from MegatronMIMO parallelism config.

    Creates grids on ALL ranks (required for consistent collective calls),
    but only ranks in each grid's range will participate in its operations.

    Each grid is built with a dense ``base`` view plus a registered ``expert`` view over the same
    rank span. Dense process groups (tp/cp/dp/pp) come from the base view; expert-parallel groups
    (expt_tp/ep/expt_dp) come from the expert view, matching the contract mcore's
    ``get_mimo_optimizer`` expects.

    Args:
        megatron_mimo_parallelism_config: MegatronMIMOParallelismConfig specifying parallelism per module.

    Returns:
        Dict mapping module names to their HyperCommGrids.
    """
    from megatron.core.hyper_comm_grid import HyperCommGrid

    grids: Dict[str, HyperCommGrid] = {}
    for module_name, parallelism in megatron_mimo_parallelism_config.module_parallelisms.items():
        shape = [
            parallelism.tensor_model_parallel_size,
            parallelism.context_parallel_size,
            parallelism.data_parallel_size,
            parallelism.pipeline_model_parallel_size,
        ]
        grid = HyperCommGrid(
            shape=shape,
            dim_names=["tp", "cp", "dp", "pp"],
            rank_offset=parallelism.rank_offset,
            backend="nccl",
        )
        grid.register_view(
            EXPERT_VIEW_NAME,
            shape=[
                parallelism.expert_tensor_parallel_size,
                parallelism.expert_model_parallel_size,
                parallelism.expert_data_parallel_size,
                parallelism.pipeline_model_parallel_size,
            ],
            dim_names=["expt_tp", "ep", "expt_dp", "pp"],
            shared_dims=["pp"],
        )

        # Create all required process groups in a stable order on every rank.
        for dims in (
            ["tp"],
            ["cp"],
            ["pp"],
            ["dp"],
            ["dp", "cp"],
            ["tp", "cp"],
            ["tp", "pp"],
            ["tp", "dp", "cp"],
            ["tp", "cp", "dp", "pp"],
        ):
            _ = grid.create_pg(dims)

        for dims in (
            ["ep"],
            ["expt_tp"],
            ["expt_dp"],
            ["expt_tp", "ep"],
            ["expt_tp", "ep", "pp"],
        ):
            _ = grid.create_pg(dims, view=EXPERT_VIEW_NAME)

        grids[module_name] = grid

    return grids


def populate_embedding_and_position_groups(
    pp_rank_groups: list[list[int]] | None,
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    """Create embedding-related process groups from globally enumerated PP ranks.

    Following MCore semantics:
    - pos_embd_pg: Only rank 0 of PP (first stage) - for position embeddings
    - embd_pg: Ranks 0 and -1 of PP (first and last stages) - for tied word embeddings

    IMPORTANT: This calls dist.new_group which is a collective operation.
    Must be called on all ranks that could participate.

    Args:
        pp_rank_groups: Every pipeline-parallel rank group in global creation order.

    Returns:
        Tuple of process groups for the current rank. Returns (None, None) when
        no pipeline-parallel rank groups are provided.
    """
    if not pp_rank_groups:
        return None, None

    current_rank = dist.get_rank()
    local_pos_embd_pg = None
    local_embd_pg = None
    for pp_rank_group in pp_rank_groups:
        pp_ranks = sorted(pp_rank_group)

        # Every global rank must create every derived group in the same order.
        pos_embd_ranks = [pp_ranks[0]]
        pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)

        embd_ranks = [pp_ranks[0]]
        if len(pp_ranks) > 1 and pp_ranks[-1] != pp_ranks[0]:
            embd_ranks.append(pp_ranks[-1])
        embd_pg = dist.new_group(ranks=embd_ranks)

        if current_rank in pos_embd_ranks:
            local_pos_embd_pg = pos_embd_pg
        if current_rank in embd_ranks:
            local_embd_pg = embd_pg

    return local_pos_embd_pg, local_embd_pg


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
