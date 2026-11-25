# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch.distributed as dist

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from torch.distributed import ProcessGroupNCCL


EXPERT_VIEW_NAME = "expert"


def _lazy_nccl_pg_options() -> "ProcessGroupNCCL.Options | None":
    """Return new NCCL options that create a MegatronMIMO group's communicator on first use.

    PyTorch builds whose ``nccl`` backend is ProcessGroupNCCL2 bind the default process group to a
    device, so every later ``new_group`` splits its NCCL communicator eagerly, even if the group
    never communicates. MegatronMIMO creates every parallelism group of every module on every
    rank, so eager splitting holds NCCL buffers for groups the rank never uses. ``lazy_init``
    instead creates a group's communicator at the group's first operation. That creation is
    collective over all members, so every member must take part in the first operation: MIMO
    groups are used by collectives over the whole group, and every pipeline stage issues its
    point-to-point operations on its pipeline group during the warm-up phase.

    Returns:
        Options with ``lazy_init=True`` when the default process group is a ProcessGroupNCCL2 group
        bound to a device and PyTorch has the option, else None (default options). ``lazy_init`` is
        specific to ProcessGroupNCCL2; on the legacy backend it would only replace the split with an
        eager per-group initialization. Each call returns a new object because PyTorch writes
        per-group settings into the options it is given.
    """
    if not dist.is_initialized() or dist.get_backend() != dist.Backend.NCCL:
        return None
    default_group = dist.group.WORLD
    device = getattr(default_group, "bound_device_id", None)
    nccl2_backend = getattr(dist, "ProcessGroupNCCL2", None)
    if device is None or nccl2_backend is None:
        return None
    try:
        backend = default_group._get_backend(device)
    except RuntimeError:
        return None
    # TORCH_DISTRIBUTED_DEBUG=DETAIL wraps backends for collective checks; look through it as PyTorch's split does.
    debug_wrapper = getattr(getattr(dist, "distributed_c10d", None), "_ProcessGroupWrapper", None)
    while debug_wrapper is not None and isinstance(backend, debug_wrapper):
        backend = backend.wrapped_pg
    if not isinstance(backend, nccl2_backend):
        return None
    options = dist.ProcessGroupNCCL.Options()
    if not hasattr(options, "lazy_init"):
        return None
    options.lazy_init = True
    return options


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

    The groups are created with lazy NCCL initialization when the backend would otherwise split
    every group eagerly (see ``_lazy_nccl_pg_options``), so a group allocates NCCL resources only
    once it communicates.

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
            _ = grid.create_pg(dims, pg_options=_lazy_nccl_pg_options())

        for dims in (
            ["ep"],
            ["expt_tp"],
            ["expt_dp"],
            ["expt_tp", "ep"],
            ["expt_tp", "ep", "pp"],
        ):
            _ = grid.create_pg(dims, view=EXPERT_VIEW_NAME, pg_options=_lazy_nccl_pg_options())

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
        pos_embd_pg = dist.new_group(ranks=pos_embd_ranks, pg_options=_lazy_nccl_pg_options())

        embd_ranks = [pp_ranks[0]]
        if len(pp_ranks) > 1 and pp_ranks[-1] != pp_ranks[0]:
            embd_ranks.append(pp_ranks[-1])
        embd_pg = dist.new_group(ranks=embd_ranks, pg_options=_lazy_nccl_pg_options())

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
