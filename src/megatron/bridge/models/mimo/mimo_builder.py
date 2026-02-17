# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig


if TYPE_CHECKING:
    from megatron.core.process_groups_config import HyperCommGrid


def build_hypercomm_grids(
    mimo_parallelism_config: MimoParallelismConfig,
) -> Dict[str, HyperCommGrid]:
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
        # Create dp_cp composite group for gradient reduction
        _ = grid.create_pg(["dp", "cp"])

        grids[module_name] = grid

    return grids


def _default_topology(mimo_parallelism_config: MimoParallelismConfig) -> Dict[str, List[str]]:
    """Infer a default multi-encoder -> LLM topology."""
    return {name: ["llm"] for name in mimo_parallelism_config.module_names if name != "llm"} | {"llm": []}
