# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""DDP wrapping utilities for MIMO models.

Called from the training layer after MimoModelProvider.provide().

Note: This module only supports DDP wrapping. FSDP is not yet implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from megatron.bridge.models.mimo.mimo_builder import is_current_rank_in_grid


if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.models.mimo import MimoModel
    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig


def wrap_mimo_model_distributed(
    mimo_model: "MimoModel",
    ddp_config: "DistributedDataParallelConfig",
    mimo_parallelism_config: "MimoParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
    pg_collections: Dict[str, Optional["ProcessGroupCollection"]],
) -> "MimoModel":
    """Wrap MIMO model's submodules with DDP.

    Modifies mimo_model in-place and returns it.

    Args:
        mimo_model: The MimoModel to wrap.
        ddp_config: DDP configuration from Bridge.
        mimo_parallelism_config: MIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping.
        pg_collections: Module name to ProcessGroupCollection mapping.

    Returns:
        The same mimo_model with wrapped submodules.
    """
    from megatron.core.distributed import DistributedDataParallel

    # Wrap language model if present and rank participates
    if mimo_model.language_model is not None:
        llm_grid = grids.get("llm")
        if llm_grid is not None and is_current_rank_in_grid(llm_grid):
            llm_pg = pg_collections.get("llm")
            if llm_pg is not None:
                mimo_model.language_model = DistributedDataParallel(
                    config=mimo_model.language_model.config,
                    ddp_config=ddp_config,
                    module=mimo_model.language_model,
                    pg_collection=llm_pg,
                )

    # Wrap modality submodules
    if hasattr(mimo_model, "modality_submodules"):
        for module_name, submodule in mimo_model.modality_submodules.items():
            if submodule is None:
                continue
            module_grid = grids.get(module_name)
            if module_grid is None:
                continue
            if not is_current_rank_in_grid(module_grid):
                continue

            module_pg = pg_collections.get(module_name)
            if module_pg is None:
                continue

            # Get config from first encoder in the submodule.
            # Note: We use the first encoder's config for DDP bucket sizing.
            # This assumes all encoders in a modality submodule share similar
            # parallelism settings, which is typical for MIMO models.
            if hasattr(submodule, "encoders") and submodule.encoders:
                encoder_key = next(iter(submodule.encoders.keys()))
                first_encoder = submodule.encoders[encoder_key]

                if not hasattr(first_encoder, "config"):
                    raise AttributeError(
                        f"Encoder '{encoder_key}' in modality '{module_name}' does not have "
                        f"a 'config' attribute. Encoders must be MegatronModule subclasses."
                    )

                wrapped = DistributedDataParallel(
                    config=first_encoder.config,
                    ddp_config=ddp_config,
                    module=submodule,
                    pg_collection=module_pg,
                )
                mimo_model.modality_submodules[module_name] = wrapped

    return mimo_model
