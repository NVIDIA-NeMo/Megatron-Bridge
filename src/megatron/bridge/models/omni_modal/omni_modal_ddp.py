# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""DDP wrapping utilities for OmniModal models.

Called from the training layer after OmniModalProvider.provide().

Note: This module only supports DDP wrapping. FSDP is not yet implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.models.mimo import MimoModel
    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.models.omni_modal.omni_modal_config import OmniModalParallelismConfig


def wrap_omni_modal_model_distributed(
    omni_modal_model: "MimoModel",
    ddp_config: "DistributedDataParallelConfig",
    omni_modal_parallelism_config: "OmniModalParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
    pg_collections: Dict[str, Optional["ProcessGroupCollection"]],
) -> "MimoModel":
    """Wrap OmniModal model's submodules with DDP.

    Modifies omni_modal_model in-place and returns it.

    Args:
        omni_modal_model: The MimoModel to wrap.
        ddp_config: DDP configuration from Bridge.
        omni_modal_parallelism_config: OmniModal parallelism configuration.
        grids: Module name to HyperCommGrid mapping.
        pg_collections: Module name to ProcessGroupCollection mapping.

    Returns:
        The same omni_modal_model with wrapped submodules.
    """
    from megatron.core.distributed import DistributedDataParallel

    # Lazy import to avoid circular dependency (models layer loads before training layer)
    from megatron.bridge.training.omni_modal_parallel_utils import is_current_rank_in_grid

    # Wrap language model if present and rank participates
    if omni_modal_model.language_model is not None:
        llm_grid = grids.get(MIMO_LANGUAGE_MODULE_KEY)
        if llm_grid is not None and is_current_rank_in_grid(llm_grid):
            llm_pg = pg_collections.get(MIMO_LANGUAGE_MODULE_KEY)
            if llm_pg is not None:
                wrapped_lm = DistributedDataParallel(
                    config=omni_modal_model.language_model.config,
                    ddp_config=ddp_config,
                    module=omni_modal_model.language_model,
                    pg_collection=llm_pg,
                )
                # MCore's DDP wrapper does not proxy arbitrary module methods.
                # MimoModel._forward_language_module() checks for and calls
                # language_model.set_input_tensor(...) on non-first PP stages.
                # Preserve that method on the wrapper so decoder input tensors
                # are wired correctly when language_model is DDP-wrapped.
                if hasattr(wrapped_lm.module, "set_input_tensor"):
                    wrapped_lm.set_input_tensor = wrapped_lm.module.set_input_tensor
                omni_modal_model.language_model = wrapped_lm

    # Wrap modality submodules
    if hasattr(omni_modal_model, "modality_submodules"):
        for module_name, submodule in omni_modal_model.modality_submodules.items():
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
            # parallelism settings, which is typical for OmniModal models.
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
                omni_modal_model.modality_submodules[module_name] = wrapped

    return omni_modal_model
