# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""DDP wrapping utilities for MegatronMIMO models.

Called from the training layer after MegatronMIMOProvider.provide().

Note: This module only supports DDP wrapping. FSDP is not yet implemented.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, Optional

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from torch import nn

from megatron.bridge.utils.common_utils import warn_rank_0


if TYPE_CHECKING:
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.models.mimo import MimoModel
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


def wrap_megatron_mimo_model_distributed(
    megatron_mimo_model: "MimoModel",
    ddp_config: "DistributedDataParallelConfig",
    megatron_mimo_parallelism_config: "MegatronMIMOParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
    pg_collections: Dict[str, Optional["ProcessGroupCollection"]],
    *,
    use_layer_wise_distributed_optimizer: bool = False,
    use_layer_wise_param_layout: bool = False,
) -> "MimoModel":
    """Wrap MegatronMIMO model's submodules with DDP.

    Modifies megatron_mimo_model in-place and returns it.

    Args:
        megatron_mimo_model: The MimoModel to wrap.
        ddp_config: DDP configuration from Bridge.
        megatron_mimo_parallelism_config: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping.
        pg_collections: Module name to ProcessGroupCollection mapping.
        use_layer_wise_distributed_optimizer: Build shard-aligned layouts for layer-wise optimizers.
        use_layer_wise_param_layout: Use shard-aligned DDP buffers for LayerWise optimizers.
            False retains compact DDP buffers and variable-size parameter synchronization.

    Returns:
        The same megatron_mimo_model with wrapped submodules.
    """
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
        tag_params_for_buffer_routing,
    )

    # Lazy import to avoid circular dependency (models layer loads before training layer)
    from megatron.bridge.training.megatron_mimo_parallel_utils import is_current_rank_in_grid

    disable_encoder_overlap = ddp_config.overlap_grad_reduce is True or ddp_config.overlap_param_gather is True
    has_modality_grid = any(
        module_name != MIMO_LANGUAGE_MODULE_KEY and grid is not None for module_name, grid in grids.items()
    )
    if disable_encoder_overlap and has_modality_grid:
        # Emit before per-grid rank filtering so global rank zero reports the
        # override even when it does not participate in an encoder grid.
        warn_rank_0(
            "Disabling gradient-reduce and parameter-gather overlap for modality encoders because "
            "not every rank is guaranteed to execute an encoder backward pass."
        )

    def wrap_module(
        module: nn.Module,
        config: "TransformerConfig",
        pg: "ProcessGroupCollection",
        module_ddp_config: "DistributedDataParallelConfig",
    ) -> "DistributedDataParallel":
        layout = None
        if use_layer_wise_distributed_optimizer:
            module_ddp_config = dataclasses.replace(
                module_ddp_config,
                use_distributed_optimizer=use_layer_wise_param_layout,
            )
        if use_layer_wise_distributed_optimizer and use_layer_wise_param_layout:
            tag_params_for_buffer_routing([module])
            groups = pg.setup_process_groups_for_ddp(pg, config, module_ddp_config)
            bucket_size = module_ddp_config.bucket_size
            if module_ddp_config.num_buckets is not None:
                bucket_size = sum(p.numel() for p in module.parameters()) // module_ddp_config.num_buckets
            elif bucket_size is None:
                bucket_size = max(40000000, 1000000 * groups["dp_group"].size())
            if not module_ddp_config.overlap_grad_reduce or groups["pp_group"].rank() > 0:
                bucket_size = None
            layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
                [p for p in module.parameters() if p.requires_grad],
                bucket_size,
                groups["intra_dp_cp_group"].size(),
                module_ddp_config,
                expert_data_parallel_world_size=groups["intra_expt_dp_group"].size(),
            )
        return DistributedDataParallel(
            config=config,
            ddp_config=module_ddp_config,
            module=module,
            pg_collection=pg,
            full_param_layout=layout,
        )

    # Wrap language model if present and rank participates
    if megatron_mimo_model.language_model is not None:
        llm_grid = grids.get(MIMO_LANGUAGE_MODULE_KEY)
        if llm_grid is not None and is_current_rank_in_grid(llm_grid):
            llm_pg = pg_collections.get(MIMO_LANGUAGE_MODULE_KEY)
            if llm_pg is not None:
                wrapped_lm = wrap_module(
                    megatron_mimo_model.language_model,
                    megatron_mimo_model.language_model.config,
                    llm_pg,
                    ddp_config,
                )
                # MCore's DDP wrapper does not proxy arbitrary module methods.
                # MimoModel._forward_language_module() checks for and calls
                # language_model.set_input_tensor(...) on non-first PP stages.
                # Preserve that method on the wrapper so decoder input tensors
                # are wired correctly when language_model is DDP-wrapped.
                if hasattr(wrapped_lm.module, "set_input_tensor"):
                    wrapped_lm.set_input_tensor = wrapped_lm.module.set_input_tensor
                megatron_mimo_model.language_model = wrapped_lm

    # Wrap modality submodules
    if hasattr(megatron_mimo_model, "modality_submodules"):
        for module_name, submodule in megatron_mimo_model.modality_submodules.items():
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
            # parallelism settings, which is typical for MegatronMIMO models.
            if hasattr(submodule, "encoders") and submodule.encoders:
                encoder_key = next(iter(submodule.encoders.keys()))
                first_encoder = submodule.encoders[encoder_key]

                if not hasattr(first_encoder, "config"):
                    raise AttributeError(
                        f"Encoder '{encoder_key}' in modality '{module_name}' does not have "
                        f"a 'config' attribute. Encoders must be MegatronModule subclasses."
                    )

                # Uneven input participation can leave encoder replicas
                # unsynchronized when communication is hook-driven.
                encoder_ddp_config = (
                    dataclasses.replace(ddp_config, overlap_grad_reduce=False, overlap_param_gather=False)
                    if disable_encoder_overlap
                    else ddp_config
                )
                wrapped = wrap_module(
                    submodule,
                    first_encoder.config,
                    module_pg,
                    encoder_ddp_config,
                )
                # MCore optimizers unwrap DDP and read process groups from the wrapped module.
                wrapped.module.pg_collection = module_pg
                megatron_mimo_model.modality_submodules[module_name] = wrapped

    return megatron_mimo_model
