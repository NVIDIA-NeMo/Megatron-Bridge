# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Multi-module process group utilities for MIMO heterogeneous parallel training.

This module provides utilities for building process group structures and handling
gradients across modules with different parallelism configurations.

Key functions:
- unwrap_mimo_model(): Unwrap Float16Module/DDP to get underlying MimoModel
- build_pg_collection_for_schedule(): Build pg_collection compatible with schedule
- multimodule_no_sync(): Context manager for gradient sync during microbatch accumulation
- finalize_model_grads_multimodule(): Finalize gradients for each module
- zero_grad_buffer_for_multimodule(): Reset gradient buffers for all modules
- validate_no_stub_ranks(): Ensure every rank participates in at least one module
- validate_data_loader_contract(): Validate data loading constraints
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch.distributed as dist
from megatron.core.distributed.finalize_model_grads import finalize_model_grads as _finalize_model_grads
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

from megatron.bridge.models.mimo.mimo_provider import MimoModelInfra


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


logger = logging.getLogger(__name__)


def unwrap_mimo_model(model) -> MimoModel:
    """Unwrap Float16Module/DDP wrappers to get the underlying MimoModel.

    When using mixed precision (bf16/fp16), models are wrapped in Float16Module.
    This function unwraps the model to access MimoModel-specific attributes
    like `role`, `mimo_config`, `language_model`, `modality_submodules`, etc.

    Args:
        model: A MimoModel or a wrapped version (Float16Module, DDP).

    Returns:
        The underlying MimoModel instance.

    Raises:
        RuntimeError: If the model cannot be unwrapped to a MimoModel.
    """
    unwrapped = model
    while not isinstance(unwrapped, MimoModel) and hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    if not isinstance(unwrapped, MimoModel):
        raise RuntimeError(f"Failed to unwrap model to MimoModel, got {type(unwrapped)}")
    return unwrapped


def is_current_rank_in_grid(grid: "HyperCommGrid") -> bool:
    """
    Determine whether the current distributed process rank lies within the grid's contiguous rank range.
    
    Parameters:
        grid (HyperCommGrid): Grid whose rank range will be checked.
    
    Returns:
        `true` if the current process rank is within the grid's rank range, `false` otherwise.
    """
    current_rank = dist.get_rank()
    return grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)


def get_module_to_grid_tuple(
    mimo_model: MimoModel,
    infra: MimoModelInfra,
) -> List[Tuple]:
    """
    Map participating submodules of the provided MimoModel to their corresponding HyperCommGrid entries.
    
    Parameters:
        mimo_model (MimoModel): The MimoModel to inspect; wrapped models (e.g., DDP/Float16Module) are unwrapped.
        infra (MimoModelInfra): Infrastructure containing `module_to_grid_map` that associates module names with grids.
    
    Returns:
        List[Tuple]: A list of (module, grid) tuples for each submodule whose grid includes the current rank.
    """
    module_to_grid_tuple = []

    # Unwrap Float16Module/DDP if present (used in mixed precision training)
    unwrapped_model = unwrap_mimo_model(mimo_model)

    for module_name, grid in infra.module_to_grid_map.items():
        if not is_current_rank_in_grid(grid):
            continue

        # Get the actual module from the unwrapped model
        if module_name == MIMO_LANGUAGE_MODULE_KEY:
            module = unwrapped_model.language_model
        elif hasattr(unwrapped_model, "modality_submodules") and module_name in unwrapped_model.modality_submodules:
            module = unwrapped_model.modality_submodules[module_name]
        else:
            logger.warning(f"Module {module_name} not found in MimoModel, skipping")
            continue

        module_to_grid_tuple.append((module, grid))

    return module_to_grid_tuple


def build_pg_collection_for_schedule(infra: MimoModelInfra):
    """
    Constructs a schedule-compatible process-group collection from the infra's per-module process groups.
    
    Prefers an aggregated MultiModuleProcessGroupCollection when available; otherwise returns a list of the existing per-module ProcessGroupCollection objects. Uses the mappings in `infra.pg_collections` and does not create or rebuild process groups.
    
    Parameters:
        infra (MimoModelInfra): Infrastructure object containing `pg_collections` mapping module names to their ProcessGroupCollection (or None).
    
    Returns:
        MultiModuleProcessGroupCollection or list: A MultiModuleProcessGroupCollection aggregating the per-module PGs when supported, or a list of the existing per-module ProcessGroupCollection instances otherwise.
    """
    try:
        from megatron.core.process_groups_config import MultiModuleProcessGroupCollection

        module_pgs = {k: v for k, v in infra.pg_collections.items() if v is not None}
        if not module_pgs:
            raise ValueError("module_pgs dict cannot be empty")
        language_model_module_name = MIMO_LANGUAGE_MODULE_KEY if MIMO_LANGUAGE_MODULE_KEY in module_pgs else None
        return MultiModuleProcessGroupCollection(
            module_pgs=module_pgs,
            language_model_module_name=language_model_module_name,
        )
    except (ImportError, ValueError, TypeError) as e:
        logger.warning(f"MultiModuleProcessGroupCollection failed ({e}), using list-based fallback")
        return [pg for pg in infra.pg_collections.values() if pg is not None]


@contextmanager
def multimodule_no_sync(*, module_to_grid_tuple: List[Tuple]):
    """
    Disable gradient synchronization for all participating modules by entering each module's `no_sync()` context.
    
    This context manager enters `no_sync()` for every module in `module_to_grid_tuple` whose grid includes the current rank, yielding control while those contexts are active and exiting them on completion.
    
    Parameters:
        module_to_grid_tuple (List[Tuple]): List of `(module, grid)` pairs; modules that are `None` or whose grid does not include the current rank are ignored.
    """
    contexts = []
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            contexts.append(module.no_sync())

    # Enter all contexts
    for ctx in contexts:
        ctx.__enter__()

    try:
        yield
    finally:
        # Exit all contexts in reverse order
        for ctx in reversed(contexts):
            ctx.__exit__(None, None, None)


def finalize_model_grads_multimodule(
    model,
    num_tokens=None,
    pg_collection=None,
    force_all_reduce=None,
    *,
    infra: MimoModelInfra,
    module_to_grid_tuple: List[Tuple],
):
    """
    Finalize gradients for each participating MIMO submodule using that module's process-group collection.
    
    For every (module, grid) in `module_to_grid_tuple` where the current rank belongs to `grid`, this calls Megatron's internal finalize routine for that single module using the per-module PG from `infra.pg_collections`. This function intentionally ignores the schedule-provided `pg_collection` and `force_all_reduce` arguments; `infra` and `module_to_grid_tuple` are expected to be pre-bound (e.g., via partial).
    
    Parameters:
        model: Ignored. Present to match the schedule's call signature.
        num_tokens: Token count forwarded to the underlying finalize call for gradient scaling.
        pg_collection: Ignored. Per-module PGs from `infra.pg_collections` are used instead.
        force_all_reduce: Ignored. Per-module PG behavior determines synchronization.
        infra: MimoModelInfra providing `module_to_grid_map` and `pg_collections`.
        module_to_grid_tuple: List of (module, grid) tuples indicating modules and their grids; only modules whose grid contains the current rank are finalized.
    """
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            # Get the module's pg_collection from infra
            # Find the module name by matching the grid
            module_pg = None
            for module_name, mod_grid in infra.module_to_grid_map.items():
                if mod_grid is grid:
                    module_pg = infra.pg_collections.get(module_name)
                    break

            if module_pg is not None:
                _finalize_model_grads([module], num_tokens=num_tokens, pg_collection=module_pg)


def zero_grad_buffer_for_multimodule(module_to_grid_tuple: List[Tuple]):
    """
    Reset gradient buffers for participating multimodule submodules that expose `zero_grad_buffer`.
    
    Parameters:
        module_to_grid_tuple (List[Tuple]): Sequence of `(module, grid)` pairs. For each pair,
            if the current rank is inside `grid` and `module` implements `zero_grad_buffer()`,
            that method will be invoked.
    """
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            if hasattr(module, "zero_grad_buffer"):
                module.zero_grad_buffer()


def validate_no_stub_ranks(module_to_grid_map: Dict[str, "HyperCommGrid"], world_size: int):
    """Ensure every rank participates in at least one module.

    Stub ranks (ranks not participating in any module) are NOT supported.
    This validation runs at setup time to fail fast with a clear error.

    Args:
        module_to_grid_map: Mapping of module names to their HyperCommGrids.
        world_size: Total number of ranks in the world.

    Raises:
        ValueError: If any rank doesn't participate in a module.
    """
    participating_ranks = set()
    for module_name, grid in module_to_grid_map.items():
        # Add all ranks in this grid's range
        for rank in range(grid.rank_offset, grid.rank_offset + grid.size):
            participating_ranks.add(rank)

    all_ranks = set(range(world_size))
    stub_ranks = all_ranks - participating_ranks

    if stub_ranks:
        raise ValueError(
            f"Ranks {sorted(stub_ranks)} do not participate in any module. "
            f"Stub ranks are not supported. Adjust parallelism config to use all {world_size} GPUs, "
            f"or reduce world_size to {len(participating_ranks)}."
        )


def validate_data_loader_contract(
    infra: MimoModelInfra,
    global_batch_size: int,
    micro_batch_size: int,
    num_microbatches: int,
):
    """
    Validate that global and microbatch sizes satisfy each module's data-parallel constraints.
    
    Checks per configured module that (1) `global_batch_size` is divisible by the module's data-parallel (DP) size and (2) `num_microbatches * micro_batch_size` equals the per-DP partition of `global_batch_size`.
    
    Parameters:
        infra (MimoModelInfra): Infrastructure containing `module_to_grid_map`.
        global_batch_size (int): Total batch size across all data-parallel ranks.
        micro_batch_size (int): Batch size for a single microbatch.
        num_microbatches (int): Number of microbatches accumulated per iteration.
    
    Raises:
        ValueError: If `global_batch_size` is not divisible by a module's DP size, or if
            `num_microbatches * micro_batch_size` does not equal `global_batch_size // dp_size`
            for any module.
    """
    for module_name, grid in infra.module_to_grid_map.items():
        # Get DP size from grid
        dp_size = grid.get_pg_size(["dp"])

        # Check global batch divisibility
        if global_batch_size % dp_size != 0:
            raise ValueError(f"Global batch size {global_batch_size} not divisible by {module_name} DP size {dp_size}")

        # Check micro-batch alignment
        per_dp_batch = global_batch_size // dp_size
        expected = num_microbatches * micro_batch_size
        if per_dp_batch != expected:
            raise ValueError(
                f"Microbatch mismatch for {module_name}: "
                f"{num_microbatches} * {micro_batch_size} = {expected} != {per_dp_batch} "
                f"(global_batch / DP_size)"
            )
