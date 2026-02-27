# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Multi-module process group utilities for MIMO heterogeneous parallel training.

This module provides utilities for building process group structures and handling
gradients across modules with different parallelism configurations.

Key functions:
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

from megatron.bridge.models.mimo.mimo_provider import MimoModelInfra


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


logger = logging.getLogger(__name__)


def is_current_rank_in_grid(grid: "HyperCommGrid") -> bool:
    """Check if current rank participates in the given grid.

    Args:
        grid: HyperCommGrid to check participation in.

    Returns:
        True if current rank is within the grid's rank range.
    """
    current_rank = dist.get_rank()
    return grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)


def get_module_to_grid_tuple(
    mimo_model: MimoModel,
    infra: MimoModelInfra,
) -> List[Tuple]:
    """Build list of (module, grid) tuples for all modules the current rank participates in.

    Args:
        mimo_model: The MimoModel instance.
        infra: MimoModelInfra containing module_to_grid_map.

    Returns:
        List of (module, grid) tuples for modules this rank participates in.
    """
    module_to_grid_tuple = []

    # Unwrap Float16Module if present
    _model = mimo_model.module if hasattr(mimo_model, "module") else mimo_model

    lang_key = _model.mimo_config.language_module_key

    for module_name, grid in infra.module_to_grid_map.items():
        if not is_current_rank_in_grid(grid):
            continue

        # Get the actual module from the model
        if module_name == lang_key:
            module = _model.language_model
        elif hasattr(_model, "modality_submodules") and module_name in _model.modality_submodules:
            module = _model.modality_submodules[module_name]
        else:
            logger.warning(f"Module {module_name} not found in MimoModel, skipping")
            continue

        module_to_grid_tuple.append((module, grid))

    return module_to_grid_tuple


def build_pg_collection_for_schedule(infra: MimoModelInfra, language_module_key: str):
    """Build pg_collection compatible with schedule.

    Primary: Use MultiModuleProcessGroupCollection if PR 3129 allows
             missing LLM PG on encoder-only ranks.
    Fallback: Return list of ProcessGroupCollections for participating modules.

    IMPORTANT: Uses infra.pg_collections directly. Do NOT rebuild PGs.

    Args:
        infra: MimoModelInfra with pg_collections for each module.
        language_module_key: Key identifying the language module (e.g. "llm").

    Returns:
        MultiModuleProcessGroupCollection or list of ProcessGroupCollections.
    """
    try:
        from megatron.core.process_groups_config import MultiModuleProcessGroupCollection

        module_pgs = {k: v for k, v in infra.pg_collections.items() if v is not None}

        # Only set language_model_module_name if this rank participates in the LLM
        lang_module_name = language_module_key if language_module_key in module_pgs else None

        return MultiModuleProcessGroupCollection(
            module_pgs=module_pgs,
            language_model_module_name=lang_module_name,
        )
    except (ImportError, ValueError, TypeError) as e:
        # Fallback: list-based approach (reference implementation pattern)
        logger.warning(f"MultiModuleProcessGroupCollection failed ({e}), using list-based fallback")
        return [pg for pg in infra.pg_collections.values() if pg is not None]


@contextmanager
def multimodule_no_sync(*, module_to_grid_tuple: List[Tuple]):
    """Context manager to disable gradient sync for all modules during microbatch accumulation.

    This function is designed to be used with functools.partial() to pre-bind
    the module_to_grid_tuple parameter, since the schedule calls no_sync_func()
    with no arguments.

    Args:
        module_to_grid_tuple: List of (module, grid) tuples (keyword-only, bound via partial).

    Yields:
        None - context manager for gradient sync control.
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
    *,
    infra: MimoModelInfra,
    module_to_grid_tuple: List[Tuple],
    **kwargs,
):
    """Finalize gradients for each module using infra.pg_collections.

    IMPORTANT: Signature matches schedule's call pattern:
        config.finalize_model_grads_func([model], num_tokens, pg_collection,
                                         force_all_reduce=...)

    The `infra` and `module_to_grid_tuple` parameters are pre-bound via partial().
    We ignore the schedule-provided `pg_collection` and use per-module PGs.
    Extra kwargs from the schedule (e.g. `force_all_reduce`) are forwarded to
    the per-module finalize_model_grads call.

    Args:
        model: Model list (passed by schedule, ignored - we use module_to_grid_tuple).
        num_tokens: Token count for gradient scaling.
        pg_collection: Schedule-provided PG (ignored - we use per-module PGs).
        infra: MimoModelInfra with per-module pg_collections (keyword-only, bound via partial).
        module_to_grid_tuple: List of (module, grid) tuples (keyword-only, bound via partial).
        **kwargs: Additional keyword args from schedule (e.g. force_all_reduce).
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
                _finalize_model_grads([module], num_tokens=num_tokens, pg_collection=module_pg, **kwargs)


def zero_grad_buffer_for_multimodule(module_to_grid_tuple: List[Tuple]):
    """Reset gradient buffers for all DDP-wrapped modules.

    Args:
        module_to_grid_tuple: List of (module, grid) tuples.
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
    """Validate data loading constraints for multimodule training.

    Checks:
    - Global batch size divisible by all module DP sizes
    - Micro-batch size consistent with per-module sharding
    - num_microbatches * micro_batch_size == global_batch_size / DP_size (per module)

    Args:
        infra: MimoModelInfra with module_to_grid_map.
        global_batch_size: Total batch size across all data parallel ranks.
        micro_batch_size: Batch size per microbatch.
        num_microbatches: Number of microbatches per iteration.

    Raises:
        ValueError: If any constraint is violated.
    """
    for module_name, grid in infra.module_to_grid_map.items():
        # Get DP size from grid shape (works on all ranks)
        dp_size = grid.shape[grid.dim_names.index("dp")]

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
