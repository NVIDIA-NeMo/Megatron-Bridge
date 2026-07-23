# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Multi-module process group utilities for MegatronMIMO heterogeneous parallel training.

This module provides utilities for building process group structures and handling
gradients across modules with different parallelism configurations.

Key functions:
- unwrap_megatron_mimo_model(): Unwrap Float16Module/DDP to get underlying MimoModel
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

from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOInfra


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.process_groups_config import ProcessGroupCollection


logger = logging.getLogger(__name__)


def _get_dp_size_from_grid(grid: "HyperCommGrid") -> int:
    """Get the DP dimension size from a grid's shape metadata.

    Uses grid.shape / grid.dim_names rather than process groups so that
    it works on ALL ranks, including those outside the grid.
    """
    dp_idx = grid.dim_names.index("dp")
    return grid.shape[dp_idx]


def unwrap_megatron_mimo_model(model) -> MimoModel:
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
    """Check if current rank participates in the given grid.

    Args:
        grid: HyperCommGrid to check participation in.

    Returns:
        True if current rank is within the grid's rank range.
    """
    current_rank = dist.get_rank()
    return grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)


def get_active_module_pg(megatron_mimo_infra: MegatronMIMOInfra) -> tuple[str, "ProcessGroupCollection"]:
    """Return the (module_name, pg_collection) for the single active module on this rank.

    Non-colocated MegatronMIMO assigns each rank to exactly one module.  This helper
    extracts that module's name and ``ProcessGroupCollection``.

    Raises:
        AssertionError: If more or fewer than one module is active on this rank.
    """
    active = [(name, pg) for name, pg in megatron_mimo_infra.pg_collections.items() if pg is not None]
    assert len(active) == 1, (
        f"Non-colocated MegatronMIMO requires exactly one active ProcessGroupCollection per rank, "
        f"got {len(active)}. Colocated MegatronMIMO is not supported by this code path."
    )
    return active[0]


def get_module_to_grid_tuple(
    megatron_mimo_model: MimoModel,
    infra: MegatronMIMOInfra,
) -> List[Tuple]:
    """Build list of (module, grid) tuples for all modules the current rank participates in.

    Args:
        megatron_mimo_model: The MimoModel instance.
        infra: MegatronMIMOInfra containing module_to_grid_map.

    Returns:
        List of (module, grid) tuples for modules this rank participates in.
    """
    module_to_grid_tuple = []

    # Unwrap Float16Module/DDP if present (used in mixed precision training)
    unwrapped_model = unwrap_megatron_mimo_model(megatron_mimo_model)

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


def build_pg_collection_for_schedule(infra: MegatronMIMOInfra):
    """Build pg_collection compatible with schedule.

    Primary: Use MultiModuleProcessGroupCollection if PR 3212 allows
             missing LLM PG on encoder-only ranks.
    Fallback: Return list of ProcessGroupCollections for participating modules.

    IMPORTANT: Uses infra.pg_collections directly. Do NOT rebuild PGs.

    Args:
        infra: MegatronMIMOInfra with pg_collections for each module.

    Returns:
        MultiModuleProcessGroupCollection or list of ProcessGroupCollections.
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
    force_all_reduce=None,
    *,
    infra: MegatronMIMOInfra,
    module_to_grid_tuple: List[Tuple],
):
    """Finalize gradients for each module using infra.pg_collections.

    IMPORTANT: Signature matches schedule's call pattern:
        config.finalize_model_grads_func([model], num_tokens, pg_collection, force_all_reduce=flag)

    The `infra` and `module_to_grid_tuple` parameters are pre-bound via partial().
    We ignore the schedule-provided `pg_collection` and use per-module PGs.
    The schedule-provided `force_all_reduce` flag is forwarded to MCore's
    standard finalizer for each active module.

    When encoder DP > LLM DP (heterogeneous), the LLM's loss normalization
    divides by tokens for ALL samples it processes, but after bridge fan-out
    each encoder DP rank only carries gradient for (encoder_dp / llm_dp) fewer
    samples.  This makes encoder gradients too small by a factor of
    encoder_dp / llm_dp.  We compensate after DDP finalization by scaling
    encoder gradients back up.

    Args:
        model: Model list (passed by schedule, ignored - we use module_to_grid_tuple).
        num_tokens: Token count for gradient scaling.
        pg_collection: Schedule-provided PG (ignored - we use per-module PGs).
        force_all_reduce: Schedule-provided flag forwarded to each per-module finalizer.
        infra: MegatronMIMOInfra with per-module pg_collections (keyword-only, bound via partial).
        module_to_grid_tuple: List of (module, grid) tuples (keyword-only, bound via partial).
    """
    llm_grid = infra.module_to_grid_map.get(MIMO_LANGUAGE_MODULE_KEY)
    llm_dp = _get_dp_size_from_grid(llm_grid) if llm_grid is not None else 1

    def _find_module(grid):
        for mn, mg in infra.module_to_grid_map.items():
            if mg is grid:
                return mn, infra.pg_collections.get(mn)
        return None, None

    if num_tokens is not None and llm_grid is not None:
        # calculate_per_token_loss=True path.
        #
        # Only LLM last-PP-stage ranks accumulated non-zero num_tokens.
        # _finalize_model_grads does PP broadcast + DP all-reduce on
        # num_tokens internally, which works correctly for the LLM because
        # each DP rank still holds its own distinct accumulated count.
        #
        # We must NOT broadcast num_tokens globally before calling
        # _finalize_model_grads — that would overwrite every rank with one
        # DP rank's value, and the subsequent DP all-reduce would sum
        # dp_size identical copies instead of distinct per-rank counts.
        #
        # Encoder ranks have num_tokens=0 (they don't compute loss).  We
        # pass num_tokens=None for them to skip the broken normalization,
        # then broadcast the correct total from LLM and apply it manually.
        #
        # With gradient_scaling_factor=1.0 (calculate_per_token_loss=True),
        # DDP does a plain SUM.  After dividing by the global token count
        # the gradient is correct — no DP compensation factor needed.

        # Phase 1: gradient all-reduce for each module.  Only the LLM gets
        # num_tokens so _finalize_model_grads can PP-broadcast + DP-all-reduce
        # the per-rank counts into the correct global total.
        for module, grid in module_to_grid_tuple:
            if module is not None and is_current_rank_in_grid(grid):
                module_name, module_pg = _find_module(grid)
                if module_pg is not None:
                    if module_name == MIMO_LANGUAGE_MODULE_KEY:
                        _finalize_model_grads(
                            [module],
                            num_tokens=num_tokens,
                            pg_collection=module_pg,
                            force_all_reduce=force_all_reduce,
                        )
                    else:
                        _finalize_model_grads(
                            [module],
                            num_tokens=None,
                            pg_collection=module_pg,
                            force_all_reduce=force_all_reduce,
                        )

        # Phase 2: broadcast the correct global total from LLM to encoder
        # ranks.  _finalize_model_grads updated num_tokens in-place on LLM
        # ranks (PP broadcast + DP all-reduce → true global total).
        llm_last_rank = llm_grid.rank_offset + llm_grid.size - 1
        dist.broadcast(num_tokens, src=llm_last_rank)

        # Phase 3: scale encoder gradients by 1 / global_total.
        for module, grid in module_to_grid_tuple:
            if module is not None and is_current_rank_in_grid(grid):
                module_name, _ = _find_module(grid)
                if module_name != MIMO_LANGUAGE_MODULE_KEY and num_tokens > 0:
                    module.scale_gradients(1.0 / num_tokens.float().item())
    else:
        # calculate_per_token_loss=False path.
        #
        # Loss was already divided by num_tokens and num_microbatches in the
        # forward pass.  DDP pre-scales gradients by 1/dp_size, producing an
        # effective MEAN across DP ranks.  When encoder_dp > llm_dp the
        # encoder mean is over fewer samples, making encoder gradients too
        # small by encoder_dp / llm_dp.  Compensate after finalization.
        for module, grid in module_to_grid_tuple:
            if module is not None and is_current_rank_in_grid(grid):
                _, module_pg = _find_module(grid)
                if module_pg is not None:
                    _finalize_model_grads(
                        [module],
                        num_tokens=None,
                        pg_collection=module_pg,
                        force_all_reduce=force_all_reduce,
                    )

                    module_dp = _get_dp_size_from_grid(grid)
                    if module_dp != llm_dp:
                        module.scale_gradients(float(module_dp) / float(llm_dp))


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
    infra: MegatronMIMOInfra,
    global_batch_size: int,
    micro_batch_size: int,
    num_microbatches: int,
):
    """Validate data loading constraints for multimodule training.

    Checks:
    - MIMO micro-batch size divisible by all module DP sizes
    - Global batch size divisible by all module DP sizes
    - num_microbatches * micro_batch_size == global_batch_size

    Args:
        infra: MegatronMIMOInfra with module_to_grid_map.
        global_batch_size: Total MIMO batch size per optimizer step.
        micro_batch_size: Global MIMO batch size per microbatch before module-local DP slicing.
        num_microbatches: Number of microbatches per iteration.

    Raises:
        ValueError: If any constraint is violated.
    """
    expected = num_microbatches * micro_batch_size
    if expected != global_batch_size:
        raise ValueError(
            f"Microbatch mismatch: {num_microbatches} * {micro_batch_size} = {expected} "
            f"!= global_batch_size ({global_batch_size})"
        )

    for module_name, grid in infra.module_to_grid_map.items():
        # Get DP size from grid
        dp_size = grid.get_pg_size(["dp"])

        if micro_batch_size % dp_size != 0:
            raise ValueError(f"Micro batch size {micro_batch_size} not divisible by {module_name} DP size {dp_size}")

        # Check global batch divisibility
        if global_batch_size % dp_size != 0:
            raise ValueError(f"Global batch size {global_batch_size} not divisible by {module_name} DP size {dp_size}")
