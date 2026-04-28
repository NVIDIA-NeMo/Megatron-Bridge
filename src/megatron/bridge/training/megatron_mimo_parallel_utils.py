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
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch.distributed as dist
from megatron.core.distributed.finalize_model_grads import finalize_model_grads as _finalize_model_grads
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.utils import get_pp_last_rank
from megatron.core.utils import get_model_config

from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOInfra


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.process_groups_config import ProcessGroupCollection


logger = logging.getLogger(__name__)


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


def get_active_module_pgs(megatron_mimo_infra: MegatronMIMOInfra) -> Dict[str, "ProcessGroupCollection"]:
    """Return every active (module_name → pg_collection) on this rank.

    Non-colocated deployments put each rank in a single module → returns a
    one-entry dict. Colocated deployments put every rank in every module →
    returns multiple entries.
    """
    return {name: pg for name, pg in megatron_mimo_infra.pg_collections.items() if pg is not None}


def get_active_module_pg(
    megatron_mimo_infra: MegatronMIMOInfra,
    module_name: Optional[str] = None,
) -> tuple[str, "ProcessGroupCollection"]:
    """Return a single (module_name, pg_collection) for this rank.

    Used by callers that need **one** pg_collection to plumb through legacy
    code paths — most importantly the ``mpu.*`` global bridging in
    ``setup_megatron_mimo`` (mcore internals read these globals during
    ``sharded_state_dict`` and other utilities).

    Selection rule:
      * If ``module_name`` is provided, return that module's pg (error if not
        active on this rank).
      * Otherwise, if exactly one module is active (non-colocated), return it.
      * Otherwise (colocated — multiple modules active), default to the
        language module if it's active, else the first module. This picks a
        canonical pg for the globals; per-module operations should iterate
        :func:`get_active_module_pgs` instead.

    Raises:
        AssertionError: If no module is active on this rank (stub rank — invalid).
        KeyError: If ``module_name`` is provided but not active on this rank.
    """
    active = get_active_module_pgs(megatron_mimo_infra)
    assert active, "MegatronMIMO requires every rank to participate in at least one module; none found."

    if module_name is not None:
        if module_name not in active:
            raise KeyError(f"Module '{module_name}' is not active on this rank. Active modules: {sorted(active)}")
        return module_name, active[module_name]

    if len(active) == 1:
        return next(iter(active.items()))

    # Colocated: prefer language module as canonical; fall back to first entry.
    if MIMO_LANGUAGE_MODULE_KEY in active:
        return MIMO_LANGUAGE_MODULE_KEY, active[MIMO_LANGUAGE_MODULE_KEY]
    return next(iter(active.items()))


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


def _validate_per_token_loss_active_modules(
    module_to_grid_tuple: List[Tuple],
    infra: MegatronMIMOInfra,
) -> None:
    """Assert every active module on this rank uses ``calculate_per_token_loss=True``.

    ``finalize_model_grads_multimodule`` applies a uniform ``1/global_num_tokens``
    scale to every module via ``module.scale_gradients(...)``. That math is only
    correct when each module's ``TransformerConfig.calculate_per_token_loss`` is
    True — that flag pins mcore's DDP ``gradient_scaling_factor`` to 1.0 so DDP
    performs pure SUM across DP. With the flag off, mcore's DDP applies its own
    per-microbatch averaging, and our global-SUM denominator on top mis-scales
    the result.

    Called from the top of ``finalize_model_grads_multimodule`` only when
    ``num_tokens is not None`` (the per-token-loss path). Symmetric MIMO setups
    with ``calculate_per_token_loss=False`` keep working — they pass
    ``num_tokens=None`` from the schedule, and this check doesn't fire.
    """
    grid_to_name = {id(grid): name for name, grid in infra.module_to_grid_map.items()}
    offenders: List[str] = []
    for module, grid in module_to_grid_tuple:
        if module is None or not is_current_rank_in_grid(grid):
            continue
        try:
            cfg = get_model_config(module)
        except Exception:
            # Couldn't read config — can't check. Skip rather than break: a real
            # config-shape problem will surface elsewhere (init/forward) with
            # a more informative trace than a precondition check could give.
            continue
        if not getattr(cfg, "calculate_per_token_loss", False):
            name = grid_to_name.get(id(grid), type(module).__name__)
            offenders.append(name)
    if offenders:
        raise ValueError(
            f"finalize_model_grads_multimodule received num_tokens != None for "
            f"per-token gradient scaling, but the following modules don't have "
            f"calculate_per_token_loss=True on their TransformerConfig: "
            f"{offenders}. Without it, mcore's DDP applies per-microbatch "
            f"averaging and the global-SUM denominator we apply on top "
            f"mis-scales gradients. Set calculate_per_token_loss=True on every "
            f"module's TransformerConfig, or use a loss function that doesn't "
            f"pass num_tokens."
        )


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

    Args:
        model: Model list (passed by schedule, ignored - we use module_to_grid_tuple).
        num_tokens: Language-local token count for per-token loss scaling.
        pg_collection: Schedule-provided PG (ignored - we use per-module PGs).
        force_all_reduce: Schedule-provided flag (ignored - per-module PGs control sync).
        infra: MegatronMIMOInfra with per-module pg_collections (keyword-only, bound via partial).
        module_to_grid_tuple: List of (module, grid) tuples (keyword-only, bound via partial).
    """
    if num_tokens is not None:
        # Per-token gradient scaling is only correct if every active module's
        # DDP performs pure SUM across DP — see helper docstring.
        _validate_per_token_loss_active_modules(module_to_grid_tuple, infra)

    global_num_tokens = None
    language_pg = infra.pg_collections.get(MIMO_LANGUAGE_MODULE_KEY)
    language_grid = infra.module_to_grid_map.get(MIMO_LANGUAGE_MODULE_KEY)
    if num_tokens is not None and language_grid is not None:
        global_num_tokens = num_tokens.clone()
        if language_pg is not None:
            dist.broadcast(global_num_tokens, src=get_pp_last_rank(language_pg.pp), group=language_pg.pp)
            language_dp_cp_group = getattr(language_pg, "dp_cp", None)
            if language_dp_cp_group is None:
                language_dp_cp_group = language_pg.dp
            dist.all_reduce(global_num_tokens, group=language_dp_cp_group, op=dist.ReduceOp.SUM)
        if any(pg is None for pg in infra.pg_collections.values()):
            dist.broadcast(global_num_tokens, src=language_grid.rank_offset, group=dist.group.WORLD)
    grad_scale = None
    if global_num_tokens is not None:
        global_num_tokens_value = global_num_tokens.item()
        if global_num_tokens_value > 0:
            grad_scale = 1.0 / global_num_tokens_value

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
                # MCore's token scaling path assumes the token count belongs
                # to the same DP/CP group as the module being finalized. In
                # MIMO the loss count is produced by the language module, so
                # compute the global denominator once from the language group
                # and apply it uniformly after each module's grad sync.
                module_num_tokens = None if global_num_tokens is not None else num_tokens
                if module_num_tokens is not None:
                    module_num_tokens = module_num_tokens.clone()
                # Propagate ``force_all_reduce``. Under the distributed optimizer:
                #   force_all_reduce=True  → full all-reduce (every DP rank gets
                #                            the full gradient tensor).
                #   force_all_reduce=False → reduce-scatter (each DP rank gets
                #                            only its parameter shard).
                # The schedule chooses based on ``overlap_grad_reduce`` and
                # ``use_distributed_optimizer`` settings. Forwarding it
                # unchanged respects that choice.
                #
                # Regression footnote: an earlier version dropped this kwarg.
                # Under ``overlap_grad_reduce=False`` that turned
                # ``finish_grad_sync`` into a no-op (no in-flight async ops to
                # wait on, no forced all-reduce to fire), and grads stayed
                # un-synced across DP. The kwarg is load-bearing for that
                # configuration; the per-token-loss oracle is what surfaced
                # the original miss.
                _finalize_model_grads(
                    [module],
                    num_tokens=module_num_tokens,
                    pg_collection=module_pg,
                    force_all_reduce=force_all_reduce,
                )
                if grad_scale is not None:
                    module.scale_gradients(grad_scale)


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
