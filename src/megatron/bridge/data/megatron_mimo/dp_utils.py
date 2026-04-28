# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data parallel utilities for MegatronMIMO data loading."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid

    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


logger = logging.getLogger(__name__)


def _find_rank_modules(
    grids: Dict[str, "HyperCommGrid"],
) -> Dict[str, "HyperCommGrid"]:
    """Return every module grid the current rank belongs to.

    In non-colocated deployments (disjoint rank ranges) this returns at most
    one entry. In colocated deployments (all grids share the same
    ``(rank_offset, size)``) every module grid is returned.
    """
    current_rank = dist.get_rank()
    return {
        name: grid for name, grid in grids.items() if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)
    }


def _needs_data_for_module(grid: "HyperCommGrid", module_name: str) -> bool:
    """Determine if the current rank needs to load data for the given module.

    LLM: first and last PP stage need data (input_ids and labels respectively).
    Encoders: only the first PP stage needs raw modality inputs.
    """
    pp_group = grid.get_pg(["pp"])
    pp_rank = pp_group.rank()
    pp_size = pp_group.size()
    if module_name == MIMO_LANGUAGE_MODULE_KEY:
        return (pp_rank == 0) or (pp_rank == pp_size - 1)
    return pp_rank == 0


def get_megatron_mimo_dp_info(
    megatron_mimo_cfg: "MegatronMIMOParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
    module_name: Optional[str] = None,
) -> Tuple[int, int, bool, str]:
    """Get **module-local** DP rank, size, data-loading flag, and module name.

    These values feed :func:`slice_batch_for_megatron_mimo`, which sub-shards a
    global micro-batch into per-module DP shards.

    .. note::
        Do **not** use these values to construct a ``DistributedSampler``.
        For sampler construction use :func:`get_megatron_mimo_sampling_info` instead,
        which returns settings that keep all data-loading ranks synchronised
        on the same sample order.

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping from build_hypercomm_grids().
        module_name: Explicit module to query. Required when the current rank
            participates in multiple modules (colocated mode) because a rank's
            encoder-DP and LLM-DP generally differ. If omitted and the rank
            serves multiple modules, defaults to the language module and emits
            a warning.

    Returns:
        Tuple of (dp_rank, dp_size, needs_data, loader_module).
    """
    rank_modules = _find_rank_modules(grids)
    if not rank_modules:
        return 0, 1, False, MIMO_LANGUAGE_MODULE_KEY

    if module_name is not None:
        grid = rank_modules.get(module_name)
        if grid is None:
            return 0, 1, False, module_name
        selected_name = module_name
    elif len(rank_modules) == 1:
        selected_name, grid = next(iter(rank_modules.items()))
    else:
        selected_name = (
            MIMO_LANGUAGE_MODULE_KEY if MIMO_LANGUAGE_MODULE_KEY in rank_modules else next(iter(rank_modules))
        )
        grid = rank_modules[selected_name]
        logger.warning(
            "get_megatron_mimo_dp_info called without module_name on a rank serving "
            "multiple modules (%s). Defaulting to '%s'. Pass module_name explicitly "
            "to disambiguate in colocated mode.",
            sorted(rank_modules),
            selected_name,
        )

    dp_rank = grid.get_pg(["dp"]).rank()
    dp_size = grid.get_pg(["dp"]).size()
    needs_data = _needs_data_for_module(grid, selected_name)
    return dp_rank, dp_size, needs_data, selected_name


def get_megatron_mimo_sampling_info(
    megatron_mimo_cfg: "MegatronMIMOParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple[int, int, bool]:
    """Get sampler DP rank, size, and data-loading flag for MegatronMIMO.

    In heterogeneous MegatronMIMO, modules may have different DP sizes.  The data
    loader must give every data-loading rank the **same global micro-batch**
    so that :func:`slice_batch_for_megatron_mimo` (called in the forward step) can
    sub-shard it consistently with the :class:`BridgeCommunicator` fan-in /
    fan-out routing.

    This function therefore returns ``dp_size=1, dp_rank=0`` for all ranks,
    disabling DP sharding at the sampler level.  Per-module DP sharding is
    deferred to :func:`slice_batch_for_megatron_mimo`.

    In colocated deployments a rank belongs to multiple module grids. We say
    the rank needs data if **any** of its modules needs data — missing a
    needed-by-LLM batch because the rank was only checked against the encoder
    grid would cause silent no-op training.

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping.

    Returns:
        Tuple of (sampler_dp_rank, sampler_dp_size, needs_data).
    """
    rank_modules = _find_rank_modules(grids)
    if not rank_modules:
        return 0, 1, False

    needs_data = any(_needs_data_for_module(grid, name) for name, grid in rank_modules.items())
    # All data-loading ranks use the same sampler settings so they load
    # identical global micro-batches.  Module-local DP slicing happens later
    # in forward_step via slice_batch_for_megatron_mimo.
    return 0, 1, needs_data


# Batch keys that always feed the LANGUAGE module, regardless of which encoders
# are present. Used by ``slice_batch_for_megatron_mimo_modules`` to route per-key
# slicing in colocated heterogeneous TP/DP. Anything not in this set and not
# under ``modality_inputs`` falls back to the language module's DP (most batch
# metadata is language-side).
_LANGUAGE_BATCH_KEYS = frozenset({"input_ids", "labels", "loss_mask", "position_ids", "attention_mask"})
_MODALITY_INPUTS_KEY = "modality_inputs"


def slice_batch_for_megatron_mimo_modules(
    batch: Dict[str, Any],
    *,
    grids: Dict[str, "HyperCommGrid"],
    language_module_name: str = MIMO_LANGUAGE_MODULE_KEY,
) -> Dict[str, Any]:
    """Slice a global micro-batch with per-key per-module DP routing.

    Required for colocated heterogeneous TP/DP where the same physical rank
    serves multiple modules with possibly different DP sizes. Language keys
    (``input_ids``, ``labels``, ``loss_mask``, ``position_ids``,
    ``attention_mask``) are sliced by the language module's DP;
    ``modality_inputs[<name>]`` is sliced by encoder ``<name>``'s DP. Anything
    else with a batch dimension falls back to language DP (most metadata is
    language-side).

    Mode behavior:
      * No ``grids`` configured (legacy / no ``module_to_grid_map``) → batch
        returned unchanged.
      * Non-colocated rank (one module on this rank) → reduces to uniform
        slicing by that one module's DP, preserving pre-task behavior.
      * Colocated rank (multiple modules) → per-key per-module slicing
        as described above.
      * Non-participating rank (rank outside every grid) → batch returned
        as-is. ``forward_step`` doesn't reach here in production
        (``validate_no_stub_ranks`` rejects stub ranks at setup), but the
        helper is defensive.

    Args:
        batch: Global batch dictionary. Tensors at the top level are sliced
            along dim 0; nested dicts under ``modality_inputs`` are recursed
            into with the encoder's DP.
        grids: Mapping module-name → ``HyperCommGrid`` (typically
            ``MegatronMIMOInfra.module_to_grid_map``).
        language_module_name: Override for the language module key
            (defaults to ``MIMO_LANGUAGE_MODULE_KEY``).

    Returns:
        Sliced batch dict. The original is not mutated.

    Note:
        Routing for new top-level batch keys is governed by
        ``_LANGUAGE_BATCH_KEYS`` and the ``modality_inputs`` nesting
        convention. New batch fields with a clear batch dimension that
        should follow encoder DP must be placed under
        ``modality_inputs[<encoder_name>]``; otherwise they default to
        language DP.
    """
    if not grids:
        return batch

    # Defensive: rank membership is undefined without an initialized process
    # group. Legacy callers (non-distributed test paths) rely on this no-op.
    if not dist.is_initialized():
        return batch

    rank_modules = _find_rank_modules(grids)
    if not rank_modules:
        # Non-participating rank — no module DP defined for this rank.
        return batch

    if len(rank_modules) == 1:
        # Non-colocated: legacy uniform slicing by the unique module's DP.
        only_grid = next(iter(rank_modules.values()))
        dp_pg = only_grid.get_pg(["dp"])
        return slice_batch_for_megatron_mimo(batch, dp_pg.rank(), dp_pg.size())

    # Colocated: per-key per-module routing.
    return _slice_batch_per_module_colocated(batch, rank_modules, language_module_name)


def _slice_batch_per_module_colocated(
    batch: Dict[str, Any],
    rank_modules: Dict[str, "HyperCommGrid"],
    language_module_name: str,
) -> Dict[str, Any]:
    """Per-key per-module slicing assuming this rank serves every module.

    Caller (``slice_batch_for_megatron_mimo_modules``) has already established
    that ``rank_modules`` has more than one entry (colocated rank) so every
    referenced module is active on this rank.
    """
    language_grid = rank_modules.get(language_module_name)
    if language_grid is None:
        # Defensive: no language module on this rank in colocated mode is
        # invalid by construction (validator + stub-rank checks). Fall back
        # to passing the batch through unchanged so the failure surfaces
        # downstream with a clearer message rather than masking it via a
        # mis-slice here.
        return batch

    language_dp = language_grid.get_pg(["dp"])
    language_dp_rank, language_dp_size = language_dp.rank(), language_dp.size()

    sliced: Dict[str, Any] = {}
    for key, value in batch.items():
        if key == _MODALITY_INPUTS_KEY:
            sliced[key] = _slice_modality_inputs(value, rank_modules)
        elif key in _LANGUAGE_BATCH_KEYS:
            sliced[key] = _slice_value(value, language_dp_rank, language_dp_size, key)
        else:
            # Fallback: language DP for any other batch field with a batch dim.
            sliced[key] = _slice_value(value, language_dp_rank, language_dp_size, key)
    return sliced


def _slice_modality_inputs(
    modality_inputs: Any,
    rank_modules: Dict[str, "HyperCommGrid"],
) -> Any:
    """Slice each modality entry by that encoder's DP, preserving non-dict values."""
    if not isinstance(modality_inputs, dict):
        return modality_inputs
    sliced: Dict[str, Any] = {}
    for modality_name, modality_value in modality_inputs.items():
        encoder_grid = rank_modules.get(modality_name)
        if encoder_grid is None:
            # Module not on this rank (would only happen in non-colocated, which
            # this function isn't called for). Pass through.
            sliced[modality_name] = modality_value
            continue
        encoder_dp = encoder_grid.get_pg(["dp"])
        sliced[modality_name] = _slice_value_recursive(
            modality_value, encoder_dp.rank(), encoder_dp.size(), modality_name
        )
    return sliced


def _slice_value(value: Any, dp_rank: int, dp_size: int, key: str) -> Any:
    """Slice a top-level batch value by ``(dp_rank, dp_size)``.

    Mirrors the per-value handling in ``slice_batch_for_megatron_mimo`` but
    operates on a single value so the caller controls routing.
    """
    if dp_size == 1:
        return value
    if isinstance(value, torch.Tensor):
        return _slice_tensor(value, dp_rank, dp_size, key)
    if isinstance(value, dict):
        # Generic nested dict (not modality_inputs — that's handled separately).
        return {k: _slice_value(v, dp_rank, dp_size, f"{key}.{k}") for k, v in value.items()}
    if isinstance(value, list) and len(value) > 0:
        if len(value) % dp_size == 0:
            local_len = len(value) // dp_size
            start_idx = dp_rank * local_len
            return value[start_idx : start_idx + local_len]
        return value
    return value


def _slice_value_recursive(value: Any, dp_rank: int, dp_size: int, key: str) -> Any:
    """Recurse into nested dicts under modality_inputs (e.g. {"clip": {"x": tensor}})."""
    if dp_size == 1:
        return value
    if isinstance(value, torch.Tensor):
        return _slice_tensor(value, dp_rank, dp_size, key)
    if isinstance(value, dict):
        return {k: _slice_value_recursive(v, dp_rank, dp_size, f"{key}.{k}") for k, v in value.items()}
    if isinstance(value, list) and len(value) > 0 and len(value) % dp_size == 0:
        local_len = len(value) // dp_size
        start_idx = dp_rank * local_len
        return value[start_idx : start_idx + local_len]
    return value


def _slice_tensor(tensor: torch.Tensor, dp_rank: int, dp_size: int, key: str) -> torch.Tensor:
    """Slice ``tensor`` along dim 0 into the rank's contiguous DP shard."""
    batch_size = tensor.size(0)
    if batch_size % dp_size != 0:
        raise ValueError(
            f"Batch size {batch_size} for key '{key}' is not divisible by DP size "
            f"{dp_size}. Ensure micro_batch_size is divisible by every module's "
            f"data_parallel_size."
        )
    local_batch_size = batch_size // dp_size
    start_idx = dp_rank * local_batch_size
    return tensor[start_idx : start_idx + local_batch_size]


def slice_batch_for_megatron_mimo(
    batch: Dict[str, Any],
    dp_rank: int,
    dp_size: int,
) -> Dict[str, Any]:
    """Slice a global micro-batch for this rank's module-local DP shard.

    All data-loading ranks receive the same global micro-batch (the sampler
    uses ``dp_size=1``).  This function contiguously slices it so that each
    module-local DP replica processes the correct subset.  The slicing is
    contiguous to match the :class:`BridgeCommunicator`'s batch-dimension
    split / concatenate logic for fan-out and fan-in routing.

    Handles nested dicts (e.g. ``modality_inputs``) by recursing.

    Args:
        batch: Global batch dictionary with tensors of shape [global_batch, ...].
            May contain nested dicts (e.g. modality_inputs → encoder → kwargs).
        dp_rank: This rank's position in its **module-local** DP group.
        dp_size: Size of the module-local DP group.

    Returns:
        Dict with tensors sliced to shape [global_batch // dp_size, ...].

    Example:
        >>> global_batch = {'tokens': torch.randn(12, 2048)}
        >>> local_batch = slice_batch_for_megatron_mimo(global_batch, dp_rank=1, dp_size=3)
        >>> local_batch['tokens'].shape  # torch.Size([4, 2048])
    """
    if dp_size == 1:
        return batch

    sliced = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Slice along batch dimension (dim=0)
            batch_size = value.size(0)
            if batch_size % dp_size != 0:
                raise ValueError(
                    f"Batch size {batch_size} for key '{key}' is not divisible "
                    f"by DP size {dp_size}. Ensure micro_batch_size is divisible "
                    f"by every module's data_parallel_size."
                )
            local_batch_size = batch_size // dp_size
            start_idx = dp_rank * local_batch_size
            end_idx = start_idx + local_batch_size
            sliced[key] = value[start_idx:end_idx]
        elif isinstance(value, dict):
            # Recurse into nested dicts (e.g. modality_inputs)
            sliced[key] = slice_batch_for_megatron_mimo(value, dp_rank, dp_size)
        elif isinstance(value, list) and len(value) > 0:
            list_len = len(value)
            if list_len % dp_size == 0:
                local_len = list_len // dp_size
                start_idx = dp_rank * local_len
                end_idx = start_idx + local_len
                sliced[key] = value[start_idx:end_idx]
            else:
                # Keep as-is if not evenly divisible (global metadata)
                sliced[key] = value
        else:
            # Keep non-tensor, non-list values as-is
            sliced[key] = value

    return sliced
