# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data parallel utilities for MegatronMIMO data loading."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any, Dict, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid

    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


def _batch_dim_for_tensor(key: str, value: torch.Tensor) -> int:
    """Return the batch dimension for known MegatronMIMO batch tensors."""
    # Qwen-VL MRoPE position_ids are [3, batch, seq] instead of [batch, seq].
    if key == "position_ids" and value.dim() >= 3 and value.size(0) == 3:
        return 1
    return 0


def _is_patch_packed_visual_dict(value: Any) -> bool:
    """Detect a patch-packed visual encoder input layout.

    Some VLM adapters pack all image patches for a microbatch into one flat
    tensor while keeping per-image grid metadata:

    - ``hidden_states``: ``[sum(patches_across_images), patch_feature_dim]``
    - ``grid_thw``:      ``[num_images, 3]``

    Dim 0 of these two tensors means different things (patches vs images), so
    they cannot be sliced independently by DP. They must be sliced jointly
    along the per-image boundary.

    TODO(mimo): This is very model-specific. We should revisit this to make it more generic.
    maybe it should be handled by the model itself that defines how to slice the data.
    """
    return (
        isinstance(value, dict)
        and isinstance(value.get("hidden_states"), torch.Tensor)
        and isinstance(value.get("grid_thw"), torch.Tensor)
        and value["hidden_states"].dim() >= 1
        and value["grid_thw"].dim() == 2
        and value["grid_thw"].size(-1) == 3
    )


def _slice_patch_packed_visual_dict(value: Dict[str, Any], dp_rank: int, dp_size: int) -> Dict[str, Any]:
    """Joint-slice a patch-packed ``{hidden_states, grid_thw, ...}`` dict.

    Shards by image count, then derives the corresponding patch range from
    ``cumsum(grid_thw.prod(dim=-1))``. Other keys in the dict are passed
    through unchanged because they are treated as global encoder metadata.

    Constraints:

    - ``num_images`` (rows of ``grid_thw``) must be divisible by ``dp_size``.
      For the encoder rank, ``num_images_per_microbatch`` is set by the
      training-time MIMO MBS and the dataset's images-per-sample. With
      single-image-per-sample data and MIMO ``MICRO_BATCH_SIZE`` divisible by
      encoder ``DP``, this always holds.

    TODO(mimo): This convention slices by image row. Multi-image samples with
    uneven image counts need explicit per-sample image split metadata so DP
    slices remain sample-aligned instead of only image-aligned.
    """
    g = value["grid_thw"]  # [num_images, 3]
    hs = value["hidden_states"]  # [sum(patches), feat]
    n_images = int(g.size(0))
    if n_images % dp_size != 0:
        raise ValueError(
            f"Patch-packed visual input has num_images_in_microbatch ({n_images}) "
            f"is not divisible by encoder DP ({dp_size}). Set MIMO MICRO_BATCH_SIZE "
            f"so that each encoder DP shard receives a whole number of images."
        )
    imgs_per_shard = n_images // dp_size
    img_lo = dp_rank * imgs_per_shard
    img_hi = img_lo + imgs_per_shard

    patches_per_image = g.prod(dim=-1).to(torch.long)  # [num_images]
    total_patches = int(patches_per_image.sum().item())
    if int(hs.size(0)) != total_patches:
        raise ValueError(
            f"Patch-packed visual input expected hidden_states dim 0 ({hs.size(0)}) "
            f"to equal sum(grid_thw products) ({total_patches})."
        )
    patch_offsets = torch.zeros(n_images + 1, dtype=torch.long, device=patches_per_image.device)
    patch_offsets[1:] = patches_per_image.cumsum(0)
    patch_lo = int(patch_offsets[img_lo].item())
    patch_hi = int(patch_offsets[img_hi].item())

    out: Dict[str, Any] = {}
    for key, sub_value in value.items():
        if key == "grid_thw":
            out[key] = g[img_lo:img_hi].contiguous()
        elif key == "hidden_states":
            out[key] = hs[patch_lo:patch_hi].contiguous()
        else:
            # Other entries (encoder kwargs, attention masks, etc.) are
            # passed through as global metadata - same convention as the
            # outer slicer's "non-divisible list" branch.
            out[key] = sub_value
    return out


def _find_rank_module(
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple["HyperCommGrid | None", "str | None"]:
    """Find which module grid the current rank belongs to."""
    current_rank = dist.get_rank()
    for module_name, grid in grids.items():
        if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
            return grid, module_name
    return None, None


def _needs_data_for_module(grid: "HyperCommGrid", module_name: str) -> bool:
    """Determine if the current rank needs to load data for the given module.

    LLM: all PP stages need batch metadata. First PP stages consume input_ids,
    last PP stages consume labels/loss_mask, and models with position-dependent
    decoder blocks (for example Qwen3-VL MRoPE) need position_ids on intermediate
    PP stages as well.
    Encoders: only the first PP stage needs raw modality inputs.
    """
    pp_group = grid.get_pg(["pp"])
    pp_rank = pp_group.rank()
    if module_name == MIMO_LANGUAGE_MODULE_KEY:
        return True
    return pp_rank == 0


def get_megatron_mimo_dp_info(
    megatron_mimo_cfg: "MegatronMIMOParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
) -> Tuple[int, int, bool, str]:
    """Get **module-local** DP rank, size, data-loading flag, and module name.

    Returns the DP settings for the module that the current rank participates
    in.  These are used by :func:`slice_batch_for_megatron_mimo` to sub-shard a global
    micro-batch into per-module DP shards.

    .. note::
        Do **not** use these values to construct a ``DistributedSampler``.
        For sampler construction use :func:`get_megatron_mimo_sampling_info` instead,
        which returns settings that keep all data-loading ranks synchronised
        on the same sample order.

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping from build_hypercomm_grids().

    Returns:
        Tuple of (dp_rank, dp_size, needs_data, loader_module).
    """
    my_grid, my_module = _find_rank_module(grids)
    if my_grid is None or my_module is None:
        return 0, 1, False, MIMO_LANGUAGE_MODULE_KEY

    dp_rank = my_grid.get_pg(["dp"]).rank()
    dp_size = my_grid.get_pg(["dp"]).size()
    needs_data = _needs_data_for_module(my_grid, my_module)
    return dp_rank, dp_size, needs_data, my_module


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

    Args:
        megatron_mimo_cfg: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping.

    Returns:
        Tuple of (sampler_dp_rank, sampler_dp_size, needs_data).
    """
    my_grid, my_module = _find_rank_module(grids)
    if my_grid is None or my_module is None:
        return 0, 1, False

    needs_data = _needs_data_for_module(my_grid, my_module)
    # All data-loading ranks use the same sampler settings so they load
    # identical global micro-batches.  Module-local DP slicing happens later
    # in forward_step via slice_batch_for_megatron_mimo.
    return 0, 1, needs_data


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
        batch: Global batch dictionary with tensors of shape [global_batch, ...],
            except known layouts such as Qwen-VL MRoPE position_ids shaped
            [3, global_batch, seq]. May contain nested dicts
            (e.g. modality_inputs -> encoder -> kwargs).
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
            batch_dim = _batch_dim_for_tensor(key, value)
            batch_size = value.size(batch_dim)
            if batch_size % dp_size != 0:
                raise ValueError(
                    f"Batch size {batch_size} for key '{key}' is not divisible "
                    f"by DP size {dp_size}. Ensure micro_batch_size is divisible "
                    f"by every module's data_parallel_size."
                )
            local_batch_size = batch_size // dp_size
            start_idx = dp_rank * local_batch_size
            end_idx = start_idx + local_batch_size
            index = [builtins.slice(None)] * value.dim()
            index[batch_dim] = builtins.slice(start_idx, end_idx)
            sliced[key] = value[tuple(index)]
        elif isinstance(value, dict):
            # Patch-packed visual encoder inputs use dim 0 for different units
            # across fields (patches for hidden_states, images for grid_thw), so
            # they need joint slicing instead of normal recursive tensor slicing.
            if _is_patch_packed_visual_dict(value):
                sliced[key] = _slice_patch_packed_visual_dict(value, dp_rank, dp_size)
            else:
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
