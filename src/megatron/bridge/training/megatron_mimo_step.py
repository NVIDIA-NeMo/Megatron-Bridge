# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO-specific forward step function for use with pipeline schedules.

This module provides the forward step function for MegatronMIMO model training.
Key design notes (per PR 3212):
- The schedule expects dict-based outputs: {module_name: tensor} instead of single tensors
- The MimoModel's forward returns output tensors that the schedule sends via MultiModulePipelineCommunicator
- The schedule's backward_step_multimodule() handles dict-based backward pass automatically
- Only the LLM module produces a loss - encoders just produce activations
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

from megatron.bridge.data.megatron_mimo.dp_utils import slice_batch_for_megatron_mimo_modules
from megatron.bridge.training.megatron_mimo_parallel_utils import unwrap_megatron_mimo_model
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)

_DATA_ALIGNMENT_CHECK_COUNT = 0


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _first_tensor_batch_dim(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        return value.size(0)
    if isinstance(value, dict):
        for nested in value.values():
            batch_dim = _first_tensor_batch_dim(nested)
            if batch_dim is not None:
                return batch_dim
    return None


def _rank_modules(grids: Dict[str, Any]) -> Dict[str, Any]:
    current_rank = dist.get_rank()
    return {
        name: grid for name, grid in grids.items() if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)
    }


def _maybe_check_colocated_data_alignment(
    original_batch: Dict[str, Any],
    sliced_batch: Dict[str, Any],
    grids: Dict[str, Any],
) -> None:
    """One-shot L3 guardrail for colocated heterogeneous-DP data slicing."""
    global _DATA_ALIGNMENT_CHECK_COUNT

    # TODO(liding): to be removed after L3 data-alignment validation is no longer needed.
    if not _env_flag("MIMO_CHECK_DATA_ALIGNMENT"):
        return
    max_checks = int(os.environ.get("MIMO_CHECK_DATA_ALIGNMENT_STEPS", "1"))
    if _DATA_ALIGNMENT_CHECK_COUNT >= max_checks:
        return
    if not dist.is_available() or not dist.is_initialized() or not grids:
        return

    rank_modules = _rank_modules(grids)
    if len(rank_modules) < 2:
        return

    language_grid = rank_modules.get(MIMO_LANGUAGE_MODULE_KEY)
    if language_grid is None:
        return

    language_dp = language_grid.get_pg(["dp"])
    language_dp_rank = language_dp.rank()
    language_dp_size = language_dp.size()
    global_batch = _first_tensor_batch_dim(original_batch.get("input_ids"))
    if global_batch is None:
        return
    if global_batch % language_dp_size != 0:
        raise RuntimeError(
            f"MegatronMIMO data alignment check failed: global language batch {global_batch} "
            f"is not divisible by language DP size {language_dp_size}."
        )

    expected_language_batch = global_batch // language_dp_size
    local_language_batch = _first_tensor_batch_dim(sliced_batch.get("input_ids"))
    if local_language_batch != expected_language_batch:
        raise RuntimeError(
            "MegatronMIMO data alignment check failed: "
            f"language local batch is {local_language_batch}, expected {expected_language_batch} "
            f"from global batch {global_batch}, language dp_rank={language_dp_rank}, "
            f"language dp_size={language_dp_size}."
        )

    modality_reports = []
    original_modalities = original_batch.get("modality_inputs") or {}
    sliced_modalities = sliced_batch.get("modality_inputs") or {}
    for modality_name, encoder_grid in sorted(rank_modules.items()):
        if modality_name == MIMO_LANGUAGE_MODULE_KEY or modality_name not in original_modalities:
            continue

        encoder_dp = encoder_grid.get_pg(["dp"])
        encoder_dp_rank = encoder_dp.rank()
        encoder_dp_size = encoder_dp.size()
        original_modality_batch = _first_tensor_batch_dim(original_modalities.get(modality_name))
        local_modality_batch = _first_tensor_batch_dim(sliced_modalities.get(modality_name))
        if original_modality_batch is None:
            continue
        if original_modality_batch != global_batch:
            raise RuntimeError(
                "MegatronMIMO data alignment check failed: "
                f"modality '{modality_name}' global batch is {original_modality_batch}, "
                f"language global batch is {global_batch}."
            )
        if original_modality_batch % encoder_dp_size != 0:
            raise RuntimeError(
                "MegatronMIMO data alignment check failed: "
                f"modality '{modality_name}' global batch {original_modality_batch} is not divisible "
                f"by encoder DP size {encoder_dp_size}."
            )

        expected_modality_batch = original_modality_batch // encoder_dp_size
        if local_modality_batch != expected_modality_batch:
            raise RuntimeError(
                "MegatronMIMO data alignment check failed: "
                f"modality '{modality_name}' local batch is {local_modality_batch}, "
                f"expected {expected_modality_batch} from encoder dp_rank={encoder_dp_rank}, "
                f"encoder dp_size={encoder_dp_size}."
            )

        if encoder_dp_size % language_dp_size == 0:
            fanin = encoder_dp_size // language_dp_size
            expected_language_dp_rank = encoder_dp_rank // fanin
            if expected_language_dp_rank != language_dp_rank:
                raise RuntimeError(
                    "MegatronMIMO data alignment check failed: "
                    f"modality '{modality_name}' encoder dp_rank={encoder_dp_rank} maps to "
                    f"language dp_rank={expected_language_dp_rank}, but this rank is language "
                    f"dp_rank={language_dp_rank}."
                )

        modality_reports.append(
            f"{modality_name}:dp={encoder_dp_rank}/{encoder_dp_size},local_batch={local_modality_batch}"
        )

    logger.info(
        "MegatronMIMO data alignment check passed on rank %s: global_batch=%s, "
        "language_dp=%s/%s, language_local_batch=%s, modalities=[%s]",
        dist.get_rank(),
        global_batch,
        language_dp_rank,
        language_dp_size,
        local_language_batch,
        "; ".join(modality_reports),
    )
    _DATA_ALIGNMENT_CHECK_COUNT += 1


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor) -> Tuple:
    """Loss function for MegatronMIMO model training.

    Called at the terminal stage (LLM's last PP stage).

    Args:
        loss_mask: Mask indicating which tokens contribute to the loss.
        output_tensor: Model output tensor (losses per token).

    Returns:
        Tuple of (total_loss, num_tokens, {'lm loss': reporting_loss}).

    Note:
        Only the LLM module produces a loss. Encoders produce activations
        that are consumed by the LLM, but don't have their own loss.
    """
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {"lm loss": reporting_loss})


def get_batch(data_iterator: Iterable) -> Optional[Dict[str, torch.Tensor]]:
    """Get batch from data iterator.

    Returns dict with:
    - input_ids, labels, loss_mask, position_ids (for LLM)
    - modality_inputs: {modality_name: preprocessed_tensors} (for encoders)

    Uses existing MegatronMIMODataset format from Phase 3.

    Args:
        data_iterator: Iterator over the dataset.

    Returns:
        Batch dictionary or None if iterator is exhausted.
    """
    if data_iterator is None:
        return None

    try:
        batch = next(data_iterator)
    except StopIteration:
        return None

    # Move tensors to GPU if not already there
    def _move_to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda(non_blocking=True) if not obj.is_cuda else obj
        if isinstance(obj, dict):
            return {k: _move_to_cuda(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [_move_to_cuda(v) for v in obj]
            return type(obj)(converted)
        return obj

    if batch is not None:
        batch = _move_to_cuda(batch)

    return batch


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: MimoModel,
) -> Tuple[torch.Tensor, Optional[partial]]:
    """Forward step for MegatronMIMO model training.

    Uses 3-arg signature with GlobalState for Bridge compatibility.
    The training loop wraps this with prepare_forward_step_func() which:
    - Injects GlobalState automatically if forward_step accepts it
    - Provides access to state.timers, state.cfg, state.train_state

    The MimoModel handles dict-based tensor flow internally:
    - Encoder modules produce activations sent via BridgeCommunicator
    - LLM module receives encoder outputs and produces loss

    At terminal stage: returns (loss_tensor, loss_func)
    At intermediate stages: returns (output_dict, None) - schedule handles communication

    GUARDRAIL: At last stage, assert output is scalar tensor (not dict) to catch
    misconfigurations early with a clear error message.

    Args:
        state: GlobalState containing timers, config, train_state.
        data_iterator: Iterator over the dataset.
        model: MimoModel instance.

    Returns:
        Tuple of (output_tensor, loss_function or None).
    """
    # Get the model's role to determine if we're at first pipeline stage
    megatron_mimo_model = unwrap_megatron_mimo_model(model)

    # Determine if this rank needs data.
    # - LLM ranks: first stage needs input_ids; last stage needs labels/loss_mask.
    # - Modality ranks: only first stage needs raw modality inputs.
    needs_data = True
    if megatron_mimo_model.role is not None:
        if megatron_mimo_model.role.has_language_module:
            is_first_stage = megatron_mimo_model.role.is_first_stage(MIMO_LANGUAGE_MODULE_KEY)
            is_last_stage = megatron_mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)
            needs_data = is_first_stage or is_last_stage
        elif megatron_mimo_model.role.has_modality_modules:
            modality_modules = megatron_mimo_model.role.modality_module_names
            needs_data = any(megatron_mimo_model.role.is_first_stage(mod) for mod in modality_modules)

    if needs_data:
        data_batch = get_batch(data_iterator)
        if data_batch is None:
            raise RuntimeError(
                "get_batch returned None at a stage that requires data. "
                "This indicates a data-loading or parallelism misconfiguration."
            )
        # Slice the global micro-batch per-module. All data-loading ranks
        # receive identical batches (sampler dp_size=1); the helper sub-shards
        # language keys by language DP and modality_inputs by each encoder's
        # DP. In non-colocated mode this reduces to uniform slicing by the
        # rank's single module's DP.
        grids = getattr(megatron_mimo_model.mimo_config, "module_to_grid_map", None) or {}
        original_data_batch = data_batch
        data_batch = slice_batch_for_megatron_mimo_modules(data_batch, grids=grids)
        _maybe_check_colocated_data_alignment(original_data_batch, data_batch, grids)
    else:
        # Non-data stages consume hidden states from pipeline input tensors.
        data_batch = {
            "input_ids": None,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "loss_mask": None,
            "modality_inputs": None,
        }

    # Extract loss_mask before forward pass
    loss_mask = data_batch.get("loss_mask")

    # Run forward pass
    # MimoModel.forward() returns (output_tensor, loss_mask) or just output_tensor
    output = model(**data_batch)

    # Handle tuple return from model
    if isinstance(output, tuple):
        output_tensor, model_loss_mask = output
        # Use model-provided loss_mask if available
        if model_loss_mask is not None:
            loss_mask = model_loss_mask
    else:
        output_tensor = output

    # Check if we're at the last pipeline stage for the language module
    # megatron_mimo_model was already unwrapped at the start of this function
    if megatron_mimo_model.role is None:
        is_last_stage = True
    elif megatron_mimo_model.role.has_language_module:
        is_last_stage = megatron_mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)
    else:
        is_last_stage = False

    if is_last_stage:
        # GUARDRAIL: Verify scalar loss at last stage
        if isinstance(output_tensor, dict):
            raise ValueError(
                f"Last pipeline stage must return scalar loss tensor, got dict with keys: {output_tensor.keys()}. "
                f"Ensure the LLM module's final stage produces a loss, not activations."
            )

        # Return output and loss function
        if loss_mask is not None:
            return output_tensor, partial(loss_func, loss_mask)
        else:
            # Create default loss mask if not provided
            logger.warning("No loss_mask provided, using all-ones mask")
            default_mask = torch.ones_like(output_tensor)
            return output_tensor, partial(loss_func, default_mask)

    # Intermediate stage - return output for activation passing
    return output_tensor, None
