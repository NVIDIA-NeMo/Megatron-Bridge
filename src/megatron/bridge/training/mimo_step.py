# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MIMO-specific forward step function for use with pipeline schedules.

This module provides the forward step function for MIMO model training.
Key design notes (per PR 3212):
- The schedule expects dict-based outputs: {module_name: tensor} instead of single tensors
- The MimoModel's forward returns output tensors that the schedule sends via MultiModulePipelineCommunicator
- The schedule's backward_step_multimodule() handles dict-based backward pass automatically
- Only the LLM module produces a loss - encoders just produce activations
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict, Iterable, Optional, Tuple

import torch
from megatron.core.models.mimo import MimoModel

from megatron.bridge.data.mimo.dp_utils import slice_batch_for_mimo
from megatron.bridge.training.mimo_parallel_utils import unwrap_mimo_model
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)


def _get_module_dp_info(
    mimo_model: MimoModel,
) -> Tuple[int, int]:
    """Get module-local DP rank and size for the current rank.

    Used to slice the global micro-batch via :func:`slice_batch_for_mimo`.
    Returns (0, 1) when grids are not configured (colocated mode).
    """
    grids = getattr(mimo_model.mimo_config, "module_to_grid_map", None)
    if not grids:
        return 0, 1

    import torch.distributed as _dist

    current_rank = _dist.get_rank()
    for _name, grid in grids.items():
        if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
            dp_rank = grid.get_pg(["dp"]).rank()
            dp_size = grid.get_pg(["dp"]).size()
            return dp_rank, dp_size

    return 0, 1


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor) -> Tuple:
    """Loss function for MIMO model training.

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

    Uses existing MimoDataset format from Phase 3.

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
    """Forward step for MIMO model training.

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
    mimo_model = unwrap_mimo_model(model)

    # Determine if this rank needs data.
    # - LLM ranks: first stage needs input_ids; last stage needs labels/loss_mask.
    # - Modality ranks: only first stage needs raw modality inputs.
    needs_data = True
    if mimo_model.role is not None:
        if mimo_model.role.has_language_module:
            module_name = mimo_model.role.language_module_name
            is_first_stage = mimo_model.role.is_first_stage(module_name)
            is_last_stage = mimo_model.role.is_last_stage(module_name)
            needs_data = is_first_stage or is_last_stage
        elif mimo_model.role.has_modality_modules:
            modality_modules = mimo_model.role.modality_module_names
            needs_data = any(mimo_model.role.is_first_stage(mod) for mod in modality_modules)

    if needs_data:
        data_batch = get_batch(data_iterator)
        if data_batch is None:
            raise RuntimeError(
                "get_batch returned None at a stage that requires data. "
                "This indicates a data-loading or parallelism misconfiguration."
            )
        # Slice the global micro-batch for this module's DP shard.
        # All data-loading ranks receive identical batches (sampler dp_size=1).
        # slice_batch_for_mimo contiguously sub-shards to match the
        # BridgeCommunicator's fan-in/fan-out batch-dimension routing.
        dp_rank, dp_size = _get_module_dp_info(mimo_model)
        data_batch = slice_batch_for_mimo(data_batch, dp_rank, dp_size)
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
    # mimo_model was already unwrapped at the start of this function
    if mimo_model.role is None:
        is_last_stage = True
    elif mimo_model.role.has_language_module:
        is_last_stage = mimo_model.role.is_last_stage(mimo_model.role.language_module_name)
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
