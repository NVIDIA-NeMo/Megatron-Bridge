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
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

import torch
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

from megatron.bridge.training.mimo_parallel_utils import unwrap_mimo_model
from megatron.bridge.training.state import GlobalState


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


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
    """
    Retrieve the next dataset batch and move any contained torch.Tensors to CUDA.
    
    Parameters:
        data_iterator (Iterable): Iterator over dataset batches; may be None.
    
    Returns:
        dict or None: The next batch dictionary with tensors transferred to CUDA (non-blocking), or `None` if `data_iterator` is None or exhausted. The batch follows the MimoDataset format and typically includes keys such as `input_ids`, `labels`, `loss_mask`, `position_ids`, and `modality_inputs` (a mapping of modality names to preprocessed tensors).
    """
    if data_iterator is None:
        return None

    try:
        batch = next(data_iterator)
    except StopIteration:
        return None

    # Move tensors to GPU if not already there
    def _move_to_cuda(obj):
        """
        Recursively move any torch.Tensor values found in obj to CUDA, preserving container types and leaving non-tensor values unchanged.
        
        Parameters:
            obj (Any): A tensor, dict, list, tuple, or nested combination thereof.
        
        Returns:
            Any: The same structure as `obj` with all torch.Tensor instances moved to CUDA (using non_blocking=True) if they were not already on CUDA.
        """
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
    """
    Perform a pipeline-compatible forward pass for a MIMO model, returning either the terminal-stage loss tensor and a loss-function partial or intermediate activations and None.
    
    Determines whether the current pipeline rank requires dataset inputs, fetches or constructs the appropriate data batch, runs model(**data_batch), and normalizes model outputs to (output_tensor, optional_loss_mask). If this rank is the language-module's last pipeline stage, returns the scalar loss tensor and a callable (partial of loss_func) that, when invoked with that tensor, computes (total_loss, total_tokens, {"lm loss": reporting_loss}). For non-terminal stages, returns activations and None so the pipeline schedule can pass them to other stages.
    
    Parameters:
        state (GlobalState): Bridge-compatible global state (timers, config, train_state).
        data_iterator (Iterable): Iterator over dataset batches; may be None for non-data ranks.
        model (MimoModel): The wrapped MIMO model to execute.
    
    Returns:
        Tuple[torch.Tensor, Optional[partial]]:
            - First element: the model's output tensor (scalar loss at terminal language stage or activations at intermediate stages).
            - Second element: a `functools.partial` of `loss_func` bound to the applicable loss mask for terminal language-stage losses, or `None` for intermediate stages.
    
    Raises:
        RuntimeError: If this rank requires data but get_batch returns None (indicates data-loading/parallelism misconfiguration).
        ValueError: If the terminal language-stage returns a dict instead of a scalar loss tensor.
    """
    # Get the model's role to determine if we're at first pipeline stage
    mimo_model = unwrap_mimo_model(model)

    # Determine if this rank needs data.
    # - LLM ranks: first stage needs input_ids; last stage needs labels/loss_mask.
    # - Modality ranks: only first stage needs raw modality inputs.
    needs_data = True
    if mimo_model.role is not None:
        if mimo_model.role.has_language_module:
            module_name = MIMO_LANGUAGE_MODULE_KEY
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
        is_last_stage = mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)
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
