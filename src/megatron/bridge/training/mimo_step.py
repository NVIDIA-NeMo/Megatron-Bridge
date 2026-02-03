# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MIMO-specific forward step function for use with pipeline schedules.

This module provides the forward step function for MIMO model training.
Key design notes (per PR 3129):
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
    
    return (total_loss, total_tokens, {'lm loss': reporting_loss})


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
    # Get batch from iterator
    data_batch = get_batch(data_iterator)
    if data_batch is None:
        data_batch = {'input_ids': None}
    
    # Extract loss_mask before forward pass
    loss_mask = data_batch.get('loss_mask')
    
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
    if model.role is None:
        is_last_stage = True
    elif model.role.has_language_module:
        is_last_stage = model.role.is_last_stage(model.role.language_module_name)
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
