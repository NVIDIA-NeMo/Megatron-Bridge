# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Collate functions for MIMO datasets."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import torch


def mimo_collate_fn(
    batch: List[Dict[str, Any]],
    modality_names: List[str],
) -> Dict[str, Any]:
    """
    Collate a list of MIMO dataset examples into a batched dict suitable for model input.
    
    Parameters:
        batch: List of example dictionaries. Each example is expected to contain:
            - input_ids: Tensor of token IDs.
            - labels: Tensor of target token IDs.
            - attention_mask: Tensor attention mask aligning with tokens.
            - position_ids: Tensor of position indices for tokens.
            - loss_mask: Tensor indicating per-token loss contribution.
            - modality_inputs: Dict[str, Dict[str, Any]] mapping modality names to modality-specific preprocessed inputs.
        modality_names: List of modality names to gather from each example's `modality_inputs`.
    
    Returns:
        A dictionary with the following entries:
            - input_ids: Tensor shaped (batch, seq) of stacked input IDs.
            - labels: Tensor shaped (batch, seq) of stacked labels.
            - loss_mask: Tensor shaped (batch, seq) of stacked loss masks.
            - attention_mask: Tensor shaped (batch, seq) of stacked attention masks.
            - position_ids: Tensor shaped (batch, seq) of stacked position indices.
            - modality_inputs: Dict mapping each present modality name to a dict of batched modality values.
              For each modality key, tensor values are stacked along the batch dimension when shapes permit; values that cannot be stacked or non-tensor values are returned as lists of per-example entries.
    """
    if not batch:
        return {}

    # Stack standard fields
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])

    # Collate modality inputs
    modality_inputs: Dict[str, Dict[str, Any]] = {}

    for modality_name in modality_names:
        # Collect all tensors for this modality across the batch
        modality_batch_items = [item.get("modality_inputs", {}).get(modality_name, {}) for item in batch]

        # Skip if no items have this modality
        if not any(modality_batch_items):
            continue

        # Get all keys from the first non-empty item
        first_non_empty = next((item for item in modality_batch_items if item), {})

        if not first_non_empty:
            continue

        modality_inputs[modality_name] = {}

        for key in first_non_empty.keys():
            values = []
            for item in modality_batch_items:
                if key in item:
                    val = item[key]
                    if isinstance(val, torch.Tensor):
                        values.append(val)
                    else:
                        # Non-tensor values are kept as lists
                        values.append(val)

            if values and isinstance(values[0], torch.Tensor):
                # Stack tensors along batch dimension
                try:
                    modality_inputs[modality_name][key] = torch.stack(values)
                except RuntimeError as e:
                    # Tensors have different shapes - keep as list but warn user
                    warnings.warn(
                        f"Cannot stack tensors for '{modality_name}.{key}' - shapes differ "
                        f"across batch. Keeping as list. This may cause issues in model "
                        f"forward pass. Consider padding inputs to uniform shapes. Error: {e}",
                        stacklevel=2,
                    )
                    modality_inputs[modality_name][key] = values
            elif values:
                # Keep non-tensor values as list
                modality_inputs[modality_name][key] = values

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "modality_inputs": modality_inputs,
    }
