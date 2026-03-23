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
    """Collate function for MIMO datasets.

    Stacks batch items and organizes modality inputs into a structure
    suitable for MIMO model forward pass.

    Args:
        batch: List of examples from MimoDataset, each containing:
            - input_ids: Token IDs with placeholder tokens
            - labels: Labels for causal LM training
            - attention_mask: Attention mask
            - position_ids: Position indices
            - modality_inputs: Dict[str, Dict[str, Any]] with preprocessed inputs
        modality_names: List of modality names to collate.

    Returns:
        Dict containing:
            - input_ids: (batch, seq) stacked token IDs
            - labels: (batch, seq) stacked labels
            - attention_mask: (batch, seq) attention mask
            - position_ids: (batch, seq) position indices
            - modality_inputs: Dict[str, Dict[str, Tensor]] with batched modality tensors
              Each modality's tensors are stacked along batch dimension.

    Example:
        >>> batch = [
        ...     {
        ...         "input_ids": torch.tensor([32000, 1, 2, 3]),
        ...         "labels": torch.tensor([32000, 1, 2, 3]),
        ...         "attention_mask": torch.ones(4),
        ...         "position_ids": torch.arange(4),
        ...         "modality_inputs": {
        ...             "vision": {"pixel_values": torch.randn(3, 224, 224)},
        ...         },
        ...     },
        ...     # ... more examples
        ... ]
        >>> collated = mimo_collate_fn(batch, modality_names=["vision"])
    """
    if not batch:
        return {}

    # Stack standard fields
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])

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
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "modality_inputs": modality_inputs,
    }
