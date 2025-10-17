# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetuning-specific data handling utilities."""

from typing import Any, Iterator

import torch


def split_batch_into_microbatches(
    batch: dict[str, Any], num_microbatches: int, enforce_divisible: bool = True
) -> list[dict[str, Any]]:
    """Split a batch dictionary into microbatches.

    Takes a global batch (e.g., [16, 240] for tokens) and splits it into
    num_microbatches smaller batches (e.g., 4 batches of [4, 240]).

    Args:
        batch: Dictionary containing tensors with batch_size = num_microbatches * micro_batch_size
        num_microbatches: Number of microbatches to split into
        enforce_divisible: Whether to enforce batch_size % num_microbatches == 0

    Returns:
        List of microbatch dictionaries, each containing the same keys as the input batch

    Example:
        >>> batch = {'tokens': torch.rand(16, 240), 'labels': torch.rand(16, 240)}
        >>> microbatches = split_batch_into_microbatches(batch, num_microbatches=4)
        >>> len(microbatches)  # 4
        >>> microbatches[0]['tokens'].shape  # torch.Size([4, 240])
    """
    # Identify tensor items vs other items (like metadata)
    tensor_items = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    other_items = {k: v for k, v in batch.items() if not isinstance(v, torch.Tensor)}

    if len(tensor_items) == 0:
        raise ValueError("Batch must contain at least one tensor")

    # Get batch size from first tensor
    first_key = next(iter(tensor_items.keys()))
    batch_size = tensor_items[first_key].shape[0]

    if enforce_divisible and batch_size % num_microbatches != 0:
        raise ValueError(
            f"Batch size {batch_size} is not divisible by num_microbatches {num_microbatches}. "
            f"Cannot split evenly into microbatches."
        )

    # Split all tensors along batch dimension (dim=0)
    split_tensors = {}
    for key, tensor in tensor_items.items():
        split_tensors[key] = torch.tensor_split(tensor, num_microbatches, dim=0)

    # Create microbatch dictionaries
    microbatches = []
    for i in range(num_microbatches):
        microbatch = {}

        # Add split tensors
        for key, splits in split_tensors.items():
            microbatch[key] = splits[i]

        # Handle non-tensor items (metadata, etc.)
        for key, value in other_items.items():
            if isinstance(value, list) and len(value) == batch_size:
                # If it's a list with length matching batch size, split it too
                micro_batch_size = batch_size // num_microbatches
                start_idx = i * micro_batch_size
                end_idx = start_idx + micro_batch_size
                microbatch[key] = value[start_idx:end_idx]
            else:
                # Otherwise copy as-is (e.g., global metadata)
                microbatch[key] = value

        microbatches.append(microbatch)

    return microbatches


def prepare_finetuning_batch(
    data_iterator: Iterator,
    num_microbatches: int,
    default_seq_length: int,
    seq_key: str = "tokens",
) -> tuple[Iterator, int]:
    """Prepare a finetuning batch by getting global batch and splitting into microbatches.

    This function handles the finetuning-specific data flow:
    1. Gets the full global batch from the iterator
    2. Extracts the dynamic sequence length from the batch
    3. Splits the batch into microbatches with consistent sequence length
    4. Returns an iterator over microbatches and the extracted sequence length

    Args:
        data_iterator: Iterator that yields global batches (e.g., from DataLoader with batch sampler)
        num_microbatches: Number of microbatches to split each global batch into
        default_seq_length: Fallback sequence length if it cannot be extracted from batch
        seq_key: Key in batch dict containing the sequence tensor (default: 'tokens')

    Returns:
        Tuple of:
        - Iterator over microbatches (each microbatch is a dict with same keys as global batch)
        - Sequence length extracted from the global batch (or default_seq_length if not found)

    Example:
        >>> # DataLoader yields global batch of shape [16, 240]
        >>> microbatch_iter, seq_len = prepare_finetuning_batch(
        ...     data_iterator=iter(dataloader),
        ...     num_microbatches=4,
        ...     default_seq_length=2048
        ... )
        >>> seq_len  # 240 (extracted from batch)
        >>> batch1 = next(microbatch_iter)
        >>> batch1['tokens'].shape  # torch.Size([4, 240])
    """
    # Get full global batch from dataloader
    global_batch = next(data_iterator)

    # Extract dynamic seq_length from the full batch
    seq_length = default_seq_length
    if seq_key in global_batch and isinstance(global_batch[seq_key], torch.Tensor):
        seq_length = global_batch[seq_key].size(1)

    # Split into microbatches
    microbatches = split_batch_into_microbatches(global_batch, num_microbatches)

    # Return iterator over microbatches and the extracted seq_length
    return iter(microbatches), seq_length
