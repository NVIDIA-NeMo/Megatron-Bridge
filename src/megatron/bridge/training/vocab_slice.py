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

"""Auto-slice output vocabulary for faster SFT training.

When fine-tuning with answer_only_loss, the output projection (logits = hidden @ embedding.T)
and cross-entropy loss are computed over the full vocabulary, even though the model only needs
to predict a small subset. This module detects which token IDs appear in answers and slices
the output projection accordingly, reducing compute proportionally.

Measured speedup (60M model, SEQ=16384, packed sequences, single GPU):
- Full vocab (64,260):     5.41s/step, 31.6 TFLOP/s
- Sliced vocab (260):      3.03s/step, 55.8 TFLOP/s  -> 1.79x speedup

The embedding layer keeps all tokens for input. Only the output projection and loss
computation are modified. Checkpoints are saved with the full vocab and convert to
HuggingFace format normally -- the slicing is training-only.

Example::

    from megatron.bridge.training.vocab_slice import (
        collect_active_vocab_ids,
        create_vocab_sliced_forward_step,
    )

    train_ds, valid_ds, test_ds = dataset_builder.build()
    active_ids = collect_active_vocab_ids(train_ds)
    fwd_step = create_vocab_sliced_forward_step(active_ids)
    finetune(config=cfg, forward_step_func=fwd_step)

See: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2473
"""

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def collect_active_vocab_ids(
    dataset: Dataset,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Scan an SFT dataset to collect unique token IDs that appear in answers.

    Iterates over the dataset and collects label token IDs at positions where
    loss_mask > 0 (i.e., answer positions when answer_only_loss is enabled).

    Works with both standard (GPTSFTDataset) and packed (GPTSFTPackedDataset)
    datasets.

    Args:
        dataset: An SFT dataset instance.
        max_samples: Optional limit on number of samples to scan. If None,
            scans the entire dataset. Useful for very large datasets where
            scanning a subset is sufficient to capture the full answer vocab.

    Returns:
        Sorted 1D tensor of unique token IDs (on CPU) that appear in answers.
    """
    active_ids: set[int] = set()
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    for i in range(n):
        item = dataset[i]
        input_ids = item["input_ids"]

        if "seq_boundaries" in item:
            # Packed dataset: multiple sub-sequences per item.
            # Labels for sub-seq j: input_ids[boundaries[j]+1 : boundaries[j+1]]
            # Loss mask for sub-seq j: loss_mask[boundaries[j] : boundaries[j+1]-1]
            seq_boundaries = item["seq_boundaries"]
            loss_mask = item["loss_mask"]
            for j in range(len(seq_boundaries) - 1):
                start = seq_boundaries[j]
                end = seq_boundaries[j + 1]
                for k in range(end - start - 1):
                    if loss_mask[start + k]:
                        active_ids.add(int(input_ids[start + k + 1]))
        elif "loss_mask" in item:
            # Standard dataset with explicit loss_mask: collect label IDs
            # at positions where loss_mask > 0 (label at k is input_ids[k+1]).
            loss_mask = item["loss_mask"]
            for k in range(len(loss_mask)):
                if loss_mask[k] and k + 1 < len(input_ids):
                    active_ids.add(int(input_ids[k + 1]))
        else:
            # Standard SFT dataset without per-item loss_mask (e.g., GPTSFTDataset
            # computes loss_mask at collation time). Collect all tokens from
            # answer_start_idx onwards -- a conservative superset.
            answer_start = item["answer_start_idx"]
            for token_id in input_ids[answer_start:]:
                active_ids.add(int(token_id))

    result = torch.tensor(sorted(active_ids), dtype=torch.long)
    logger.info(f"Collected {len(result)} unique active vocab IDs from {n} samples")
    return result


def install_vocab_slice(model: torch.nn.Module, active_ids: torch.Tensor) -> None:
    """Patch model to compute logits only for active vocabulary tokens.

    Modifies the model in-place by monkey-patching two methods:

    1. ``output_layer.forward``: slices the shared embedding weight to
       ``weight[active_ids]`` (shape ``[N, H]`` instead of ``[V, H]``).
    2. ``compute_language_model_loss``: remaps label token IDs to their
       index in ``active_ids`` before computing cross-entropy. Labels at
       masked positions (loss_mask=0) may contain IDs outside ``active_ids``;
       these are mapped to index 0 (safe since their loss contribution is
       zeroed by the mask).

    This produces ``[batch, seq, N]`` logits instead of ``[batch, seq, V]``,
    reducing both the output matmul and cross-entropy costs by ``V/N``.

    Args:
        model: The GPT model. May be wrapped with DDP / Float16Module.
        active_ids: Sorted 1D tensor of active vocabulary token IDs,
            typically from :func:`collect_active_vocab_ids`.
    """
    # Unwrap DDP / Float16Module to reach the inner GPTModel
    gpt_model = model
    while hasattr(gpt_model, "module"):
        gpt_model = gpt_model.module

    # Build remap table: original_label_id -> index in active_ids
    vocab_size = gpt_model.vocab_size
    device = next(gpt_model.parameters()).device
    active_ids = active_ids.to(device)

    if len(active_ids) > 0 and active_ids.max().item() >= vocab_size:
        raise ValueError(
            f"active_ids contains token ID {active_ids.max().item()} "
            f">= vocab_size {vocab_size}"
        )

    remap = torch.zeros(vocab_size, dtype=torch.long, device=device)
    remap[active_ids] = torch.arange(len(active_ids), dtype=torch.long, device=device)

    n_active = len(active_ids)

    # 1. Patch output_layer to slice weight to active_ids rows.
    # runtime_gather_output is part of the Megatron ColumnParallelLinear
    # interface but unused here since we bypass the parallel output layer.
    def _sliced_output_forward(input_, weight=None, runtime_gather_output=None):
        if weight is not None:
            weight = weight[active_ids]
        else:
            weight = gpt_model.shared_embedding_or_output_weight()[active_ids]
        return F.linear(input_, weight), None

    gpt_model.output_layer.forward = _sliced_output_forward

    # 2. Patch loss to remap labels to active_ids indices
    original_loss_fn = gpt_model.compute_language_model_loss

    def _remapped_loss(labels, logits):
        return original_loss_fn(remap[labels], logits)

    gpt_model.compute_language_model_loss = _remapped_loss

    # Log reduction ratio
    embed_weight = gpt_model.shared_embedding_or_output_weight()
    full_vocab = embed_weight.shape[0]
    logger.info(
        f"Vocab slice installed: {n_active} active tokens out of {full_vocab} "
        f"({n_active / full_vocab:.1%} of output layer compute)"
    )


def create_vocab_sliced_forward_step(
    active_ids: torch.Tensor,
    base_forward_step: Callable | None = None,
) -> Callable:
    """Create a forward_step wrapper that auto-installs vocab slicing.

    The vocab slice is installed lazily on the first call, when the model
    is first available.

    Args:
        active_ids: Sorted 1D tensor of active vocabulary token IDs,
            typically from :func:`collect_active_vocab_ids`.
        base_forward_step: The base forward_step function to wrap.
            If None, imports and uses
            :func:`megatron.bridge.training.gpt_step.forward_step`.

    Returns:
        A forward_step function with the same signature as the base.

    Example::

        active_ids = collect_active_vocab_ids(train_dataset)
        fwd_step = create_vocab_sliced_forward_step(active_ids)
        finetune(config=cfg, forward_step_func=fwd_step)
    """
    if base_forward_step is None:
        from megatron.bridge.training.gpt_step import forward_step

        base_forward_step = forward_step

    installed_model_id: int | None = None

    def _wrapper(state, data_iterator, model, return_schedule_plan=False):
        nonlocal installed_model_id
        if installed_model_id != id(model):
            install_vocab_slice(model, active_ids)
            installed_model_id = id(model)
        return base_forward_step(state, data_iterator, model, return_schedule_plan)

    return _wrapper
