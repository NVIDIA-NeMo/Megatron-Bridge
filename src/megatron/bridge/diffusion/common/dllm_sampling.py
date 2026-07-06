# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Sampling primitives for block-diffusion / masked-dLLM generation.

The sampling primitives (``add_gumbel_noise``, ``get_num_transfer_tokens``,
``get_transfer_index``) implement the iterative-denoising step shared by every
block-diffusion / masked-dLLM generation loop in this repo (NemotronLabsDiffusion,
LLaDA1.5, ...). They are model-agnostic: each model keeps its own generation loop
with its own attention semantics (causal-with-KV-cache vs fully bidirectional) but
calls these helpers to score confidence and choose which masked positions to
unmask at each step.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise to logits for stochastic sampling.

    At ``temperature == 0`` this is a no-op (returns ``logits`` unchanged), so an
    ``argmax`` over the result is plain greedy decoding.

    Args:
        logits: Unnormalized scores of shape ``[..., vocab_size]``.
        temperature: Sampling temperature. ``0`` disables noise (greedy).

    Returns:
        Noised scores (float64 when noise is applied) whose ``argmax`` samples
        from the temperature-scaled distribution.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute how many masked tokens to unmask at each diffusion step.

    Distributes the number of masked positions as evenly as possible across
    ``steps``, giving the earlier steps the remainder.

    Args:
        mask_index: Boolean tensor ``[batch, seq_len]`` (True where masked).
        steps: Number of denoising steps to spread the unmasking over.

    Returns:
        Int64 tensor ``[batch, steps]`` whose rows sum to each sequence's mask
        count.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
    threshold: Optional[float] = None,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select which masked positions to unmask at one diffusion step.

    Samples candidate tokens (``x0``) from ``logits`` and, among currently
    masked positions, transfers the highest-confidence ones from mask to real
    token. Used identically by every block-diffusion generation loop in the repo
    regardless of attention semantics.

    Args:
        logits: Per-position scores ``[batch, seq_len, vocab_size]``.
        temperature: Sampling temperature for Gumbel noise (``0`` = greedy).
        remasking: Confidence source for ranking: ``"low_confidence"`` uses the
            softmax probability of the chosen token; ``"random"`` uses uniform
            noise.
        mask_index: Boolean ``[batch, seq_len]`` marking still-masked positions.
        x: Current token ids ``[batch, seq_len]``; non-masked positions are kept.
        num_transfer_tokens: Per-sequence count of tokens to unmask this step
            (``[batch]`` slice of :func:`get_num_transfer_tokens`). Ignored when
            ``threshold`` is set.
        threshold: If set, transfer every masked position whose confidence
            exceeds this value instead of a fixed count.
        neg_entropy: If True, rank by negative entropy of the distribution
            instead of the chosen token's probability.

    Returns:
        Tuple ``(x0, transfer_index)`` where ``x0`` is the candidate token ids
        (non-masked positions unchanged) and ``transfer_index`` is a boolean mask
        of positions to commit this step.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    if neg_entropy:
        p = F.softmax(logits, dim=-1)
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        confidence_scores = torch.sum(p * log_probs, dim=-1)
    else:
        confidence_scores = x0_p

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence_scores, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index
