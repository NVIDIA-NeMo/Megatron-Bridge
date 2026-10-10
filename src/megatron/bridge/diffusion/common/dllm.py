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

"""Diffusion language model utilities: masking, block attention masks, and sampling.

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
from torch.nn.attention.flex_attention import create_block_mask


def forward_process_simple_masking(input_ids, mask_token_id, eps=1e-3, loss_mask=None, generator=None):
    """Uniform random masking for diffusion LM training.

    For each sequence in the batch, sample a masking ratio t ~ U(eps, 1) and
    independently mask each token with probability t.

    Returns:
        noisy_batch: input_ids with masked positions replaced by mask_token_id
        masked_indices: boolean mask of shape (b, l)
        p_mask: per-token masking probability of shape (b, l)
    """
    b, seq_len = input_ids.shape
    device = input_ids.device

    t = torch.rand(b, device=device, generator=generator)

    p_mask = (1 - eps) * t + eps  # shape: (b,)
    p_mask = p_mask[:, None].expand(-1, seq_len)  # shape: (b, l)

    masked_indices = torch.rand((b, seq_len), device=device, generator=generator) < p_mask

    if loss_mask is not None:
        masked_indices[loss_mask == 0] = 0

    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

    return noisy_batch, masked_indices, p_mask


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


def compute_block_mask(block_size, max_seq_length):
    """Compute the sbd_block_diff attention mask.

    The semi-block-diffusion mask is composed of three sub-masks over a
    doubled sequence [xt | x0] of length 2*max_seq_length:
      - Block Diagonal (M_BD): self-attention within noised blocks (xt only)
      - Offset Block-Causal (M_OBC): cross-attention from xt to past x0 blocks
      - Fully Causal (M_FC): fully causal attention within x0

    Args:
        block_size: Block size for block-based attention.
        max_seq_length: Length of one half (xt or x0) of the sequence.

    Returns:
        BlockMask for use with ``flex_attention``.
    """
    n = max_seq_length

    def sbd_block_diff_mask(b, h, q_idx, kv_idx):
        x0_flag_q = q_idx >= n
        x0_flag_kv = kv_idx >= n

        block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

        block_diagonal = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
        fully_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

        return block_diagonal | offset_block_causal | fully_causal

    q_len = max_seq_length * 2
    return create_block_mask(sbd_block_diff_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len)


def compute_block_bias(block_size, max_seq_length, dtype, device):
    """Dense additive ``post_scale_bias`` equivalent of the sbd_block_diff mask.

    flex_attention's ``BlockMask`` is not compatible with context parallelism
    (Transformer Engine disables the unfused backend under CP). The same mask can
    instead be supplied to ``TEDotProductAttention`` as an additive bias
    (``core_attention_bias_type="post_scale_bias"``), which TE *does* support
    under CP and slices per zigzag chunk internally.

    The returned bias is ``0`` where attention is allowed and a large negative
    value where it is disallowed, shaped ``[1, 1, 2L, 2L]`` (broadcastable
    ``11ss``). Every query row has at least its own position allowed (xt within
    its block; x0 via causal self), so no row is fully masked.

    Args:
        block_size: Block size for block-based attention.
        max_seq_length: Length of one half (xt or x0) of the doubled sequence.
        dtype: Bias dtype (match the attention compute dtype).
        device: Device to build the bias on.

    Returns:
        Tensor of shape ``[1, 1, 2*max_seq_length, 2*max_seq_length]``.
    """
    n = max_seq_length
    q_len = 2 * n
    idx = torch.arange(q_len, device=device)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]

    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n
    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    fully_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q
    allowed = block_diagonal | offset_block_causal | fully_causal

    bias = torch.zeros(q_len, q_len, dtype=dtype, device=device)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias.view(1, 1, q_len, q_len)
