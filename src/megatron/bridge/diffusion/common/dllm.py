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

"""Diffusion language model utilities: masking and block attention masks."""

import torch
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
    b, l = input_ids.shape
    device = input_ids.device

    t = torch.rand(b, device=device, generator=generator)

    p_mask = (1 - eps) * t + eps  # shape: (b,)
    p_mask = p_mask[:, None].expand(-1, l)  # shape: (b, l)

    masked_indices = torch.rand((b, l), device=device, generator=generator) < p_mask

    if loss_mask is not None:
        masked_indices[loss_mask == 0] = 0

    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

    return noisy_batch, masked_indices, p_mask


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
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)

        block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

        block_diagonal = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
        fully_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

        return block_diagonal | offset_block_causal | fully_causal

    q_len = max_seq_length * 2
    return create_block_mask(sbd_block_diff_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len)
