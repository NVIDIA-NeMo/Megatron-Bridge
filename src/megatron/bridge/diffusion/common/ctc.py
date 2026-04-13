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

"""CTC block mask for diffusion language models with asymmetric block sizes."""

import torch
from torch.nn.attention.flex_attention import create_block_mask


def compute_ctc_block_mask(xt_block_size, x0_block_size, max_seq_length):
    """Compute the attention mask for CTC-based dLLM with asymmetric block sizes.

    The sequence is [xt | x0] of length 2*max_seq_length:
      - xt: CTC output blocks of size xt_block_size (e.g. 128), filled with blank tokens
      - x0: clean context blocks of size x0_block_size (e.g. 64)

    Attention pattern:
      - Block Diagonal (bidirectional) within each xt block (for CTC prediction)
      - Offset Block-Causal: xt block i attends to x0 blocks 0..i-1
      - Fully Causal within x0

    xt block i predicts target tokens[i*x0_block_size : (i+1)*x0_block_size] using CTC,
    conditioned on tokens[0 : i*x0_block_size] from x0.

    Args:
        xt_block_size: Block size for CTC output blocks (e.g. 128).
        x0_block_size: Block size for clean context blocks (e.g. 64).
        max_seq_length: Length of one half (xt or x0) of the sequence.

    Returns:
        BlockMask for use with ``flex_attention``.
    """
    n = max_seq_length

    def ctc_block_mask_fn(b, h, q_idx, kv_idx):
        x0_flag_q = q_idx >= n
        x0_flag_kv = kv_idx >= n

        # Compute block indices using appropriate block sizes
        block_q = torch.where(x0_flag_q, (q_idx - n) // x0_block_size, q_idx // xt_block_size)
        block_kv = torch.where(x0_flag_kv, (kv_idx - n) // x0_block_size, kv_idx // xt_block_size)

        # Bidirectional within each xt block
        block_diagonal = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)

        # xt block i attends to x0 blocks 0..i-1
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)

        # Fully causal within x0
        fully_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

        return block_diagonal | offset_block_causal | fully_causal

    q_len = max_seq_length * 2
    return create_block_mask(ctc_block_mask_fn, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len)
