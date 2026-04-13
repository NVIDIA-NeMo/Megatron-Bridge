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

"""CTC block mask for diffusion language models with asymmetric sequence lengths."""

import torch
from torch.nn.attention.flex_attention import create_block_mask


def compute_ctc_block_mask(xt_block_size, x0_block_size, x0_len):
    """Compute the attention mask for CTC-based dLLM.

    The sequence layout is [xt | x0] where:
      - xt has length xt_len = x0_len * 2 (2x expansion for CTC)
      - x0 has length x0_len (clean context tokens)
      - Total sequence = xt_len + x0_len = 3 * x0_len

    Each xt block of xt_block_size (128) positions predicts a target of
    x0_block_size (64) tokens from the corresponding x0 block via CTC loss.

    Number of blocks:
      - xt: xt_len / xt_block_size = (2 * x0_len) / 128 = 64 blocks (for x0_len=4096)
      - x0: x0_len / x0_block_size = 4096 / 64 = 64 blocks

    Attention pattern:
      - Block Diagonal (bidirectional) within each xt block
      - Offset Block-Causal: xt block i attends to x0 blocks 0..i-1
      - Fully Causal within x0

    Args:
        xt_block_size: CTC output block size (e.g. 128).
        x0_block_size: Clean context block size (e.g. 64).
        x0_len: Length of the clean context (e.g. 4096).

    Returns:
        BlockMask for use with ``flex_attention``.
    """
    xt_len = x0_len * 2  # 2x expansion
    total_len = xt_len + x0_len

    def ctc_block_mask_fn(b, h, q_idx, kv_idx):
        x0_flag_q = q_idx >= xt_len
        x0_flag_kv = kv_idx >= xt_len

        # Block indices: xt uses xt_block_size, x0 uses x0_block_size
        block_q = torch.where(x0_flag_q, (q_idx - xt_len) // x0_block_size, q_idx // xt_block_size)
        block_kv = torch.where(x0_flag_kv, (kv_idx - xt_len) // x0_block_size, kv_idx // xt_block_size)

        # Bidirectional within each xt block
        block_diagonal = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)

        # xt block i attends to x0 blocks 0..i-1
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)

        # Fully causal within x0
        fully_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

        return block_diagonal | offset_block_causal | fully_causal

    return create_block_mask(ctc_block_mask_fn, B=None, H=None, Q_LEN=total_len, KV_LEN=total_len)
