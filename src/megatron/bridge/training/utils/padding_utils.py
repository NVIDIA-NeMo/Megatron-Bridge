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

"""Padding and truncation helpers for training batches.

These utilities centralize common sequence length adjustments used to ensure
fixed or efficient shapes for tensors such as tokens, labels, position ids,
and attention masks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = [
    "pad_or_truncate_2d_to_len",
    "pad_or_truncate_pos_to_len",
    "pad_or_truncate_attn_to_len",
]


def pad_or_truncate_2d_to_len(
    x: torch.Tensor | None, target_len: int, max_cap: int, pad_value: int | float
) -> torch.Tensor | None:
    """Pad or truncate a 2D tensor to a desired target length with an upper cap.

    Expects input of shape (batch, seq_len). Pads/truncates along the last dimension.
    """
    if x is None:
        return None
    current_len = x.size(1)
    if current_len < target_len:
        return F.pad(x, (0, target_len - current_len), value=pad_value)
    if current_len > max_cap:
        return x[:, :max_cap]
    return x


def pad_or_truncate_pos_to_len(pos: torch.Tensor | None, target_len: int, max_cap: int) -> torch.Tensor | None:
    """Pad or truncate position ids to a target length with an upper cap.

    Extends positions by appending a monotonically increasing range starting
    from the current length to the target length.
    """
    if pos is None:
        return None
    current_len = pos.size(1)
    if current_len < target_len:
        addition = (
            torch.arange(current_len, target_len, device=pos.device, dtype=pos.dtype)
            .unsqueeze(0)
            .expand(pos.size(0), -1)
        )
        return torch.cat([pos, addition], dim=1)
    if current_len > max_cap:
        return pos[:, :max_cap]
    return pos


def pad_or_truncate_attn_to_len(mask: torch.Tensor | None, target_len: int, max_cap: int) -> torch.Tensor | None:
    """Pad or truncate a 4D attention mask to the target length with an upper cap.

    Expects input of shape (batch, heads, seq_len, seq_len). Pads the last two dims.
    """
    if mask is None:
        return None
    _, _, s1, s2 = mask.shape
    pad_value = False if mask.dtype == torch.bool else 0
    if s1 < target_len:
        return F.pad(mask, (0, target_len - s2, 0, target_len - s1), value=pad_value)
    if s1 > max_cap:
        return mask[:, :, :max_cap, :max_cap]
    return mask
