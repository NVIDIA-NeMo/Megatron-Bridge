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

import torch


def get_rope_index(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index for Qwen3-ASR.

    Simplified version for audio-only model: just cumulative position IDs
    expanded to 3 MRoPE dimensions. No special vision/video handling needed.

    Ported from HF Qwen3ASRPreTrainedModelForConditionalGeneration.get_rope_index.

    Args:
        input_ids: Input token IDs of shape (batch_size, sequence_length).
        attention_mask: Attention mask of shape (batch_size, sequence_length).

    Returns:
        position_ids: Position IDs of shape (3, batch_size, sequence_length).
        mrope_position_deltas: Position deltas of shape (batch_size, 1).
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    attention_mask = attention_mask.to(input_ids.device)
    position_ids = attention_mask.float().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

    return position_ids, mrope_position_deltas
