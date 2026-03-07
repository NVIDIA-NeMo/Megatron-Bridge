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

from dataclasses import dataclass, field
from typing import List

from megatron.core.transformer.transformer_config import TransformerConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


@dataclass
class Qwen3OmniTransformerConfig(TransformerConfig):
    """Configuration for Qwen3 Omni transformer with vision and language components."""

    vocab_size: int = 152064
    language_max_sequence_length: int = 4096

    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2048

    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    fp16_lm_cross_entropy: bool = False
    share_embeddings_and_output_weights: bool = False
    rotary_percent: float = 1.0
    rotary_base: float = 10000

    # Multimodal rope section for [temporal, height, width] dimensions
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    apply_rope_fusion: bool = False

    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151675
    vision_start_token_id: int = 151652
    audio_start_token_id: int = 151669
    position_id_per_seconds: int = 13
    hf_text_config: Qwen3VLTextConfig | None = None
    vision_dp_when_cp: bool = False
    use_hf_vision_model: bool = False
