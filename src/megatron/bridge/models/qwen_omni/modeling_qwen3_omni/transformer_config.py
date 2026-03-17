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

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig


@dataclass
class Qwen3OmniTransformerConfig(Qwen3VLTransformerConfig):
    """Stage-1 Qwen3-Omni transformer config.

    Stage 1 focuses on the thinker language backbone and checkpoint conversion.
    Vision and audio runtime support are added in later stages.
    """

    vocab_size: int = 152064
    language_max_sequence_length: int = 32768
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    fp16_lm_cross_entropy: bool = False
    rotary_percent: float = 1.0
    rotary_base: float = 1000000.0
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151646
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648

    qk_layernorm: bool = True
