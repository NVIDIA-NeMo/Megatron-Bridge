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

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Qwen3ASRTransformerConfig(TransformerConfig):
    """Configuration for Qwen3-ASR transformer with audio and language components."""

    vocab_size: int = 152064
    language_max_sequence_length: int = 4096

    apply_rotary_pos_emb_in_fp32: bool = False
    fp16_lm_cross_entropy: bool = False
    share_embeddings_and_output_weights: bool = False
    rotary_percent: float = 1.0
    rotary_base: float = 5000000.0

    # Multimodal rope section for 3 dimensions (same position IDs across all dims for ASR)
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    apply_rope_fusion: bool = False

    audio_token_id: int = 151646
    audio_start_token_id: int = 151647

    qk_layernorm: bool = True
