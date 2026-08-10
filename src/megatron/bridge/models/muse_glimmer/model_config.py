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

"""Builder-backed configuration objects for Muse Glimmer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class MuseGlimmerTransformerConfig(TransformerConfig):
    """Muse-specific decoder fields not represented by MCore's base config."""

    post_norm_epsilon: float = 1e-8
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0


@dataclass
class MuseGlimmerVisionModelConfig:
    """Serializable configuration for the replicated Muse vision encoder."""

    hidden_size: int = 1_536
    intermediate_size: int = 8_960
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    max_position_embeddings: int = 1_024
    layer_norm_epsilon: float = 1e-5
    hidden_activation: str = "gelu"
    rotary_base: float = 10_000.0
    layer_types: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class MuseGlimmerModelConfig(BridgeGPTModelConfig):
    """Complete builder configuration for Muse Glimmer."""

    builder: ClassVar[str] = "megatron.bridge.models.muse_glimmer.MuseGlimmerModelBuilder"

    vision: MuseGlimmerVisionModelConfig = field(default_factory=MuseGlimmerVisionModelConfig)
    image_token_id: int = 200_092
    video_token_id: int = 200_091
    bos_token_id: int | None = 200_000
    eos_token_id: int | list[int] | None = 200_001
    pad_token_id: int | None = None
    vision_output_size: int = 6_144
    projector_hidden_size: int = 4_096
    projector_hidden_activation: str = "gelu"
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return media token IDs used by multimodal data pipelines."""
        return {"images": self.image_token_id, "videos": self.video_token_id}


__all__ = [
    "MuseGlimmerModelConfig",
    "MuseGlimmerTransformerConfig",
    "MuseGlimmerVisionModelConfig",
]
