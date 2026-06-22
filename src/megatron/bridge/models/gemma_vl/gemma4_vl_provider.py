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

"""Gemma 4 Vision-Language model providers.

Gemma4VLModelProvider: MoE Vision-Language provider (extends Gemma4ModelProvider).
Gemma4DenseVLProvider: Dense Vision-Language provider (extends Gemma4DenseProvider).

Text-only providers (Gemma4DenseProvider, Gemma4ModelProvider) live in:
    megatron.bridge.models.gemma.gemma4_provider
"""

from dataclasses import dataclass
from typing import Any

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel


# ---------------------------------------------------------------------------
# VL providers
# ---------------------------------------------------------------------------


@dataclass
class Gemma4VLModelProvider(Gemma4ModelProvider):
    """Model provider for Gemma 4 MoE Vision-Language models."""

    scatter_embedding_sequence_parallel: bool = False

    vision_config: Any = None
    text_config: Any = None
    audio_config: Any = None

    vision_soft_tokens_per_image: int = 280

    bos_token_id: int = 2
    eos_token_id: int = 1
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Gemma4VLModel:
        model = Gemma4VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class Gemma4DenseVLProvider(Gemma4DenseProvider):
    """Model provider for Dense Gemma 4 Vision-Language checkpoints."""

    scatter_embedding_sequence_parallel: bool = False

    vision_config: Any = None
    text_config: Any = None
    audio_config: Any = None

    vision_soft_tokens_per_image: int = 280

    bos_token_id: int = 2
    eos_token_id: int = 1
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Gemma4VLModel:
        model = Gemma4VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
