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

"""Gemma 4 VL model provider."""

from dataclasses import dataclass, field
from typing import Any

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel


@dataclass
class Gemma4VLModelProvider(Gemma4ModelProvider):
    """Model provider for Gemma 4 Vision-Language models.

    Extends Gemma4ModelProvider with vision tower config, multimodal projector
    config, and token IDs for vision-text fusion.
    """

    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False

    # Vision configuration (set by bridge from HF config)
    vision_config: Any = None
    text_config: Any = None  # HF text config, needed for multimodal embedder init

    # Multimodal token counts
    vision_soft_tokens_per_image: int = 280

    # Token IDs
    bos_token_id: int = 2
    eos_token_id: int = 1
    image_token_id: int = 258_880
    video_token_id: int = 258_884

    # Freeze options
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
