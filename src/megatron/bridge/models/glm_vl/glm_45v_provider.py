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

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig

from megatron.bridge.models import GLM45AirModelProvider106B

from .modeling_glm_45v import GLM45VModel


@dataclass
class GLM45VModelProvider(GLM45AirModelProvider106B):
    """
    Base model provider for GLM 4.5 Vision-Language (VL) Models.
    """

    # Language configuration inherited from GLM45ModelProvider (GLM 4.5 Air)
    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [8, 12, 12])

    # Vision configuration
    vision_config: Glm4vVisionConfig = field(default_factory=Glm4vVisionConfig)

    # Token IDs
    eos_token_id: int = 151329
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342
    image_token_id: int = 151363
    video_token_id: int = 151364

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GLM45VModel:
        model = GLM45VModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
