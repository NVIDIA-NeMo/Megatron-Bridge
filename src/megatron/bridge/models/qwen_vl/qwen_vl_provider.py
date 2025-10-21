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

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models import (
    Qwen2ModelProvider,
)

from .modeling_qwen25_vl import Qwen25VLModel


# =============================================================================
# Qwen 2.5 VL Model Providers
# =============================================================================


@dataclass
class Qwen25VLModelProvider(Qwen2ModelProvider):
    """
    Base model provider for Qwen 2.5 VL Models.
    """

    # Language configuration inherited from Qwen25ModelProvider3B
    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False
    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])

    # Vision configuration
    vision_config: Qwen2_5_VLVisionConfig = field(default_factory=Qwen2_5_VLVisionConfig)

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen25VLModel:
        model = Qwen25VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

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
