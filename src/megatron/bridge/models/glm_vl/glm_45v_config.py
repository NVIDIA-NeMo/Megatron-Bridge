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

"""Builder-backed model config for GLM-4.5V and GLM-4.6V.

The language half is the GLM-4.5 MoE decoder, configured with the same fields as the GLM-4.5 text
bridge's ``BridgeGPTModelConfig``. ``GLM45VModel`` binds Hugging Face ``Glm4vModel`` methods that
read the vision config, the image and video token IDs and ``return_dict`` from the model's config,
so the transformer config declares them. The other vision-language settings live on the outer config.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class GLM45VTransformerConfig(TransformerConfig):
    """GLM-4.5V fields that ``GLM45VModel`` reads from the transformer config it is built with."""

    vision_config: Any = None
    """Hugging Face ``Glm4vMoeVisionConfig``."""

    return_dict: bool = True
    image_token_id: int = 151363
    video_token_id: int = 151364

    spatial_merge_size: int = 2
    """Set by ``GLM45VModel`` from ``vision_config``."""


@dataclass(kw_only=True)
class GLM45VModelConfig(BridgeGPTModelConfig):
    """Builder config for GLM-4.5V and GLM-4.6V vision-language models.

    The remaining token IDs are the provider fields of the same names.
    """

    builder: ClassVar[str] = "megatron.bridge.models.glm_vl.glm_45v_builder.GLM45VModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = GLM45VTransformerConfig

    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "yarn", "none"] = "mrope"
    scatter_embedding_sequence_parallel: bool = False

    eos_token_id: int | list[int] = 151329
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False


__all__ = ["GLM45VModelConfig", "GLM45VTransformerConfig"]
