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

"""Builder-backed model config for EXAONE 4.5 vision-language models.

The language half is the EXAONE 4.0 Post-LN decoder with hybrid sliding-window attention and an
optional MTP layer, configured with the same fields as the EXAONE 4.0 text bridge's
``BridgeGPTModelConfig``. The vision-language settings live on the outer config;
``Exaone45ModelBuilder`` combines the decoder with the EXAONE 4.5 vision encoder.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class Exaone45ModelConfig(BridgeGPTModelConfig):
    """Builder config for EXAONE 4.5 vision-language models.

    ``vision_config`` holds the Hugging Face vision config, as on the provider. The token IDs and
    freeze options are the provider fields of the same names. ``language_max_sequence_length``
    replaces the provider's ``hf_text_config``, which the model read only for its
    ``max_position_embeddings``.
    """

    builder: ClassVar[str] = "megatron.bridge.models.exaone.exaone45.exaone45_builder.Exaone45ModelBuilder"

    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "yarn", "none"] = "rope"
    scatter_embedding_sequence_parallel: bool = False

    vision_config: Any = None
    """Hugging Face ``Exaone4_5_VisionConfig``."""

    image_token_id: int = 67
    video_token_id: int = 68
    vision_token_id: int = 67
    vision_start_token_id: int = 73
    vision_end_token_id: int = 74
    bos_token_id: int = 1
    eos_token_id: int = 53
    spatial_merge_size: int = 2

    language_max_sequence_length: int = 131072
    """Language model ``max_position_embeddings`` from the Hugging Face text config."""

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    freeze_mtp_model: bool = False


__all__ = ["Exaone45ModelConfig"]
