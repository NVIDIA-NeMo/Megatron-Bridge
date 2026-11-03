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

"""Builder-backed model config for GLM-5.3-Flash (``glm5_next``).

The language model is a Megatron Core ``HybridModel`` of KDA, DSA (MLA with a kpool indexer),
dense MLP and MoE layers wrapped in mHC hyper-connections. ``GLM53FlashModel`` and the Hugging Face
feature methods bound to it read the vision config and multimodal token IDs from the config the
model is built with, so the transformer config declares them. Freeze options live on the outer config.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from megatron.core.ssm.kda_layer_config import KDALayerConfig

from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.hybrid_mla_provider import apply_dsa_pipeline_separators
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig


@dataclass
class _GLM53FlashVLTransformerFields:
    """Vision-language fields that ``GLM53FlashModel`` and its bound Hugging Face methods read."""

    vision_config: Any = None
    """Hugging Face ``Glm5NextVisionConfig``."""

    spatial_merge_size: int = 2
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833

    return_dict: bool = True
    """Read by Hugging Face's ``can_return_tuple``-decorated feature extraction methods."""


@dataclass
class GLM53FlashTransformerConfig(_GLM53FlashVLTransformerFields, MLATransformerConfig, KDALayerConfig):
    """MLA transformer config with KDA layer fields and the vision-language fields of GLM-5.3-Flash."""


@dataclass(kw_only=True)
class GLM53FlashModelConfig(BridgeHybridModelConfig):
    """GLM-5.3-Flash as a Hybrid MLA language model with a Hugging Face vision tower.

    The builder counterpart of ``GLM53FlashModelProvider``. ``finalize()`` places DSA-safe
    pipeline separators in ``hybrid_layer_pattern`` before the Hybrid finalization, as the
    provider does. The nested transformer config is an MLA config, but ``rotary_base`` and
    ``rotary_percent`` stay on this outer config, as in ``BridgeHybridModelConfig``; GLM-5.3-Flash
    applies no rotary embedding (``qk_pos_emb_head_dim=0``) and keeps their defaults.
    """

    builder: ClassVar[str] = "megatron.bridge.models.glm5next.glm53flash_builder.GLM53FlashModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = GLM53FlashTransformerConfig

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def finalize(self) -> None:
        """Add DSA-safe pipeline separators, then finalize the Hybrid config."""
        apply_dsa_pipeline_separators(self)
        super().finalize()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config, with the Hugging Face vision config in the form provider checkpoints use.

        Megatron Core serializes nested dataclasses field by field; Hugging Face configs are
        serialized through Bridge's config container instead.
        """
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

        transformer = self.transformer
        vision_config = transformer.vision_config
        transformer.vision_config = None
        try:
            data = super().as_dict()
        finally:
            transformer.vision_config = vision_config
        data["transformer"]["vision_config"] = _ConfigContainerBase._convert_value_to_dict(vision_config)
        return data


__all__ = ["GLM53FlashModelConfig", "GLM53FlashTransformerConfig"]
