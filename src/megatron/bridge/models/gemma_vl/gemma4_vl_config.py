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

"""Builder-backed model configs for Gemma 4 vision-language models.

Gemma4VLModelConfig: MoE 26B-A4B, built by ``Gemma4VLModelBuilder``.
Gemma4DenseVLModelConfig: dense E2B, E4B and 31B, built by ``Gemma4DenseVLModelBuilder``.

The language halves are the Gemma 4 text configs. ``Gemma4VLModel`` reads the Hugging Face
sub-configs and the image and audio token IDs from the config it is built with, as it does from
a provider, so the vision-language transformer configs declare them. The other vision-language
settings live on the outer configs.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from megatron.bridge.models.gemma.gemma4_config import (
    Gemma4DenseModelConfig,
    Gemma4DenseTransformerConfig,
    Gemma4ModelConfig,
    Gemma4TransformerConfig,
)
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class _Gemma4VLTransformerFields:
    """Vision-language fields ``Gemma4VLModel`` reads from the transformer config it is built with."""

    vision_config: Any = None
    """Hugging Face ``Gemma4VisionConfig``."""

    text_config: Any = None
    """Hugging Face ``Gemma4TextConfig``, for the multimodal embedders, padding and attention masks."""

    audio_config: Any = None
    """Hugging Face ``Gemma4AudioConfig``; None for checkpoints without an audio encoder."""

    image_token_id: int = 258_880
    audio_token_id: int = 258_881


_HF_CONFIG_FIELDS = ("vision_config", "text_config", "audio_config")


@dataclass(kw_only=True)
class _Gemma4VLModelFields:
    """Vision-language fields of the outer config: export-only token IDs and freeze options."""

    vision_soft_tokens_per_image: int = 280
    bos_token_id: int = 2
    eos_token_id: int | list[int] = 1
    video_token_id: int = 258_884

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config, with the Hugging Face sub-configs in the form provider checkpoints use.

        Megatron Core serializes nested dataclasses field by field, which fails for
        ``Gemma4TextConfig``: its per-layer attributes raise when read for the whole model.
        """
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

        transformer = self.transformer
        hf_configs = {name: getattr(transformer, name) for name in _HF_CONFIG_FIELDS}
        for name in _HF_CONFIG_FIELDS:
            setattr(transformer, name, None)
        try:
            data = super().as_dict()
        finally:
            for name, value in hf_configs.items():
                setattr(transformer, name, value)
        for name, value in hf_configs.items():
            data["transformer"][name] = _ConfigContainerBase._convert_value_to_dict(value)
        return data


@dataclass
class Gemma4VLTransformerConfig(_Gemma4VLTransformerFields, Gemma4TransformerConfig):
    """Gemma 4 MoE decoder fields plus the vision-language fields ``Gemma4VLModel`` reads."""


@dataclass
class Gemma4DenseVLTransformerConfig(_Gemma4VLTransformerFields, Gemma4DenseTransformerConfig):
    """Gemma 4 dense decoder fields plus the vision-language fields ``Gemma4VLModel`` reads."""


@dataclass(kw_only=True)
class Gemma4VLModelConfig(_Gemma4VLModelFields, Gemma4ModelConfig):
    """Builder config for Gemma 4 MoE vision-language models (26B-A4B and similar).

    ``Gemma4VLModel`` merges image features into the language embeddings before it scatters
    them for sequence parallelism, so the language model's embedding does not scatter them.
    """

    builder: ClassVar[str] = "megatron.bridge.models.gemma_vl.gemma4_vl_builder.Gemma4VLModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Gemma4VLTransformerConfig

    scatter_embedding_sequence_parallel: bool = False


@dataclass(kw_only=True)
class Gemma4DenseVLModelConfig(_Gemma4VLModelFields, Gemma4DenseModelConfig):
    """Builder config for Gemma 4 dense vision-language models (E2B, E4B, 31B)."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma_vl.gemma4_vl_builder.Gemma4DenseVLModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Gemma4DenseVLTransformerConfig


__all__ = [
    "Gemma4DenseVLModelConfig",
    "Gemma4DenseVLTransformerConfig",
    "Gemma4VLModelConfig",
    "Gemma4VLTransformerConfig",
]
