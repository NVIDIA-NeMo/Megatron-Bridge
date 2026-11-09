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

"""Builder-backed model configs for MiniMax-M3.

MiniMaxM3VLModelConfig: the complete vision-language model the released checkpoint ships, built by
``MiniMaxM3VLModelBuilder``.
MiniMaxM3ModelConfig: the sparse-MoE language model alone, built by ``MiniMaxM3ModelBuilder``. The
text pretraining and SFT recipes use it.

``MiniMaxM3VLModel`` reads the vision config, the token IDs, the projector shape and the Lightning
Indexer layout from the config it is built with, as it does from a provider, so the
vision-language transformer config declares them. The other vision-language settings live on the
outer config.
"""

import copy
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from transformers import PretrainedConfig

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.instantiate_utils import _resolve_target


@dataclass(kw_only=True)
class MiniMaxM3ModelConfig(BridgeGPTModelConfig):
    """Builder config for the MiniMax-M3 sparse-MoE language model.

    ``MiniMaxM3ModelBuilder`` keeps the router weights in FP32, as the checkpoint stores them.
    """

    builder: ClassVar[str] = "megatron.bridge.models.minimax_m3.minimax_m3_builder.MiniMaxM3ModelBuilder"


@dataclass
class MiniMaxM3VLTransformerConfig(TransformerConfig):
    """Language-model fields plus the vision-language fields ``MiniMaxM3VLModel`` reads."""

    vision_config: Any = None
    """Hugging Face vision config of the checkpoint."""

    image_token_id: int = 200025
    video_token_id: int = 200026
    projector_hidden_size: int = 6144
    multimodal_projector_bias: bool = True
    spatial_merge_size: int = 2

    lightning_indexer_layers: list[int] = field(default_factory=list)
    """Decoder layers whose Lightning Indexer weights the model keeps as frozen state."""

    index_n_heads: int = 4
    index_head_dim: int = 128


@dataclass(kw_only=True)
class MiniMaxM3VLModelConfig(MiniMaxM3ModelConfig):
    """Builder config for the complete MiniMax-M3 vision-language model.

    ``MiniMaxM3VLModel`` merges image features into the language embeddings before it scatters
    them for sequence parallelism, so the language model's embedding does not scatter them.
    """

    builder: ClassVar[str] = "megatron.bridge.models.minimax_m3.minimax_m3_builder.MiniMaxM3VLModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = MiniMaxM3VLTransformerConfig

    scatter_embedding_sequence_parallel: bool = False

    hf_config_dict: dict[str, Any] = field(default_factory=dict)
    """The source Hugging Face config, which export starts from."""

    temporal_patch_size: int = 2

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the token IDs used by multimodal data pipelines."""
        return {"images": self.image_token_id}

    def to_text_config(self) -> MiniMaxM3ModelConfig:
        """Return the equivalent language-model config.

        The text pretraining and SFT recipes build the language model alone, which keeps the
        native checkpoint keys of the language model.
        """
        transformer = TransformerConfig(
            **{
                config_field.name: getattr(self.transformer, config_field.name)
                for config_field in fields(TransformerConfig)
                if config_field.init
            }
        )
        return MiniMaxM3ModelConfig(
            transformer=transformer,
            **{
                config_field.name: getattr(self, config_field.name)
                for config_field in fields(MiniMaxM3ModelConfig)
                if config_field.init and config_field.name not in {"transformer", "pre_wrap_hooks", "post_wrap_hooks"}
            },
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config, with the Hugging Face vision config in Hugging Face's own form.

        Field-by-field serialization keeps only a Hugging Face config's declared fields. The
        checkpoint's config sets the vision tower's shape as undeclared attributes, for example
        on a generic ``PretrainedConfig`` when loaded with remote code, so it would not survive.
        """
        vision_config = self.transformer.vision_config
        self.transformer.vision_config = None
        try:
            data = super().as_dict()
        finally:
            self.transformer.vision_config = vision_config
        if isinstance(vision_config, PretrainedConfig):
            vision_config = {
                "_target_": f"{type(vision_config).__module__}.{type(vision_config).__qualname__}",
                **vision_config.to_dict(),
            }
        data["transformer"]["vision_config"] = copy.deepcopy(vision_config)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiniMaxM3VLModelConfig":
        """Deserialize a config, restoring the Hugging Face vision config with its ``from_dict()``."""
        transformer_data = dict(data["transformer"])
        vision_config = transformer_data.pop("vision_config", None)
        result = super().from_dict({**data, "transformer": transformer_data})
        if isinstance(vision_config, dict) and "_target_" in vision_config:
            vision_config = dict(vision_config)
            config_class = _resolve_target(
                vision_config.pop("_target_"), full_key="transformer.vision_config._target_"
            )
            if not isinstance(config_class, type) or not issubclass(config_class, PretrainedConfig):
                raise TypeError(f"vision_config must restore a Hugging Face config, got {config_class!r}.")
            vision_config = config_class.from_dict(vision_config)
        result.transformer.vision_config = vision_config
        return result


__all__ = ["MiniMaxM3ModelConfig", "MiniMaxM3VLModelConfig", "MiniMaxM3VLTransformerConfig"]
