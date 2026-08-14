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

"""Native HybridModel configuration objects for Muse Glimmer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.training.models.hybrid import HybridModelConfig

from megatron.bridge.models.common import ModelConfigOverrideMixin, deserialize_model_config
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable
from megatron.bridge.utils.instantiate_utils import _resolve_target


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

    @property
    def depth(self) -> int:
        """Expose the vision depth expected by Bridge's VLM FLOPs estimator."""
        return self.num_hidden_layers

    @property
    def spatial_merge_size(self) -> int:
        """Expose the spatial merge factor expected by VLM data utilities."""
        return self.merge_size


@dataclass(kw_only=True)
class MuseGlimmerModelConfig(ModelConfigOverrideMixin, HybridModelConfig):
    """Complete builder configuration for Muse Glimmer."""

    builder: ClassVar[str] = "megatron.bridge.models.muse_glimmer.MuseGlimmerModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = MuseGlimmerTransformerConfig
    hybrid_attention_layers_include_mlp: ClassVar[bool] = True

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
    recompute_vision_layers: bool = False

    @property
    def vision_config(self) -> MuseGlimmerVisionModelConfig:
        """Expose the vision config to shared VLM training utilities."""
        return self.vision

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return media token IDs used by multimodal data pipelines."""
        return {"images": self.image_token_id, "videos": self.video_token_id}

    def get_builder_cls(self) -> type:
        """Resolve the Muse builder through Bridge's target allowlist."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize the Hybrid config with a symbolic activation function."""
        data = super().as_dict()
        transformer_data = data.get("transformer")
        if not isinstance(transformer_data, dict):
            raise TypeError("Serialized Muse Glimmer config must contain a transformer mapping.")

        activation_func = self.transformer.activation_func
        if isinstance(activation_func, str):
            str_to_callable(activation_func)
            activation_name = activation_func
        else:
            activation_name = callable_to_str(activation_func)
        if activation_name is None:
            raise ValueError(f"Cannot serialize unregistered activation callable: {activation_func!r}.")

        transformer_data["activation_func"] = activation_name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MuseGlimmerModelConfig":
        """Deserialize a Muse Hybrid config and restore its activation callable."""
        restored_data = dict(data)
        transformer_data = restored_data.get("transformer")
        if isinstance(transformer_data, dict):
            restored_transformer = dict(transformer_data)
            activation_name = restored_transformer.get("activation_func")
            if isinstance(activation_name, str):
                restored_transformer["activation_func"] = str_to_callable(activation_name)
            restored_data["transformer"] = restored_transformer

        result = deserialize_model_config(restored_data)
        if not isinstance(result, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(result).__name__}.")
        return result


__all__ = [
    "MuseGlimmerModelConfig",
    "MuseGlimmerTransformerConfig",
    "MuseGlimmerVisionModelConfig",
]
