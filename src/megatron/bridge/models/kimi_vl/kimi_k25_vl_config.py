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

"""Builder-backed model config for Kimi K2.5 VL.

The language half is Kimi K2's MoE decoder with Multi-Latent Attention, configured with the fields of
``BridgeGPTMLAModelConfig``. The vision-language settings, the provider fields of the same names, live
on the outer config. ``KimiK25VLModelBuilder`` combines the decoder with the MoonViT vision tower and
the projector, which ``KimiK25VLModel`` loads from the checkpoint's own code.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from transformers import PretrainedConfig

from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig
from megatron.bridge.utils.instantiate_utils import _resolve_target


@dataclass(kw_only=True)
class KimiK25VLModelConfig(BridgeGPTMLAModelConfig):
    """Builder config for Kimi K2.5 VL, the builder counterpart of ``KimiK25VLModelProvider``.

    ``KimiK25VLModel`` merges image features into the language embeddings before it scatters them
    for sequence parallelism, so the language model's embedding does not scatter them.
    """

    builder: ClassVar[str] = "megatron.bridge.models.kimi_vl.kimi_k25_vl_builder.KimiK25VLModelBuilder"

    scatter_embedding_sequence_parallel: bool = False

    vision_config: Any = None
    """Hugging Face ``KimiK25VisionConfig``. The model reloads it from ``hf_model_path`` when it builds
    the vision tower."""

    hf_model_path: str | None = None
    """Hugging Face checkpoint whose code defines the vision tower and the projector."""

    trust_remote_code: bool = False
    """Whether the model may load the vision tower and the projector from the code in ``hf_model_path``."""

    bos_token_id: int = 163584
    eos_token_id: int = 163585
    image_token_id: int = 163605
    media_placeholder_token_id: int = 163605
    pad_token_id: int = 163839
    ignore_index: int = -100

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config, with the Hugging Face vision config in the form provider checkpoints use.

        Field-by-field serialization keeps only the declared fields of a Hugging Face config, and the
        remote ``KimiK25VisionConfig`` sets the vision tower's shape as undeclared attributes.
        """
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

        vision_config = self.vision_config
        self.vision_config = None
        try:
            data = super().as_dict()
        finally:
            self.vision_config = vision_config
        data["vision_config"] = _ConfigContainerBase._convert_value_to_dict(vision_config)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KimiK25VLModelConfig":
        """Deserialize a config, restoring the Hugging Face vision config with its own ``from_dict()``.

        Raises:
            TypeError: If the serialized vision config does not name a Hugging Face config class.
        """
        restored_data = dict(data)
        vision_config = restored_data.pop("vision_config", None)
        result = super().from_dict(restored_data)
        if isinstance(vision_config, dict) and "_target_" in vision_config:
            vision_config = dict(vision_config)
            config_class = _resolve_target(vision_config.pop("_target_"), full_key="vision_config._target_")
            if not isinstance(config_class, type) or not issubclass(config_class, PretrainedConfig):
                raise TypeError(f"vision_config must restore a Hugging Face config, got {config_class!r}.")
            vision_config = config_class.from_dict(vision_config)
        result.vision_config = vision_config
        return result


__all__ = ["KimiK25VLModelConfig"]
