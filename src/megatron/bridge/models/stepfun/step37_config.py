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

"""Builder-backed model config for Step-3.7-Flash.

The language half is the Step-3.5 decoder, configured with the fields of
``Step35ModelConfig``. The vision-language settings live on the outer config;
``Step37ModelBuilder`` combines the decoder with the Perception-Encoder vision
tower.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from megatron.bridge.models.stepfun.step35_config import Step35ModelConfig


@dataclass(kw_only=True)
class Step37ModelConfig(Step35ModelConfig):
    """Builder config for Step-3.7-Flash.

    ``vision_config`` holds the Hugging Face vision config, as on the provider. The
    token ID, projector and freeze settings are the provider fields of the same names.
    """

    builder: ClassVar[str] = "megatron.bridge.models.stepfun.step37_builder.Step37ModelBuilder"

    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "yarn", "none"] = "rope"

    vision_config: Any = None
    """Hugging Face Perception-Encoder config (``Step37VisionConfig`` or the checkpoint's own class)."""

    image_token_id: int = 128001
    understand_projector_stride: int = 2
    projector_bias: bool = False
    language_max_sequence_length: int = 262144

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config, with the Hugging Face vision config in the form provider checkpoints use.

        Megatron Core serializes a nested dataclass field by field. Hugging Face configs are
        dataclasses, but the Step-3.7 vision configs set their shape attributes in ``__init__``
        without declaring them as fields, so that form would drop them.
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
    def from_dict(cls, data: dict[str, Any]) -> "Step37ModelConfig":
        """Deserialize a config, rebuilding the vision config from every saved value as provider reloads do."""
        from megatron.bridge.utils.instantiate_utils import instantiate

        restored_data = dict(data)
        vision_data = restored_data.pop("vision_config", None)
        result = super().from_dict(restored_data)
        result.vision_config = instantiate(vision_data) if isinstance(vision_data, dict) else vision_data
        return result


__all__ = ["Step37ModelConfig"]
