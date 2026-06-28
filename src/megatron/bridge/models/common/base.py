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

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any

from megatron.training.models.base import (
    BuildConfigT,  # noqa: F401
    ModelBuilder,  # noqa: F401
    ModelT,  # noqa: F401
    Serializable,  # noqa: F401
    compose_hooks,  # noqa: F401
)
from megatron.training.models.base import (
    ModelConfig as _MegatronModelConfig,
)

from megatron.bridge.utils.instantiate_utils import _resolve_target, _validate_target_prefix


class ModelConfig(_MegatronModelConfig):
    """Bridge compatibility wrapper for Megatron-LM model configs."""

    def get_builder_cls(self) -> type:
        """Get the appropriate builder type for this config."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Deserialize config from dictionary with Bridge target validation."""

        def _from_dict(subdata: dict[str, Any], full_key: str) -> Any:
            target = subdata.get("_target_")
            if target is None:
                raise ValueError("Cannot deserialize: missing '_target_' field")
            if not isinstance(target, str):
                raise ValueError(f"Cannot deserialize: '_target_' must be a string, got {type(target).__name__}")

            config_cls = _resolve_target(target, full_key=full_key)
            if not isinstance(config_cls, type) or not is_dataclass(config_cls):
                raise ValueError(f"Cannot deserialize: target '{target}' did not resolve to a dataclass type")

            valid_fields = {f.name for f in dataclass_fields(config_cls)}
            filtered_data = {k: v for k, v in subdata.items() if k in valid_fields and not k.startswith("_")}

            subconfigs = {}
            for k, v in filtered_data.items():
                if isinstance(v, dict) and "_target_" in v:
                    subconfigs[k] = _from_dict(v, full_key=f"{full_key}.{k}")
            filtered_data.update(subconfigs)

            return config_cls(**filtered_data)

        builder = data.get("_builder_")
        if not isinstance(builder, str):
            raise ValueError("Cannot deserialize: missing '_builder_' field")
        _validate_target_prefix(target=builder, full_key="_builder_")

        result = _from_dict(data, full_key="_target_")
        result.builder = builder

        return result
