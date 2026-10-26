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
from typing import Any, ClassVar

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

from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.bridge.utils.instantiate_utils import _resolve_target, _validate_target_prefix, instantiate


def deserialize_model_config(data: dict[str, Any]) -> _MegatronModelConfig:
    """Deserialize a builder config, including persisted enum and dtype values."""

    def _restore_value(value: Any, full_key: str) -> Any:
        if isinstance(value, list):
            return [_restore_value(item, f"{full_key}.{index}") for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value
        if "_target_" not in value:
            return {key: _restore_value(item, f"{full_key}.{key}") for key, item in value.items()}
        return _from_dict(value, full_key)

    def _from_dict(subdata: dict[str, Any], full_key: str) -> Any:
        target = subdata.get("_target_")
        if target is None:
            raise ValueError("Cannot deserialize: missing '_target_' field")
        if not isinstance(target, str):
            raise ValueError(f"Cannot deserialize: '_target_' must be a string, got {type(target).__name__}")

        config_cls = _resolve_target(target, full_key=full_key, check_callable=False)
        if not isinstance(config_cls, type) or not is_dataclass(config_cls):
            return instantiate(subdata)

        valid_fields = {f.name for f in dataclass_fields(config_cls) if f.init}
        filtered_data = {
            key: _restore_value(value, f"{full_key}.{key}")
            for key, value in subdata.items()
            if key in valid_fields and not key.startswith("_")
        }
        return config_cls(**filtered_data)

    builder = data.get("_builder_")
    if not isinstance(builder, str):
        raise ValueError("Cannot deserialize: missing '_builder_' field")
    _validate_target_prefix(target=builder, full_key="_builder_")

    result = _from_dict(data, full_key="_target_")
    if not isinstance(result, _MegatronModelConfig):
        raise ValueError(f"Cannot deserialize: outer target produced {type(result).__name__}, not ModelConfig")
    result.builder = builder
    return result


class ModelConfigOverrideMixin:
    """Route flat overrides to their declared owner and reject unknown fields.

    Builder-backed model configs store Megatron-Core transformer settings in a
    nested ``transformer`` dataclass while exposing flat assignment for recipe
    compatibility. This mixin keeps that convenience without allowing typos to
    create phantom configuration attributes. Subclasses declare the nested
    config type through ``transformer_config_class``.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]]

    def __setattr__(self, name: str, value: Any, /) -> None:
        """Assign a declared outer or nested field and reject phantom fields."""
        try:
            transformer = object.__getattribute__(self, "transformer")
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        model_fields = getattr(type(self), "__dataclass_fields__", {})
        transformer_fields = getattr(type(transformer), "__dataclass_fields__", {})
        descriptor = getattr(type(self), name, None)

        if name == "transformer" or name in model_fields:
            object.__setattr__(self, name, value)
        elif name in transformer_fields:
            setattr(transformer, name, value)
        elif name == "builder" or name.startswith("_") or hasattr(descriptor, "__set__"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Neither {type(self).__name__} nor {type(transformer).__name__} declares a field named {name!r}."
            )


class MLAModelConfigMixin:
    """Nest an MLA transformer config, which owns the rotary settings that MLA attention applies.

    Megatron Core's MLA transformer config declares ``rotary_base`` and
    ``rotary_percent``, as do the GPT and Hybrid model configs. MLA attention
    reads the nested values, so the outer fields here read and write them
    instead of keeping a second copy that could disagree. Construction assigns
    the outer values, so pass them to the model config rather than only to its
    transformer config. List this mixin before the model config class it
    extends.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]] = MLATransformerConfig

    @property
    def rotary_base(self) -> float:
        """Rotary base of the nested MLA transformer config."""
        return self.transformer.rotary_base

    @rotary_base.setter
    def rotary_base(self, value: float) -> None:
        self.transformer.rotary_base = value

    @property
    def rotary_percent(self) -> float:
        """Rotary percent of the nested MLA transformer config."""
        return self.transformer.rotary_percent

    @rotary_percent.setter
    def rotary_percent(self, value: float) -> None:
        self.transformer.rotary_percent = value


class ModelConfig(_MegatronModelConfig):
    """Bridge compatibility wrapper for Megatron-LM model configs."""

    def get_builder_cls(self) -> type:
        """Get the appropriate builder type for this config."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _MegatronModelConfig:
        """Deserialize config from dictionary with Bridge target validation."""
        return deserialize_model_config(data)
