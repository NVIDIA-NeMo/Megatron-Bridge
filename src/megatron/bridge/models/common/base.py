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

import functools
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

from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.instantiate_utils import _resolve_target, _validate_target_prefix, instantiate


_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "pre_wrap_hooks",
        "post_wrap_hooks",
        "timers",
        "finalize_model_grads_func",
        "grad_scale_func",
        "moe_grad_scale_func",
        "mtp_grad_scale_func",
        "no_sync_func",
        "grad_sync_func",
        "param_sync_func",
    }
)


def serialize_model_config(config: _MegatronModelConfig) -> dict[str, Any]:
    """Serialize model configs while preserving importable callable fields."""

    def _serialize_value(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            result = {"_target_": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}
            for config_field in dataclass_fields(value):
                if config_field.name.startswith("_") or config_field.name in _RUNTIME_CONFIG_FIELDS:
                    continue
                result[config_field.name] = _serialize_value(getattr(value, config_field.name))
            return result
        if isinstance(value, (list, tuple)):
            return [_serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: _serialize_value(item) for key, item in value.items()}
        if isinstance(value, functools.partial):
            module = getattr(value.func, "__module__", None)
            qualname = getattr(value.func, "__qualname__", None)
            if not module or not qualname or "<locals>" in qualname:
                raise TypeError(f"Cannot serialize partial with non-importable callable {value.func!r}")
            result = {"_target_": f"{module}.{qualname}", "_partial_": True}
            if value.args:
                result["_args_"] = [_serialize_value(item) for item in value.args]
            for key, item in (value.keywords or {}).items():
                if key in {"_target_", "_partial_", "_args_"}:
                    raise TypeError(f"Cannot serialize partial with reserved keyword {key!r}")
                result[key] = _serialize_value(item)
            return result
        if callable(value):
            module = getattr(value, "__module__", None)
            qualname = getattr(value, "__qualname__", None)
            if not module or not qualname or "<locals>" in qualname:
                raise TypeError(f"Cannot serialize non-importable callable {value!r}")
            return {"_target_": f"{module}.{qualname}", "_callable_": True}
        return value

    result = _serialize_value(config)
    result["_builder_"] = config.builder
    return result


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
        if subdata.get("_callable_") is True:
            if set(subdata) != {"_target_", "_callable_"}:
                raise ValueError(f"Cannot deserialize callable target with extra fields: {sorted(subdata)}")
            return config_cls
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


class ModelConfig(_MegatronModelConfig):
    """Bridge compatibility wrapper for Megatron-LM model configs."""

    def get_builder_cls(self) -> type:
        """Get the appropriate builder type for this config."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize this config without dropping importable callable fields."""
        return serialize_model_config(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _MegatronModelConfig:
        """Deserialize config from dictionary with Bridge target validation."""
        return deserialize_model_config(data)
