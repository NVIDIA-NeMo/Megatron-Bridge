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

"""Serializable Bridge extension of Megatron-LM's GPT model config."""

import functools
import inspect
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any

from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.config_proxy import FlatTransformerConfigMixin
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable


ACTIVATION_FUNC_METADATA_KEY = "megatron_bridge.activation_func"


def _callable_target(value: object) -> str:
    """Return an importable target path for a layer-spec callable."""
    module = inspect.getmodule(value)
    qualname = getattr(value, "__qualname__", None)
    if module is None or not isinstance(qualname, str) or "<locals>" in qualname:
        raise ValueError(f"Cannot serialize non-importable transformer layer spec: {value!r}.")
    return f"{module.__name__}.{qualname}"


def _serialize_layer_spec(value: object) -> dict[str, Any]:
    """Serialize a function or partial as an allow-listed code reference."""
    if isinstance(value, functools.partial):
        result: dict[str, Any] = {
            "_target_": _callable_target(value.func),
            "_partial_": True,
            "_args_": list(value.args),
        }
        result.update(value.keywords or {})
        return result
    if inspect.isfunction(value):
        return {"_target_": _callable_target(value), "_call_": False}
    raise ValueError(
        "transformer_layer_spec must be an importable function, functools.partial, ModuleSpec, or None; "
        f"got {type(value).__name__}."
    )


def _same_layer_spec(left: object, right: object) -> bool:
    """Return whether two layer-spec callables have the same construction."""
    if left is right:
        return True
    if isinstance(left, functools.partial) and isinstance(right, functools.partial):
        return left.func is right.func and left.args == right.args and left.keywords == right.keywords
    return False


def _uses_default_layer_spec(config: object) -> bool:
    """Return whether a dataclass instance still uses its class field default."""
    config_field = next(field for field in fields(config) if field.name == "transformer_layer_spec")
    if config_field.default is not MISSING:
        default = config_field.default
    elif config_field.default_factory is not MISSING:
        default = config_field.default_factory()
    else:
        return False
    return _same_layer_spec(getattr(config, config_field.name), default)


def _drop_non_init_fields(data: dict[str, Any], config: object) -> None:
    """Remove derived dataclass fields that cannot be passed to constructors."""
    for config_field in fields(config):
        if not config_field.init:
            data.pop(config_field.name, None)
            continue
        value = getattr(config, config_field.name)
        serialized_value = data.get(config_field.name)
        if is_dataclass(value) and isinstance(serialized_value, dict):
            _drop_non_init_fields(serialized_value, value)


def restore_model_config_callables(data: dict[str, Any]) -> dict[str, Any]:
    """Restore symbolic callable fields before nested dataclass construction.

    Args:
        data: Serialized model config dictionary.

    Returns:
        A shallow copy with the activation callable restored in the serialized
        transformer config. The input dictionary is not mutated.

    Raises:
        ValueError: If activation metadata is present without a nested
            transformer dictionary or names an unsupported activation.
    """
    metadata = data.get("extra_checkpoint_metadata")
    if not isinstance(metadata, dict) or ACTIVATION_FUNC_METADATA_KEY not in metadata:
        return data

    activation_name = metadata[ACTIVATION_FUNC_METADATA_KEY]
    if not isinstance(activation_name, str):
        raise ValueError(f"{ACTIVATION_FUNC_METADATA_KEY} must be a string, got {type(activation_name).__name__}.")

    transformer = data.get("transformer")
    if not isinstance(transformer, dict):
        raise ValueError(f"{ACTIVATION_FUNC_METADATA_KEY} requires a serialized transformer config.")

    restored = dict(data)
    restored_transformer = dict(transformer)
    restored_transformer["activation_func"] = str_to_callable(activation_name)
    restored["transformer"] = restored_transformer
    return restored


@dataclass(kw_only=True)
class BridgeGPTModelConfig(FlatTransformerConfigMixin, GPTModelConfig):
    """GPTModelConfig with strict flat updates and callable round trips.

    The upstream config supplies the builder contract. Bridge keeps inherited
    outer/nested duplicate fields synchronized, rejects phantom overrides, and
    serializes activation and layer-spec callables through allow-listed targets.
    """

    def as_dict(self) -> dict[str, Any]:
        """Serialize config while preserving the activation symbol in metadata.

        Returns:
            Serialized model config with a symbolic activation name.

        Raises:
            ValueError: If the activation callable is not registered with
                Bridge's activation map.
        """
        data = super().as_dict()
        _drop_non_init_fields(data, self)
        activation_name = callable_to_str(self.transformer.activation_func)
        if activation_name is None:
            raise ValueError(
                "Cannot serialize unregistered transformer activation callable. "
                "Register a symbolic activation before saving this model config."
            )

        metadata = dict(data.get("extra_checkpoint_metadata") or {})
        metadata[ACTIVATION_FUNC_METADATA_KEY] = activation_name
        data["extra_checkpoint_metadata"] = metadata
        if callable(self.transformer_layer_spec) and not _uses_default_layer_spec(self):
            data["transformer_layer_spec"] = _serialize_layer_spec(self.transformer_layer_spec)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeGPTModelConfig":
        """Deserialize through Bridge's validated ModelConfig loader."""
        from megatron.bridge.models.common.base import ModelConfig

        result = ModelConfig.from_dict(data)
        if not isinstance(result, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(result).__name__}.")
        return result


__all__ = ["ACTIVATION_FUNC_METADATA_KEY", "BridgeGPTModelConfig", "restore_model_config_callables"]
