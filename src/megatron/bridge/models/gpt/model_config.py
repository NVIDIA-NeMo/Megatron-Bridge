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

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.config_proxy import FlatTransformerConfigMixin
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable


ACTIVATION_FUNC_METADATA_KEY = "megatron_bridge.activation_func"


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
    """GPTModelConfig that preserves registered activation functions on save."""

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
