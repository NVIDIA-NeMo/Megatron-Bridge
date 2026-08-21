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

from dataclasses import dataclass
from typing import Any

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable
from megatron.bridge.utils.instantiate_utils import _resolve_target


@dataclass(kw_only=True)
class BridgeGPTModelConfig(GPTModelConfig):
    """GPT model config with strict overrides and callable serialization.

    Outer GPT build fields and nested transformer fields keep one declared
    owner. Flat assignment remains convenient, but unknown names fail instead
    of silently creating phantom configuration.

    Hugging Face source provenance (model id and revision) is not declared as a
    dedicated field here. It is recorded in the inherited serializable
    ``extra_checkpoint_metadata`` mapping so it round-trips through
    ``run_config.yaml`` without adding model-specific config fields.
    """

    def __setattr__(self, name: str, value: Any, /) -> None:
        """Assign a declared outer or nested field and reject phantom fields."""
        try:
            transformer = object.__getattribute__(self, "transformer")
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        model_fields = getattr(type(self), "__dataclass_fields__", {})
        transformer_fields = getattr(type(transformer), "__dataclass_fields__", {})

        if name == "transformer":
            object.__setattr__(self, name, value)
        elif name in model_fields:
            object.__setattr__(self, name, value)
        elif name in transformer_fields:
            setattr(transformer, name, value)
        elif name == "builder" or name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Neither {type(self).__name__} nor {type(transformer).__name__} declares a field named {name!r}."
            )

    def get_builder_cls(self) -> type:
        """Resolve the configured builder through Bridge's target allowlist."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config with a symbolic activation function."""
        data = super().as_dict()
        transformer_data = data.get("transformer")
        if not isinstance(transformer_data, dict):
            raise TypeError("Serialized GPT model config must contain a transformer mapping.")

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
    def from_dict(cls, data: dict[str, Any]) -> "BridgeGPTModelConfig":
        """Deserialize a config while restoring its activation callable."""
        restored_data = dict(data)
        transformer_data = restored_data.get("transformer")
        if isinstance(transformer_data, dict):
            restored_transformer = dict(transformer_data)
            activation_name = restored_transformer.get("activation_func")
            if isinstance(activation_name, str):
                restored_transformer["activation_func"] = str_to_callable(activation_name)
            restored_data["transformer"] = restored_transformer

        result = ModelConfig.from_dict(restored_data)
        if not isinstance(result, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(result).__name__}.")
        return result


__all__ = ["BridgeGPTModelConfig"]
