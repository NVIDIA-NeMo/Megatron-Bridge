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

"""Bridge wrapper classes for Megatron Core transformer configurations.

These classes provide deferred post-initialization to support the Bridge configuration
override system while maintaining compatibility with Megatron Core's post_init behavior.
"""

import copy
from dataclasses import dataclass, fields, is_dataclass

from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig as MCoreHeterogeneousTransformerConfig,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig as MCoreMLATransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig


def _safe_asdict(obj, skip_keys: set[str]) -> dict:
    """Shallow asdict variant that preserves handles like process groups.

    dataclasses.asdict performs a deep copy of every leaf value, which breaks objects that
    should remain shared references (e.g., ProcessGroupCollection). This helper mirrors the
    structure of asdict but returns leaf objects as-is so they are not deep-copied.
    """
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = value if f.name in skip_keys else _safe_asdict(value, skip_keys)
        return result
    if isinstance(obj, (list, tuple)):
        return obj.__class__(_safe_asdict(v, skip_keys) for v in obj)
    if isinstance(obj, dict):
        return obj.__class__((_safe_asdict(k, skip_keys), _safe_asdict(v, skip_keys)) for k, v in obj.items())
    return obj


@dataclass
class TransformerConfig(MCoreTransformerConfig):
    """Megatron Core TransformerConfig with deferred post-init.

    This class inherits from Megatron Core's TransformerConfig but defers the
    execution of post_init() until finalize() is explicitly called. This allows
    for field modifications after construction but before computed fields are
    calculated.

    Usage:
        # Create config with deferred post-init
        config = TransformerConfig(num_layers=32, hidden_size=4096)

        # Modify fields as needed
        config.seq_length = 8192
        config.tensor_model_parallel_size = 2

        # Finalize to compute derived fields
        config.finalize()
    """

    _NO_COPY_KEYS = {"_pg_collection"}

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        This allows for field modifications after construction without
        invalidating computed fields.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core TransformerConfig.__post_init__()
        to compute derived fields based on the current field values. It can be
        called multiple times safely.
        """
        if self.pipeline_model_parallel_size > 1 and self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        MCoreTransformerConfig.__post_init__(self)

    def __deepcopy__(self, memo):
        """Custom deepcopy to preserve process group handles when cloning configs.

        Certain attributes (_pg_collection, etc.) should remain shared references
        rather than being wiped or re-created during deepcopy.
        TODO: This is a temporary hack. Once providers stop embedding the Transformer
        config and instead hold the MCore config as an attribute, we can remove this.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key in self._NO_COPY_KEYS:
                # Keep the same reference to avoid losing initialized process groups.
                setattr(result, key, value)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result

    def asdict(self) -> dict:
        """Return a dict view without deep-copying shared handles (e.g., process groups)."""
        return _safe_asdict(self, self._NO_COPY_KEYS)


@dataclass
class MLATransformerConfig(TransformerConfig, MCoreMLATransformerConfig):
    """Megatron Core MLATransformerConfig with deferred post-init.

    This class inherits from Megatron Core's MLATransformerConfig but defers the
    execution of post_init() until finalize() is explicitly called. This allows
    for field modifications after construction but before computed fields are
    calculated.

    Usage:
        # Create config with deferred post-init
        config = MLATransformerConfig(num_layers=32, hidden_size=4096)

        # Modify fields as needed
        config.q_lora_rank = 1536
        config.kv_lora_rank = 512

        # Finalize to compute derived fields
        config.finalize()
    """

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        This allows for field modifications after construction without
        invalidating computed fields.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core MLATransformerConfig.__post_init__()
        to compute derived fields based on the current field values. It can be
        called multiple times safely.
        """
        if self.pipeline_model_parallel_size > 1 and self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        MCoreMLATransformerConfig.__post_init__(self)


@dataclass
class HeterogeneousTransformerConfig(TransformerConfig, MCoreHeterogeneousTransformerConfig):
    """Megatron Core HeterogeneousTransformerConfig with deferred post-init.

    This class inherits from both our lazy TransformerConfig and Megatron Core's
    HeterogeneousTransformerConfig. The MRO ensures that our lazy post-init behavior
    is preserved while maintaining all heterogeneous functionality.

    CRITICAL: The inheritance order is important for MRO:
    1. TransformerConfig (our lazy version) comes first
    2. MCoreHeterogeneousTransformerConfig comes second

    Usage:
        # Create config with deferred post-init
        config = HeterogeneousTransformerConfig(
            num_layers=32,
            hidden_size=4096,
            heterogeneous_layers_config_encoded_json=json_string
        )

        # Modify fields as needed
        config.seq_length = 8192
        config.tensor_model_parallel_size = 2

        # Finalize to compute derived fields and parse heterogeneous config
        config.finalize()
    """

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        This allows for field modifications after construction without
        invalidating computed fields.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core HeterogeneousTransformerConfig.__post_init__()
        to compute derived fields and parse the heterogeneous block configurations.
        It can be called multiple times safely.
        """
        MCoreHeterogeneousTransformerConfig.__post_init__(self)

    def get_config_for_layer(self, layer_number: int) -> MCoreTransformerConfig:
        """Return a layer-specific TransformerConfig without deep-copying process groups."""
        # TODO: This is a temporary hack; replace once providers hold the MCore config directly.

        layer_idx = layer_number - 1  # layer number starts from 1
        if layer_idx < 0 or layer_idx >= len(self.per_block_parameters):
            raise ValueError(
                f"Invalid layer number: {layer_number}. Should be in range [1, {len(self.per_block_parameters)}]."
            )
        block_config = self.per_block_parameters[layer_idx]

        keys_to_update = {}

        # attention config updates
        if block_config.attention.num_query_groups is not None:
            assert not block_config.attention.replace_with_linear and not block_config.attention.no_op
            keys_to_update["num_query_groups"] = block_config.attention.num_query_groups

        # mlp config updates
        if block_config.mlp.ffn_hidden_size is not None:
            assert not block_config.mlp.replace_with_linear and not block_config.mlp.no_op
            keys_to_update["ffn_hidden_size"] = block_config.mlp.ffn_hidden_size

        transformer_config_dict = self.asdict()

        # remove keys that are not in TransformerConfig
        transformer_config_field_names = {f.name for f in fields(MCoreTransformerConfig)}
        transformer_config_dict = {
            k: v for k, v in transformer_config_dict.items() if k in transformer_config_field_names
        }

        transformer_config_dict.update(keys_to_update)

        return MCoreTransformerConfig(**transformer_config_dict)
