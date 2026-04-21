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

from dataclasses import dataclass

from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig as MCoreHeterogeneousTransformerConfig,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig as MCoreMLATransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig


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
