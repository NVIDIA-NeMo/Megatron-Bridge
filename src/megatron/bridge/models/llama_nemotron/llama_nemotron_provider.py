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

import logging
from dataclasses import dataclass
from typing import Callable, Union

import torch

# Import heterogeneous layer spec dependencies
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import get_gpt_heterogeneous_layer_spec
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.llama.llama_provider import (
    Llama31ModelProvider8B,
    Llama31ModelProvider70B,
    Llama31ModelProvider405B,
)
from megatron.bridge.models.llama_nemotron.llama_nemotron_config import (
    LLAMA_31_NEMOTRON_ULTRA_253B_HETEROGENEOUS_CONFIG,
    LLAMA_33_NEMOTRON_SUPER_49B_HETEROGENEOUS_CONFIG,
)
from megatron.bridge.models.transformer_config import HeterogeneousTransformerConfig


logger = logging.getLogger(__name__)


def heterogeneous_layer_spec(config) -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability.

    Uses Transformer Engine specs since TE is a required dependency.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The selected module specification
    """
    return get_gpt_heterogeneous_layer_spec(config, use_te=True)


@dataclass
class Llama31NemotronNano8BProvider(Llama31ModelProvider8B):
    """
    Configuration class for the Llama3.1-Nemotron-Nano-8B model.
    Maps to: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
    Based on Llama31Config8B with kv_channels=128
    """

    kv_channels: int = 128
    # Data type settings to match HF models
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16


@dataclass
class Llama31Nemotron70BProvider(Llama31ModelProvider70B):
    """
    Configuration class for the Llama3.1-Nemotron-70B model.
    Maps to: nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
    Based on Llama31Config70B with kv_channels=128
    """

    kv_channels: int = 128
    # Data type settings to match HF models
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16


@dataclass
class Llama33NemotronSuper49BProvider(Llama31ModelProvider70B, HeterogeneousTransformerConfig):
    """
    Configuration class for the Llama3.3-Nemotron-Super-49B model.
    Maps to: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    Based on Llama31Config70B with heterogeneous architecture and kv_channels=128

    CRITICAL MRO: Llama31ModelProvider70B must come first to ensure proper
    provider functionality, then HeterogeneousTransformerConfig for heterogeneous support.
    """

    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_layers: int = 80
    kv_channels: int = 128
    # Data type settings to match HF models
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    heterogeneous_layers_config_path: str = ""
    heterogeneous_layers_config_encoded_json: str = LLAMA_33_NEMOTRON_SUPER_49B_HETEROGENEOUS_CONFIG
    transformer_layer_spec: Union[ModuleSpec, Callable] = heterogeneous_layer_spec


@dataclass
class Llama31NemotronUltra253BProvider(Llama31ModelProvider405B, HeterogeneousTransformerConfig):
    """
    Configuration class for the Llama3.1-Nemotron-Ultra-253B model.
    Maps to: nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
    Based on Llama31Config405B with heterogeneous architecture and kv_channels=128

    CRITICAL MRO: Llama31ModelProvider must come first to ensure proper
    provider functionality, then HeterogeneousTransformerConfig for heterogeneous support.
    """

    # Override base config for Ultra model specifics
    num_layers: int = 162
    hidden_size: int = 16384
    num_attention_heads: int = 128
    kv_channels: int = 128
    # Data type settings to match HF models
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    # Heterogeneous configuration fields
    heterogeneous_layers_config_path: str = ""
    heterogeneous_layers_config_encoded_json: str = LLAMA_31_NEMOTRON_ULTRA_253B_HETEROGENEOUS_CONFIG
    transformer_layer_spec: Union[ModuleSpec, Callable] = heterogeneous_layer_spec
