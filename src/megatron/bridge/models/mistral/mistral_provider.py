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
from typing import Callable

import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)


@dataclass
class MistralModelProvider(GPTModelProvider):
    """
    Base model provider for Mistral 7B Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    seq_length: int = 32768
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    window_size: list[int] = None
    rotary_base: float = 1000000.0
    params_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 32768
    bf16: bool = True


# =============================================================================
# Mistral Small 3 24B Model Providers
# =============================================================================


@dataclass
class MistralSmall3ModelProvider24B(MistralModelProvider):
    """
    Config for Mistral Small 3 24B: https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
    """

    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 32
    kv_channels: int = 128
    seq_length: int = 32768

    window_size: list[int] = None
    cp_comm_type: str = None
    rotary_percent: float = 1.0
    rotary_base: float = 100000000.0
    params_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 131072
