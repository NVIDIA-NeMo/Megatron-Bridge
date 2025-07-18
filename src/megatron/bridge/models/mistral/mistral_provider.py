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
from typing import Optional, Tuple

import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)


@dataclass
class MistralModelProvider(GPTModelProvider):
    """Base model provider for Mistral Models."""

    normalization: str = "RMSNorm"
    activation_func = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    seq_length: int = 32768
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    layernorm_epsilon: float = 1e-5
    rotary_base: float = 10000.0
    position_embedding_type: str = "rope"


@dataclass
class MistralModelProvider7B(MistralModelProvider):
    """
    Config for Mistral 7B: https://huggingface.co/mistralai/Mistral-7B-v0.1
    """

    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8  # GQA
    vocab_size: int = 32000
    # Sliding window attention
    window_size: Optional[Tuple[int, int]] = (4096, 0)


@dataclass
class MistralModelProvider7BInstruct(MistralModelProvider7B):
    """
    Config for Mistral 7B Instruct: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    """

    seq_length: int = 32768


@dataclass
class MistralNemoModelProvider(MistralModelProvider):
    """
    Config for Mistral Nemo (12B): https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
    """

    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8  # GQA
    kv_channels: int = 128
    vocab_size: int = 131072  # Larger vocab
    seq_length: int = 128000  # Very long context
    rotary_base: float = 1000000.0  # Different rope theta
    # No sliding window for Nemo
    window_size: Optional[Tuple[int, int]] = None


@dataclass
class MistralLargeModelProvider(MistralModelProvider):
    """
    Config for Mistral Large (123B): https://huggingface.co/mistralai/Mistral-Large-Instruct-2407
    """

    num_layers: int = 88
    hidden_size: int = 12288
    ffn_hidden_size: int = 28672
    num_attention_heads: int = 96
    num_query_groups: int = 8  # GQA
    kv_channels: int = 128
    vocab_size: int = 32768
    seq_length: int = 131072
    rotary_base: float = 1000000.0
    # No sliding window for Large
    window_size: Optional[Tuple[int, int]] = None
