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

import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)


@dataclass
class Phi3ModelProvider(GPTModelProvider):
    """Base model provider for Phi-3 Models."""

    normalization: str = "RMSNorm"
    activation_func = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    seq_length: int = 4096
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    layernorm_epsilon: float = 1e-5
    rotary_base: float = 10000.0
    position_embedding_type: str = "rope"


@dataclass
class Phi3ModelProviderMini(Phi3ModelProvider):
    """
    Config for Phi-3 Mini (3.8B): https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
    """

    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    num_query_groups: int = 32  # No GQA
    vocab_size: int = 32064


@dataclass
class Phi3ModelProviderMini128K(Phi3ModelProviderMini):
    """
    Config for Phi-3 Mini 128K: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
    """

    seq_length: int = 131072
    rotary_base: float = 10000.0  # Uses different RoPE scaling


@dataclass
class Phi3ModelProviderSmall(Phi3ModelProvider):
    """
    Config for Phi-3 Small (7B): https://huggingface.co/microsoft/Phi-3-small-8k-instruct
    """

    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 32  # No GQA
    vocab_size: int = 100352
    seq_length: int = 8192


@dataclass
class Phi3ModelProviderSmall128K(Phi3ModelProviderSmall):
    """
    Config for Phi-3 Small 128K: https://huggingface.co/microsoft/Phi-3-small-128k-instruct
    """

    seq_length: int = 131072


@dataclass
class Phi3ModelProviderMedium(Phi3ModelProvider):
    """
    Config for Phi-3 Medium (14B): https://huggingface.co/microsoft/Phi-3-medium-4k-instruct
    """

    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 17920
    num_attention_heads: int = 40
    num_query_groups: int = 10  # Uses GQA
    vocab_size: int = 32064
    seq_length: int = 4096


@dataclass
class Phi3ModelProviderMedium128K(Phi3ModelProviderMedium):
    """
    Config for Phi-3 Medium 128K: https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
    """

    seq_length: int = 131072


# Phi-4 models use the same Phi3ForCausalLM architecture
@dataclass
class Phi4ModelProviderMini(Phi3ModelProvider):
    """
    Config for Phi-4 Mini (4B): https://huggingface.co/microsoft/Phi-4-mini-instruct
    """

    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 24
    num_query_groups: int = 8  # Uses GQA
    vocab_size: int = 200064
    seq_length: int = 131072
    rotary_base: float = 250000.0
    layernorm_epsilon: float = 1e-5


@dataclass
class Phi4ModelProvider(Phi3ModelProvider):
    """
    Config for Phi-4 (15B): https://huggingface.co/microsoft/Phi-4
    """

    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 17920
    num_attention_heads: int = 40
    num_query_groups: int = 10  # Uses GQA
    vocab_size: int = 100352
    seq_length: int = 16384
    rotary_base: float = 250000.0
    layernorm_epsilon: float = 1e-5
