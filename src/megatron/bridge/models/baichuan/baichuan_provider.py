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
class BaichuanModelProvider(GPTModelProvider):
    """Base model provider for Baichuan Models."""

    normalization: str = "RMSNorm"
    activation_func = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    seq_length: int = 4096
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    layernorm_epsilon: float = 1e-6
    position_embedding_type: str = "rope"
    rotary_base: float = 10000.0


@dataclass
class Baichuan2ModelProvider7B(BaichuanModelProvider):
    """
    Config for Baichuan2 7B: https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
    """

    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 11008
    num_attention_heads: int = 32
    num_query_groups: int = 32  # No GQA in 7B model
    vocab_size: int = 125696


@dataclass
class Baichuan2ModelProvider13B(BaichuanModelProvider):
    """
    Config for Baichuan2 13B: https://huggingface.co/baichuan-inc/Baichuan2-13B-Base

    Note: 13B model uses ALiBi position embeddings instead of RoPE
    """

    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 13696
    num_attention_heads: int = 40
    num_query_groups: int = 40  # No GQA
    vocab_size: int = 125696
    position_embedding_type: str = "alibi"  # 13B uses ALiBi
