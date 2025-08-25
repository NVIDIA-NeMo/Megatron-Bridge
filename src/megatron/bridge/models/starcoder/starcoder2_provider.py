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

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class Starcoder2ModelProvider(GPTModelProvider):
    """
    Model Provider class for Starcoder2, inheriting from GPTModelProvider.
    """

    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = F.gelu
    add_bias_linear: bool = True
    seq_length: int = 16384
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    init_method_std: float = 0.01
    share_embeddings_and_output_weights: bool = False
    kv_channels: int = None
    num_query_groups: int = None
    window_size: Optional[List[int]] = None
    attention_softmax_in_fp32: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    layernorm_epsilon: float = 1e-5


@dataclass
class Starcoder2ModelProvider3B(Starcoder2ModelProvider):
    """
    Model Provider for the Starcoder2 3B, inheriting from Starcoder2ModelProvider.
    """

    num_layers: int = 30
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_query_groups: int = 2
    num_attention_heads: int = 24
    init_method_std: float = 0.018042
    rotary_base: float = 999999.4420358813


@dataclass
class Starcoder2ModelProvider7B(Starcoder2ModelProvider):
    """
    Model Provider for the Starcoder2 7B, inheriting from Starcoder2ModelProvider.
    """

    num_layers: int = 32
    hidden_size: int = 4608
    ffn_hidden_size: int = 18432
    num_query_groups: int = 4
    num_attention_heads: int = 36
    init_method_std: float = 0.018042
    rotary_base: float = 1_000_000


@dataclass
class Starcoder2ModelProvider15B(Starcoder2ModelProvider):
    """
    Model Provider for the Starcoder2 15B, inheriting from Starcoder2ModelProvider.
    """

    num_layers: int = 40
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_query_groups: int = 4
    num_attention_heads: int = 48
    init_method_std: float = 0.01275
    rotary_base: float = 100_000
