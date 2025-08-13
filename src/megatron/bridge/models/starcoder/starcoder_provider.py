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
from typing import Callable

import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class StarcoderModelProvider(GPTModelProvider):
    """
    Model Provider class for Starcoder, inheriting from GPTModelProvider.
    """

    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = F.gelu
    add_bias_linear: bool = True
    seq_length: int = 8192
    position_embedding_type: str = "learned_absolute"
    hidden_dropout: float = 0.2
    attention_dropout: float = 0.2
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-5
    share_embeddings_and_output_weights: bool = False
    kv_channels: int = None
    num_query_groups: int = 1
    attention_softmax_in_fp32: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class StarcoderConfig15B(StarcoderModelProvider):
    """
    Model Provider for the Starcoder 15B, inheriting from StarcoderModelProvider.
    """

    num_layers: int = 40
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    init_method_std: float = 0.02
