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
from typing import Callable, Optional

import torch.nn.functional as F

from megatron.hub.models.gpt import GPTConfig


@dataclass
class Qwen2Config(GPTConfig):
    """
    Base config for Qwen 2 Models
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    seq_length: int = 4096
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"


@dataclass
class Qwen2Config500M(Qwen2Config):
    """
    Config for Qwen 2 0.5B: https://huggingface.co/Qwen/Qwen2-0.5B
    """

    num_layers: int = 24
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_query_groups: int = 2
    ffn_hidden_size: int = 4864


@dataclass
class Qwen25Config500M(Qwen2Config500M):
    """
    Config for Qwen 2.5 0.5B: https://huggingface.co/Qwen/Qwen2.5-0.5B
    """

    seq_length: int = 32768


@dataclass
class Qwen2Config1P5B(Qwen2Config):
    """
    Config for Qwen 2 1.5B: https://huggingface.co/Qwen/Qwen2-1.5B
    """

    num_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_query_groups: int = 2
    ffn_hidden_size: int = 8960


@dataclass
class Qwen25Config1P5B(Qwen2Config1P5B):
    """
    Config for Qwen 2.5 1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B
    """

    seq_length: int = 131072


@dataclass
class Qwen2Config7B(Qwen2Config):
    """
    Config for Qwen 2 7B: https://huggingface.co/Qwen/Qwen2-7B
    """

    num_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_query_groups: int = 4
    ffn_hidden_size: int = 18944
    vocab_size: int = 152064


@dataclass
class Qwen25Config7B(Qwen2Config7B):
    """
    Config for Qwen 2.5 7B: https://huggingface.co/Qwen/Qwen2.5-7B
    """

    seq_length: int = 131072


@dataclass
class Qwen25Config14B(Qwen2Config):
    """
    Config for Qwen 2.5 14B: https://huggingface.co/Qwen/Qwen2.5-14B
    """

    num_layers: int = 48
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 8
    ffn_hidden_size: int = 13824
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-5
    seq_length: int = 131072


@dataclass
class Qwen25Config32B(Qwen2Config):
    """
    Config for Qwen 2.5 32B: https://huggingface.co/Qwen/Qwen2.5-32B
    """

    num_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 8
    ffn_hidden_size: int = 27648
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-5
    seq_length: int = 131072


@dataclass
class Qwen2Config72B(Qwen2Config):
    """
    Config for Qwen 2 72B: https://huggingface.co/Qwen/Qwen2-72B
    """

    num_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 29568
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-5


@dataclass
class Qwen25Config72B(Qwen2Config72B):
    """
    Config for Qwen 2.5 72B: https://huggingface.co/Qwen/Qwen2.5-72B
    """

    seq_length: int = 131072
