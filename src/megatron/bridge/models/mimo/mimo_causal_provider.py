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

import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class MiMoModelProvider7B(GPTModelProvider):
    """Base provider for Xiaomi MiMo 7B Causal LM family."""

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    qk_layernorm: bool = False
    position_embedding_type: str = "rope"

    num_layers: int = 36
    hidden_size: int = 4096
    ffn_hidden_size: int = 11008
    num_attention_heads: int = 32
    num_query_groups: int = 8
    kv_channels: int = 128

    seq_length: int = 32768
    vocab_size: int = 151680
    share_embeddings_and_output_weights: bool = False
    layernorm_epsilon: float = 1e-5
    rotary_base: float = 640000.0
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    mtp_num_layers: int = 1
    mtp_loss_scaling_factor: float = 0.1

    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
    fp16: bool = False


@dataclass
class MiMoModelProvider7BBase(MiMoModelProvider7B):
    """Provider for XiaomiMiMo/MiMo-7B-Base."""


@dataclass
class MiMoModelProvider7BSFT(MiMoModelProvider7B):
    """Provider for XiaomiMiMo/MiMo-7B-SFT."""


@dataclass
class MiMoModelProvider7BRL(MiMoModelProvider7B):
    """Provider for XiaomiMiMo/MiMo-7B-RL."""


@dataclass
class MiMoModelProvider7BRLZero(MiMoModelProvider7B):
    """Provider for XiaomiMiMo/MiMo-7B-RL-Zero."""


@dataclass
class MiMoModelProvider7BRL0530(MiMoModelProvider7B):
    """Provider for XiaomiMiMo/MiMo-7B-RL-0530."""

    seq_length: int = 65536
