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


G_LOGGER = logging.getLogger(__name__)


@dataclass
class MixtralModelProvider(GPTModelProvider):
    """
    Model provider for Mixtral MoE models.

    Mixtral is a sparse mixture-of-experts (MoE) model that uses 8 experts
    per token with a top-2 routing strategy.

    Example:
        Mixtral 8x7B: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    add_qkv_bias: bool = False

    # Model architecture
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    seq_length: int = 32768

    # Attention
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    qk_layernorm: bool = False

    # RoPE
    rotary_base: float = 1000000.0
    rotary_percent: float = 1.0

    # Attention - explicitly set kv_channels to ensure it's computed
    kv_channels: int | None = None

    # Embedding
    share_embeddings_and_output_weights: bool = False
    vocab_size: int = 32000

    # MoE specific parameters
    num_moe_experts: int = 8
    moe_router_topk: int = 2
    moe_ffn_hidden_size: int = 14336
    moe_aux_loss_coeff: float = 0.01
    moe_router_pre_softmax: bool = True
    moe_router_load_balancing_type: str = "none"
    moe_router_score_function: str = "softmax"
    moe_shared_expert_intermediate_size: int | None = None
    moe_shared_expert_overlap: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True

    # Optimization
    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
