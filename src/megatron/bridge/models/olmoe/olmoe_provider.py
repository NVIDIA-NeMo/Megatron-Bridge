# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Deprecated OLMoE provider compatibility API."""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.gpt_provider import GPTModelProvider, default_layer_spec
from megatron.bridge.models.olmoe.modeling_olmoe import OLMoESelfAttention


def olmoe_layer_spec(config: GPTModelProvider) -> ModuleSpec:
    """Layer spec for legacy OLMoE providers."""
    layer_spec = default_layer_spec(config)
    layer_spec.submodules.self_attention.module = OLMoESelfAttention
    return layer_spec


@dataclass
class OlMoEModelProvider(GPTModelProvider):
    """Legacy provider for OLMoE models."""

    transformer_layer_spec: Union[ModuleSpec, Callable[[GPTModelProvider], ModuleSpec]] = olmoe_layer_spec
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    seq_length: int = 4096
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    vocab_size: int = 50304
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-5
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.float32
    bf16: bool = False
    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 1024
    moe_ffn_hidden_size: int = 1024
    kv_channels: int = 2048 // 16
    num_query_groups: int = 16
    num_attention_heads: int = 16
    attention_dropout: float = 0.0
    qk_layernorm: bool = True
    position_embedding_type: str = "rope"
    rotary_base: float = 10000.0
    num_moe_experts: int = 64
    moe_router_topk: int = 8
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_aux_loss_coeff: float = 1e-2
    moe_router_pre_softmax: bool = True
    moe_grouped_gemm: bool = True
    moe_router_score_function: str = "softmax"
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    persist_layer_norm: bool = True


__all__ = ["OLMoESelfAttention", "OlMoEModelProvider", "olmoe_layer_spec"]
