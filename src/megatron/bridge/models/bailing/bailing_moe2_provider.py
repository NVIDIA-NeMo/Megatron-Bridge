# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

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
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from megatron.bridge.models.gpt_provider import GPTModelProvider


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False
if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec


logger = logging.getLogger(__name__)


@dataclass
class BailingMoeV2ModelProvider(GPTModelProvider):
    """
    Base model provider for Bailing MoE V2 Model: https://huggingface.co/inclusionAI/Ling-mini-2.0/
    """

    transformer_layer_spec: Union["ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]] = partial(
        get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
    )

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    qk_layernorm: bool = True
    seq_length: int = 4096
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    vocab_size: int = 157184
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-6
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
    position_embedding_type: str = "rope"
    rotary_base: float = 10000
    rotary_percent: float = 0.5

    # Attention
    attention_dropout: float = 0.0
    kv_channels: int = 128

    # MoE specific parameters
    num_moe_experts: int = 256
    moe_router_topk: int = 8
    # moe_token_dispatcher_type: str = "flex"
    moe_router_load_balancing_type: str = "none"
    moe_router_score_function: str = "sigmoid"
    moe_router_pre_softmax: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_num_groups: int = 8
    moe_router_group_topk: int = 4
    moe_router_enable_expert_bias: bool = True
    moe_grouped_gemm: bool = True
    moe_router_topk_scaling_factor: float = 2.5
    moe_layer_freq: Union[int, list[int]] = field(default_factory=lambda: [1])  # Default: all MoE layers


@dataclass
class LingMini2ModelProvider(BailingMoeV2ModelProvider):
    """
    Config for Ling Mini 2.0 (16B activate 1.4B)
    """

    num_layers: int = 20
    hidden_size: int = 2048
    ffn_hidden_size: int = 5120
    num_attention_heads: int = 16
    num_query_groups: int = 4

    # MoE
    moe_layer_freq: Union[int, list[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 19
    )  # first layer is dense only
    moe_ffn_hidden_size: int = 512
    moe_shared_expert_intermediate_size: int = 512

    # MTP
    mtp_num_layers: Optional[int] = 1


@dataclass
class LingFlash2ModelProvider(BailingMoeV2ModelProvider):
    """
    Config for Ling Flash 2.0 (100B activate 4.8B)
    """

    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 9216
    num_attention_heads: int = 32
    num_query_groups: int = 4

    # MoE
    moe_layer_freq: Union[int, list[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 31
    )  # first layer is dense only
    moe_ffn_hidden_size: int = 1024
    moe_shared_expert_intermediate_size: int = 1024

    # MTP
    mtp_num_layers: Optional[int] = 1


@dataclass
class Ling1TModelProvider(BailingMoeV2ModelProvider):
    """
    Config for Ling 1T (1TB activate 50B)
    """

    num_layers: int = 80
    hidden_size: int = 8192
    ffn_hidden_size: int = 18432
    num_attention_heads: int = 64
    num_query_groups: int = 8

    # MoE
    moe_layer_freq: Union[int, list[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 79
    )  # first layer is dense only
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048

    # MTP
    mtp_num_layers: Optional[int] = 0
