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
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch.nn.functional as F
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

logger = logging.getLogger(__name__)

try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec

if HAVE_TE:
    from megatron.core.utils import is_te_min_version


@dataclass
class SarvamMoEModelProvider(GPTModelProvider):

    transformer_layer_spec: Union[
        "ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]
    ] = partial(
        get_gpt_decoder_block_spec,
        use_transformer_engine=HAVE_TE,
        normalization="RMSNorm",
        vp_stage=None,
    )

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 128
    add_qkv_bias: bool = False
    qk_layernorm: bool = True

    init_method_std: float = 0.006
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6

    moe_aux_loss_coeff: float = 0
    moe_router_pre_softmax: bool = True
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    moe_grouped_gemm: bool = True
    moe_permute_fusion: bool = True
    moe_router_topk_scaling_factor: float = 2.5
    moe_z_loss_coeff: float = 3.5e-5
    moe_shared_expert_overlap: bool = False
    moe_router_dtype: Optional[str] = "fp32"
    moe_router_score_function: str = "sigmoid"
    moe_token_dispatcher_type: str = "alltoall"

    attention_softmax_in_fp32: bool = True
    persist_layer_norm: bool = True

    cross_entropy_fusion_impl: str = "te"
    cp_comm_type: str = "p2p"
    recompute_granularity: str = "selective"
    recompute_modules: List[str] = field(
        default_factory=lambda: ["layernorm", "shared_experts", "moe", "mlp", "moe_act"]
    )

    # Configured through hf config

    kv_channels: Optional[int] = 64
    seq_length: int = 131072
    rotary_base: float = 8_000_000.0
    vocab_size: int = 262144
    num_moe_experts: int = 128
    moe_router_topk: int = 6
    num_layers: int = 19
    hidden_size: int = 4096
    num_attention_heads: int = 64
    ffn_hidden_size: int = 8192
    moe_ffn_hidden_size: int = 1024
    moe_shared_expert_intermediate_size: int = 1024
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] + [1] * 18
    )
    bf16: bool = True

    # GQA
    num_query_groups: int = 4


@dataclass
class SarvamMLAModelProvider(MLATransformerConfig, GPTModelProvider):

    transformer_layer_spec: Union[
        "ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]
    ] = partial(
        get_gpt_decoder_block_spec,
        use_transformer_engine=HAVE_TE,
        normalization="RMSNorm",
        vp_stage=None,
    )

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 128
    add_qkv_bias: bool = False
    qk_layernorm: bool = True

    init_method_std: float = 0.006
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6

    moe_aux_loss_coeff: float = 0
    moe_router_pre_softmax: bool = True
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    moe_grouped_gemm: bool = True
    moe_permute_fusion: bool = True
    moe_router_topk_scaling_factor: float = 2.5
    moe_z_loss_coeff: float = 3.5e-5
    moe_shared_expert_overlap: bool = False
    moe_router_dtype: Optional[str] = "fp32"
    moe_router_score_function: str = "sigmoid"
    moe_token_dispatcher_type: str = "alltoall"

    attention_softmax_in_fp32: bool = True
    persist_layer_norm: bool = True

    cross_entropy_fusion_impl: str = "te"
    cp_comm_type: str = "p2p"
    recompute_granularity: str = "selective"
    recompute_modules: List[str] = field(
        default_factory=lambda: ["layernorm", "shared_experts", "moe", "mlp", "moe_act"]
    )

    multi_latent_attention: bool = True

    # Configured through hf config
    
    kv_channels: Optional[int] = 64
    seq_length: int = 131072
    rotary_base: float = 8_000_000.0
    vocab_size: int = 262144
    num_moe_experts: int = 128
    moe_router_topk: int = 8
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 64
    ffn_hidden_size: int = 16384
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] + [1] * 31
    )
    bf16: bool = True

    # MLA
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 40
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    
