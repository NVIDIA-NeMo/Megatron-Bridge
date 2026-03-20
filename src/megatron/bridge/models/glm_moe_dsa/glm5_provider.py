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

"""GLM5 MoE+DSA Model Provider.

GLM-5 is a Mixture-of-Experts model with Multi-Latent Attention (MLA) and
Dynamic Sparse Attention (DSA) indexer layers.

Reference: https://huggingface.co/zai-org/GLM-5
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec

try:
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    _DEFAULT_LAYER_SPEC = get_transformer_block_with_experimental_attention_variant_spec
except (ImportError, ModuleNotFoundError):
    _DEFAULT_LAYER_SPEC = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)


@dataclass
class GLM5ModelProvider(MLATransformerConfig, GPTModelProvider):
    """
    Model provider for GLM-5 (MoE + MLA + DSA).

    GLM-5 uses:
    - Multi-Latent Attention (MLA) for efficient KV compression
    - Dynamic Sparse Attention (DSA) indexer for sparse attention patterns
    - Mixture-of-Experts (MoE) feed-forward layers
    - Multi-Token Prediction (MTP) auxiliary heads

    Reference: https://huggingface.co/zai-org/GLM-5
    """

    transformer_layer_spec: Union["ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]] = _DEFAULT_LAYER_SPEC

    # Model dimensions
    num_layers: int = 40
    hidden_size: int = 4096
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    kv_channels: int = 128
    vocab_size: int = 154880

    # Normalization & activation
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    share_embeddings_and_output_weights: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # RoPE
    position_embedding_type: str = "rope"
    rotary_base: float = 1000000.0
    rotary_scaling_factor: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # MLA (Multi-Latent Attention)
    multi_latent_attention: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    qk_layernorm: bool = False

    # DSA (Dynamic Sparse Attention)
    experimental_attention_variant: str = "dsa"
    dsa_indexer_head_dim: Optional[int] = None
    dsa_indexer_n_heads: Optional[int] = None
    dsa_indexer_topk: Optional[int] = None
    dsa_indexer_loss_coeff: float = 0.001
    dsa_indexer_use_sparse_loss: bool = True

    # MoE
    num_moe_experts: int = 64
    moe_ffn_hidden_size: int = 1024
    moe_shared_expert_intermediate_size: int = 1024
    moe_layer_freq: Union[int, List[int]] = 1
    moe_router_topk: int = 8
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    moe_router_topk_scaling_factor: float = 1.0
    moe_aux_loss_coeff: float = 1e-3
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = False
    moe_router_bias_update_rate: float = 0.0
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"

    # MTP (Multi-Token Prediction)
    mtp_num_layers: Optional[int] = 1
    mtp_loss_scaling_factor: Optional[float] = 0.1

    # Misc
    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    make_vocab_size_divisible_by: int = 1280
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16
    attention_softmax_in_fp32: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
