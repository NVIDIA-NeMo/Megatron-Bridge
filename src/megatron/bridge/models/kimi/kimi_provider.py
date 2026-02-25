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
from dataclasses import dataclass, field
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

if HAVE_TE:
    from megatron.core.utils import is_te_min_version


@dataclass
class KimiK2Provider(MLATransformerConfig, GPTModelProvider):
    """
    https://moonshotai.github.io/Kimi-K2/
    """

    transformer_layer_spec: Union["ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]] = partial(
        get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
    )

    # Model
    num_layers: int = 61
    hidden_size: int = 7168
    ffn_hidden_size: int = 18432
    num_moe_experts: int = 384
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048  # 2048 * 1 shared expert
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] + [1] * 60)  # first layer are dense
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True  # swiglu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    num_attention_heads: int = 64
    kv_channels: int = 64
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    rotary_base: float = 50000.0
    make_vocab_size_divisible_by: int = 1280
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: Optional[float] = None

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    qk_layernorm: bool = True

    # MoE
    moe_router_topk: int = 8
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 2.827
    moe_aux_loss_coeff: float = 1e-3
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True
    moe_router_dtype: Optional[str] = "fp32"

    # MLA
    multi_latent_attention: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 32
    beta_fast: float = 1.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # Miscellaneous
    init_method_std: float = 0.006
    layernorm_epsilon: float = 1e-6
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    attention_softmax_in_fp32: bool = False
    persist_layer_norm: bool = True
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False
    vocab_size: int = 163840

    # fusions
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = True
    gradient_accumulation_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    cross_entropy_fusion_impl: str = "te"
    moe_permute_fusion: bool = is_te_min_version("2.1.0") if HAVE_TE else False
