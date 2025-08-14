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
from typing import Callable, List, Literal, Optional, Tuple, Union

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer.enums import AttnBackend

logger = logging.getLogger(__name__)


@dataclass
class GPTOSSProvider(GPTModelProvider):
    """
    Base config for GPT-OSS
    """
    hidden_size: int = 2880
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 2880
    kv_channels: Optional[int] = 64
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = True
    share_embeddings_and_output_weights: bool = False
    vocab_size: int = 201088
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    position_embedding_type: str = "yarn"
    rotary_base: int = 150000
    rotary_scaling_factor: float = 32.
    yarn_original_max_position_embeddings: int = 131072
    yarn_beta_fast: float = 32.
    yarn_beta_slow: float = 1.
    yarn_correction_range_round_to_int: bool = False

    moe_router_topk: int = 4
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_ffn_hidden_size: int = 2880
    moe_router_load_balancing_type: str = "none"
    seq_length: int = 131072
    window_size: Optional[Tuple[int, int]] = (128, 0)
    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = "learnable"
    activation_func: Callable = quick_gelu
    glu_linear_offset: float = 1.0
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = False
    window_attn_skip_freq: Optional[Union[int, List[int]]] = 2  # alternative SWA/full
    attention_backend: AttnBackend = AttnBackend.local  # currently only "local" is supported
    activation_func_clamp_value: Optional[float] = 7.0

@dataclass
class GPTOSSProvider120B(GPTOSSProvider):
    """Config for GPT-OSS 120B """
    num_layers: int = 36
    num_moe_experts: int = 128


@dataclass
class GPTOSSProvider20B(GPTOSSProvider):
    """Config for GPT-OSS 20B """
    num_layers: int = 24
    num_moe_experts: int = 32
