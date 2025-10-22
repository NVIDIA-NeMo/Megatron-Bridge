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
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import SelfAttention as MCoreSelfAttention
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider, default_layer_spec


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


def olmoe_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Layer spec for OlMoE models."""
    layer_spec = default_layer_spec(config)
    layer_spec.submodules.self_attention.module = OLMoESelfAttention
    return layer_spec


@dataclass
class OlMoEModelProvider(GPTModelProvider):
    """Base provider for OlMoE Models."""

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = olmoe_layer_spec
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

    # Model specific parameters
    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 1024
    moe_ffn_hidden_size: int = 1024
    kv_channels: int = 2048 // 16

    # Attention
    num_query_groups: int = 16
    num_attention_heads: int = 16
    attention_dropout: float = 0.0
    qk_layernorm: bool = True
    # RoPE
    position_embedding_type: str = "rope"
    rotary_base: float = 10000.0
    # MoE specific parameters
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
    # Optimization
    persist_layer_norm: bool = True


class OLMoESelfAttention(MCoreSelfAttention):
    """Custom self-attention module for OlMoE models."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        # Unlike Mcore QK Layernorm, OlMoE layernorm has hidden_size = hidden_size_per_attention_head * num_attention_heads
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head
            * self.config.num_attention_heads,  # Main difference between Mcore QK Layernorm
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head
            * self.config.num_attention_heads,  # Main difference between Mcore QK Layernorm
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        # Main difference between Mcore QK Layernorm
        query = query.reshape(query.size(0), query.size(1), -1)
        key = key.reshape(key.size(0), key.size(1), -1)
        query = self.q_layernorm(query)
        key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        query = query.view(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        key = key.view(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value
