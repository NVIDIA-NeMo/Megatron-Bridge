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
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec
logger = logging.getLogger(__name__)


@dataclass
class Qwen2ModelProvider(GPTModelProvider):
    """Base model provider for Qwen 2 Models."""

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
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True


# =============================================================================
# Qwen 2 Model Providers
# =============================================================================


@dataclass
class Qwen2ModelProvider500M(Qwen2ModelProvider):
    """
    Config for Qwen 2 0.5B: https://huggingface.co/Qwen/Qwen2-0.5B
    """

    num_layers: int = 24
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_query_groups: int = 2
    ffn_hidden_size: int = 4864
    share_embeddings_and_output_weights: bool = True
    seq_length: int = 32768


@dataclass
class Qwen2ModelProvider1P5B(Qwen2ModelProvider):
    """
    Config for Qwen 2 1.5B: https://huggingface.co/Qwen/Qwen2-1.5B
    """

    num_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_query_groups: int = 2
    ffn_hidden_size: int = 8960
    seq_length: int = 32768
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen2ModelProvider7B(Qwen2ModelProvider):
    """
    Config for Qwen 2 7B: https://huggingface.co/Qwen/Qwen2-7B
    """

    num_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_query_groups: int = 4
    ffn_hidden_size: int = 18944
    vocab_size: int = 152064
    seq_length: int = 32768


@dataclass
class Qwen2ModelProvider72B(Qwen2ModelProvider):
    """
    Config for Qwen 2 72B: https://huggingface.co/Qwen/Qwen2-72B
    """

    num_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 29568
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-6
    seq_length: int = 32768


# =============================================================================
# Qwen 2.5 Model Providers
# =============================================================================


@dataclass
class Qwen25ModelProvider500M(Qwen2ModelProvider):
    """
    Config for Qwen 2.5 0.5B: https://huggingface.co/Qwen/Qwen2.5-0.5B
    """

    num_layers: int = 24
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_query_groups: int = 2
    ffn_hidden_size: int = 4864
    share_embeddings_and_output_weights: bool = True
    seq_length: int = 32768


@dataclass
class Qwen25ModelProvider1P5B(Qwen2ModelProvider):
    """
    Config for Qwen 2.5 1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B
    """

    num_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_query_groups: int = 2
    ffn_hidden_size: int = 8960
    seq_length: int = 32768
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen25ModelProvider3B(Qwen2ModelProvider):
    """
    Config for Qwen 2.5 3B: https://huggingface.co/Qwen/Qwen2.5-3B
    """

    num_layers: int = 36
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_query_groups: int = 2
    ffn_hidden_size: int = 11008
    vocab_size: int = 151936
    share_embeddings_and_output_weights: bool = True
    seq_length: int = 32768


@dataclass
class Qwen25ModelProvider7B(Qwen2ModelProvider):
    """
    Config for Qwen 2.5 7B: https://huggingface.co/Qwen/Qwen2.5-7B
    """

    num_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_query_groups: int = 4
    ffn_hidden_size: int = 18944
    vocab_size: int = 152064
    seq_length: int = 32768


@dataclass
class Qwen25ModelProvider14B(Qwen2ModelProvider):
    """
    Config for Qwen 2.5 14B: https://huggingface.co/Qwen/Qwen2.5-14B
    """

    num_layers: int = 48
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 8
    ffn_hidden_size: int = 13824
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-6
    seq_length: int = 32768


@dataclass
class Qwen25ModelProvider32B(Qwen2ModelProvider):
    """
    Config for Qwen 2.5 32B: https://huggingface.co/Qwen/Qwen2.5-32B
    """

    num_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 8
    ffn_hidden_size: int = 27648
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-6
    seq_length: int = 32768


@dataclass
class Qwen25ModelProvider72B(Qwen2ModelProvider):
    """
    Config for Qwen 2.5 72B: https://huggingface.co/Qwen/Qwen2.5-72B
    """

    num_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 29568
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-6
    seq_length: int = 32768


# =============================================================================
# Qwen 3 Model Provider (based on GPTProvider)
# =============================================================================


@dataclass
class Qwen3ModelProvider(GPTModelProvider):
    """Base model provider for Qwen 3 Models."""

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: Optional[int] = 128
    num_query_groups: int = 8
    seq_length: int = 40960
    max_position_embeddings: int = 40960
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True


@dataclass
class Qwen3ModelProvider600M(Qwen3ModelProvider):
    """
    Config for Qwen 3 0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
    """

    num_layers: int = 28
    hidden_size: int = 1024
    num_attention_heads: int = 16
    ffn_hidden_size: int = 3072
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen3ModelProvider1P7B(Qwen3ModelProvider):
    """
    Config for Qwen 3 1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
    """

    num_layers: int = 28
    hidden_size: int = 2048
    num_attention_heads: int = 16
    ffn_hidden_size: int = 6144
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen3ModelProvider4B(Qwen3ModelProvider):
    """
    Config for Qwen 3 4B: https://huggingface.co/Qwen/Qwen3-4B
    """

    num_layers: int = 36
    hidden_size: int = 2560
    num_attention_heads: int = 32
    ffn_hidden_size: int = 9728
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen3ModelProvider8B(Qwen3ModelProvider):
    """
    Config for Qwen 3 8B: https://huggingface.co/Qwen/Qwen3-8B
    """

    num_layers: int = 36
    hidden_size: int = 4096
    num_attention_heads: int = 32
    ffn_hidden_size: int = 12288


@dataclass
class Qwen3ModelProvider14B(Qwen3ModelProvider):
    """
    Config for Qwen 3 14B: https://huggingface.co/Qwen/Qwen3-14B
    """

    num_layers: int = 40
    hidden_size: int = 5120
    num_attention_heads: int = 40
    ffn_hidden_size: int = 17408


@dataclass
class Qwen3ModelProvider32B(Qwen3ModelProvider):
    """
    Config for Qwen 3 32B: https://huggingface.co/Qwen/Qwen3-32B
    """

    num_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 64
    ffn_hidden_size: int = 25600


# =============================================================================
# Qwen 3 MoE Model Provider (based on GPTProvider)
# =============================================================================


@dataclass
class Qwen3MoEModelProvider(GPTModelProvider):
    """Base provider for Qwen 3 MoE Models."""

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: Optional[int] = 128
    num_query_groups: int = 8
    seq_length: int = 40960
    max_position_embeddings: int = 40960
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True

    # MoE specific parameters
    num_moe_experts: int = 128
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_topk: int = 8
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True


@dataclass
class Qwen3MoEModelProvider30B_A3B(Qwen3MoEModelProvider):
    """
    Provider for Qwen 3 30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
    """

    num_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_query_groups: int = 4
    ffn_hidden_size: int = 6144
    moe_ffn_hidden_size: int = 768


@dataclass
class Qwen3MoEModelProvider235B_A22B(Qwen3MoEModelProvider):
    """
    Provider for Qwen 3 235B-A22B: https://huggingface.co/Qwen/Qwen3-235B-A22B
    """

    num_layers: int = 94
    hidden_size: int = 4096
    num_attention_heads: int = 64
    num_query_groups: int = 4
    ffn_hidden_size: int = 12288
    moe_ffn_hidden_size: int = 1536


# =============================================================================
# Qwen 3 Next Model Provider (based on Qwen3MoEModelProvider)
# =============================================================================


@dataclass
class Qwen3NextModelProvider(Qwen3MoEModelProvider):
    """Base provider for Qwen 3 Next Models."""

    transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = (
        get_transformer_block_with_experimental_attention_variant_spec
    )

    layernorm_zero_centered_gamma: bool = True  # Zero-centered RMSNorm
    kv_channels: int | None = 256
    num_query_groups: int = 2
    seq_length: int = 262144  # 256k tokens
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25  # 25% of the hidden size is used for RoPE
    attention_output_gate: bool = True  # Gated Attention

    # MoE specific parameters
    num_moe_experts: int = 512
    moe_router_topk: int = 10  # 10 routed experts per token
    moe_shared_expert_gate: bool = True  # Qwen3-Next uses a gate for the shared expert
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "global_aux_loss"  # Qwen3-Next uses global aux loss for load balancing

    # Linear Attention specific parameters
    experimental_attention_variant: str = "gated_delta_net"  # Gated Delta Net used in 75% of the model layers
    linear_attention_freq: int | list[int] = 4  # 1 gated standard attention layer per 4 layers
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # Checkpointing
    hetereogenous_dist_checkpoint: bool = True


@dataclass
class Qwen3NextModelProvider80B_A3B(Qwen3NextModelProvider):
    """
    Provider for Qwen 3 Next 80B-A3B: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct and https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking
    """

    num_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_query_groups: int = 2
    ffn_hidden_size: int = 5120
    moe_ffn_hidden_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    mtp_num_layers: Optional[int] = None


# =============================================================================
# Qwen 3 Next MambaModel Provider (based on MambaModelProvider)
# =============================================================================


@dataclass
class Qwen3NextMambaModelProvider(MambaModelProvider):
    """Base provider for Qwen 3 Next Models using MambaModel with GDN layers."""

    # Architecture
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    layernorm_zero_centered_gamma: bool = True
    layernorm_epsilon: float = 1e-6
    kv_channels: Optional[int] = 256
    num_query_groups: int = 2
    seq_length: int = 262144
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: bool = False
    init_method_std: int = 0.02
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True

    # Position embeddings (override MambaModel default of "none")
    position_embedding_type: str = "rope"
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25

    # Attention
    attention_output_gate: bool = True

    # MoE specific parameters
    num_moe_experts: int = 512
    moe_router_topk: int = 10
    moe_shared_expert_gate: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "global_aux_loss"
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_aux_loss_coeff: float = 1e-3

    # GDN linear attention parameters
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # Checkpointing
    hetereogenous_dist_checkpoint: bool = True


@dataclass
class Qwen3NextMambaModelProvider80B_A3B(Qwen3NextMambaModelProvider):
    """
    Provider for Qwen 3 Next 80B-A3B using MambaModel with GDN layers.

    In MambaModel, each layer type is a separate physical layer, so each original
    Qwen3-Next transformer layer (attention/GDN + MoE FFN) becomes two physical
    layers: one G/* layer for attention and one E layer for MoE FFN.

    Original architecture (48 logical layers, linear_attention_freq=4):
        Every 4th layer uses standard attention, the rest use GDN.
        Each layer has MoE FFN (512 experts, top-10 routing).

    MambaModel mapping (96 physical layers):
        GEGEGE*E repeated 12 times = 96 layers
        - GE = GDN attention + MoE FFN (one logical layer)
        - *E = Standard attention + MoE FFN (one logical layer)

    MTP: /GE = 1 MTP depth with GDN attention + MoE FFN
    """

    # GEGEGE*E repeated 12 times = 96 physical layers, /GE = 1 MTP depth
    hybrid_override_pattern: str = "GEGEGE*E" * 12 + "/GE"
    num_layers: int = 96
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_query_groups: int = 2
    ffn_hidden_size: int = 5120
    moe_ffn_hidden_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    mtp_num_layers: Optional[int] = None
