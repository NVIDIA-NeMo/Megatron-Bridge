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

"""Model provider and custom layer specifications for EXAONE 4.0.

EXAONE 4.0 uses a pure Post-LayerNorm architecture:
    h = x + Attn(x)            # no pre-norm before attention
    h = PostAttnNorm(h)         # RMSNorm after residual add
    o = h + MLP(h)              # no pre-norm before MLP
    o = PostFFNNorm(o)          # RMSNorm after residual add

This requires a custom layer spec because the standard Megatron GPT spec
assumes Pre-LN (fusing layernorm into the column-parallel linear via
TELayerNormColumnParallelLinear). EXAONE instead needs:
- Plain column-parallel linears for QKV and FC1 (no fused pre-norm)
- Row-parallel linears with post-layernorm for output projection and FC2

The Post-LN implementation reuses the TERowParallelLinearLayerNorm pattern
established by Gemma2 bridge.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer import (
    ModuleSpec,
    TransformerConfig,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)


# =============================================================================
# Custom Modules for EXAONE Post-LN Architecture
# =============================================================================


class TERowParallelLinearLayerNorm(TERowParallelLinear):
    """Row-parallel linear with an additional Post-LayerNorm on the output.

    Used for attention output projection (o_proj) and MLP down projection (down_proj)
    in Post-LN architectures where normalization is applied after the residual add.

    This is the same pattern used by Gemma2 bridge for Post-LN support.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            **kwargs,
        )
        self.post_layernorm = TENorm(config, output_size)

    def forward(self, x):
        """Forward with Post-LN applied to the linear output."""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias


# =============================================================================
# EXAONE 4.0 Layer Specification
# =============================================================================


def exaone4_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """EXAONE 4.0 layer specification with pure Post-LayerNorm.

    Key differences from standard GPT layer spec:
    - linear_qkv: TEColumnParallelLinear (no fused pre-norm, since no input_layernorm)
    - linear_proj: TERowParallelLinearLayerNorm (post-attention norm)
    - linear_fc1: TEColumnParallelLinear (no fused pre-norm, since no pre_feedforward_layernorm)
    - linear_fc2: TERowParallelLinearLayerNorm (post-feedforward norm)
    - QK layernorm is handled by qk_layernorm=True in TransformerConfig

    Args:
        config: GPTModelProvider configuration

    Returns:
        ModuleSpec for EXAONE 4.0 transformer layer
    """
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,  # No Pre-LN (pure Post-LN arch)
                    core_attention=None,  # Use default DotProductAttention
                    linear_proj=TERowParallelLinearLayerNorm,  # Post-attention RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,  # No Pre-LN (pure Post-LN arch)
                    linear_fc2=TERowParallelLinearLayerNorm,  # Post-feedforward RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


# =============================================================================
# EXAONE 4.0 Model Providers
# =============================================================================


@dataclass
class Exaone4ModelProvider(GPTModelProvider):
    """Base configuration for EXAONE 4.0 models (LG AI Research).

    Architecture features:
    - Pure Post-LayerNorm (no input_layernorm / pre_feedforward_layernorm)
    - QK RMSNorm after Q/K projection
    - GQA (Grouped Query Attention)
    - SwiGLU activation
    - RoPE with llama3-style scaling
    - Tied word embeddings (embed_tokens shared with lm_head)
    """

    # Architecture defaults common across EXAONE 4.0 model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True  # EXAONE 4.0 uses QK RMSNorm
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = True  # tie_word_embeddings
    rotary_percent: float = 1.0

    # Custom layer spec for Post-LN architecture
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = exaone4_layer_spec

    # Dtype defaults
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True


@dataclass
class Exaone4ModelProvider1P2B(Exaone4ModelProvider):
    """Configuration for EXAONE 4.0 1.2B.

    Model: LGAI-EXAONE/EXAONE-4.0-1.2B
    - 30 layers, 2048 hidden, 32 attention heads, 8 KV heads
    - Full attention only (no sliding window / hybrid attention)
    - RoPE: theta=1M, llama3 scaling factor=16, original_max_pos=8192
    - Vocab: 102,400 tokens, Context: 65,536 tokens
    """

    num_layers: int = 30
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    kv_channels: int = 64
    seq_length: int = 65536
    vocab_size: int = 102400
    rotary_base: float = 1000000.0
    layernorm_epsilon: float = 1e-5
    init_method_std: float = 0.02

    # RoPE scaling (llama3-style)
    rope_scaling: bool = True
    rope_scaling_factor: float = 16.0


# TODO: Add Exaone4ModelProvider32B when 32B model details are confirmed
# The 32B model introduces hybrid attention (LLLG pattern: 3 local + 1 global)
# with sliding_window_pattern and layer_types configuration.
# This will require:
# - Layer-wise attention type branching (local vs global)
# - RoPE disable for global attention layers
# - Sliding window size configuration per layer
