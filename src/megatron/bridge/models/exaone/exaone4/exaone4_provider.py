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

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer import (
    ModuleSpec,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron.bridge.models.common.te_layers import TERowParallelLinearLayerNorm
from megatron.bridge.models.gpt_provider import GPTModelProvider


# =============================================================================
# EXAONE 4.0 Layer Specification
# =============================================================================


def exaone4_layer_spec(config: "GPTModelProvider") -> ModuleSpec:  # noqa: ARG001
    """EXAONE 4.0 layer specification with pure Post-LayerNorm.

    Key differences from standard GPT layer spec:
    - linear_qkv: TEColumnParallelLinear (no fused pre-norm, since no input_layernorm)
    - linear_proj: TERowParallelLinearLayerNorm (post-attention norm)
    - linear_fc1: TEColumnParallelLinear (no fused pre-norm, since no pre_feedforward_layernorm)
    - linear_fc2: TERowParallelLinearLayerNorm (post-feedforward norm)
    - QK layernorm is handled by qk_layernorm=True in TransformerConfig

    Args:
        config: Reserved for future use (e.g., 32B hybrid attention with
            layer-wise branching between local and global attention).

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
                    core_attention=DotProductAttention,  # Explicit attention class
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
