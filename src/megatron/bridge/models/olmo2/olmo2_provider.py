# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Provider and layer spec for OLMo-2 dense causal LMs.

OLMo-2's decoder block applies normalization *only after* each sub-block::

    x = x + post_attention_layernorm(self_attn(x))
    x = x + post_feedforward_layernorm(mlp(x))

This differs from the standard Megatron pre-norm spec (which normalizes the
input of each sub-block) and from Gemma2's sandwich norm (which normalizes
both the input and the output). To realize OLMo-2 in Megatron-Core, the layer
spec built here:

* uses ``IdentityOp`` for ``input_layernorm`` and ``pre_mlp_layernorm`` so
  the pre-block normalizations are no-ops,
* wraps ``linear_proj`` and ``linear_fc2`` in ``TERowParallelLinearPostLN``
  so an RMSNorm is applied to each sub-block's output before the residual,
* keeps Q/K RMSNorm via the standard ``q_layernorm`` / ``k_layernorm``
  submodule slots (enabled by ``qk_layernorm=True`` on the provider).

The post-LN wrapper is identical in spirit to Gemma2's
``TERowParallelLinearLayerNorm``; we redefine it locally per the project's
"keep model-specific logic in the family directory" guideline.
"""

from dataclasses import dataclass
from typing import Callable, Union

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
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
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron.bridge.models.gpt_provider import GPTModelProvider


class TERowParallelLinearPostLN(TERowParallelLinear):
    """``TERowParallelLinear`` with a trailing RMSNorm applied to its output.

    Used at the output of attention (`linear_proj`) and MLP (`linear_fc2`)
    sub-blocks so that OLMo-2's post-norm placement
    ``residual + post_norm(sub_block(x))`` can be expressed in the standard
    Megatron-Core ``TransformerLayer`` template.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        **kwargs: object,
    ) -> None:
        super().__init__(input_size, output_size, config=config, **kwargs)
        self.post_layernorm = TENorm(config, output_size)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        """Forward with a trailing RMSNorm."""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias


def olmo2_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Layer spec for OLMo-2 dense models.

    * No pre-norms (``input_layernorm`` / ``pre_mlp_layernorm`` → ``IdentityOp``)
    * Post-attention RMSNorm fused into ``linear_proj`` via ``TERowParallelLinearPostLN``
    * Post-feedforward RMSNorm fused into ``linear_fc2`` via ``TERowParallelLinearPostLN``
    * QK-RMSNorm via standard ``q_layernorm`` / ``k_layernorm`` submodule slots,
      activated by ``provider.qk_layernorm = True``.
    """
    del config  # spec is independent of the runtime config; signature kept for symmetry.
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinearPostLN,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinearPostLN,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


@dataclass
class Olmo2ModelProvider(GPTModelProvider):
    """Base provider for OLMo-2 dense causal LMs.

    Architectural choices (from `allenai/OLMo-2-1124-7B/config.json` and
    `transformers/models/olmo2/modeling_olmo2.py`):

    * RMSNorm with `layernorm_epsilon=1e-6`
    * SwiGLU MLP (``activation_func=F.silu`` + ``gated_linear_unit=True``)
    * No biases anywhere (``add_bias_linear=False``, ``add_qkv_bias=False``)
    * RoPE with ``rotary_base=500000``
    * Tied embeddings = False (untied input/output)
    * QK-RMSNorm (``qk_layernorm=True``)
    * Pure post-norm via the custom ``olmo2_layer_spec``.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = olmo2_layer_spec
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 500000.0
    seq_length: int = 4096
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    persist_layer_norm: bool = True
    autocast_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 100352


@dataclass
class Olmo2ModelProvider1B(Olmo2ModelProvider):
    """OLMo-2 1B (`allenai/OLMo-2-0425-1B`): 16 layers, hidden=2048, MHA=16, ffn=8192."""

    num_layers: int = 16
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_query_groups: int = 16
    ffn_hidden_size: int = 8192
    kv_channels: int = 128


@dataclass
class Olmo2ModelProvider7B(Olmo2ModelProvider):
    """OLMo-2 7B (`allenai/OLMo-2-1124-7B`): 32 layers, hidden=4096, MHA=32, ffn=11008."""

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 32
    ffn_hidden_size: int = 11008
    kv_channels: int = 128


@dataclass
class Olmo2ModelProvider13B(Olmo2ModelProvider):
    """OLMo-2 13B (`allenai/OLMo-2-1124-13B`): 40 layers, hidden=5120, MHA=40, ffn=13824."""

    num_layers: int = 40
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 40
    ffn_hidden_size: int = 13824
    kv_channels: int = 128
