# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""MCore Hybrid stack specification for Inkling2."""

import copy
from functools import partial

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from megatron.bridge.models.inkling.modeling_inkling2.attention import Inkling2Attention
from megatron.bridge.models.inkling.modeling_inkling2.layers import (
    Inkling2AttentionLayer,
    Inkling2MoELayer,
    Inkling2SiTU,
)


def _with_situ_activation(builder):
    """Clone an MLP or MoE builder and install the softcapped SwiGLU activation."""
    if not isinstance(builder, partial):
        raise TypeError(f"Expected a partial module builder, got {type(builder).__name__}")
    keywords = dict(builder.keywords or {})
    submodules = copy.deepcopy(keywords.get("submodules"))
    if isinstance(submodules, (MLPSubmodules, GroupedMLPSubmodules)):
        submodules.activation_func = Inkling2SiTU
    elif isinstance(submodules, MoESubmodules):
        submodules.experts = _with_situ_activation(submodules.experts)
        if submodules.shared_experts is not None:
            submodules.shared_experts = _with_situ_activation(submodules.shared_experts)
    else:
        raise TypeError(f"Cannot install the Inkling2 activation in {type(submodules).__name__}")
    keywords["submodules"] = submodules
    return partial(builder.func, *builder.args, **keywords)


def build_inkling2_hybrid_stack_spec(config) -> ModuleSpec:
    """Compose Inkling2 attention and SiTU MoE layers for a repeated ``*E`` Hybrid stack.

    Args:
        config: The Inkling2 transformer config.

    Returns:
        A ``HybridStack`` spec whose attention and MoE layers form the decoder blocks.
    """
    moe = get_moe_module_spec(
        use_te=True, num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(
            attention_layer=ModuleSpec(
                module=Inkling2AttentionLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=TENorm,
                    self_attention=ModuleSpec(module=Inkling2Attention),
                ),
            ),
            moe_layer=ModuleSpec(
                module=Inkling2MoELayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=TENorm,
                    mlp=_with_situ_activation(moe),
                ),
            ),
        ),
    )
