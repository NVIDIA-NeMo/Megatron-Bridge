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

"""MCore layer specification for Kimi K3."""

import copy
from functools import partial

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention, KimiK3MoELayer, KimiK3TransformerLayer
from megatron.bridge.models.kimi.kimi_k3_ops import SiTUAndMul


def _with_situ_activation(builder):
    """Clone an MLP/MoE builder and replace every activation with SiTU."""
    if not isinstance(builder, partial):
        return builder

    keywords = dict(builder.keywords or {})
    submodules = copy.deepcopy(keywords.get("submodules"))
    if isinstance(submodules, (MLPSubmodules, GroupedMLPSubmodules)):
        submodules.activation_func = SiTUAndMul
    elif isinstance(submodules, MoESubmodules):
        submodules.experts = _with_situ_activation(submodules.experts)
        if submodules.shared_experts is not None:
            submodules.shared_experts = _with_situ_activation(submodules.shared_experts)
    if submodules is not None:
        keywords["submodules"] = submodules

    function = KimiK3MoELayer if builder.func is MoELayer else builder.func
    return partial(function, *builder.args, **keywords)


def build_kimi_k3_spec(config, vp_stage=None):
    """Build the heterogeneous KDA/MLA and dense/MoE Kimi K3 block."""
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Kimi K3 does not support virtual pipeline parallelism yet")

    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )
    layer_offset = get_transformer_layer_offset(config, vp_stage)
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.module = KimiK3TransformerLayer
        layer_spec.submodules.self_attention = ModuleSpec(module=KimiK3Attention)
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.pre_mlp_layernorm = TENorm
        layer_spec.submodules.mlp = _with_situ_activation(layer_spec.submodules.mlp)

        if not config.moe_layer_freq[layer_offset + len(layer_specs)]:
            layer_spec.submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec
