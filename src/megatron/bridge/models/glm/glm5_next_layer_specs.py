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

"""Megatron block spec for GLM-5.3-Flash: KDA or DSA attention, dense or MoE MLP, mHC residuals.

megatron-core's ``get_transformer_layer_with_experimental_attention_variant_spec`` builds a
hybrid of one experimental attention variant against *standard* attention. GLM-5.3-Flash mixes
two experimental variants -- KDA on 3 of every 4 layers, NoPE-MLA DSA on the rest -- so the
per-layer spec is assembled here from the same public helpers instead.
"""

import copy
from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, get_backend_from_config
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_kda_module_spec,
    get_linear_attention_pattern,
    get_moe_layer_pattern,
)
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from megatron.bridge.models.glm.glm5_next_dsa import Glm5NextDSAttention


def build_glm5_next_layer_spec(
    config: TransformerConfig, vp_stage: Optional[int] = None
) -> TransformerBlockSubmodules:
    """Build the GLM-5.3-Flash decoder block spec for this pipeline stage.

    Layer ``i`` is KDA when ``config.linear_attention_freq[i]`` is 1 and DSA (NoPE MLA with the
    lightning indexer) otherwise; its MLP is MoE when ``config.moe_layer_freq[i]`` is 1 and a
    dense (layernorm-fused) MLP otherwise. Every layer is an mHC
    :class:`~megatron.core.transformer.transformer_layer.HyperConnectionTransformerLayer`.
    """
    if config.transformer_impl != "transformer_engine":
        raise ValueError("The GLM-5.3-Flash block spec requires transformer_impl='transformer_engine'.")
    if not config.enable_mhc_connections:
        raise ValueError("GLM-5.3-Flash requires enable_mhc_connections=True.")
    backend: BackendSpecProvider = get_backend_from_config(config)
    rms_norm = config.normalization == "RMSNorm"

    attention_pattern = get_linear_attention_pattern(config)
    moe_pattern = get_moe_layer_pattern(config)

    kda_spec = get_kda_module_spec(config, backend)
    dsa_spec = copy.deepcopy(get_dsa_module_spec_for_backend(config, backend))
    dsa_spec.submodules.core_attention.module = Glm5NextDSAttention

    moe_mlp = get_moe_module_spec_for_backend(
        backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )
    # The TE dense MLP fuses the pre-MLP layernorm into linear_fc1.
    dense_mlp = get_mlp_module_spec_for_backend(
        backend, num_experts=None, use_te_activation_func=config.use_te_activation_func
    )

    layer_specs = []
    for layer_idx in range(config.num_layers):
        is_moe = bool(moe_pattern[layer_idx])
        layer_specs.append(
            ModuleSpec(
                module=HyperConnectionTransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
                    self_attention=kda_spec if attention_pattern[layer_idx] else dsa_spec,
                    self_attn_bda=get_bias_dropout_add,
                    self_attention_hyper_connection=HyperConnectionModule,
                    pre_mlp_layernorm=(backend.layer_norm(rms_norm=rms_norm, for_qk=False) if is_moe else IdentityOp),
                    mlp=moe_mlp if is_moe else dense_mlp,
                    mlp_bda=get_bias_dropout_add,
                    mlp_hyper_connection=HyperConnectionModule,
                ),
            )
        )

    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    return TransformerBlockSubmodules(
        layer_specs=layer_specs, layer_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )
