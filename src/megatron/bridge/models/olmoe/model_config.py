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

"""Provider-neutral OLMoE model configuration."""

from dataclasses import dataclass
from typing import Any, Callable

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.olmoe.modeling_olmoe import OLMoESelfAttention


def olmoe_model_config_layer_spec(config: Any) -> ModuleSpec:
    """Build the OLMoE layer spec from pure model-config fields."""
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        fp8=bool(config.num_moe_experts and config.fp8 is not None),
    )
    layer_spec.submodules.self_attention.module = OLMoESelfAttention
    return layer_spec


@dataclass(kw_only=True)
class OlMoEModelConfig(BridgeGPTModelConfig):
    """Builder-backed OLMoE config with its custom self-attention layer spec."""

    transformer_layer_spec: Callable[..., ModuleSpec] = olmoe_model_config_layer_spec


__all__ = ["OlMoEModelConfig", "olmoe_model_config_layer_spec"]
