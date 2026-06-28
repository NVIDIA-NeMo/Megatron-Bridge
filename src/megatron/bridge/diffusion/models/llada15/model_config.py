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

"""Provider-neutral builder configuration for LLaDA1.5."""

from dataclasses import dataclass, field
from typing import Callable

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import ModuleSpec

from megatron.bridge.diffusion.models.llada15.llada15_attention import LLaDA15TEDotProductAttention
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def llada15_layer_spec(config: BridgeGPTModelConfig) -> ModuleSpec:
    """Build the bidirectional LLaDA1.5 Transformer Engine layer spec."""
    transformer = config.transformer
    spec = get_gpt_layer_with_transformer_engine_spec(
        transformer.num_moe_experts,
        transformer.moe_grouped_gemm,
        transformer.qk_layernorm,
        transformer.multi_latent_attention,
    )
    spec.submodules.self_attention.submodules.core_attention = LLaDA15TEDotProductAttention
    return spec


@dataclass(kw_only=True)
class LLaDA15ModelConfig(BridgeGPTModelConfig):
    """Serializable LLaDA1.5 GPT model configuration."""

    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: llada15_layer_spec)


__all__ = ["LLaDA15ModelConfig", "llada15_layer_spec"]
