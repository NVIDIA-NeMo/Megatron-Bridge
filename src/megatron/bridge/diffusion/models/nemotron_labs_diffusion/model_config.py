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

"""Provider-neutral model configuration for Nemotron Labs Diffusion."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import ModuleSpec

from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import NemotronLabsDiffusionAttention
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def _namespace(data: Any) -> Any:
    if isinstance(data, dict):
        values = dict(data)
        text_config = values.get("text_config")
        if isinstance(text_config, dict):
            values["text_config"] = SimpleNamespace(**text_config)
        return SimpleNamespace(**values)
    return data


def nemotron_labs_diffusion_layer_spec(config: "NemotronLabsDiffusionModelConfig") -> ModuleSpec:
    """Build a GPT layer spec with semi-block-diffusion core attention."""
    transformer = config.transformer
    spec = get_gpt_layer_with_transformer_engine_spec(
        transformer.num_moe_experts,
        transformer.moe_grouped_gemm,
        transformer.qk_layernorm,
        transformer.multi_latent_attention,
    )
    spec.submodules.self_attention.submodules.core_attention = ModuleSpec(
        module=NemotronLabsDiffusionAttention,
        params={
            "hf_config": _namespace(config.hf_config),
            "block_size": config.block_size,
            "apply_llama4_style_query_key_layer_scaling": config.apply_llama4_style_query_key_layer_scaling,
        },
    )
    return spec


@dataclass(kw_only=True)
class NemotronLabsDiffusionModelConfig(BridgeGPTModelConfig):
    """Serializable GPT build config for Nemotron Labs Diffusion."""

    transformer_layer_spec: Callable[..., ModuleSpec] = field(
        default_factory=lambda: nemotron_labs_diffusion_layer_spec
    )
    hf_config: dict[str, Any] = field(default_factory=dict)
    mask_token_id: int = 100
    dlm_paradigm: str = "sbd_block_diff"
    block_size: int = 64
    different_seed_per_dp: bool = True
    apply_llama4_style_query_key_layer_scaling: bool = True
    dlm_loss_weight: float = 0.3
    ar_loss_weight: float = 1.0


__all__ = ["NemotronLabsDiffusionModelConfig", "nemotron_labs_diffusion_layer_spec"]
