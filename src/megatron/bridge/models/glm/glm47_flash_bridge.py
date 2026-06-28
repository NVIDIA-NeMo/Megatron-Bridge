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

"""Megatron Bridge for GLM-4.7-Flash (Glm4MoeLiteForCausalLM).

This module registers only the ``glm4_moe_lite`` / Flash variant. The full
GLM-4.7 model uses the existing ``GLM45Bridge`` registration for
``Glm4MoeForCausalLM`` / ``glm4_moe``.

GLM-4.7-Flash combines Multi-Latent Attention (MLA, inherited from DeepSeek V3)
with GLM-style MoE routing.  The safetensors checkpoint uses per-expert weight
naming (``experts.{i}.gate_proj``), not the fused ``gate_up_proj`` tensor used
by the runtime model, so the DeepSeek common mapping list works directly.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass(kw_only=True)
class GLM47FlashModelConfig(BridgeGPTModelConfig):
    """Serializable GLM-4.7-Flash GPT build configuration."""

    transformer_layer_spec: ModuleSpec | Callable[[BridgeGPTModelConfig], ModuleSpec] | None = field(
        default_factory=lambda: partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
    )


@MegatronModelBridge.register_bridge(
    source="Glm4MoeLiteForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="glm4_moe_lite",
)
class GLM47FlashBridge(MegatronModelBridge):
    """Megatron Bridge for GLM-4.7-Flash.

    GLM-4.7-Flash uses Multi-Latent Attention (MLA) with MoE routing,
    combining DeepSeek V3-style compressed attention with GLM MoE architecture.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.7-Flash")
        >>> provider = bridge.to_megatron_provider()
    """

    TRANSFORMER_CONFIG_CLASS = MLATransformerConfig
    MODEL_CONFIG_CLASS = GLM47FlashModelConfig
    CUSTOM_PROVIDER_MODEL_CONFIG_SUPPORTED = True

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        """Convert HuggingFace config to MLAModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.multi_latent_attention = True
        provider.qk_layernorm = True

        provider.moe_shared_expert_overlap = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_permute_fusion = True
        provider.moe_router_dtype = "fp32"
        provider.moe_router_bias_update_rate = 0
        provider.moe_aux_loss_coeff = 0.001

        provider.apply_rope_fusion = False
        provider.persist_layer_norm = True
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16

        provider.mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", None)
        provider.mtp_loss_scaling_factor = 0.3

        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * getattr(
            hf_config, "n_shared_experts", 1
        )

        first_k = getattr(hf_config, "first_k_dense_replace", 1)
        provider.moe_layer_freq = [0] * first_k + [1] * (hf_config.num_hidden_layers - first_k)

        return provider

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Convert a Hugging Face GLM-4.7-Flash config to Megatron model-config kwargs."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        first_k = getattr(hf_config, "first_k_dense_replace", 1)
        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            share_embeddings_and_output_weights=False,
            multi_latent_attention=True,
            qk_layernorm=True,
            moe_shared_expert_overlap=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_grouped_gemm=True,
            moe_router_pre_softmax=True,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_permute_fusion=True,
            moe_router_dtype="fp32",
            moe_router_bias_update_rate=0,
            moe_aux_loss_coeff=0.001,
            apply_rope_fusion=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            hidden_dropout=0.0,
            mtp_num_layers=getattr(hf_config, "num_nextn_predict_layers", None),
            mtp_loss_scaling_factor=0.3,
            moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size
            * getattr(hf_config, "n_shared_experts", 1),
            moe_layer_freq=[0] * first_k + [1] * (hf_config.num_hidden_layers - first_k),
        )
        return config_kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = getattr(self, "hf_config", None)
        mapping_list = get_common_mapping_list(hf_config=hf_config)
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.expert_bias",
                hf_param="model.layers.*.mlp.gate.e_score_correction_bias",
            )
        )
        return MegatronMappingRegistry(*mapping_list)
