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

from typing import Any

from megatron.core.activations import fast_gelu
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import Gemma2ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gemma.gemma2_provider import Gemma2ModelProvider
from megatron.bridge.models.gemma.model_config import Gemma2ModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register custom Gemma2 modules for AutoMapping
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")
AutoMapping.register_module_type("Gemma2OutputLayer", "column")


@MegatronModelBridge.register_bridge(
    source=Gemma2ForCausalLM,
    target=GPTModel,
    provider=Gemma2ModelProvider,
    model_type="gemma2",
)
class Gemma2Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma2 Causal LM.
    """

    MODEL_CONFIG_CLASS = Gemma2ModelConfig
    TRANSFORMER_CONFIG_CLASS = TransformerConfig
    CUSTOM_PROVIDER_MODEL_CONFIG_SUPPORTED = True

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma2ModelProvider:
        """Convert HuggingFace config to Gemma2ModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.query_pre_attn_scalar = hf_config.query_pre_attn_scalar
        provider.attn_logit_softcapping = hf_config.attn_logit_softcapping
        provider.final_logit_softcapping = hf_config.final_logit_softcapping
        provider.window_size = (hf_config.sliding_window - 1, 0)

        provider.normalization = "RMSNorm"
        provider.activation_func = fast_gelu
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.attention_dropout = 0.0
        provider.hidden_dropout = 0.0
        provider.share_embeddings_and_output_weights = True
        provider.layernorm_zero_centered_gamma = True

        return provider

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Convert a Hugging Face Gemma2 config to model-config kwargs.

        Args:
            hf_config: Hugging Face Gemma2 configuration.

        Returns:
            Flat model and transformer config keyword arguments.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            query_pre_attn_scalar=hf_config.query_pre_attn_scalar,
            attn_logit_softcapping=hf_config.attn_logit_softcapping,
            final_logit_softcapping=hf_config.final_logit_softcapping,
            window_size=(hf_config.sliding_window - 1, 0),
            normalization="RMSNorm",
            activation_func=fast_gelu,
            gated_linear_unit=True,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            share_embeddings_and_output_weights=True,
            layernorm_zero_centered_gamma=True,
        )
        return config_kwargs

    @classmethod
    def megatron_to_hf_config(cls, provider: Gemma2ModelProvider | Gemma2ModelConfig) -> dict:
        """Convert a Gemma2 provider or model config back to Hugging Face config."""
        hf_config = super().megatron_to_hf_config(provider)

        hf_config["query_pre_attn_scalar"] = provider.query_pre_attn_scalar
        hf_config["attn_logit_softcapping"] = provider.attn_logit_softcapping
        hf_config["final_logit_softcapping"] = provider.final_logit_softcapping

        if getattr(provider, "window_size", None) is not None:
            hf_config["sliding_window"] = provider.window_size[0] + 1

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.
        Returns:
            MegatronMappingRegistry: Registry of parameter mappings
        """
        # Dictionary maps HF parameter names -> Megatron parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.pre_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
