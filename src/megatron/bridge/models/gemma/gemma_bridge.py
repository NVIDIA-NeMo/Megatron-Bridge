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

from megatron.core.activations import fast_gelu
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnBackend
from transformers import GemmaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gemma.gemma_provider import GemmaModelProvider
from megatron.bridge.models.hf_pretrained import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(
    source=GemmaForCausalLM,
    target=GPTModel,
    provider=GemmaModelProvider,
    model_type="gemma",
)
class GemmaBridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma Causal LM.

    This bridge handles the conversion between HuggingFace GemmaForCausalLM
    and Megatron-Core GPTModel formats, including weight mappings and
    configuration translation.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("google/gemma-2b")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GemmaModelProvider:
        """Convert HuggingFace config to GemmaModelProvider.

        Args:
            hf_pretrained: HuggingFace pretrained model wrapper

        Returns:
            GemmaModelProvider: Configured provider for Megatron model
        """
        provider = super().provider_bridge(hf_pretrained)

        provider.normalization = "RMSNorm"
        provider.activation_func = fast_gelu
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.attention_dropout = 0.0
        provider.hidden_dropout = 0.0
        provider.share_embeddings_and_output_weights = True
        provider.layernorm_zero_centered_gamma = True
        provider.attention_backend = AttnBackend.flash

        return provider

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
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
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
