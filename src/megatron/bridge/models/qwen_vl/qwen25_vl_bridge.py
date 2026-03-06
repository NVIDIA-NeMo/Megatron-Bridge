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

from transformers import Qwen2_5_VLForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.modeling_qwen25_vl import Qwen25VLModel
from megatron.bridge.models.qwen_vl.qwen25_vl_provider import Qwen25VLModelProvider


@MegatronModelBridge.register_bridge(
    source=Qwen2_5_VLForConditionalGeneration,
    target=Qwen25VLModel,
    provider=Qwen25VLModelProvider,
    model_type="qwen2_5_vl",
)
class Qwen25VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen2.5-VL Conditional Generation.

    This bridge handles the conversion between HuggingFace Qwen2_5_VLForConditionalGeneration
    and Megatron-Core GPTModel formats, including weight mappings and
    configuration translation for vision-language models.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen25VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = Qwen25VLModelProvider(**provider_kwargs)

        # Qwen2-specific settings
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = True
        provider.add_bias_linear = False
        provider.hidden_dropout = 0.0
        provider.rotary_base = rope_theta_from_hf(text_config)

        # For VLMs, tie_word_embeddings lives on the top-level config, not text_config.
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # VL-specific overrides
        provider.position_embedding_type = "mrope"
        provider.vision_config = hf_config.vision_config
        provider.bos_token_id = getattr(text_config, "bos_token_id", 151643)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 151645)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 151652)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 151653)
        provider.vision_token_id = getattr(hf_config, "vision_token_id", 151654)
        provider.image_token_id = getattr(hf_config, "image_token_id", 151655)
        provider.video_token_id = getattr(hf_config, "video_token_id", 151656)

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "language_model.embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="visual.**",
                    hf_param="visual.**",
                ),
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias: Combine separate Q, K, V biases into single QKV bias (Qwen2 specific)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
