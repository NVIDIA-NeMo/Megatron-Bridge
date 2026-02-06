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

import torch
from transformers import Qwen2_5OmniForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.modeling_qwen25_vl import Qwen25VLModel
from megatron.bridge.models.qwen_vl.qwen_vl_provider import Qwen25VLModelProvider


@MegatronModelBridge.register_bridge(source=Qwen2_5OmniForConditionalGeneration, target=Qwen25VLModel)
class Qwen25OmniBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen 2.5-Omni Conditional Generation.

    This bridge handles the conversion between HuggingFace Qwen2_5OmniForConditionalGeneration
    and Megatron-Core GPTModel formats, including weight mappings and
    configuration translation for multimodal models.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen25VLModelProvider:

        hf_config = hf_pretrained.config

        thinker_config = hf_config.thinker_config
        text_config = thinker_config.text_config
        
        mrope_section = None
        if hasattr(text_config, "rope_parameters") and text_config.rope_parameters is not None:
            if isinstance(text_config.rope_parameters, dict):
                mrope_section = text_config.rope_parameters.get("mrope_section", None)
            elif hasattr(text_config.rope_parameters, "mrope_section"):
                mrope_section = text_config.rope_parameters.mrope_section
        
        rope_theta = getattr(text_config, "rope_theta", None) or (
            text_config.rope_parameters.rope_theta
            if hasattr(text_config, "rope_parameters") and 
               text_config.rope_parameters is not None and
               hasattr(text_config.rope_parameters, "rope_theta")
            else 1000000.0
        )
        
        model_dtype = self.dtype_from_hf(text_config, default=torch.float32)
        
        image_token_id = getattr(thinker_config, "image_token_id", None) or getattr(thinker_config, "image_token_index", 151655)
        video_token_id = getattr(thinker_config, "video_token_id", None) or getattr(thinker_config, "video_token_index", 151656)
        audio_token_id = getattr(thinker_config, "audio_token_id", None) or getattr(thinker_config, "audio_token_index", 151646)
        
        provider = Qwen25VLModelProvider(
            # Language model configuration from text_config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rope_theta,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            generation_config=hf_pretrained.generation_config,
            add_qkv_bias=True,
            # mRoPE configuration
            mrope_section=mrope_section if mrope_section is not None else [16, 24, 24],

            # Vision configuration from thinker_config
            vision_config=thinker_config.vision_config,
            # VL-specific token IDs from thinker_config
            bos_token_id=getattr(text_config, "bos_token_id", 151643),
            eos_token_id=getattr(text_config, "eos_token_id", 151645),
            vision_start_token_id=getattr(thinker_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(thinker_config, "vision_end_token_id", 151653),
            vision_token_id=getattr(thinker_config, "vision_token_id", 151654),
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format.
        """
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            # Direct mappings
            "language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
            "language_model.output_layer.weight": "thinker.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
            
            # Layer-specific direct mappings
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "thinker.model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "thinker.model.layers.*.mlp.down_proj.weight",
            
            # Bias mappings (if add_bias_linear=True in config)
            "language_model.decoder.layers.*.self_attention.linear_proj.bias": "thinker.model.layers.*.self_attn.o_proj.bias",
            "language_model.decoder.layers.*.mlp.linear_fc2.bias": "thinker.model.layers.*.mlp.down_proj.bias",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="visual.**",
                    hf_param="thinker.visual.**",
                ),
                ReplicatedMapping(
                    megatron_param="audio.**",
                    hf_param="thinker.audio_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="talker.**",
                    hf_param="talker.**",
                ),
                ReplicatedMapping(
                    megatron_param="token2wav.**",
                    hf_param="token2wav.**",
                ),
                # Based on ms-swift: concatenate [Q, K, V] along head dimension
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="thinker.model.layers.*.self_attn.q_proj.weight",
                    k="thinker.model.layers.*.self_attn.k_proj.weight",
                    v="thinker.model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias: Combine separate Q, K, V biases into single QKV bias
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="thinker.model.layers.*.self_attn.q_proj.bias",
                    k="thinker.model.layers.*.self_attn.k_proj.bias",
                    v="thinker.model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Stack gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.up_proj.weight",
                ),
                # Gated MLP bias: Stack gate and up biases into single FC1 bias
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.bias",
                    gate="thinker.model.layers.*.mlp.gate_proj.bias",
                    up="thinker.model.layers.*.mlp.up_proj.bias",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)