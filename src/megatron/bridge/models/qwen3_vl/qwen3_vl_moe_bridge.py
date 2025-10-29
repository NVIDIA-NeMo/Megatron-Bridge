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
from transformers.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen3_vl.qwen3_vl_moe_provider import Qwen3VLMoEModelProvider
from .model import Qwen3VLModel  # Using same model for MoE


@MegatronModelBridge.register_bridge(source=Qwen3VLMoeForConditionalGeneration, target=Qwen3VLModel)
class Qwen3VLMoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-VL MoE (Mixture of Experts) Conditional Generation.

    This bridge handles the conversion between HuggingFace Qwen3VLMoeForConditionalGeneration
    and Megatron-Core Qwen3VL MoE model formats, including weight mappings and
    configuration translation for vision-language MoE models.

    The weight mappings handle:
    - Vision model weights (same as dense model)
    - Language model MoE layers with expert routing
    - Shared embeddings and output layers
    - QK layernorm specific to Qwen3 architecture

    This bridge works with any Qwen3VL MoE model size and automatically extracts
    the MoE configuration from the HuggingFace model.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3VLMoEModelProvider:
        """
        Create a Qwen3VLMoEModelProvider from a HuggingFace pretrained MoE model.

        Args:
            hf_pretrained: HuggingFace pretrained VLM MoE model

        Returns:
            Qwen3VLMoEModelProvider configured with the HF MoE model's parameters
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        # Create the generic MoE provider with configuration from the HF model
        provider = Qwen3VLMoEModelProvider(
            # Language model configuration from text_config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,  # Dense FFN size (for non-MoE layers if any)
            moe_ffn_hidden_size=text_config.moe_intermediate_size,  # Expert FFN size
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,  # GQA configuration
            head_dim=getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,  # Qwen3 MoE uses gated linear units
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=getattr(text_config, "rope_theta", 5000000.0),  # Default Qwen3 rope theta
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(text_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(text_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(text_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            # Qwen3 specific parameters
            add_qkv_bias=text_config.attention_bias,  # Qwen3 can have bias in QKV
            qk_layernorm=True,  # Qwen3 uses QK layernorm
            # MoE specific parameters
            num_moe_experts=text_config.num_experts,
            moe_router_topk=text_config.num_experts_per_tok,
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),  # Default to every layer being MoE
            mlp_only_layers=getattr(text_config, "mlp_only_layers", []),  # Default to all layers using MoE
            # Vision configuration
            vision_config=hf_config.vision_config,
            # Vision-Language token IDs
            bos_token_id=getattr(text_config, "bos_token_id", 151643),
            eos_token_id=getattr(text_config, "eos_token_id", 151645),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 151653),
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            video_token_id=getattr(hf_config, "video_token_id", 151656),
            # MRoPE configuration for multimodal position embeddings
            mrope_section=getattr(text_config, "rope_scaling", {}).get("mrope_section", [24, 20, 20]),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for MoE models.
        
        The MoE mappings include:
        1. Standard language model mappings (embeddings, layer norms, output)
        2. Vision model mappings (same as dense model)
        3. QKV mappings with QK layernorm
        4. MoE-specific mappings:
           - Router weights for expert selection
           - Expert MLPs (multiple experts per layer)
           - Pre-MLP layernorm
        5. Deepstack visual merger mappings

        Returns:
            MegatronMappingRegistry with all MoE parameter mappings
        """
        # Language model direct mappings (same as dense model)
        param_mappings = {
            # Embeddings and output layers
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            
            # Layer normalization for attention
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            
            # MoE-specific: pre-MLP layernorm
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            
            # QK layernorm weights (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            
            # MoE router weights
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
        }

        mapping_list = []
        
        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend([
            # Vision model weights are replicated directly
            ReplicatedMapping(
                megatron_param="vision_model.**",
                hf_param="model.visual.**",
            ),
            
            # QKV mapping: Combine separate Q, K, V matrices
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            ),
            
            # QKV bias mapping (if attention_bias is True)
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                q="model.language_model.layers.*.self_attn.q_proj.bias",
                k="model.language_model.layers.*.self_attn.k_proj.bias",
                v="model.language_model.layers.*.self_attn.v_proj.bias",
            ),
            
            # MoE expert weights mapping
            # Note: Expert weights are handled specially in yan-mbridge
            # Each expert has gate_up_proj and down_proj
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight",
                hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight",
                hf_param="model.language_model.layers.*.mlp.experts.down_proj",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)
