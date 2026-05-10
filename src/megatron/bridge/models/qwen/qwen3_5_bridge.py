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

import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Qwen3_5ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (  # noqa: F401
    AutoMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMappingSeparate,
    QKVMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf


@MegatronModelBridge.register_bridge(source=Qwen3_5ForCausalLM, target=GPTModel, model_type="qwen3_5_text")
class Qwen3_5Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3.5 Dense Language Model.

    This bridge handles the conversion between HuggingFace Qwen3.5 language 
    model and Megatron-Core Qwen3.5 Model formats, including weight mappings and
    configuration translation for the hybrid GDN+Attention LM architecture.

    The weight mappings handle:
    - Language model hybrid layers (GDN + standard attention)
    - Dense MLP with gated SiLU activation (fused pre-MLP layernorm)
    - QK layernorm, zero-centered RMSNorm for GDN output norm

    Architecture (27B): 16 × (3 × GDN + 1 × Attention) = 64 layers

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B")
        >>> model.save_pretrained("./Qwen3.5-27B-LM")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
        >>> tokenizer.save_pretrained("./Qwen3.5-27B-LM")
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("./Qwen3.5-27B-LM")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained):
        """Convert HuggingFace Qwen3.5 text model config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Standard GPT settings (shared with Qwen3-Next)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = getattr(hf_config, "attention_bias", False)
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True
        provider.autocast_dtype = torch.bfloat16
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        provider.rotary_percent = getattr(hf_config, "rope_parameters", {}).get("partial_rotary_factor", 0.25)
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 248045)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 248046)

        # Qwen3.5: zero-centered RMSNorm and gated attention
        provider.layernorm_zero_centered_gamma = True
        provider.attention_output_gate = True

        # Qwen3.5: hybrid gated delta net + standard attention
        provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
        provider.experimental_attention_variant = "gated_delta_net"
        # full_attention_interval defines how often standard attention appears:
        # e.g., 4 means every 4th layer is standard attention (3 GDN + 1 Attn)
        provider.linear_attention_freq = full_attention_interval_from_hf(hf_config)
        provider.linear_conv_kernel_dim = getattr(hf_config, "linear_conv_kernel_dim", 4)
        provider.linear_key_head_dim = getattr(hf_config, "linear_key_head_dim", 128)
        provider.linear_value_head_dim = getattr(hf_config, "linear_value_head_dim", 128)
        provider.linear_num_key_heads = getattr(hf_config, "linear_num_key_heads", 16)
        provider.linear_num_value_heads = getattr(hf_config, "linear_num_value_heads", 48)

        # Heterogeneous checkpointing for mixed attention layers
        provider.hetereogenous_dist_checkpoint = True

        # MTP (Multi-Token Prediction)
        if provider.mtp_num_layers:
            provider.mtp_loss_scaling_factor = 0.1

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            # Embedding and output
            "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # MLP
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            # Standard attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # Linear attention
            "decoder.layers.*.self_attention.in_proj.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.out_proj.weight": "model.language_model.layers.*.linear_attn.out_proj.weight",
            "decoder.layers.*.self_attention.A_log": "model.language_model.layers.*.linear_attn.A_log",
            "decoder.layers.*.self_attention.dt_bias": "model.language_model.layers.*.linear_attn.dt_bias",
            # MTP projection and norms
            "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
            # MTP MLP
            "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight": "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight": "mtp.layers.0.mlp.down_proj.weight",
            # MTP standard attention
            "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                # Note: Qwen3.5 does NOT have bias in QKV projections
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
                    q="mtp.layers.*.self_attn.q_proj.weight",
                    k="mtp.layers.*.self_attn.k_proj.weight",
                    v="mtp.layers.*.self_attn.v_proj.weight",
                ),
                # GDNLinear: Combine separate QKVZ_proj and BA_proj into single in_proj for GDN
                # Note: Qwen3.5 does NOT have bias in the input linear projections
                GDNConv1dMapping(
                    megatron_param="decoder.layers.*.self_attention.conv1d.weight",
                    hf_param="model.language_model.layers.*.linear_attn.conv1d.weight",
                ),
                GDNLinearMappingSeparate(
                    megatron_param="decoder.layers.*.self_attention.in_proj.weight",
                    qkv="model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
                    z="model.language_model.layers.*.linear_attn.in_proj_z.weight",
                    b="model.language_model.layers.*.linear_attn.in_proj_b.weight",
                    a="model.language_model.layers.*.linear_attn.in_proj_a.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight",
                    gate="mtp.layers.0.mlp.gate_proj.weight",
                    up="mtp.layers.0.mlp.up_proj.weight",
                ),
                # Qwen 3.5 implements the output norm as a standard RMSNorm while initializing weight to ones,
                # while other norms are regular zero-centered RMSNorms.
                # To correctly load the output norm weight, we need to subtract 1 from it.
                RMSNorm2ZeroCenteredRMSNormMapping(
                    "decoder.layers.*.self_attention.out_norm.weight",
                    "model.language_model.layers.*.linear_attn.norm.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
