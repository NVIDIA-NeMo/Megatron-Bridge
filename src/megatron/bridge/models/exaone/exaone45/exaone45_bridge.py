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
from transformers import Exaone4_5_ForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.exaone.exaone45.exaone45_provider import Exaone45ModelProvider
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.model import Exaone45Model
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")


@MegatronModelBridge.register_bridge(
    source=Exaone4_5_ForConditionalGeneration,
    target=Exaone45Model,
    provider=Exaone45ModelProvider,
    model_type="exaone4_5",
)
class Exaone45Bridge(MegatronModelBridge):
    """
    Megatron Bridge for EXAONE 4.5 conditional generation.

    This bridge handles the conversion between HuggingFace Exaone4_5_ForConditionalGeneration
    and Megatron-Core Exaone45Model formats, including weight mappings and
    configuration translation for vision-language models.

    The weight mappings are based on the yan-mbridge implementation which defines:
    - Vision model direct mappings
    - Vision attention layer mappings
    - Vision MLP layer mappings
    - Language model mappings

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("LGAI-EXAONE/EXAONE-4.5-33B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Exaone45ModelProvider:
        """
        Create a Exaone45ModelProvider from a HuggingFace pretrained model.

        Args:
            hf_pretrained: HuggingFace pretrained VLM model

        Returns:
            Exaone45ModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config
        self.hf_config = hf_config
        text_config = hf_config.text_config

        no_rope_freq, window_attn_skip_freq, window_size = None, None, None

        window_attn_skip_freq, no_rope_freq = [], []
        layer_types = getattr(text_config, "layer_types", None) or []
        has_sliding_attention = False
        for layer_idx in range(text_config.num_hidden_layers):
            layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "sliding_attention"
            is_sliding = layer_type == "sliding_attention"
            has_sliding_attention = has_sliding_attention or is_sliding
            no_rope_freq.append(0 if is_sliding else 1)
            window_attn_skip_freq.append(1 if is_sliding else 0)

        sliding_window = getattr(text_config, "sliding_window", None)
        if has_sliding_attention and sliding_window is not None:
            window_size = (sliding_window - 1, 0)

        rope_parameters = getattr(text_config, "rope_parameters")
        position_embedding_type = "rope"
        rope_theta = rope_parameters["rope_theta"]
        rope_scaling_factor = rope_parameters.get("factor")
        rope_scaling = rope_scaling_factor is not None

        # Get the model dtype from text config
        model_dtype = self.dtype_from_hf(text_config, default=torch.float32)
        moe_layer_freq = [0] * text_config.num_hidden_layers

        # Set vision config dtype to match the language model dtype
        # This ensures vision model parameters are initialized in the same dtype
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = model_dtype

        # Create the provider with text model configuration
        provider = Exaone45ModelProvider(
            # Language model configuration from text_config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,  # GQA configuration
            kv_channels=getattr(text_config, "head_dim", None),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            # VLM weight tying is defined on the top-level config.
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            add_qkv_bias=getattr(text_config, "attention_bias", False),
            # Hybrid attention config for EXAONE architecture
            no_rope_freq=no_rope_freq,
            window_attn_skip_freq=window_attn_skip_freq,
            window_size=window_size,
            qk_layernorm=True,
            # RoPE
            rotary_base=rope_theta,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            position_embedding_type=position_embedding_type,
            # Vision configuration
            vision_config=vision_config,
            # Store the original HF text config for RoPE initialization
            hf_text_config=text_config,
            # VL-specific token IDs
            bos_token_id=getattr(hf_config, "bos_token_id", 1),
            eos_token_id=getattr(hf_config, "eos_token_id", 53),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 73),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 74),
            vision_token_id=getattr(hf_config, "vision_token_id", 67),
            image_token_id=getattr(hf_config, "image_token_id", 67),
            video_token_id=getattr(hf_config, "video_token_id", 68),
            moe_layer_freq=moe_layer_freq,
            # MTP
            mtp_num_layers=getattr(text_config, "num_nextn_predict_layers", None),
            mtp_loss_scaling_factor=getattr(text_config, "mtp_loss_scaling_factor", 0.1),
            mtp_use_repeated_layer=getattr(text_config, "mtp_share_layers", False),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format.

        The mappings are organized into:
        1. Simple 1:1 mappings for embeddings, layer norms, and output layers
        2. Vision model mappings (replicated without modification)
        3. QKV mappings that combine separate Q, K, V matrices
        4. Gated MLP mappings that combine gate and up projections

        Returns:
            MegatronMappingRegistry with all parameter mappings
        """
        # Dictionary maps Megatron parameter names -> HF parameter names
        # Based on yan-mbridge weight mappings in __init__.py

        # Language model direct mappings
        hf_config = getattr(self, "hf_config", None)
        text_config = getattr(hf_config, "text_config", hf_config)
        share_embeddings_and_output_weights = getattr(
            hf_config,
            "share_embeddings_and_output_weights",
            getattr(hf_config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)),
        )
        output_layer_hf_param = (
            "model.language_model.embed_tokens.weight" if share_embeddings_and_output_weights else "lm_head.weight"
        )
        mtp_num_layers = getattr(text_config, "num_nextn_predict_layers", 0) or 0
        add_qkv_bias = getattr(text_config, "attention_bias", False)

        param_mappings = {
            # Language
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.language_model.layers.*.post_attention_layernorm.weight"
            ),
            "language_model.decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.language_model.layers.*.post_feedforward_layernorm.weight"
            ),
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # vision module self-attention
            "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "model.visual.blocks.*.attn.proj.weight",
            "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "model.visual.blocks.*.attn.proj.bias",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.visual.blocks.*.norm1.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.visual.blocks.*.norm1.bias",
            # vision module mlp
            "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "model.visual.blocks.*.mlp.down_proj.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "model.visual.blocks.*.mlp.down_proj.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.visual.blocks.*.norm2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.visual.blocks.*.norm2.bias",
            # vision module merger
            "vision_model.merger.patch_norm.weight": "model.visual.merger.ln_q.weight",
            "vision_model.merger.linear_fc1.weight": "model.visual.merger.mlp.0.weight",
            "vision_model.merger.linear_fc1.bias": "model.visual.merger.mlp.0.bias",
            "vision_model.merger.linear_fc2.weight": "model.visual.merger.mlp.2.weight",
            "vision_model.merger.linear_fc2.bias": "model.visual.merger.mlp.2.bias",
        }

        output_layer_mappings = {
            "language_model.output_layer.weight": output_layer_hf_param,
        }

        mtp_param_mappings = {
            "language_model.mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "language_model.mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "language_model.mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "language_model.mtp.layers.*.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.*.self_attn.o_proj.weight",
            "language_model.mtp.layers.*.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.*.self_attn.q_norm.weight",
            "language_model.mtp.layers.*.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.*.self_attn.k_norm.weight",
            "language_model.mtp.layers.*.mtp_model_layer.mlp.linear_fc2.weight": "mtp.layers.*.mlp.down_proj.weight",
            "language_model.mtp.layers.*.mtp_model_layer.self_attention.linear_proj.post_layernorm.weight": (
                "mtp.layers.*.post_attention_layernorm.weight"
            ),
            "language_model.mtp.layers.*.mtp_model_layer.mlp.linear_fc2.post_layernorm.weight": (
                "mtp.layers.*.post_feedforward_layernorm.weight"
            ),
            "language_model.mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
        }

        mapping_list = []
        # Convert simple 1:1 mappings to AutoMapping objects
        param_mappings.update(output_layer_mappings)
        if hf_config is None or mtp_num_layers > 0:
            param_mappings.update(mtp_param_mappings)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # Language Setting
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                # Vision setting
                # Vision MLP
                GatedMLPMapping(
                    megatron_param="vision_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.visual.blocks.*.mlp.gate_proj.weight",
                    up="model.visual.blocks.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="vision_model.decoder.layers.*.mlp.linear_fc1.bias",
                    gate="model.visual.blocks.*.mlp.gate_proj.bias",
                    up="model.visual.blocks.*.mlp.up_proj.bias",
                ),
                # Vision Patch Embedding
                ReplicatedMapping(
                    megatron_param="vision_model.patch_embed.proj.**",
                    hf_param="model.visual.patch_embed.proj.**",
                ),
                ## Vision Self Attention
                ConcatenatedQKVMapping(
                    megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="model.visual.blocks.*.attn.qkv.weight",
                ),
                ConcatenatedQKVMapping(
                    megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    hf_param="model.visual.blocks.*.attn.qkv.bias",
                ),
            ]
        )

        if hf_config is None or add_qkv_bias:
            mapping_list.append(
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                )
            )

        if hf_config is None or mtp_num_layers > 0:
            mapping_list.extend(
                [
                    # MTP setting
                    # MTP QKV
                    QKVMapping(
                        megatron_param="language_model.mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
                        q="mtp.layers.*.self_attn.q_proj.weight",
                        k="mtp.layers.*.self_attn.k_proj.weight",
                        v="mtp.layers.*.self_attn.v_proj.weight",
                    ),
                    # MTP Gate
                    GatedMLPMapping(
                        megatron_param="language_model.mtp.layers.*.mtp_model_layer.mlp.linear_fc1.weight",
                        gate="mtp.layers.*.mlp.gate_proj.weight",
                        up="mtp.layers.*.mlp.up_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
