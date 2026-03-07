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
from transformers import Qwen3OmniMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniMoeModel
from megatron.bridge.models.qwen_omni.qwen3_omni_provider import Qwen3OmniMoeModelProvider


@MegatronModelBridge.register_bridge(source=Qwen3OmniMoeForConditionalGeneration, target=Qwen3OmniMoeModel)
class Qwen3OmniMoeBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-Omni-Moe Conditional Generation.

    This bridge handles the conversion between HuggingFace Qwen3OmniMoeForConditionalGeneration
    and Megatron-Core Qwen3OmniMoeModel formats, including weight mappings and
    configuration translation for Omni Moe models.

    The weight mappings handle:
    1. Standard language model mappings (embeddings, layer norms, output)
    2. Vision model mappings
    3. QKV mappings with QK layernorm
    4. MoE-specific mappings:
        - Router weights for expert selection
        - Expert MLPs (multiple experts per layer)
        - Pre-MLP layernorm
    5. Deepstack visual merger mappings
    6. Audio model mappings

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3OmniMoeModelProvider:
        """
        Create a Qwen3OmniMoeModelProvider from a HuggingFace pretrained model.

        Args:
            hf_pretrained: HuggingFace pretrained VLM model

        Returns:
            Qwen3OmniMoeModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config
        thinker_config = hf_config.thinker_config
        talker_config = hf_config.talker_config
        code2wav_config = hf_config.code2wav_config

        text_config = thinker_config.text_config
        model_dtype = self.dtype_from_hf(thinker_config, default=torch.float32)

        provider = Qwen3OmniMoeModelProvider(
            thinker_config=thinker_config,
            talker_config=talker_config,
            code2wav_config=code2wav_config,
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
            rotary_base=getattr(text_config, "rope_theta", 1000000),  # Default Qwen3 omni rope theta
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # Qwen3 specific parameters
            add_qkv_bias=text_config.attention_bias,  # Qwen3 can have bias in QKV
            qk_layernorm=True,  # Qwen3 uses QK layernorm
            # MoE specific parameters
            num_moe_experts=text_config.num_experts,
            moe_router_topk=text_config.num_experts_per_tok,
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),  # Default to every layer being MoE
            mlp_only_layers=getattr(text_config, "mlp_only_layers", []),  # Default to all layers using MoE
            # Store the original HF text config for RoPE initialization
            hf_text_config=text_config,
            # Vision-Language token IDs
            bos_token_id=getattr(text_config, "bos_token_id", 151643),
            eos_token_id=getattr(text_config, "eos_token_id", 151645),
            vision_start_token_id=getattr(thinker_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(thinker_config, "vision_end_token_id", 151653),
            audio_start_token_id=getattr(thinker_config, "audio_start_token_id", 151669),
            audio_end_token_id=getattr(thinker_config, "audio_end_token_id", 151670),
            image_token_id=getattr(thinker_config, "image_token_id", 151655),
            video_token_id=getattr(thinker_config, "video_token_id", 151656),
            audio_token_id=getattr(thinker_config, "audio_token_id", 151675),
            tts_bos_token_id=getattr(hf_config, "tts_bos_token_id", 151672),
            tts_eos_token_id=getattr(hf_config, "tts_eos_token_id", 151673),
            tts_pad_token_id=getattr(hf_config, "tts_pad_token_id", 151671),
            # MRoPE configuration for multimodal position embeddings
            mrope_section=getattr(text_config, "rope_scaling", {}).get("mrope_section", [24, 20, 20]),
            position_id_per_seconds=getattr(thinker_config, "position_id_per_seconds", 13),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for MoE models.

        The MoE mappings include:
        1. Standard language model mappings (embeddings, layer norms, output)
        2. Vision model mappings
        3. QKV mappings with QK layernorm
        4. MoE-specific mappings:
           - Router weights for expert selection
           - Expert MLPs (multiple experts per layer)
           - Pre-MLP layernorm
        5. Deepstack visual merger mappings
        6. Audio model mappings

        Returns:
            MegatronMappingRegistry with all MoE parameter mappings
        """
        # Language model direct mappings (same as dense model)
        param_mappings = {
            # Embeddings and output layers
            "thinker.language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
            "thinker.language_model.output_layer.weight": "thinker.lm_head.weight",
            "thinker.language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
            # Layer normalization for attention
            "thinker.language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.model.layers.*.input_layernorm.weight",
            # MoE-specific: pre-MLP layernorm
            "thinker.language_model.decoder.layers.*.pre_mlp_layernorm.weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "thinker.language_model.decoder.layers.*.self_attention.linear_proj.weight": "thinker.model.layers.*.self_attn.o_proj.weight",
            # QK layernorm weights (Qwen3 specific)
            "thinker.language_model.decoder.layers.*.self_attention.q_layernorm.weight": "thinker.model.layers.*.self_attn.q_norm.weight",
            "thinker.language_model.decoder.layers.*.self_attention.k_layernorm.weight": "thinker.model.layers.*.self_attn.k_norm.weight",
            # MoE router weights
            "thinker.language_model.decoder.layers.*.mlp.router.weight": "thinker.model.layers.*.mlp.gate.weight",
            # MLP output projection
            "thinker.language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*": "thinker.model.layers.*.mlp.experts.*.down_proj.weight",
            # vision module attn
            "thinker.vision_model.decoder.layers.*.self_attention.linear_proj.weight": "thinker.visual.blocks.*.attn.proj.weight",
            "thinker.vision_model.decoder.layers.*.self_attention.linear_proj.bias": "thinker.visual.blocks.*.attn.proj.bias",
            # vision module mlp
            "thinker.vision_model.decoder.layers.*.mlp.linear_fc1.weight": "thinker.visual.blocks.*.mlp.linear_fc1.weight",
            "thinker.vision_model.decoder.layers.*.mlp.linear_fc1.bias": "thinker.visual.blocks.*.mlp.linear_fc1.bias",
            "thinker.vision_model.decoder.layers.*.mlp.linear_fc2.weight": "thinker.visual.blocks.*.mlp.linear_fc2.weight",
            "thinker.vision_model.decoder.layers.*.mlp.linear_fc2.bias": "thinker.visual.blocks.*.mlp.linear_fc2.bias",
            # vision module norm
            "thinker.vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.visual.blocks.*.norm1.weight",
            "thinker.vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "thinker.visual.blocks.*.norm1.bias",
            "thinker.vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "thinker.visual.blocks.*.norm2.weight",
            "thinker.vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "thinker.visual.blocks.*.norm2.bias",
            # # vision module deepstack merger
            "thinker.vision_model.decoder.deepstack_merger_list.*.patch_norm.weight": "thinker.visual.merger_list.*.ln_q.weight",
            "thinker.vision_model.decoder.deepstack_merger_list.*.patch_norm.bias": "thinker.visual.merger_list.*.ln_q.bias",
            "thinker.vision_model.decoder.deepstack_merger_list.*.linear_fc1.weight": "thinker.visual.merger_list.*.mlp.0.weight",
            "thinker.vision_model.decoder.deepstack_merger_list.*.linear_fc1.bias": "thinker.visual.merger_list.*.mlp.0.bias",
            "thinker.vision_model.decoder.deepstack_merger_list.*.linear_fc2.weight": "thinker.visual.merger_list.*.mlp.2.weight",
            "thinker.vision_model.decoder.deepstack_merger_list.*.linear_fc2.bias": "thinker.visual.merger_list.*.mlp.2.bias",
            # vision module merger
            "thinker.vision_model.merger.patch_norm.**": "thinker.visual.merger.ln_q.**",
            "thinker.vision_model.merger.linear_fc1.weight": "thinker.visual.merger.mlp.0.weight",
            "thinker.vision_model.merger.linear_fc1.bias": "thinker.visual.merger.mlp.0.bias",
            "thinker.vision_model.merger.linear_fc2.weight": "thinker.visual.merger.mlp.2.weight",
            "thinker.vision_model.merger.linear_fc2.bias": "thinker.visual.merger.mlp.2.bias",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # Audio model weights are replicated directly
                ReplicatedMapping(
                    megatron_param="thinker.audio_model.**",
                    hf_param="thinker.audio_tower.**",
                ),
                # QKV mapping: Combine separate Q, K, V matrices
                QKVMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="thinker.model.layers.*.self_attn.q_proj.weight",
                    k="thinker.model.layers.*.self_attn.k_proj.weight",
                    v="thinker.model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias mapping (if attention_bias is True)
                QKVMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="thinker.model.layers.*.self_attn.q_proj.bias",
                    k="thinker.model.layers.*.self_attn.k_proj.bias",
                    v="thinker.model.layers.*.self_attn.v_proj.bias",
                ),
                GatedMLPMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="thinker.model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                # QKV mapping for vision model
                ConcatenatedQKVMapping(
                    megatron_param="thinker.vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="thinker.visual.blocks.*.attn.qkv.weight",
                ),
                ConcatenatedQKVMapping(
                    megatron_param="thinker.vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    hf_param="thinker.visual.blocks.*.attn.qkv.bias",
                ),
                ReplicatedMapping(  # These patch_embed are conv, we need to use ReplicatedMapping
                    megatron_param="thinker.vision_model.patch_embed.proj.**",
                    hf_param="thinker.visual.patch_embed.proj.**",
                ),
                ReplicatedMapping(
                    megatron_param="thinker.vision_model.pos_embed.weight",
                    hf_param="thinker.visual.pos_embed.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
