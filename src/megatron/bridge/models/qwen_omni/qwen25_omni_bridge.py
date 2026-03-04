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
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.model import Qwen25OmniModel
from megatron.bridge.models.qwen_omni.qwen25_omni_provider import Qwen25OmniModelProvider


@MegatronModelBridge.register_bridge(source=Qwen2_5OmniForConditionalGeneration, target=Qwen25OmniModel)
class Qwen25OmniBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen2.5-Omni Conditional Generation.

    Handles conversion between HuggingFace Qwen2_5OmniForConditionalGeneration
    and Megatron-Core Qwen25OmniModel formats.

    Key differences from Qwen3OmniMoeBridge:
    - Dense LLM (Qwen2), not MoE -> no router/expert mappings
    - QKV bias mappings (Qwen2 has attention bias)
    - No QK layernorm weight mappings
    - Vision: ReplicatedMapping for HF vision encoder (thinker.visual.**)
    - Audio: ReplicatedMapping for HF audio encoder (thinker.audio_model.** -> thinker.audio_tower.**)
    - LLM layer norms use mlp.linear_fc1.layer_norm_weight (not pre_mlp_layernorm)
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen25OmniModelProvider:
        """Create a Qwen25OmniModelProvider from a HuggingFace pretrained model."""
        hf_config = hf_pretrained.config
        thinker_config = hf_config.thinker_config
        talker_config = hf_config.talker_config
        token2wav_config = hf_config.token2wav_config
        text_config = thinker_config.text_config
        model_dtype = self.dtype_from_hf(thinker_config, default=torch.float32)

        provider = Qwen25OmniModelProvider(
            thinker_config=thinker_config,
            talker_config=talker_config,
            token2wav_config=token2wav_config,
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            head_dim=getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=getattr(text_config, "rope_theta", 1000000),
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            add_qkv_bias=True,  # Qwen2 always has QKV bias
            qk_layernorm=False,  # Qwen2 has no QK layernorm
            # Token IDs from thinker config
            image_token_id=getattr(thinker_config, "image_token_index", 151655),
            video_token_id=getattr(thinker_config, "video_token_index", 151656),
            audio_token_id=getattr(thinker_config, "audio_token_index", 151646),
            vision_start_token_id=getattr(thinker_config, "vision_start_token_id", 151652),
            audio_start_token_id=getattr(thinker_config, "audio_start_token_id", 151647),
            audio_end_token_id=getattr(thinker_config, "audio_end_token_id", 151648),
            mrope_section=getattr(text_config, "rope_scaling", {}).get("mrope_section", [16, 24, 24]),
            position_id_per_seconds=getattr(thinker_config, "position_id_per_seconds", 25),
            seconds_per_chunk=getattr(thinker_config, "seconds_per_chunk", 2),
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings for dense Qwen2.5 Omni models."""
        # LLM parameter mappings (same pattern as Qwen25VL bridge but prefixed with thinker.)
        param_mappings = {
            # Embeddings and output layers
            "thinker.language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
            "thinker.language_model.output_layer.weight": "thinker.lm_head.weight",
            "thinker.language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
            # Layer normalization
            "thinker.language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.model.layers.*.input_layernorm.weight",
            "thinker.language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "thinker.language_model.decoder.layers.*.self_attention.linear_proj.weight": "thinker.model.layers.*.self_attn.o_proj.weight",
            # MLP down projection
            "thinker.language_model.decoder.layers.*.mlp.linear_fc2.weight": "thinker.model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # Vision: ReplicatedMapping (HF vision encoder used directly)
                ReplicatedMapping(
                    megatron_param="thinker.visual.**",
                    hf_param="thinker.visual.**",
                ),
                # Audio: ReplicatedMapping (HF audio encoder used directly)
                # HF uses thinker.audio_tower, Megatron uses thinker.audio_model
                ReplicatedMapping(
                    megatron_param="thinker.audio_model.**",
                    hf_param="thinker.audio_tower.**",
                ),
                # QKV weight: Combine separate Q, K, V weights into single QKV matrix
                QKVMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="thinker.model.layers.*.self_attn.q_proj.weight",
                    k="thinker.model.layers.*.self_attn.k_proj.weight",
                    v="thinker.model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias: Combine separate Q, K, V biases into single QKV bias (Qwen2 has attention bias)
                QKVMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="thinker.model.layers.*.self_attn.q_proj.bias",
                    k="thinker.model.layers.*.self_attn.k_proj.bias",
                    v="thinker.model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
