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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.model import Qwen3ASRModel
from megatron.bridge.models.qwen3_asr.qwen3_asr_provider import Qwen3ASRModelProvider


# Use string-based registration because Qwen3ASRForConditionalGeneration is not in
# the standard transformers library (it's a custom model in qwen_asr package).
# auto_bridge.py resolves custom architectures via config.auto_map or string fallback.
@MegatronModelBridge.register_bridge(source="Qwen3ASRForConditionalGeneration", target=Qwen3ASRModel)
class Qwen3ASRBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-ASR Conditional Generation.

    Handles conversion between HuggingFace Qwen3ASRForConditionalGeneration
    and Megatron-Core Qwen3ASRModel formats.

    Key differences from Qwen25OmniBridge:
    - Qwen3-based LLM (not Qwen2): no QKV bias, has QK layernorm
    - Audio-only: no vision/video mappings
    - Audio: ReplicatedMapping for HF audio encoder (thinker.audio_model.** -> thinker.audio_tower.**)
    - QK layernorm weight mappings (q_layernorm, k_layernorm)
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3ASRModelProvider:
        """Create a Qwen3ASRModelProvider from a HuggingFace pretrained model."""
        hf_config = hf_pretrained.config
        thinker_config = hf_config.thinker_config
        text_config = thinker_config.text_config
        model_dtype = self.dtype_from_hf(thinker_config, default=torch.float32)

        provider = Qwen3ASRModelProvider(
            thinker_config=thinker_config,
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=getattr(text_config, "rope_theta", 5000000.0),
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            add_qkv_bias=False,  # Qwen3 has no QKV bias
            qk_layernorm=True,  # Qwen3 has QK layernorm
            # Token IDs from thinker config
            audio_token_id=getattr(thinker_config, "audio_token_id", 151646),
            audio_start_token_id=getattr(thinker_config, "audio_start_token_id", 151647),
            mrope_section=(getattr(text_config, "rope_scaling", None) or {}).get("mrope_section", [24, 20, 20]),
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings for Qwen3-ASR models."""
        # LLM parameter mappings (Qwen3-style with QK layernorm, prefixed with thinker.)
        param_mappings = {
            # Embeddings and output layers
            "thinker.language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
            "thinker.language_model.output_layer.weight": "thinker.lm_head.weight",
            "thinker.language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
            # Layer normalization
            "thinker.language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.model.layers.*.input_layernorm.weight",
            "thinker.language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            # QK layernorm (Qwen3-specific)
            "thinker.language_model.decoder.layers.*.self_attention.q_layernorm.weight": "thinker.model.layers.*.self_attn.q_norm.weight",
            "thinker.language_model.decoder.layers.*.self_attention.k_layernorm.weight": "thinker.model.layers.*.self_attn.k_norm.weight",
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
                # Audio: ReplicatedMapping (HF audio encoder used directly)
                # HF uses thinker.audio_tower, Megatron uses thinker.audio_model
                ReplicatedMapping(
                    megatron_param="thinker.audio_model.**",
                    hf_param="thinker.audio_tower.**",
                ),
                # QKV weight: Combine separate Q, K, V weights into single QKV matrix (no bias for Qwen3)
                QKVMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="thinker.model.layers.*.self_attn.q_proj.weight",
                    k="thinker.model.layers.*.self_attn.k_proj.weight",
                    v="thinker.model.layers.*.self_attn.v_proj.weight",
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
