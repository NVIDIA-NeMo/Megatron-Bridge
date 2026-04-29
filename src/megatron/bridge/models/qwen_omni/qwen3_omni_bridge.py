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

import logging

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniModel
from megatron.bridge.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(
    source=Qwen3OmniMoeForConditionalGeneration,
    target=Qwen3OmniModel,
    provider=Qwen3OmniModelProvider,
    model_type="qwen3_omni",
)
class Qwen3OmniBridge(MegatronModelBridge):
    """Bridge for Qwen3-Omni."""

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3OmniModelProvider:
        hf_config = hf_pretrained.config
        if getattr(hf_config, "enable_audio_output", False):
            logger.warning(
                "Qwen3-Omni talker/code2wav audio-output is not supported yet; "
                "converting thinker-side weights only. Talker/code2wav modules will be ignored."
            )
        thinker_config = hf_config.thinker_config
        talker_config = getattr(hf_config, "talker_config", None)
        code2wav_config = getattr(hf_config, "code2wav_config", None)
        text_config = thinker_config.text_config
        dtype_config = thinker_config if hasattr(thinker_config, "torch_dtype") else hf_config
        model_dtype = self.dtype_from_hf(dtype_config, default=torch.float32)

        rope_scaling = (
            getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None) or {}
        )
        vision_config = thinker_config.vision_config

        provider = Qwen3OmniModelProvider(
            thinker_config=thinker_config,
            talker_config=talker_config,
            code2wav_config=code2wav_config,
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            moe_ffn_hidden_size=getattr(text_config, "moe_intermediate_size", None),
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", None),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=getattr(text_config, "rope_theta", 1000000.0),
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            language_max_sequence_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            add_qkv_bias=getattr(text_config, "attention_bias", False),
            qk_layernorm=True,
            num_moe_experts=getattr(text_config, "num_experts", 128),
            moe_router_topk=getattr(text_config, "num_experts_per_tok", 8),
            image_token_id=getattr(thinker_config, "image_token_id", 151655),
            video_token_id=getattr(thinker_config, "video_token_id", 151656),
            audio_token_id=getattr(thinker_config, "audio_token_id", 151646),
            vision_start_token_id=getattr(thinker_config, "vision_start_token_id", 151652),
            audio_start_token_id=getattr(thinker_config, "audio_start_token_id", 151647),
            position_id_per_seconds=getattr(thinker_config, "position_id_per_seconds", 25),
            seconds_per_chunk=getattr(thinker_config, "seconds_per_chunk", 2),
            patch_size=getattr(vision_config, "patch_size", 16),
            temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
            spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
            position_embedding_type="mrope",
            mrope_section=rope_scaling.get("mrope_section", [24, 20, 20]),
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            "thinker.language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
            "thinker.language_model.output_layer.weight": "thinker.lm_head.weight",
            "thinker.language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
            "thinker.language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.model.layers.*.input_layernorm.weight",
            "thinker.language_model.decoder.layers.*.pre_mlp_layernorm.weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            "thinker.language_model.decoder.layers.*.self_attention.q_layernorm.weight": "thinker.model.layers.*.self_attn.q_norm.weight",
            "thinker.language_model.decoder.layers.*.self_attention.k_layernorm.weight": "thinker.model.layers.*.self_attn.k_norm.weight",
            "thinker.language_model.decoder.layers.*.self_attention.linear_proj.weight": "thinker.model.layers.*.self_attn.o_proj.weight",
            "thinker.language_model.decoder.layers.*.mlp.router.weight": "thinker.model.layers.*.mlp.gate.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="thinker.visual.**",
                    hf_param="thinker.visual.**",
                ),
                ReplicatedMapping(
                    megatron_param="thinker.audio_model.**",
                    hf_param="thinker.audio_tower.**",
                ),
                QKVMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="thinker.model.layers.*.self_attn.q_proj.weight",
                    k="thinker.model.layers.*.self_attn.k_proj.weight",
                    v="thinker.model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="thinker.model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="thinker.model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="thinker.language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="thinker.model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
