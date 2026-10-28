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

from typing import Any

from transformers import Glm4vMoeForConditionalGeneration, PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.glm.glm45_bridge import GLM45Bridge
from megatron.bridge.models.glm_vl.glm_45v_config import GLM45VModelConfig
from megatron.bridge.models.glm_vl.glm_45v_provider import GLM45VModelProvider
from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel


@MegatronModelBridge.register_bridge(
    source=Glm4vMoeForConditionalGeneration, target=GLM45VModel, provider=GLM45VModelProvider
)
class GLM45VBridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM 4.5 Vision-Language (VL) Models.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.5V")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = GLM45VModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face GLM-4.5V config to builder config kwargs.

        The language-model fields are the GLM-4.5 text bridge's, read from ``text_config``, with the
        values the vision-language provider has always used where they differ.
        """
        text_config = hf_config.text_config
        config_kwargs = GLM45Bridge().hf_config_to_model_config_kwargs(text_config)
        config_kwargs.update(
            position_embedding_type="mrope",
            mrope_section=[8, 12, 12],
            # The vision-language models have no MTP layers and route without expert groups.
            mtp_num_layers=0,
            moe_router_num_groups=None,
            moe_router_group_topk=None,
            vision_config=hf_config.vision_config,
            # As on the provider, the token IDs are read from text_config, falling back to the GLM-4.5V values.
            eos_token_id=getattr(text_config, "eos_token_id", 151329),
            image_start_token_id=getattr(text_config, "image_start_token_id", 151339),
            image_end_token_id=getattr(text_config, "image_end_token_id", 151340),
            video_start_token_id=getattr(text_config, "video_start_token_id", 151341),
            video_end_token_id=getattr(text_config, "video_end_token_id", 151342),
            image_token_id=getattr(text_config, "image_token_id", 151363),
            video_token_id=getattr(text_config, "video_token_id", 151364),
        )
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        return self.hf_config_to_model_config_kwargs(hf_config)

    @classmethod
    def get_hf_tokenizer_kwargs(cls) -> dict:
        """Return HuggingFace tokenizer kwargs specific to GLM 4.5V models.

        GLM 4.5V requires use_fast=True to properly load the tokenizer.
        """
        return {"use_fast": True}

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        mapping_list = []
        use_fused_experts = self._uses_fused_experts()
        gate_up_suffix = self._hf_expert_suffix("mlp.experts.gate_up_proj")
        down_suffix = self._hf_expert_suffix("mlp.experts.down_proj")

        param_mappings = {
            # Embed
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            # LM Head
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.output_layer.weight": "lm_head.weight",
        }

        layer_specific_mappings = {
            # Attention
            "language_model.decoder.layers.*.input_layernorm.weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
            #  In GLM, HF weight `model.language_model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `language_model.decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # MLP
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.language_model.layers.*.mlp.shared_experts.down_proj.weight",
            "language_model.decoder.layers.*.mlp.shared_experts.router.weight": "model.language_model.layers.*.mlp.shared_experts.gate.weight",
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
            "language_model.decoder.layers.*.mlp.router.expert_bias": "model.language_model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for megatron_param, hf_param in layer_specific_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="visual.**",
                    hf_param="model.visual.**",
                ),
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )
        if use_fused_experts:
            mapping_list.extend(
                [
                    FusedGatedExpertMapping(
                        megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param=f"model.language_model.layers.*.mlp.experts.gate_up_proj{gate_up_suffix}",
                    ),
                    FusedExpertMapping(
                        megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param=f"model.language_model.layers.*.mlp.experts.down_proj{down_suffix}",
                    ),
                ]
            )
        else:
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                        up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )
        return MegatronMappingRegistry(*mapping_list)

    def _hf_state_source(self):
        """Return HF state source when mappings can inspect checkpoint weights."""
        hf_state = getattr(self.hf_pretrained, "state", None)
        if hf_state is None:
            return None
        return getattr(hf_state, "source", None)

    def _uses_fused_experts(self) -> bool:
        hf_source = self._hf_state_source()
        return bool(
            hf_source is not None
            and (hf_source.has_glob("*mlp.experts.gate_up_proj*") or hf_source.has_glob("*mlp.experts.down_proj*"))
        )

    def _hf_expert_suffix(self, base_name: str) -> str:
        hf_source = self._hf_state_source()
        if hf_source is not None and hf_source.has_glob(f"*{base_name}.weight"):
            return ".weight"

        return ""
