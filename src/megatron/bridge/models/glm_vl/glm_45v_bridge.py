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

from typing import Dict

import torch
from megatron.core import parallel_state
from transformers import Glm4vMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.glm.glm_moe_mappings import (
    GLMExpertDownProjMapping,
    GLMExpertGateUpProjMapping,
)
from megatron.bridge.models.glm_vl.glm_45v_provider import GLM45VModelProvider
from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


@MegatronModelBridge.register_bridge(source=Glm4vMoeForConditionalGeneration, target=GLM45VModel)
class GLM45VBridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM 4.5 Vision-Language (VL) Models.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> GLM45VModelProvider:
        hf_config = hf_pretrained.config

        # GLM 4.5 has separate text_config and vision_config
        text_config = getattr(hf_config, "text_config", hf_config)
        moe_layer_freq = [0] * text_config.first_k_dense_replace + [1] * (
            text_config.num_hidden_layers - text_config.first_k_dense_replace
        )

        provider = GLM45VModelProvider(
            add_qkv_bias=text_config.attention_bias,
            kv_channels=text_config.head_dim,
            hidden_size=text_config.hidden_size,
            rotary_base=rope_theta_from_hf(text_config),
            rotary_percent=text_config.partial_rotary_factor,
            init_method_std=text_config.initializer_range,
            ffn_hidden_size=text_config.intermediate_size,
            seq_length=text_config.max_position_embeddings,
            moe_ffn_hidden_size=text_config.moe_intermediate_size,
            # norm topk prob
            num_attention_heads=text_config.num_attention_heads,
            # n group, topk group
            num_moe_experts=text_config.n_routed_experts,
            # n shared expert
            moe_shared_expert_intermediate_size=text_config.moe_intermediate_size,
            moe_router_topk_scaling_factor=text_config.routed_scaling_factor,
            moe_router_topk=text_config.num_experts_per_tok,
            moe_layer_freq=moe_layer_freq,
            num_layers=text_config.num_hidden_layers,
            num_query_groups=text_config.num_key_value_heads,
            layernorm_epsilon=text_config.rms_norm_eps,
            mtp_num_layers=0,  # No MTP for VL models
            qk_layernorm=text_config.use_qk_norm,
            vocab_size=text_config.vocab_size,
            fp16=(self.dtype_from_hf(text_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(text_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(text_config, default=torch.float32),
            vision_config=hf_config.vision_config,
            # VL-specific token IDs
            eos_token_id=getattr(text_config, "eos_token_id", 151329),
            image_start_token_id=getattr(text_config, "image_start_token_id", 151339),
            image_end_token_id=getattr(text_config, "image_end_token_id", 151340),
            video_start_token_id=getattr(text_config, "video_start_token_id", 151341),
            video_end_token_id=getattr(text_config, "video_end_token_id", 151342),
            image_token_id=getattr(text_config, "image_token_id", 151363),
            video_token_id=getattr(text_config, "video_token_id", 151364),
        )
        return provider

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Override to store config before mapping_registry is called."""
        self._hf_config = hf_pretrained.config
        self._hf_state_source = hf_pretrained.state.source
        self._hf_keys = list(self._hf_state_source.get_all_keys())
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

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
                    GLMExpertGateUpProjMapping(
                        megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param=f"model.language_model.layers.*.mlp.experts.gate_up_proj{gate_up_suffix}",
                    ),
                    GLMExpertDownProjMapping(
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

    def _uses_fused_experts(self) -> bool:
        hf_keys = getattr(self, "_hf_keys", None)
        if hf_keys:
            if any("mlp.experts.gate_up_proj" in key for key in hf_keys) or any(
                "mlp.experts.down_proj" in key for key in hf_keys
            ):
                return True

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None:
            return hf_source.has_glob("*mlp.experts.gate_up_proj*") or hf_source.has_glob("*mlp.experts.down_proj*")

        return False

    def _hf_expert_suffix(self, base_name: str) -> str:
        hf_keys = getattr(self, "_hf_keys", None) or []
        if any(f"{base_name}.weight" in key for key in hf_keys):
            return ".weight"

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None and hf_source.has_glob(f"*{base_name}.weight"):
            return ".weight"

        return ""

    def maybe_modify_converted_hf_weight(
        self,
        task,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(task.mapping, (GLMExpertGateUpProjMapping, GLMExpertDownProjMapping)):
            return converted_weights_dict

        if not converted_weights_dict:
            return {}

        text_config = getattr(self._hf_config, "text_config", self._hf_config)
        num_experts = text_config.n_routed_experts
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = extract_expert_number_from_param(task.param_name) % experts_per_rank
        except ValueError:
            return converted_weights_dict

        if not hasattr(self, "hf_weights_cache"):
            self.hf_weights_cache = {}

        for key, value in converted_weights_dict.items():
            if key not in self.hf_weights_cache:
                self.hf_weights_cache[key] = {}

            if ep_size == 1:
                self.hf_weights_cache[key][local_expert_number] = value
            else:
                if value.shape[0] != ep_size:
                    raise ValueError(f"Expected EP dim {ep_size} for {key}, got {value.shape}.")
                for i, exp_val in enumerate(value):
                    global_expert_number = local_expert_number + (i * experts_per_rank)
                    self.hf_weights_cache[key][global_expert_number] = exp_val

            if len(self.hf_weights_cache[key]) == num_experts:
                merged = torch.stack([self.hf_weights_cache[key][i] for i in range(num_experts)], dim=0)
                if key in hf_state_dict:
                    expected = hf_state_dict[key].shape
                    if merged.shape != expected and merged.transpose(-1, -2).shape == expected:
                        merged = merged.transpose(-1, -2).contiguous()
                del self.hf_weights_cache[key]
                return {key: merged}

            return {}

        return {}
