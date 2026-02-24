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
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.sarvam.common import get_common_config
from megatron.bridge.models.sarvam.sarvam_provider import SarvamMLAModelProvider


@MegatronModelBridge.register_bridge(source="SarvamMLAForCausalLM", target=GPTModel)
class SarvamMLABridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Sarvam MLA Causal LM.

    This bridge handles the conversion between HuggingFace SarvamMLAForCausalLM
    and Megatron-Core GPTModel formats. Sarvam MLA models use multi-latent attention
    architecture.
    """

    def provider_bridge(
        self, hf_pretrained: PreTrainedCausalLM
    ) -> SarvamMLAModelProvider:
        hf_config = hf_pretrained.config
        config = get_common_config(hf_pretrained)

        config["fp16"] = (
            self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        )
        config["bf16"] = (
            self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        )
        config["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)
        config["kv_channels"] = hf_config.hidden_size // hf_config.num_attention_heads

        # MLA
        config["kv_lora_rank"] = hf_config.kv_lora_rank
        config["qk_head_dim"] = hf_config.qk_nope_head_dim
        config["qk_pos_emb_head_dim"] = hf_config.qk_rope_head_dim
        config["v_head_dim"] = hf_config.v_head_dim

        return SarvamMLAModelProvider(**config)

    def mapping_registry(self) -> MegatronMappingRegistry:

        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            #  In sarvam, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # Mcore local spec
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # Moe
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(
                AutoMapping(hf_param=hf_param, megatron_param=megatron_param)
            )

        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
