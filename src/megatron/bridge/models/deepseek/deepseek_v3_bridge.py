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
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.conversion.utils import get_causal_lm_class_via_auto_map
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekV3Provider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(
    source=get_causal_lm_class_via_auto_map(model_name_or_path="deepseek-ai/DeepSeek-V3-Base"), target=GPTModel
)
@MegatronModelBridge.register_bridge(
    source=get_causal_lm_class_via_auto_map(model_name_or_path="deepseek-ai/DeepSeek-V3"), target=GPTModel
)
# @MegatronModelBridge.register_bridge(
#     source=get_causal_lm_class_via_auto_map(model_name_or_path="/lustre/fsw/portfolios/coreai/users/yifuw/hf_checkpoints/dsv3/DeepSeek-V3-BF16"), target=GPTModel
# )
class DeepSeekV3Bridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for DeepSeek-V3.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3-Base")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV3Provider:
        hf_config = hf_pretrained.config

        optional_kwargs = {}
        # Not all deepseek configs have aux_loss_alpha
        if hasattr(hf_config, "aux_loss_alpha"):
            optional_kwargs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        n_moe_layers = hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        provider = DeepSeekV3Provider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            kv_channels=hf_config.num_key_value_heads,
            q_lora_rank=hf_config.q_lora_rank,
            num_moe_experts=hf_config.n_routed_experts,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,  # Maps to moe_intermediate_size in HF
            moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
            moe_layer_freq=[0] * hf_config.first_k_dense_replace + [1] * n_moe_layers,
            moe_router_topk=hf_config.num_experts_per_tok,  # Maps to num_experts_per_tok in HF
            moe_router_num_groups=hf_config.n_group,
            moe_router_group_topk=hf_config.topk_group,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            kv_lora_rank=hf_config.kv_lora_rank,
            qk_head_dim=hf_config.qk_nope_head_dim,
            qk_pos_emb_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            vocab_size=hf_config.vocab_size,
            **optional_kwargs,
        )

        # provider.gradient_accumulation_fusion = False

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()

        param_mappings = {
            # MLA
            "model.layers.*.self_attn.q_a_proj.weight": "decoder.layers.*.self_attention.linear_q_down_proj.weight",
            "model.layers.*.self_attn.q_b_proj.weight": "decoder.layers.*.self_attention.linear_q_up_proj.weight",
            "model.layers.*.self_attn.q_a_layernorm.weight": "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight",

            # expert bias
            "model.layers.*.mlp.gate.e_score_correction_bias": "decoder.layers.*.mlp.router.expert_bias",
        }

        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        # Mcore local spec
        mapping_list.append(
            AutoMapping(
                hf_param="model.layers.*.self_attn.q_a_layernorm.weight",
                megatron_param="decoder.layers.*.self_attention.q_layernorm.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)
