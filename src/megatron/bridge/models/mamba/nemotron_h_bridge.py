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
from typing import Dict, Optional, Tuple
from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.nemotron_h_provider import NemotronHModelProvider

@MegatronModelBridge.register_bridge(source="NemotronHForCausalLM", target=MambaModel)
class NemotronHBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Nemotron-H.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/nemotron-h-4b", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def __init__(self):
        super().__init__()
        self.hf_weights_cache = {}
        self.megatron_weights_cache = {}

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> NemotronHModelProvider:
        hf_config = hf_pretrained.config

        configs = {
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "ffn_hidden_size": hf_config.intermediate_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "kv_channels": hf_config.attention_head_dim,
            "vocab_size": hf_config.vocab_size,
            "rotary_base": getattr(hf_config, "rope_theta", 10000.0),
            "layernorm_epsilon": hf_config.layer_norm_epsilon,
            "hybrid_override_pattern": hf_config.hybrid_override_pattern,
            "mamba_num_heads": hf_config.mamba_num_heads,
            "mamba_num_groups": hf_config.n_groups,
            "mamba_head_dim": hf_config.mamba_head_dim,
            "mamba_state_dim": hf_config.ssm_state_size,
        }

        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        # MoE configurations
        if hasattr(hf_config, "n_routed_experts") and hf_config.n_routed_experts > 0:
            configs.update({
                "num_moe_experts": hf_config.n_routed_experts,
                "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
                "moe_shared_expert_intermediate_size": hf_config.moe_shared_expert_intermediate_size,
                "moe_router_topk": hf_config.num_experts_per_tok,
                "moe_router_num_groups": hf_config.n_group,
                "moe_router_group_topk": hf_config.topk_group,
                "moe_router_topk_scaling_factor": hf_config.routed_scaling_factor,
                "moe_router_score_function": "sigmoid",
                "moe_router_enable_expert_bias": True,
            })

        provider = NemotronHModelProvider(**configs)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        # Basic parameter mappings from decoder (Megatron) to backbone (HF)
        param_mappings = {
            # Embedding
            "embedding.word_embeddings.weight": "backbone.embeddings.weight",

            # Attention layers
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "backbone.layers.*.mixer.o_proj.weight",

            # MoE layers
            "decoder.layers.*.mlp.router.weight": "backbone.layers.*.mixer.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "backbone.layers.*.mixer.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.experts.linear_fc1.weight*": "backbone.layers.*.mixer.experts.*.up_proj.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "backbone.layers.*.mixer.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "backbone.layers.*.mixer.shared_experts.up_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "backbone.layers.*.mixer.shared_experts.down_proj.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "backbone.layers.*.norm.weight",

            # Mamba mixer layers
            "decoder.layers.*.mixer.A_log": "backbone.layers.*.mixer.A_log",
            "decoder.layers.*.mixer.D": "backbone.layers.*.mixer.D",
            "decoder.layers.*.mixer.conv1d.weight": "backbone.layers.*.mixer.conv1d.weight",
            "decoder.layers.*.mixer.conv1d.bias": "backbone.layers.*.mixer.conv1d.bias",
            "decoder.layers.*.mixer.in_proj.weight": "backbone.layers.*.mixer.in_proj.weight",
            "decoder.layers.*.mixer.dt_bias": "backbone.layers.*.mixer.dt_bias",
            "decoder.layers.*.mixer.out_proj.weight": "backbone.layers.*.mixer.out_proj.weight",
            "decoder.layers.*.mixer.norm.weight": "backbone.layers.*.mixer.norm.weight",
            "decoder.layers.*.mixer.in_proj.layer_norm_weight": "backbone.layers.*.norm.weight",

            # Final layers
            "decoder.final_norm.weight": "backbone.norm_f.weight",
            "output_layer.weight": "lm_head.weight",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add QKV mapping for attention weights
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="backbone.layers.*.mixer.q_proj.weight",
                k="backbone.layers.*.mixer.k_proj.weight",
                v="backbone.layers.*.mixer.v_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)
