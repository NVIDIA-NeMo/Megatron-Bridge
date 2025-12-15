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

from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


try:
    import apex  # noqa: F401

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False


def get_common_configs(hf_pretrained: PreTrainedCausalLM) -> dict:
    """
    Returns a dictionary of common configurations for the DeepSeek family of models.
    """
    hf_config = hf_pretrained.config

    configs = {}

    if not HAVE_APEX:
        configs["gradient_accumulation_fusion"] = False

    if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
        configs["rotary_scaling_factor"] = hf_config.rope_scaling["factor"]
        configs["mscale"] = hf_config.rope_scaling["mscale"]
        configs["mscale_all_dim"] = hf_config.rope_scaling["mscale_all_dim"]
    else:
        configs["rotary_scaling_factor"] = 1.0
        configs["mscale"] = 1.0
        configs["mscale_all_dim"] = 1.0

    configs["num_layers"] = hf_config.num_hidden_layers
    configs["hidden_size"] = hf_config.hidden_size
    configs["ffn_hidden_size"] = hf_config.intermediate_size
    configs["num_attention_heads"] = hf_config.num_attention_heads
    configs["kv_channels"] = hf_config.num_key_value_heads
    configs["q_lora_rank"] = hf_config.q_lora_rank
    configs["num_moe_experts"] = hf_config.n_routed_experts
    configs["moe_ffn_hidden_size"] = hf_config.moe_intermediate_size
    configs["moe_shared_expert_intermediate_size"] = hf_config.moe_intermediate_size * hf_config.n_shared_experts
    configs["moe_layer_freq"] = [0] * hf_config.first_k_dense_replace + [1] * (
        hf_config.num_hidden_layers - hf_config.first_k_dense_replace
    )
    configs["moe_router_topk"] = hf_config.num_experts_per_tok
    configs["moe_router_num_groups"] = hf_config.n_group
    configs["moe_router_group_topk"] = hf_config.topk_group
    configs["moe_router_topk_scaling_factor"] = hf_config.routed_scaling_factor
    configs["kv_lora_rank"] = hf_config.kv_lora_rank
    configs["qk_head_dim"] = hf_config.qk_nope_head_dim
    configs["qk_pos_emb_head_dim"] = hf_config.qk_rope_head_dim
    configs["v_head_dim"] = hf_config.v_head_dim

    # Ensure MLA is enabled
    configs["multi_latent_attention"] = True
    configs["generation_config"] = hf_pretrained.generation_config
    configs["vocab_size"] = hf_config.vocab_size
    configs["rotary_base"] = hf_config.rope_theta
    configs["init_method_std"] = hf_config.initializer_range
    configs["layernorm_epsilon"] = hf_config.rms_norm_eps

    return configs


def get_common_mapping_list() -> list:
    """
    Returns a list of common parameter mappings for the DeepSeek family of models.
    """
    param_mappings = {
        # Embed
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        # Attention
        "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
        "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
        #  In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
        #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
        #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
        "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        # Mcore local spec
        "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        # Dense MLP
        "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        # MoE
        "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
        "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        # LM Head
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
        # MLA
        "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # Mcore local spec
        "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # For models without MLA
        "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
    }

    # TODO: mtp layers

    mapping_list = []
    # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
    for megatron_param, hf_param in param_mappings.items():
        mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

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

    return mapping_list
