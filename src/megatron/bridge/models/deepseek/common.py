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


def get_common_mapping_list() -> list:
    """
    Returns a list of common parameter mappings for the DeepSeek family of models.
    """
    param_mappings = {
        # Embed
        "model.embed_tokens.weight": "embedding.word_embeddings.weight",
        # Attention
        "model.layers.*.input_layernorm.weight": "decoder.layers.*.input_layernorm.weight",
        "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
        "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
        "model.layers.*.self_attn.kv_a_proj_with_mqa.weight": "decoder.layers.*.self_attention.linear_kv_down_proj.weight",
        "model.layers.*.self_attn.kv_b_proj.weight": "decoder.layers.*.self_attention.linear_kv_up_proj.weight",
        "model.layers.*.self_attn.kv_a_layernorm.weight": "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight",
        # Dense MLP
        "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
        # MoE
        "model.layers.*.mlp.gate.weight": "decoder.layers.*.mlp.router.weight",
        "model.layers.*.mlp.experts.*.down_proj.weight": "decoder.layers.*.mlp.experts.linear_fc2.weight*",
        "model.layers.*.mlp.shared_experts.down_proj.weight": "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
        # LM Head
        "model.norm.weight": "decoder.final_layernorm.weight",
        "lm_head.weight": "output_layer.weight",
    }

    # TODO: mtp layers

    mapping_list = []
    # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
    for hf_param, megatron_param in param_mappings.items():
        mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

    # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
    #  In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
    #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
    #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
    #
    # (a) is defined in the `param_mappings` above, so we need to add (b) here separately (to avoid dictionary key conflict)
    mapping_list.append(
        AutoMapping(
            hf_param="model.layers.*.post_attention_layernorm.weight",
            megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
        )
    )

    # Mcore local spec
    mapping_list.append(
        AutoMapping(
            hf_param="model.layers.*.self_attn.kv_a_layernorm.weight",
            megatron_param="decoder.layers.*.self_attention.kv_layernorm.weight",
        )
    )

    # Add special mappings that require parameter concatenation/transformation
    mapping_list.extend(
        [
            GatedMLPMapping(
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
            ),
            GatedMLPMapping(
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            ),
            GatedMLPMapping(
                gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                up="model.layers.*.mlp.shared_experts.up_proj.weight",
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
            ),
        ]
    )

    return mapping_list
