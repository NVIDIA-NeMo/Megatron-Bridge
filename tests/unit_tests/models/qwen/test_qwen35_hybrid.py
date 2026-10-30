# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pytest
from transformers import Qwen3_5MoeTextConfig, Qwen3_5TextConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.qwen.modeling_qwen35.hybrid import qwen35_hybrid_mappings
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge


@pytest.mark.parametrize("moe", [False, True])
def test_split_layer_mapping_preserves_hf_block_and_expert_indices(moe):
    templates = (
        Qwen35MoEBridge._get_moe_lm_mappings(experts_packed=False) if moe else Qwen35Bridge._get_dense_lm_mappings()
    )
    registry = MegatronMappingRegistry(*qwen35_hybrid_mappings(templates, num_layers=4))
    qkv = registry.megatron_to_hf_lookup("decoder.layers.6.self_attention.linear_qkv.weight")
    assert qkv.hf_param["q"] == "model.layers.3.self_attn.q_proj.weight"
    assert registry.hf_to_megatron_lookup(qkv.hf_param["q"]).megatron_param == qkv.megatron_param

    if moe:
        mlp_name = "decoder.layers.7.mlp.experts.linear_fc1.weight12"
        hf_gate = "model.layers.3.mlp.experts.12.gate_proj.weight"
    else:
        mlp_name = "decoder.layers.7.mlp.linear_fc1.weight"
        hf_gate = "model.layers.3.mlp.gate_proj.weight"
    mlp = registry.megatron_to_hf_lookup(mlp_name)
    assert mlp.hf_param["gate"] == hf_gate
    assert registry.hf_to_megatron_lookup(hf_gate).megatron_param == mlp_name
    assert registry.megatron_to_hf_lookup("decoder.final_norm.weight").hf_param == "model.norm.weight"


@pytest.mark.parametrize(
    "bridge_class,config_class", [(Qwen35Bridge, Qwen3_5TextConfig), (Qwen35MoEBridge, Qwen3_5MoeTextConfig)]
)
def test_hybrid_config_export_preserves_nonperiodic_attention_schedule(bridge_class, config_class):
    hf_config = config_class(
        num_hidden_layers=4, layer_types=["full_attention", "linear_attention", "linear_attention", "full_attention"]
    )
    bridge = bridge_class()
    config = bridge.hf_config_to_model_config(hf_config)
    restored = config_class(**bridge.megatron_to_hf_config(config))
    assert restored.num_hidden_layers == hf_config.num_hidden_layers
    assert restored.layer_types == hf_config.layer_types
    assert config.num_layers == 2 * hf_config.num_hidden_layers


def test_hybrid_mtp_mapping_preserves_zero_centered_gdn_conversion():
    mappings = qwen35_hybrid_mappings(Qwen35MoEBridge._get_moe_lm_mappings(), num_layers=4)
    registry = MegatronMappingRegistry(*mappings)
    norm = registry.megatron_to_hf_lookup("decoder.layers.4.self_attention.out_norm.weight")
    assert type(norm).__name__ == "RMSNorm2ZeroCenteredRMSNormMapping"
    assert norm.hf_param == "model.layers.2.linear_attn.norm.weight"

    registry = MegatronMappingRegistry(*qwen35_hybrid_mappings(Qwen35MoEBridge._get_moe_mtp_mappings(), num_layers=4))
    mapping = registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.layers.1.mlp.router.weight")
    assert mapping.hf_param == "mtp.layers.0.mlp.gate.weight"


def test_vl_mapping_splits_only_language_blocks():
    from transformers import Qwen3_5MoeConfig

    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge, _get_vision_mappings

    bridge = Qwen35VLMoEBridge()
    bridge.hf_config = Qwen3_5MoeConfig(text_config={"num_hidden_layers": 4})
    registry = bridge.mapping_registry()
    language = registry.megatron_to_hf_lookup("language_model.decoder.layers.7.mlp.router.weight")
    assert language.hf_param == "model.language_model.layers.3.mlp.gate.weight"
    vision = MegatronMappingRegistry(*_get_vision_mappings())
    for template in vision.mappings:
        name = template.megatron_param.replace("*", "0")
        assert registry.megatron_to_hf_lookup(name).hf_param == vision.megatron_to_hf_lookup(name).hf_param
