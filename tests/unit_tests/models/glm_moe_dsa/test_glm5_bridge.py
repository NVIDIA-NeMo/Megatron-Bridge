# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the GLM-5 MoE DSA bridge."""

from types import SimpleNamespace

import pytest

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge


pytestmark = pytest.mark.unit


@pytest.fixture
def glm5_bridge() -> GLM5Bridge:
    """Create a GLM-5 bridge with only the config fields read by mapping_registry."""
    bridge = GLM5Bridge()
    bridge.hf_config = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=1)
    return bridge


def _mapping_by_megatron_param(bridge: GLM5Bridge) -> dict[str, object]:
    return {mapping.megatron_param: mapping for mapping in bridge.mapping_registry()}


def _provider_from_hf_config(monkeypatch: pytest.MonkeyPatch, **config_overrides: int) -> SimpleNamespace:
    config = {
        "first_k_dense_replace": 3,
        "num_hidden_layers": 78,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "rope_parameters": {"rope_theta": 1_000_000},
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "indexer_rope_interleave": True,
    }
    config.update(config_overrides)

    monkeypatch.setattr(
        MegatronModelBridge,
        "provider_bridge",
        lambda _self, _hf_pretrained: SimpleNamespace(),
    )
    return GLM5Bridge().provider_bridge(SimpleNamespace(config=SimpleNamespace(**config)))


@pytest.mark.parametrize(
    ("config_overrides", "expected_topk_freq", "expected_skip_topk_offset"),
    [
        ({}, 1, 0),
        ({"index_topk_freq": 4, "index_skip_topk_offset": 3}, 4, 3),
    ],
)
def test_provider_bridge_maps_dsa_architecture_from_hf_config(
    monkeypatch: pytest.MonkeyPatch,
    config_overrides: dict[str, int],
    expected_topk_freq: int,
    expected_skip_topk_offset: int,
) -> None:
    """GLM-5 version-specific DSA architecture stays owned by the bridge."""
    provider = _provider_from_hf_config(monkeypatch, **config_overrides)

    assert provider.experimental_attention_variant == "dsa"
    assert provider.cp_comm_type == "allgather"
    assert provider.dsa_indexer_head_dim == 128
    assert provider.dsa_indexer_n_heads == 32
    assert provider.dsa_indexer_topk == 2048
    assert provider.dsa_indexer_rope_interleaved is True
    assert provider.dsa_indexer_topk_freq == expected_topk_freq
    assert provider.dsa_indexer_skip_topk_offset == expected_skip_topk_offset
    assert provider.dsa_indexer_rotate_activation is False
    assert provider.dsa_indexer_k_norm_epsilon == 1e-6
    assert provider.dsa_indexer_loss_coeff == 0.001
    assert provider.dsa_indexer_use_sparse_loss is True


def test_mapping_registry_includes_grouped_and_local_expert_fc2_paths(glm5_bridge: GLM5Bridge) -> None:
    """GLM-5 MoE export supports both packed and local-expert down-projection names."""
    mappings = _mapping_by_megatron_param(glm5_bridge)

    grouped_mapping = mappings["decoder.layers.*.mlp.experts.linear_fc2.weight*"]
    assert isinstance(grouped_mapping, AutoMapping)
    assert grouped_mapping.hf_param == "model.layers.*.mlp.experts.*.down_proj.weight"

    local_expert_mapping = mappings["decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight"]
    assert isinstance(local_expert_mapping, AutoMapping)
    assert local_expert_mapping.hf_param == "model.layers.*.mlp.experts.*.down_proj.weight"

    registry = glm5_bridge.mapping_registry()
    grouped_lookup = registry.megatron_to_hf_lookup("decoder.layers.2.mlp.experts.linear_fc2.weight3")
    assert grouped_lookup is not None
    assert grouped_lookup.hf_param == "model.layers.2.mlp.experts.3.down_proj.weight"

    local_expert_lookup = registry.megatron_to_hf_lookup(
        "decoder.layers.2.mlp.experts.local_experts.3.linear_fc2.weight"
    )
    assert local_expert_lookup is not None
    assert local_expert_lookup.hf_param == "model.layers.2.mlp.experts.3.down_proj.weight"


@pytest.mark.parametrize("layer_prefix", ["transformer_layer", "mtp_model_layer"])
def test_mapping_registry_includes_mtp_moe_mappings(glm5_bridge: GLM5Bridge, layer_prefix: str) -> None:
    """Each GLM-5 MTP block mirrors the decoder MoE mappings for both layer replicas."""
    mappings = _mapping_by_megatron_param(glm5_bridge)

    router_mapping = mappings[f"mtp.layers.0.{layer_prefix}.mlp.router.expert_bias"]
    assert isinstance(router_mapping, AutoMapping)
    assert router_mapping.hf_param == "model.layers.4.mlp.gate.e_score_correction_bias"

    expert_fc1_mapping = mappings[f"mtp.layers.0.{layer_prefix}.mlp.experts.local_experts.*.linear_fc1.weight"]
    assert isinstance(expert_fc1_mapping, GatedMLPMapping)
    assert expert_fc1_mapping.hf_param == {
        "gate": "model.layers.4.mlp.experts.*.gate_proj.weight",
        "up": "model.layers.4.mlp.experts.*.up_proj.weight",
    }

    expert_fc2_mapping = mappings[f"mtp.layers.0.{layer_prefix}.mlp.experts.local_experts.*.linear_fc2.weight"]
    assert isinstance(expert_fc2_mapping, AutoMapping)
    assert expert_fc2_mapping.hf_param == "model.layers.4.mlp.experts.*.down_proj.weight"

    registry = glm5_bridge.mapping_registry()
    expert_fc2_lookup = registry.megatron_to_hf_lookup(
        f"mtp.layers.0.{layer_prefix}.mlp.experts.local_experts.7.linear_fc2.weight"
    )
    assert expert_fc2_lookup is not None
    assert expert_fc2_lookup.hf_param == "model.layers.4.mlp.experts.7.down_proj.weight"


@pytest.mark.parametrize("layer_prefix", ["transformer_layer", "mtp_model_layer"])
def test_mapping_registry_includes_mtp_attention_and_dense_mlp_mappings(
    glm5_bridge: GLM5Bridge, layer_prefix: str
) -> None:
    """MTP attention and dense MLP mappings point at the appended HF layer index."""
    mappings = _mapping_by_megatron_param(glm5_bridge)

    qkv_mapping = mappings[f"mtp.layers.0.{layer_prefix}.self_attention.linear_qkv.weight"]
    assert isinstance(qkv_mapping, QKVMapping)
    assert qkv_mapping.hf_param == {
        "q": "model.layers.4.self_attn.q_proj.weight",
        "k": "model.layers.4.self_attn.k_proj.weight",
        "v": "model.layers.4.self_attn.v_proj.weight",
    }

    mlp_mapping = mappings[f"mtp.layers.0.{layer_prefix}.mlp.linear_fc1.weight"]
    assert isinstance(mlp_mapping, GatedMLPMapping)
    assert mlp_mapping.hf_param == {
        "gate": "model.layers.4.mlp.gate_proj.weight",
        "up": "model.layers.4.mlp.up_proj.weight",
    }


def test_mapping_registry_includes_mtp_standalone_weights(glm5_bridge: GLM5Bridge) -> None:
    """GLM-5 MTP-only weights map to the appended HF MTP layer."""
    mappings = _mapping_by_megatron_param(glm5_bridge)

    expected_hf_params = {
        "mtp.layers.0.enorm.weight": "model.layers.4.enorm.weight",
        "mtp.layers.0.hnorm.weight": "model.layers.4.hnorm.weight",
        "mtp.layers.0.eh_proj.weight": "model.layers.4.eh_proj.weight",
        "mtp.layers.0.final_layernorm.weight": "model.layers.4.shared_head.norm.weight",
    }
    for megatron_param, hf_param in expected_hf_params.items():
        mapping = mappings[megatron_param]
        assert isinstance(mapping, AutoMapping)
        assert mapping.hf_param == hf_param


def test_mapping_registry_omits_mtp_mappings_without_nextn_layers() -> None:
    """No MTP mappings are registered when the HF config has no MTP layers."""
    bridge = GLM5Bridge()
    bridge.hf_config = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=0)

    assert all(not mapping.megatron_param.startswith("mtp.") for mapping in bridge.mapping_registry())
