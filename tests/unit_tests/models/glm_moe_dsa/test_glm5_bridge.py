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
from unittest.mock import patch

import pytest
import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.identity_op import IdentityOp

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.glm_moe_dsa.glm5_bridge import (
    GLM5Bridge,
    _GLM5IndexShareMapping,
    _indexer_types_for_schedule,
    glm5_hybrid_stack_spec,
)
from megatron.bridge.models.glm_moe_dsa.glm5_provider import GLM5ModelProvider
from megatron.bridge.models.hybrid_mla_provider import HybridMLAModelProvider


pytestmark = pytest.mark.unit


@pytest.fixture
def glm5_bridge() -> GLM5Bridge:
    """Create a GLM-5 bridge with only the config fields read by mapping_registry."""
    bridge = GLM5Bridge()
    bridge.hf_config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        indexer_types=["full", "full", "full", "shared"],
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    return bridge


def _hf_config(
    *,
    num_hidden_layers: int = 4,
    mlp_layer_types: list[str] | None = None,
    indexer_types: list[str] | None = None,
) -> SimpleNamespace:
    if mlp_layer_types is None:
        mlp_layer_types = ["dense", "sparse", "sparse", "sparse"][:num_hidden_layers]
    if indexer_types is None:
        indexer_types = ["full"] * num_hidden_layers
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        attention_bias=False,
        vocab_size=128,
        hidden_act="silu",
        torch_dtype=torch.bfloat16,
        rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        rope_scaling=None,
        mlp_layer_types=mlp_layer_types,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=32,
        indexer_types=indexer_types,
        num_nextn_predict_layers=1,
    )


def _provider_from_config(hf_config: SimpleNamespace) -> GLM5ModelProvider:
    bridge = GLM5Bridge()
    return bridge.provider_bridge(SimpleNamespace(config=hf_config))


def _mapping_by_megatron_param(bridge: GLM5Bridge) -> dict[str, object]:
    return {mapping.megatron_param: mapping for mapping in bridge.mapping_registry()}


def test_glm5_uses_hybrid_mla_provider() -> None:
    """The compatibility provider is the shared Hybrid MLA provider."""
    assert GLM5Bridge.PROVIDER_CLASS is GLM5ModelProvider
    assert GLM5ModelProvider is HybridMLAModelProvider

    provider = _provider_from_config(_hf_config())
    assert isinstance(provider, HybridMLAModelProvider)
    assert provider.q_lora_rank == 16
    assert provider.kv_lora_rank == 8


def test_shared_config_conversion_detects_hybrid_mla_provider() -> None:
    """Hybrid MLA providers use MLA's direct YaRN config field names."""
    hf_config = _hf_config()
    hf_config.rope_scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "mscale": 0.7,
    }

    provider_kwargs = GLM5Bridge().hf_config_to_provider_kwargs(hf_config)

    assert provider_kwargs["_mla_rope_params"] == {
        "rotary_scaling_factor": 2.0,
        "mscale": 0.7,
    }
    assert "yarn_rotary_scaling_factor" not in provider_kwargs


def test_glm5_stack_uses_native_norm_for_dsa_q_and_kv() -> None:
    """GLM DSA uses the selected Hybrid stack's norm for both MLA latent paths."""
    stack_spec = glm5_hybrid_stack_spec(GLM5ModelProvider())
    dsa_submodules = stack_spec.submodules.dsa_layer.submodules
    attention_submodules = dsa_submodules.self_attention.submodules

    assert attention_submodules.q_layernorm is dsa_submodules.input_layernorm
    assert attention_submodules.kv_layernorm is dsa_submodules.input_layernorm
    assert attention_submodules.q_layernorm is not IdentityOp


def test_glm5_provider_translates_logical_layers_to_hybrid_pattern() -> None:
    """Dense and sparse logical blocks become D-/DE physical pairs."""
    provider = _provider_from_config(_hf_config())

    assert provider.hybrid_layer_pattern == "D-DEDEDE"
    assert provider.num_layers == 8
    assert provider.moe_layer_freq == [0, 0, 0, 1, 0, 1, 0, 1]
    assert provider.mtp_num_layers is None
    assert provider.sequence_parallel is True
    assert provider.rotary_interleaved is False
    assert provider.dsa_indexer_rope_interleaved is False
    assert provider.apply_rope_fusion is False


def test_glm5_provider_falls_back_to_first_k_dense_replace() -> None:
    """Legacy GLM configs without mlp_layer_types retain their dense prefix."""
    hf_config = _hf_config()
    hf_config.mlp_layer_types = None
    hf_config.first_k_dense_replace = 2

    provider = _provider_from_config(hf_config)

    assert provider.hybrid_layer_pattern == "D-D-DEDE"


def test_glm52_provider_has_156_physical_layers() -> None:
    """The official GLM-5.2 layout has 78 DSA, 3 dense, and 75 MoE layers."""
    num_logical_layers = 78
    hf_config = _hf_config(
        num_hidden_layers=num_logical_layers,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
        indexer_types=_indexer_types_for_schedule(
            num_logical_layers,
            topk_freq=4,
            skip_topk_offset=3,
        ),
    )
    hf_config.index_topk_freq = 4
    hf_config.index_skip_topk_offset = 3

    provider = _provider_from_config(hf_config)

    assert provider.num_layers == 156
    assert provider.hybrid_layer_pattern.count(Symbols.DS_ATTENTION) == 78
    assert provider.hybrid_layer_pattern.count(Symbols.MLP) == 3
    assert provider.hybrid_layer_pattern.count(Symbols.MOE) == 75
    assert provider.dsa_indexer_topk_freq == 8
    assert provider.dsa_indexer_skip_topk_offset == 5


def test_glm5_provider_rejects_non_periodic_indexshare() -> None:
    """Hybrid DSA cannot represent an arbitrary explicit IndexShare pattern."""
    hf_config = _hf_config(indexer_types=["full", "shared", "full", "shared"])
    hf_config.index_topk_freq = 4
    hf_config.index_skip_topk_offset = 3

    with pytest.raises(ValueError, match="does not match"):
        _provider_from_config(hf_config)


def test_glm5_export_restores_logical_layer_count() -> None:
    """HF export counts DSA layers rather than physical Hybrid layers."""
    provider = _provider_from_config(_hf_config())
    provider.hybrid_layer_pattern = "D-DE|DEDE"

    with patch.object(MegatronModelBridge, "megatron_to_hf_config", return_value={}):
        hf_config = GLM5Bridge.megatron_to_hf_config(provider)

    assert hf_config["num_hidden_layers"] == 4


def test_mapping_registry_maps_logical_layers_to_physical_pairs(glm5_bridge: GLM5Bridge) -> None:
    """Logical GLM layer indices map to their DSA and FFN physical layers."""
    mappings = _mapping_by_megatron_param(glm5_bridge)

    assert mappings["decoder.layers.0.input_layernorm.weight"].hf_param == ("model.layers.0.input_layernorm.weight")
    assert mappings["decoder.layers.1.mlp.linear_fc2.weight"].hf_param == ("model.layers.0.mlp.down_proj.weight")
    assert mappings["decoder.layers.6.self_attention.linear_proj.weight"].hf_param == (
        "model.layers.3.self_attn.o_proj.weight"
    )
    assert mappings["decoder.layers.7.mlp.router.weight"].hf_param == "model.layers.3.mlp.gate.weight"
    assert mappings["decoder.final_norm.weight"].hf_param == "model.norm.weight"


def test_mapping_registry_exports_indexshare_weights_from_source_layer(glm5_bridge: GLM5Bridge) -> None:
    """A live MCore indexer supplies both its full and shared HF modules."""
    mappings = _mapping_by_megatron_param(glm5_bridge)

    source_mapping = mappings["decoder.layers.4.self_attention.core_attention.indexer.linear_wq_b.weight"]
    assert source_mapping.hf_param == "model.layers.2.self_attn.indexer.wq_b.weight"
    assert source_mapping.shared_hf_params == ("model.layers.3.self_attn.indexer.wq_b.weight",)
    assert "decoder.layers.6.self_attention.core_attention.indexer.linear_wq_b.weight" not in mappings


def test_indexshare_export_copies_shared_hf_weights() -> None:
    """Shared HF indexer weights have equal values without aliasing source storage."""
    source_weight = torch.ones(2)
    mapping = _GLM5IndexShareMapping(
        megatron_param="decoder.indexer.weight",
        hf_param="model.source.indexer.weight",
        shared_hf_params=("model.shared.indexer.weight",),
    )

    with patch.object(
        AutoMapping,
        "megatron_to_hf",
        return_value={"model.source.indexer.weight": source_weight},
    ):
        result = mapping.megatron_to_hf(None, None)

    shared_weight = result["model.shared.indexer.weight"]
    assert torch.equal(shared_weight, source_weight)
    assert shared_weight.data_ptr() != source_weight.data_ptr()


def test_mapping_registry_includes_grouped_and_local_expert_fc2_paths(glm5_bridge: GLM5Bridge) -> None:
    """GLM-5 MoE export supports both packed and local-expert down-projection names."""
    mappings = _mapping_by_megatron_param(glm5_bridge)

    grouped_mapping = mappings["decoder.layers.3.mlp.experts.linear_fc2.weight*"]
    assert isinstance(grouped_mapping, AutoMapping)
    assert grouped_mapping.hf_param == "model.layers.1.mlp.experts.*.down_proj.weight"

    local_expert_mapping = mappings["decoder.layers.3.mlp.experts.local_experts.*.linear_fc2.weight"]
    assert isinstance(local_expert_mapping, AutoMapping)
    assert local_expert_mapping.hf_param == "model.layers.1.mlp.experts.*.down_proj.weight"

    registry = glm5_bridge.mapping_registry()
    grouped_lookup = registry.megatron_to_hf_lookup("decoder.layers.3.mlp.experts.linear_fc2.weight3")
    assert grouped_lookup is not None
    assert grouped_lookup.hf_param == "model.layers.1.mlp.experts.3.down_proj.weight"

    local_expert_lookup = registry.megatron_to_hf_lookup(
        "decoder.layers.3.mlp.experts.local_experts.3.linear_fc2.weight"
    )
    assert local_expert_lookup is not None
    assert local_expert_lookup.hf_param == "model.layers.1.mlp.experts.3.down_proj.weight"


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
    bridge.hf_config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        indexer_types=["full"] * 4,
    )

    assert all(not mapping.megatron_param.startswith("mtp.") for mapping in bridge.mapping_registry())


def test_mapping_registry_omits_mtp_mappings_when_nextn_layers_are_disabled() -> None:
    """An exported config uses None to represent disabled runtime MTP."""
    bridge = GLM5Bridge()
    bridge.hf_config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=None,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        indexer_types=["full"] * 4,
    )

    assert all(not mapping.megatron_param.startswith("mtp.") for mapping in bridge.mapping_registry())
