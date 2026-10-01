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


from dataclasses import fields
from types import SimpleNamespace

from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    DirectMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.shensi import ShensiBridge, ShensiModelProvider
from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig


def _hf_config(layers=("sliding_attention", "compressed_sparse_attention"), mlp=("hash_moe", "moe")):
    return SimpleNamespace(
        num_hidden_layers=len(layers),
        layer_types=list(layers),
        mlp_layer_types=list(mlp),
        n_routed_experts=8,
        hc_conv_kernels=(4, 8, 12),
        hidden_size=128,
        num_attention_heads=4,
        vocab_size=128,
        scoring_func="sqrtsoftplus",
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def _registry(hf_config=None):
    return ShensiBridge.mapping_registry(SimpleNamespace(hf_config=hf_config or _hf_config()))


def _by_megatron(registry):
    return {m.megatron_param: m for m in registry.get_all_mappings()}


class TestShensiDispatch:
    def test_register_bridge_resolves(self):
        got = get_model_bridge("ShensiForCausalLM")
        assert got is ShensiBridge or isinstance(got, ShensiBridge), type(got)

    def test_provider_is_mcore_shensi_config(self):
        assert issubclass(ShensiModelProvider, ShensiTransformerConfig)
        names = {f.name for f in fields(ShensiModelProvider)}
        for key in ("hc_active_streams", "hc_fixed_streams", "hc_conv_kernels", "attn_res_block_size"):
            assert key in names, key

    def test_provider_implements_provide(self):
        assert callable(ShensiModelProvider.provide)


class TestShensiMappings:
    def test_top_level(self):
        reg = _by_megatron(_registry())
        assert isinstance(reg["embedding.word_embeddings.weight"], AutoMapping)
        assert isinstance(reg["output_norm.weight"], AutoMapping)
        assert isinstance(reg["hc_head.hc_fn"], ReplicatedMapping)
        assert isinstance(reg["output_attn_res.g_b_proj.bias"], DirectMapping)

    def test_hyper_connection_and_attn_res_are_same_named(self):
        reg = _by_megatron(_registry())
        for layer in (0, 1):
            for hc in ("attn_hc", "ffn_hc"):
                m = reg[f"decoder.layers.{layer}.{hc}.pre_fn"]
                assert isinstance(m, ReplicatedMapping)
                assert str(m.hf_param) == f"model.layers.{layer}.{hc}.pre_fn"
            for slot in ("self_attention_attn_res", "mlp_attn_res"):
                for suffix in (
                    "g_scale",
                    "t",
                    "q_a_proj.weight",
                    "q_b_proj.weight",
                    "k_a_proj.weight",
                    "k_b_proj.weight",
                    "g_a_proj.weight",
                    "g_a_proj.bias",
                    "g_b_proj.weight",
                    "g_b_proj.bias",
                ):
                    assert f"decoder.layers.{layer}.{slot}.{suffix}" in reg
        for k in range(3):
            assert f"decoder.layers.0.ffn_hc.temporal_convs.{k}.weight" in reg

    def test_hash_layer_maps_gate_up_into_linear_fc1(self):
        reg = _by_megatron(_registry())
        m = reg["decoder.layers.0.mlp.linear_fc1.weight"]
        assert isinstance(m, GatedMLPMapping)
        assert m.hf_param["gate"] == "model.layers.0.mlp.gate_proj.weight"
        assert m.hf_param["up"] == "model.layers.0.mlp.up_proj.weight"
        assert "decoder.layers.0.mlp.deepemb.weight" in reg

    def test_moe_layer_uses_fused_expert_mappings(self):
        reg = _by_megatron(_registry())
        assert isinstance(reg["decoder.layers.1.mlp.experts.linear_fc1.weight*"], FusedGatedExpertMapping)
        assert isinstance(reg["decoder.layers.1.mlp.experts.linear_fc2.weight*"], FusedExpertMapping)
        assert isinstance(reg["decoder.layers.1.mlp.router.weight"], ReplicatedMapping)

    def test_csa_layer_has_indexer_and_compressor(self):
        reg = _by_megatron(_registry())
        for key in (
            "decoder.layers.1.self_attention.core_attention.compressor.linear_wkv.weight",
            "decoder.layers.1.self_attention.core_attention.compressor.ape",
            "decoder.layers.1.self_attention.core_attention.indexer.linear_wq_b.weight",
            "decoder.layers.1.self_attention.core_attention.indexer.compressor.ape",
        ):
            assert key in reg, key

    def test_sliding_layer_has_no_compressor(self):
        reg = _by_megatron(_registry())
        assert "decoder.layers.0.self_attention.core_attention.compressor.linear_wkv.weight" not in reg
        assert "decoder.layers.0.self_attention.core_attention.indexer.linear_wq_b.weight" not in reg

    def test_hash_layer_has_no_experts(self):
        reg = _by_megatron(_registry())
        assert "decoder.layers.0.mlp.experts.linear_fc1.weight*" not in reg
        assert "decoder.layers.1.mlp.linear_fc1.weight" not in reg
