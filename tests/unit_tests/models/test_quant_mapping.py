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

import os
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.quant_mapping import (
    AmaxFanoutMapping,
    AmaxMapping,
)


class TestAmaxMapping:
    def test_inherits_replicated_mapping(self):
        m = AmaxMapping("mcore.weight_quantizer._amax", "hf.weight_quantizer._amax")
        assert isinstance(m, ReplicatedMapping)
        assert m.allow_hf_name_mismatch is True


class TestAmaxFanoutMapping:
    def test_canonical_hf_param_is_first_target(self):
        targets = ["hf.q._amax", "hf.k._amax", "hf.v._amax"]
        m = AmaxFanoutMapping("mcore.qkv._amax", targets)
        assert isinstance(m, AmaxMapping)
        assert m.hf_param == "hf.q._amax"
        assert m.hf_targets == targets

    def test_empty_hf_params_raises(self):
        with pytest.raises(AssertionError):
            AmaxFanoutMapping("mcore._amax", [])

    def test_megatron_to_hf_fans_out(self):
        targets = ["hf.q._amax", "hf.k._amax", "hf.v._amax"]
        m = AmaxFanoutMapping("mcore.qkv._amax", targets)
        weight = torch.tensor([1.0])

        with patch.object(ReplicatedMapping, "megatron_to_hf", return_value={"hf.q._amax": weight}):
            result = m.megatron_to_hf(weight, None)

        assert set(result.keys()) == set(targets)
        for t in targets:
            assert torch.equal(result[t], weight)

    def test_megatron_to_hf_empty_base(self):
        m = AmaxFanoutMapping("mcore._amax", ["hf.a._amax", "hf.b._amax"])
        with patch.object(ReplicatedMapping, "megatron_to_hf", return_value={}):
            result = m.megatron_to_hf(None, None)
        assert result == {}

    def test_resolve_replaces_wildcards(self):
        m = AmaxFanoutMapping(
            "decoder.layers.*.self_attention.linear_qkv.weight_quantizer._amax",
            [
                "model.layers.*.self_attn.q_proj.weight_quantizer._amax",
                "model.layers.*.self_attn.k_proj.weight_quantizer._amax",
                "model.layers.*.self_attn.v_proj.weight_quantizer._amax",
            ],
        )
        resolved = m.resolve(("5",))
        assert resolved.megatron_param == "decoder.layers.5.self_attention.linear_qkv.weight_quantizer._amax"
        assert isinstance(resolved, AmaxFanoutMapping)
        expected = {
            "model.layers.5.self_attn.q_proj.weight_quantizer._amax",
            "model.layers.5.self_attn.k_proj.weight_quantizer._amax",
            "model.layers.5.self_attn.v_proj.weight_quantizer._amax",
        }
        assert set(resolved.hf_targets) == expected


class TestQuantMappingRegistryIntegration:
    """Test quantization mappings inside MegatronMappingRegistry with a Llama-like bridge."""

    @pytest.fixture
    def llama_like_mappings(self):
        return [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            AutoMapping("decoder.final_layernorm.weight", "model.norm.weight"),
            AutoMapping(
                "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "model.layers.*.post_attention_layernorm.weight",
            ),
            AutoMapping(
                "decoder.layers.*.self_attention.linear_proj.weight",
                "model.layers.*.self_attn.o_proj.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.linear_fc2.weight",
                "model.layers.*.mlp.down_proj.weight",
            ),
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
        ]

    @pytest.fixture
    def registry(self, llama_like_mappings):
        with patch.dict(os.environ, {"ENABLE_BRIDGE_QUANT_MAPPING": "1"}, clear=False):
            return MegatronMappingRegistry(*llama_like_mappings)

    def test_quant_mappings_disabled_by_default(self, llama_like_mappings):
        with patch.dict(os.environ, {"ENABLE_BRIDGE_QUANT_MAPPING": "0"}, clear=False):
            registry = MegatronMappingRegistry(*llama_like_mappings)
        assert not any(isinstance(m, AmaxMapping) for m in registry.get_all_mappings())

    def test_quant_mappings_count(self, registry):
        """weight_quantizer and input_quantizer amax mappings are added in equal numbers."""
        amax_mappings = [m for m in registry.get_all_mappings() if isinstance(m, AmaxMapping)]
        weight_q = [m for m in amax_mappings if "weight_quantizer" in m.megatron_param]
        input_q = [m for m in amax_mappings if "input_quantizer" in m.megatron_param]
        assert len(weight_q) == len(input_q)
        assert len(weight_q) > 0

    def test_original_weight_mappings_unaffected(self, registry):
        m = registry.megatron_to_hf_lookup("embedding.word_embeddings.weight")
        assert m is not None
        assert m.hf_param == "model.embed_tokens.weight"

    def test_nonexistent_amax_returns_none(self, registry):
        assert registry.megatron_to_hf_lookup("decoder.layers.0.nonexistent.weight_quantizer._amax") is None

    @pytest.mark.parametrize(
        "megatron_amax, expected_hf_amax",
        [
            (
                "decoder.layers.0.self_attention.linear_proj.weight_quantizer._amax",
                "model.layers.0.self_attn.o_proj.weight_quantizer._amax",
            ),
            (
                "decoder.layers.0.mlp.linear_fc2.weight_quantizer._amax",
                "model.layers.0.mlp.down_proj.weight_quantizer._amax",
            ),
            (
                "embedding.word_embeddings.weight_quantizer._amax",
                "model.embed_tokens.weight_quantizer._amax",
            ),
            (
                "output_layer.weight_quantizer._amax",
                "lm_head.weight_quantizer._amax",
            ),
            (
                "decoder.final_layernorm.weight_quantizer._amax",
                "model.norm.weight_quantizer._amax",
            ),
            (
                "decoder.layers.0.self_attention.linear_proj.input_quantizer._amax",
                "model.layers.0.self_attn.o_proj.input_quantizer._amax",
            ),
            (
                "decoder.layers.0.mlp.linear_fc2.input_quantizer._amax",
                "model.layers.0.mlp.down_proj.input_quantizer._amax",
            ),
        ],
    )
    def test_simple_amax_forward_lookup(self, registry, megatron_amax, expected_hf_amax):
        m = registry.megatron_to_hf_lookup(megatron_amax)
        assert m is not None, f"No mapping found for {megatron_amax}"
        assert isinstance(m, AmaxMapping)
        assert m.hf_param == expected_hf_amax

    @pytest.mark.parametrize(
        "megatron_amax, expected_hf_targets",
        [
            (
                "decoder.layers.0.self_attention.linear_qkv.weight_quantizer._amax",
                [
                    "model.layers.0.self_attn.q_proj.weight_quantizer._amax",
                    "model.layers.0.self_attn.k_proj.weight_quantizer._amax",
                    "model.layers.0.self_attn.v_proj.weight_quantizer._amax",
                ],
            ),
            (
                "decoder.layers.0.mlp.linear_fc1.weight_quantizer._amax",
                [
                    "model.layers.0.mlp.gate_proj.weight_quantizer._amax",
                    "model.layers.0.mlp.up_proj.weight_quantizer._amax",
                ],
            ),
            (
                "decoder.layers.0.self_attention.linear_qkv.input_quantizer._amax",
                [
                    "model.layers.0.self_attn.q_proj.input_quantizer._amax",
                    "model.layers.0.self_attn.k_proj.input_quantizer._amax",
                    "model.layers.0.self_attn.v_proj.input_quantizer._amax",
                ],
            ),
            (
                "decoder.layers.0.mlp.linear_fc1.input_quantizer._amax",
                [
                    "model.layers.0.mlp.gate_proj.input_quantizer._amax",
                    "model.layers.0.mlp.up_proj.input_quantizer._amax",
                ],
            ),
        ],
    )
    def test_fanout_amax_forward_lookup(self, registry, megatron_amax, expected_hf_targets):
        m = registry.megatron_to_hf_lookup(megatron_amax)
        assert m is not None, f"No mapping found for {megatron_amax}"
        assert isinstance(m, AmaxFanoutMapping)
        assert set(m.hf_targets) == set(expected_hf_targets)

    def test_layer_index_independence(self, registry):
        """Different layer indices resolve correctly."""
        for layer_idx in [0, 5, 31]:
            m = registry.megatron_to_hf_lookup(
                f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight_quantizer._amax"
            )
            assert m is not None
            assert m.hf_param == f"model.layers.{layer_idx}.self_attn.o_proj.weight_quantizer._amax"

            m = registry.megatron_to_hf_lookup(
                f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight_quantizer._amax"
            )
            assert m is not None
            assert isinstance(m, AmaxFanoutMapping)
            for proj in ["q", "k", "v"]:
                assert f"model.layers.{layer_idx}.self_attn.{proj}_proj.weight_quantizer._amax" in m.hf_targets
