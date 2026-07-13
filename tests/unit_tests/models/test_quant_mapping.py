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
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    QKVGMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.quant_mapping import (
    AmaxFanoutMapping,
    AmaxMapping,
    MoeAmaxFanoutMapping,
    derive_kv_bmm_amax_map,
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


class TestDeriveKvBmmAmaxMap:
    def test_derives_kv_bmm_amax_mappings_from_qkv_mapping(self):
        mapping = QKVMapping(
            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
            q="model.layers.*.self_attn.q_proj.weight",
            k="model.layers.*.self_attn.k_proj.weight",
            v="model.layers.*.self_attn.v_proj.weight",
        )

        result = derive_kv_bmm_amax_map([mapping])

        assert [(m.megatron_param, m.hf_param) for m in result] == [
            (
                "decoder.layers.*.self_attention.core_attention.k_bmm_quantizer._amax",
                "model.layers.*.self_attn.k_bmm_quantizer._amax",
            ),
            (
                "decoder.layers.*.self_attention.core_attention.v_bmm_quantizer._amax",
                "model.layers.*.self_attn.v_bmm_quantizer._amax",
            ),
        ]
        assert all(isinstance(mapping, AmaxMapping) for mapping in result)

    def test_derives_kv_bmm_amax_mappings_from_step_qkvg_mapping(self):
        mapping = QKVGMapping(
            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
            q="model.layers.*.self_attn.q_proj.weight",
            k="model.layers.*.self_attn.k_proj.weight",
            v="model.layers.*.self_attn.v_proj.weight",
            g="model.layers.*.self_attn.g_proj.weight",
        )

        result = derive_kv_bmm_amax_map([mapping])

        assert [(m.megatron_param, m.hf_param) for m in result] == [
            (
                "decoder.layers.*.self_attention.core_attention.k_bmm_quantizer._amax",
                "model.layers.*.self_attn.k_bmm_quantizer._amax",
            ),
            (
                "decoder.layers.*.self_attention.core_attention.v_bmm_quantizer._amax",
                "model.layers.*.self_attn.v_bmm_quantizer._amax",
            ),
        ]

    def test_preserves_wildcards_and_language_model_prefixes(self):
        mapping = QKVMapping(
            megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
            q="language_model.model.layers.*.self_attn.q_proj.weight",
            k="language_model.model.layers.*.self_attn.k_proj.weight",
            v="language_model.model.layers.*.self_attn.v_proj.weight",
        )

        result = derive_kv_bmm_amax_map([mapping])

        assert [(m.megatron_param, m.hf_param) for m in result] == [
            (
                "language_model.decoder.layers.*.self_attention.core_attention.k_bmm_quantizer._amax",
                "language_model.model.layers.*.self_attn.k_bmm_quantizer._amax",
            ),
            (
                "language_model.decoder.layers.*.self_attention.core_attention.v_bmm_quantizer._amax",
                "language_model.model.layers.*.self_attn.v_bmm_quantizer._amax",
            ),
        ]

    @pytest.mark.parametrize(
        ("mapping", "reason"),
        [
            pytest.param(
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                "bias",
                id="bias-mapping",
            ),
            pytest.param(
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                "non-common-parent",
                id="non-common-parent",
            ),
            pytest.param(
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.kernel",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                "malformed-projection",
                id="malformed-projection",
            ),
            pytest.param(
                QKVMapping(
                    megatron_param="mtp.layers.*.self_attention.linear_qkv.weight",
                    q="model.mtp_layers.*.self_attn.q_proj.weight",
                    k="model.mtp_layers.*.self_attn.k_proj.weight",
                    v="model.mtp_layers.*.self_attn.v_proj.weight",
                ),
                "mtp-path",
                id="mtp-path",
            ),
            pytest.param(
                QKVMapping(
                    megatron_param="draft.layers.*.self_attention.linear_qkv.weight",
                    q="model.draft_layers.*.self_attn.q_proj.weight",
                    k="model.draft_layers.*.self_attn.k_proj.weight",
                    v="model.draft_layers.*.self_attn.v_proj.weight",
                ),
                "draft-path",
                id="draft-path",
            ),
            pytest.param(
                ConcatenatedQKVMapping(
                    megatron_param="vision.blocks.*.self_attention.linear_qkv.weight",
                    hf_param="vision_model.layers.*.self_attn.qkv.weight",
                ),
                "vision-concatenated-qkv",
                id="vision-concatenated-qkv",
            ),
        ],
    )
    def test_skips_disallowed_qkv_shapes(self, mapping, reason):
        assert derive_kv_bmm_amax_map([mapping]) == [], reason

    def test_skips_mappings_that_allow_missing_hf_projections(self):
        mapping = QKVMapping(
            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
            q="model.layers.*.self_attn.q_proj.weight",
            k="model.layers.*.self_attn.k_proj.weight",
            v="model.layers.*.self_attn.v_proj.weight",
        )
        mapping.allow_hf_name_mismatch = True

        assert derive_kv_bmm_amax_map([mapping]) == []


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
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.core_attention.k_bmm_quantizer._amax")
            is None
        )

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

    @pytest.mark.parametrize(
        "megatron_amax, expected_hf_amax",
        [
            (
                "decoder.layers.0.self_attention.core_attention.k_bmm_quantizer._amax",
                "model.layers.0.self_attn.k_bmm_quantizer._amax",
            ),
            (
                "decoder.layers.0.self_attention.core_attention.v_bmm_quantizer._amax",
                "model.layers.0.self_attn.v_bmm_quantizer._amax",
            ),
        ],
    )
    def test_kv_bmm_amax_forward_lookup(self, registry, megatron_amax, expected_hf_amax):
        mapping = registry.megatron_to_hf_lookup(megatron_amax)

        assert mapping is not None
        assert isinstance(mapping, AmaxMapping)
        assert mapping.megatron_param == megatron_amax
        assert mapping.hf_param == expected_hf_amax

    @pytest.mark.parametrize(
        "hf_amax, expected_megatron_amax",
        [
            (
                "model.layers.7.self_attn.k_bmm_quantizer._amax",
                "decoder.layers.7.self_attention.core_attention.k_bmm_quantizer._amax",
            ),
            (
                "model.layers.7.self_attn.v_bmm_quantizer._amax",
                "decoder.layers.7.self_attention.core_attention.v_bmm_quantizer._amax",
            ),
        ],
    )
    def test_kv_bmm_amax_reverse_lookup(self, registry, hf_amax, expected_megatron_amax):
        mapping = registry.hf_to_megatron_lookup(hf_amax)

        assert mapping is not None
        assert isinstance(mapping, AmaxMapping)
        assert mapping.megatron_param == expected_megatron_amax
        assert mapping.hf_param == hf_amax

    def test_kv_bmm_amax_coexists_with_weight_and_input_quantizer_mappings(self, registry):
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.core_attention.k_bmm_quantizer._amax")
            is not None
        )
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_qkv.weight_quantizer._amax")
            is not None
        )
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_qkv.input_quantizer._amax")
            is not None
        )


class TestKvBmmQuantMappingPrefixes:
    def test_registry_preserves_prefixes_and_wildcards(self):
        mappings = [
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="language_model.model.layers.*.self_attn.q_proj.weight",
                k="language_model.model.layers.*.self_attn.k_proj.weight",
                v="language_model.model.layers.*.self_attn.v_proj.weight",
            )
        ]

        with patch.dict(os.environ, {"ENABLE_BRIDGE_QUANT_MAPPING": "1"}, clear=False):
            registry = MegatronMappingRegistry(*mappings)

        mapping = registry.megatron_to_hf_lookup(
            "language_model.decoder.layers.9.self_attention.core_attention.k_bmm_quantizer._amax"
        )

        assert mapping is not None
        assert (
            mapping.megatron_param
            == "language_model.decoder.layers.9.self_attention.core_attention.k_bmm_quantizer._amax"
        )
        assert mapping.hf_param == "language_model.model.layers.9.self_attn.k_bmm_quantizer._amax"


class TestMoeAmaxFanoutMapping:
    """Tests for grouped-MoE expert amax fanout via all_gather across the EP group."""

    def test_inherits_amax_mapping(self):
        m = MoeAmaxFanoutMapping(
            "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax",
            ["model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"],
            num_experts=4,
        )
        assert isinstance(m, AmaxMapping)
        assert m.allow_hf_name_mismatch is True

    def test_empty_hf_patterns_raises(self):
        with pytest.raises(AssertionError):
            MoeAmaxFanoutMapping("mcore._amax", [], num_experts=4)

    def test_megatron_to_hf_fans_out_across_ep_ranks(self):
        """For EP=2, num_experts=4: rank-0's amax goes to experts 0,1; rank-1's to experts 2,3."""
        hf_pattern = "model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"
        megatron_param = "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax"
        m = MoeAmaxFanoutMapping(megatron_param, [hf_pattern], num_experts=4)

        rank0_weight = torch.tensor([10.0])
        rank1_weight = torch.tensor([20.0])

        def fake_all_gather(out_list, tensor, group=None):
            out_list[0].copy_(rank0_weight)
            out_list[1].copy_(rank1_weight)

        m.ep_group = object()  # any non-None sentinel; all_gather is mocked
        with (
            patch.object(ReplicatedMapping, "megatron_to_hf", return_value={hf_pattern: rank0_weight}),
            patch.object(MoeAmaxFanoutMapping, "ep_size", new=2),
            patch("torch.distributed.all_gather", side_effect=fake_all_gather),
        ):
            result = m.megatron_to_hf(rank0_weight, None)

        expected_keys = {
            "model.layers.0.mlp.experts.0.gate_proj.weight_quantizer._amax",
            "model.layers.0.mlp.experts.1.gate_proj.weight_quantizer._amax",
            "model.layers.0.mlp.experts.2.gate_proj.weight_quantizer._amax",
            "model.layers.0.mlp.experts.3.gate_proj.weight_quantizer._amax",
        }
        assert set(result.keys()) == expected_keys
        # rank-0 amax fanned out to experts 0, 1
        assert torch.equal(result["model.layers.0.mlp.experts.0.gate_proj.weight_quantizer._amax"], rank0_weight)
        assert torch.equal(result["model.layers.0.mlp.experts.1.gate_proj.weight_quantizer._amax"], rank0_weight)
        # rank-1 amax fanned out to experts 2, 3
        assert torch.equal(result["model.layers.0.mlp.experts.2.gate_proj.weight_quantizer._amax"], rank1_weight)
        assert torch.equal(result["model.layers.0.mlp.experts.3.gate_proj.weight_quantizer._amax"], rank1_weight)

    def test_megatron_to_hf_ep1_no_all_gather(self):
        """EP=1: the single rank's amax is broadcast to all expert names with no all_gather call."""
        hf_pattern = "model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"
        m = MoeAmaxFanoutMapping(
            "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax",
            [hf_pattern],
            num_experts=2,
        )
        weight = torch.tensor([7.0])

        with (
            patch.object(ReplicatedMapping, "megatron_to_hf", return_value={hf_pattern: weight}),
            patch.object(MoeAmaxFanoutMapping, "ep_size", new=1),
            patch("torch.distributed.all_gather") as mock_all_gather,
        ):
            result = m.megatron_to_hf(weight, None)

        mock_all_gather.assert_not_called()
        assert set(result.keys()) == {
            "model.layers.0.mlp.experts.0.gate_proj.weight_quantizer._amax",
            "model.layers.0.mlp.experts.1.gate_proj.weight_quantizer._amax",
        }
        for v in result.values():
            assert torch.equal(v, weight)

    def test_resolve_preserves_expert_wildcard(self):
        """resolve() should fill layer wildcards but leave the expert wildcard intact."""
        m = MoeAmaxFanoutMapping(
            "decoder.layers.*.mlp.experts.linear_fc1.weight_quantizer._amax",
            [
                "model.layers.*.mlp.experts.*.gate_proj.weight_quantizer._amax",
                "model.layers.*.mlp.experts.*.up_proj.weight_quantizer._amax",
            ],
            num_experts=4,
        )
        resolved = m.resolve(("3",))
        assert isinstance(resolved, MoeAmaxFanoutMapping)
        assert resolved.megatron_param == "decoder.layers.3.mlp.experts.linear_fc1.weight_quantizer._amax"
        assert resolved.hf_patterns == [
            "model.layers.3.mlp.experts.*.gate_proj.weight_quantizer._amax",
            "model.layers.3.mlp.experts.*.up_proj.weight_quantizer._amax",
        ]
        assert resolved.num_experts == 4

    def test_megatron_to_hf_ep_must_divide_num_experts(self):
        m = MoeAmaxFanoutMapping(
            "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax",
            ["model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"],
            num_experts=3,
        )
        weight = torch.tensor([1.0])
        m.ep_group = object()
        with (
            patch.object(
                ReplicatedMapping,
                "megatron_to_hf",
                return_value={"model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax": weight},
            ),
            patch.object(MoeAmaxFanoutMapping, "ep_size", new=2),
        ):
            with pytest.raises(RuntimeError, match="must be divisible"):
                m.megatron_to_hf(weight, None)

    def test_get_num_experts_from_megatron_module_config_num_moe_experts(self):
        """Production path: num_experts not in ctor; pulled from megatron_module.config.num_moe_experts."""
        hf_pattern = "model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"
        megatron_param = "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax"
        m = MoeAmaxFanoutMapping(megatron_param, [hf_pattern])  # no num_experts kwarg
        assert m.num_experts is None

        megatron_module = SimpleNamespace(config=SimpleNamespace(num_moe_experts=4))

        rank0_weight = torch.tensor([10.0])
        rank1_weight = torch.tensor([20.0])

        def fake_all_gather(out_list, tensor, group=None):
            out_list[0].copy_(rank0_weight)
            out_list[1].copy_(rank1_weight)

        m.ep_group = object()
        with (
            patch.object(ReplicatedMapping, "megatron_to_hf", return_value={hf_pattern: rank0_weight}),
            patch.object(MoeAmaxFanoutMapping, "ep_size", new=2),
            patch.object(MoeAmaxFanoutMapping, "broadcast_obj_from_pp_rank", side_effect=lambda obj, **_: obj),
            patch("torch.distributed.all_gather", side_effect=fake_all_gather),
        ):
            result = m.megatron_to_hf(rank0_weight, megatron_module)

        # rank-0 amax fanned out to experts 0, 1
        assert torch.equal(result["model.layers.0.mlp.experts.0.gate_proj.weight_quantizer._amax"], rank0_weight)
        assert torch.equal(result["model.layers.0.mlp.experts.1.gate_proj.weight_quantizer._amax"], rank0_weight)
        # rank-1 amax fanned out to experts 2, 3
        assert torch.equal(result["model.layers.0.mlp.experts.2.gate_proj.weight_quantizer._amax"], rank1_weight)
        assert torch.equal(result["model.layers.0.mlp.experts.3.gate_proj.weight_quantizer._amax"], rank1_weight)

    def test_get_num_experts_from_megatron_module_config_num_experts_fallback(self):
        """Fallback resolution: when num_moe_experts is missing, num_experts on config is used."""
        hf_pattern = "model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"
        megatron_param = "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax"
        m = MoeAmaxFanoutMapping(megatron_param, [hf_pattern])  # no num_experts kwarg

        # config has num_experts (not num_moe_experts) — exercises the fallback attribute.
        megatron_module = SimpleNamespace(config=SimpleNamespace(num_experts=4))

        rank0_weight = torch.tensor([10.0])
        rank1_weight = torch.tensor([20.0])

        def fake_all_gather(out_list, tensor, group=None):
            out_list[0].copy_(rank0_weight)
            out_list[1].copy_(rank1_weight)

        m.ep_group = object()
        with (
            patch.object(ReplicatedMapping, "megatron_to_hf", return_value={hf_pattern: rank0_weight}),
            patch.object(MoeAmaxFanoutMapping, "ep_size", new=2),
            patch.object(MoeAmaxFanoutMapping, "broadcast_obj_from_pp_rank", side_effect=lambda obj, **_: obj),
            patch("torch.distributed.all_gather", side_effect=fake_all_gather),
        ):
            result = m.megatron_to_hf(rank0_weight, megatron_module)

        # Same EP=2 / 4-expert layout: rank-0 -> {0,1}, rank-1 -> {2,3}.
        assert torch.equal(result["model.layers.0.mlp.experts.0.gate_proj.weight_quantizer._amax"], rank0_weight)
        assert torch.equal(result["model.layers.0.mlp.experts.1.gate_proj.weight_quantizer._amax"], rank0_weight)
        assert torch.equal(result["model.layers.0.mlp.experts.2.gate_proj.weight_quantizer._amax"], rank1_weight)
        assert torch.equal(result["model.layers.0.mlp.experts.3.gate_proj.weight_quantizer._amax"], rank1_weight)

    def test_get_num_experts_from_megatron_module_config_n_routed_experts_fallback(self):
        m = MoeAmaxFanoutMapping(
            "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax",
            ["model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"],
        )
        megatron_module = SimpleNamespace(config=SimpleNamespace(n_routed_experts=8))

        assert m._get_num_experts(megatron_module) == 8

    def test_megatron_to_hf_raises_when_num_experts_undeterminable(self):
        """If num_experts is neither in the ctor nor on the module/config, raise RuntimeError."""
        hf_pattern = "model.layers.0.mlp.experts.*.gate_proj.weight_quantizer._amax"
        megatron_param = "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax"
        m = MoeAmaxFanoutMapping(megatron_param, [hf_pattern], num_experts=None)

        # Config exposes none of num_moe_experts / num_experts / n_routed_experts.
        megatron_module = SimpleNamespace(config=SimpleNamespace())

        weight = torch.tensor([1.0])
        m.ep_group = object()
        with (
            patch.object(ReplicatedMapping, "megatron_to_hf", return_value={hf_pattern: weight}),
            patch.object(MoeAmaxFanoutMapping, "ep_size", new=1),
            patch.object(MoeAmaxFanoutMapping, "broadcast_obj_from_pp_rank", side_effect=lambda obj, **_: obj),
        ):
            with pytest.raises(RuntimeError, match="Could not determine num_experts"):
                m.megatron_to_hf(weight, megatron_module)


class TestMoeQuantMappingRegistryIntegration:
    """Registry-level resolution must produce MoeAmaxFanoutMapping for `.weight*` MoE mappings."""

    @pytest.fixture
    def moe_like_mappings(self):
        return [
            AutoMapping(
                "decoder.layers.*.mlp.experts.linear_fc2.weight*",
                "model.layers.*.mlp.experts.*.down_proj.weight",
            ),
        ]

    @pytest.fixture
    def registry(self, moe_like_mappings):
        with patch.dict(os.environ, {"ENABLE_BRIDGE_QUANT_MAPPING": "1"}, clear=False):
            return MegatronMappingRegistry(*moe_like_mappings)

    def test_moe_weight_star_resolves_to_moe_fanout(self, registry):
        m = registry.megatron_to_hf_lookup("decoder.layers.0.mlp.experts.linear_fc2.weight_quantizer._amax")
        assert m is not None, "registry did not resolve MoE amax mapping"
        assert isinstance(m, MoeAmaxFanoutMapping)
        # Layer wildcard should be resolved; expert wildcard preserved for runtime fanout.
        assert m.megatron_param == "decoder.layers.0.mlp.experts.linear_fc2.weight_quantizer._amax"
        assert m.hf_patterns == [
            "model.layers.0.mlp.experts.*.down_proj.weight_quantizer._amax",
        ]

    def test_moe_weight_star_amax_is_export_only(self, registry):
        hf_name = "model.layers.0.mlp.experts.1.down_proj.weight_quantizer._amax"
        m = registry.megatron_to_hf_lookup("decoder.layers.0.mlp.experts.linear_fc2.weight_quantizer._amax")

        assert m is not None
        assert m.hf_param == {}
        assert registry.hf_to_megatron_lookup(hf_name) is None
        assert m.hf_to_megatron({}, SimpleNamespace()) is None

    def test_moe_weight_star_amax_stream_import_yields_no_weight(self, registry):
        from megatron.bridge.models.conversion import model_bridge as mb
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

        class DummyBridge(MegatronModelBridge):
            def provider_bridge(self, hf_pretrained):
                return None

            def mapping_registry(self):
                return registry

        class EmptyState(dict):
            def __init__(self, hf_keys):
                super().__init__()
                self.source = SimpleNamespace(get_all_keys=lambda: hf_keys)

        local_name = "decoder.layers.0.mlp.experts.linear_fc2.weight_quantizer._amax"
        hf_name = "model.layers.0.mlp.experts.1.down_proj.weight_quantizer._amax"
        model_config = SimpleNamespace(num_moe_experts=2, share_embeddings_and_output_weights=False)
        model = SimpleNamespace(
            config=model_config,
            named_parameters=lambda: iter(()),
        )
        module = SimpleNamespace(config=model.config)
        param = torch.tensor([1.0])
        hf_pretrained = SimpleNamespace(state=EmptyState({hf_name}))
        bridge = DummyBridge()

        with (
            patch.object(bridge, "_megatron_global_param_names_all_pp_ranks", return_value=[local_name]),
            patch.object(mb, "unwrap_model", return_value=[model]),
            patch.object(mb, "_megatron_local_name_to_global", side_effect=lambda _models, _config, name, *_: name),
            patch.object(mb, "persistent_buffers", return_value=[(local_name, param)]),
            patch.object(mb, "get_module_and_param_from_name", return_value=(module, param)),
            patch.object(mb.parallel_state, "get_pipeline_model_parallel_rank", return_value=0),
        ):
            tasks = bridge.build_conversion_tasks(hf_pretrained, [model])
            result = list(bridge.stream_weights_hf_to_megatron(hf_pretrained, [model], tasks))

        assert len(tasks) == 1
        assert tasks[0] is not None
        assert tasks[0].megatron_module is module
        assert result == []

    def test_sequential_mlp_local_expert_weight_uses_regular_expert_mapping(self):
        mappings = [
            AutoMapping(
                "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                "model.layers.*.mlp.experts.*.gate_proj.weight",
            ),
        ]
        with patch.dict(os.environ, {"ENABLE_BRIDGE_QUANT_MAPPING": "1"}, clear=False):
            registry = MegatronMappingRegistry(*mappings)

        m = registry.megatron_to_hf_lookup(
            "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight_quantizer._amax"
        )
        assert m is not None, "registry did not resolve SequentialMLP amax mapping"
        assert isinstance(m, AmaxMapping)
        assert not isinstance(m, MoeAmaxFanoutMapping)
        assert m.is_expert is True
        assert m.megatron_param == "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight_quantizer._amax"
        assert m.hf_param == "model.layers.0.mlp.experts.1.gate_proj.weight_quantizer._amax"


class TestConvertToAmaxMapMoeWeightWildcard:
    """convert_to_amax_map should construct MoeAmaxFanoutMapping for `.weight*` mappings."""

    def test_grouped_moe_weight_star_creates_moe_fanout(self):
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
        from megatron.bridge.models.conversion.quant_mapping import convert_to_amax_map

        moe_mapping = AutoMapping(
            megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
            hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
        )
        result = convert_to_amax_map([moe_mapping])
        assert len(result) == 1
        out = result[0]
        assert isinstance(out, MoeAmaxFanoutMapping)
        assert out.megatron_param == "decoder.layers.*.mlp.experts.linear_fc2.weight_quantizer._amax"
        assert out.hf_patterns == [
            "model.layers.*.mlp.experts.*.down_proj.weight_quantizer._amax",
        ]

    def test_grouped_moe_weight_star_converts_dict_hf_params(self):
        from megatron.bridge.models.conversion.quant_mapping import convert_to_amax_map

        moe_mapping = GatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            up="model.layers.*.mlp.experts.*.up_proj.weight",
        )
        result = convert_to_amax_map([moe_mapping])
        assert len(result) == 1
        out = result[0]
        assert isinstance(out, MoeAmaxFanoutMapping)
        assert out.hf_patterns == [
            "model.layers.*.mlp.experts.*.gate_proj.weight_quantizer._amax",
            "model.layers.*.mlp.experts.*.up_proj.weight_quantizer._amax",
        ]


class TestEPRenumberRegexSkipsAmax:
    """Regression: `_megatron_local_name_to_global` must skip amax buffers under EP renumber.

    Before the regex fix, any name containing the substring ``.weight`` was sent through
    ``_update_expert_number``. For names like
    ``decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax`` the body called
    ``int(param_name.split(".weight")[-1])`` = ``int("_quantizer._amax")`` which raises
    ``ValueError`` and aborts quantized MoE export. The fix anchors the match to a digit
    suffix (``\\.weight\\d+$`` / ``\\.bias\\d+$``) so quantizer buffer names skip
    renumbering cleanly.
    """

    # Representative param names produced by Megatron-Core for grouped-MoE experts:
    #   - .weight0 / .weight1 / ...   per-expert weight (must renumber)
    #   - .bias0 / .bias1 / ...        per-expert bias (must renumber)
    #   - .weight_quantizer._amax      shared quantizer buffer (must NOT renumber)
    #   - .input_quantizer._amax       shared quantizer buffer (must NOT renumber)
    AMAX_NAMES = [
        "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax",
        "decoder.layers.0.mlp.experts.linear_fc1.input_quantizer._amax",
        "decoder.layers.0.mlp.experts.linear_fc2.weight_quantizer._amax",
        "decoder.layers.3.mlp.experts.linear_fc1.weight_quantizer._amax",
    ]
    EXPERT_WEIGHT_NAMES = [
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "decoder.layers.0.mlp.experts.linear_fc1.weight7",
        "decoder.layers.0.mlp.experts.linear_fc2.weight15",
    ]
    EXPERT_BIAS_NAMES = [
        "decoder.layers.0.mlp.experts.linear_fc1.bias0",
        "decoder.layers.0.mlp.experts.linear_fc2.bias3",
    ]

    @pytest.mark.parametrize("name", AMAX_NAMES)
    def test_regex_does_not_match_amax_buffers(self, name):
        assert re.search(r"\.weight\d+$", name) is None
        assert re.search(r"\.bias\d+$", name) is None

    @pytest.mark.parametrize("name", EXPERT_WEIGHT_NAMES)
    def test_regex_matches_expert_weight_indices(self, name):
        assert re.search(r"\.weight\d+$", name) is not None
        assert re.search(r"\.bias\d+$", name) is None

    @pytest.mark.parametrize("name", EXPERT_BIAS_NAMES)
    def test_regex_matches_expert_bias_indices(self, name):
        assert re.search(r"\.bias\d+$", name) is not None
        assert re.search(r"\.weight\d+$", name) is None

    def _call_local_to_global(self, param_name, num_moe_experts=4, ep_size=2, ep_rank=0):
        """Invoke the real `_megatron_local_name_to_global` with parallel_state mocked.

        Mocks ``parallel_state.get_expert_model_parallel_group`` and
        ``parallel_state.get_pipeline_model_parallel_group`` plus ``get_pg_size`` so the
        function takes the EP-renumber branch (PP=1 to keep the test focused on EP).
        """
        from megatron.bridge.models.conversion import model_bridge as mb

        ep_group = MagicMock()
        ep_group.size.return_value = ep_size
        ep_group.rank.return_value = ep_rank
        pp_group = MagicMock()

        config = SimpleNamespace(num_moe_experts=num_moe_experts)

        def _pg_size(group):
            if group is ep_group:
                return ep_size
            return 1  # PP=1 -> skip the PP layer-renumber branch

        with (
            patch.object(mb.parallel_state, "get_expert_model_parallel_group", return_value=ep_group),
            patch.object(mb.parallel_state, "get_pipeline_model_parallel_group", return_value=pp_group),
            patch.object(mb, "get_pg_size", side_effect=_pg_size),
        ):
            return mb._megatron_local_name_to_global(models=None, config=config, param_name=param_name)

    @pytest.mark.parametrize("name", AMAX_NAMES)
    def test_amax_name_passes_through_unchanged_under_ep(self, name):
        """Quantizer amax buffers must NOT go through EP renumbering."""
        out = self._call_local_to_global(name, num_moe_experts=4, ep_size=2, ep_rank=0)
        assert out == name

    def test_expert_weight_renumbered_under_ep(self):
        """A real per-expert weight name is renumbered local -> global on a non-zero EP rank."""
        # EP=2, num_experts=4 -> 2 experts per rank. Rank 1 owns experts {2, 3}.
        # Local weight0 on rank 1 -> global weight2.
        out = self._call_local_to_global(
            "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            num_moe_experts=4,
            ep_size=2,
            ep_rank=1,
        )
        assert out == "decoder.layers.0.mlp.experts.linear_fc1.weight2"

    def test_expert_bias_renumbered_under_ep(self):
        """A real per-expert bias name is renumbered local -> global on a non-zero EP rank."""
        out = self._call_local_to_global(
            "decoder.layers.0.mlp.experts.linear_fc1.bias1",
            num_moe_experts=4,
            ep_size=2,
            ep_rank=1,
        )
        assert out == "decoder.layers.0.mlp.experts.linear_fc1.bias3"
