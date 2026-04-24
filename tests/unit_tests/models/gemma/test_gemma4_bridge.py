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

"""Unit tests for Gemma4Bridge (text-only CausalLM bridge)."""

import math
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge, _infer_attn_pattern
from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hf_config():
    """Flat Gemma4 CausalLM config (26B-A4B)."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 62
    cfg.hidden_size = 2816
    cfg.intermediate_size = 2112        # shared expert FFN
    cfg.moe_intermediate_size = 704     # routed expert FFN
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 4
    cfg.head_dim = 256
    cfg.global_head_dim = 512
    cfg.num_global_key_value_heads = 2
    cfg.initializer_range = 0.02
    cfg.rms_norm_eps = 1e-6
    cfg.vocab_size = 262144
    cfg.max_position_embeddings = 131072
    cfg.sliding_window = 1024
    cfg.rope_theta = 1000000.0
    cfg.rope_local_base_freq = 10000.0
    cfg.rope_parameters = {"full_attention": {"partial_rotary_factor": 0.25}}
    cfg.query_pre_attn_scalar = 1.0
    cfg.hidden_act = "gelu_pytorch_tanh"
    cfg.torch_dtype = "bfloat16"
    cfg.enable_moe_block = True
    cfg.num_experts = 128
    cfg.top_k_experts = 8
    cfg.layer_types = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
    cfg.final_logit_softcapping = 30.0
    return cfg


@pytest.fixture
def mock_pretrained(mock_hf_config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = mock_hf_config
    return pretrained


@pytest.fixture
def bridge():
    return Gemma4Bridge()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestGemma4BridgeRegistration:
    def test_is_subclass_of_model_bridge(self):
        assert issubclass(Gemma4Bridge, MegatronModelBridge)

    def test_registered_for_gemma4_causal_lm(self):
        # Verify bridge can be instantiated and has the right provider class
        b = Gemma4Bridge()
        assert b is not None

    def test_initialization(self, bridge):
        assert isinstance(bridge, Gemma4Bridge)

    def test_has_required_methods(self, bridge):
        assert callable(getattr(bridge, "provider_bridge", None))
        assert callable(getattr(bridge, "mapping_registry", None))
        assert callable(getattr(bridge, "maybe_modify_loaded_hf_weight", None))
        assert callable(getattr(bridge, "maybe_modify_converted_hf_weight", None))


# ---------------------------------------------------------------------------
# provider_bridge
# ---------------------------------------------------------------------------


class TestGemma4BridgeProviderBridge:
    def test_returns_provider_instance(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, Gemma4ModelProvider)

    def test_basic_transformer_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_layers == 62
        assert provider.hidden_size == 2816
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 4
        assert provider.kv_channels == 256
        assert provider.vocab_size == 262144
        assert provider.seq_length == 131072
        assert provider.init_method_std == 0.02
        assert provider.layernorm_epsilon == 1e-6

    def test_moe_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_moe_experts == 128
        assert provider.moe_router_topk == 8
        assert provider.moe_ffn_hidden_size == 704
        assert provider.moe_shared_expert_intermediate_size == 2112
        assert provider.moe_layer_freq == 1
        assert provider.moe_shared_expert_overlap is False
        assert provider.moe_shared_expert_gate is False

    def test_window_size(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.window_size == 1024

    def test_rotary_base_tuple(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        # Should be (local_freq, global_freq) tuple
        assert isinstance(provider.rotary_base, tuple)
        assert len(provider.rotary_base) == 2
        assert provider.rotary_base[0] == 10000.0   # rope_local_base_freq
        assert provider.rotary_base[1] == 1000000.0  # rope_theta

    def test_softmax_scale_is_one(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.softmax_scale == 1.0

    def test_qk_layernorm_enabled(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.qk_layernorm is True

    def test_global_attention_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.global_head_dim == 512
        assert provider.num_global_key_value_heads == 2

    def test_global_rotary_percent(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.global_rotary_percent == 0.25

    def test_interleaved_attn_pattern(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        # 5 sliding + 1 full pattern
        assert provider.interleaved_attn_pattern == (5, 1)

    def test_logit_softcapping(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.final_logit_softcapping == 30.0

    def test_dtype_is_bf16(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_different_hidden_sizes(self, bridge, mock_pretrained):
        for hidden_size in [2048, 2816, 4096]:
            mock_pretrained.config.hidden_size = hidden_size
            provider = bridge.provider_bridge(mock_pretrained)
            assert provider.hidden_size == hidden_size

    def test_different_layer_counts(self, bridge, mock_pretrained):
        for num_layers in [32, 46, 62]:
            mock_pretrained.config.num_hidden_layers = num_layers
            provider = bridge.provider_bridge(mock_pretrained)
            assert provider.num_layers == num_layers

    def test_vocab_size_variants(self, bridge, mock_pretrained):
        for vocab_size in [256000, 262144, 300000]:
            mock_pretrained.config.vocab_size = vocab_size
            provider = bridge.provider_bridge(mock_pretrained)
            assert provider.vocab_size == vocab_size


# ---------------------------------------------------------------------------
# _infer_attn_pattern
# ---------------------------------------------------------------------------


class TestInferAttnPattern:
    def test_5_sliding_1_global(self):
        layer_types = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
        assert _infer_attn_pattern(layer_types) == (5, 1)

    def test_all_sliding(self):
        layer_types = ["sliding_attention"] * 8
        assert _infer_attn_pattern(layer_types) == (8, 0)

    def test_single_sliding_then_global(self):
        layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
        assert _infer_attn_pattern(layer_types) == (1, 1)

    def test_consecutive_global_layers(self):
        # 3 sliding + 2 consecutive global
        layer_types = ["sliding_attention"] * 3 + ["full_attention", "full_attention"]
        assert _infer_attn_pattern(layer_types) == (3, 2)

    def test_global_at_start(self):
        layer_types = ["full_attention"] + ["sliding_attention"] * 5
        assert _infer_attn_pattern(layer_types) == (0, 1)


# ---------------------------------------------------------------------------
# maybe_modify_loaded_hf_weight
# ---------------------------------------------------------------------------


class TestMaybeModifyLoadedHFWeight:
    """Tests for weight modification during HF → Megatron loading."""

    def _make_state_dict(self, layer_idx=0, hidden=8, num_experts=4):
        """Build a minimal HF state dict for one MoE layer."""
        sd = {}
        prefix = f"model.layers.{layer_idx}"
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden // 2, hidden)
        # v_proj absent (global attention layer with K=V)
        sd[f"{prefix}.router.proj.weight"] = torch.randn(num_experts, hidden)
        sd[f"{prefix}.router.scale"] = torch.ones(hidden)
        sd[f"{prefix}.pre_feedforward_layernorm_2.weight"] = torch.ones(hidden) * 2.0
        sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(16, hidden)
        sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(16, hidden)
        sd[f"{prefix}.pre_feedforward_layernorm.weight"] = torch.ones(hidden) * 3.0
        return sd

    def test_kv_synthesis_when_v_proj_absent(self, bridge):
        """V is synthesized from K when v_proj is absent (global attention layer)."""
        sd = self._make_state_dict()
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",  # absent from sd
        }
        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert isinstance(result, dict)
        assert "v" in result
        # V should equal K
        torch.testing.assert_close(result["v"], result["k"])

    def test_kv_no_synthesis_when_v_present(self, bridge):
        """Normal QKV loading when v_proj is present (sliding layer)."""
        sd = self._make_state_dict()
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(4, 8)
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }
        # With v_proj present, base class handles it (no synthesis)
        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        # Should fall through to super() which just returns the base dict
        assert result is not None

    def test_router_weight_fusion(self, bridge):
        """Router weight is fused with scale * hidden^-0.5 / ln2_weight."""
        hidden = 8
        sd = self._make_state_dict(hidden=hidden)
        hf_param = "model.layers.0.router.proj.weight"

        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert isinstance(result, torch.Tensor)
        assert result.shape == sd[hf_param].shape

        # Verify: fused = orig * (scale * hidden^-0.5 / ln2_weight)
        # scale=1, ln2_weight=2.0 → factor = 1 * hidden^-0.5 / 2
        expected_factor = 1.0 * (hidden ** -0.5) / 2.0
        expected = (sd[hf_param].float() * expected_factor).to(sd[hf_param].dtype)
        torch.testing.assert_close(result, expected)

    def test_router_fusion_missing_keys_passthrough(self, bridge):
        """Router fusion is skipped if scale or ln2 keys are absent."""
        sd = {"model.layers.0.router.proj.weight": torch.randn(4, 8)}
        result = bridge.maybe_modify_loaded_hf_weight("model.layers.0.router.proj.weight", sd)
        torch.testing.assert_close(result, sd["model.layers.0.router.proj.weight"])

    def test_shared_expert_prenorm_fusion(self, bridge):
        """Shared expert gate/up weights are fused with pffl/pffl2 ratio."""
        hidden = 8
        sd = self._make_state_dict(hidden=hidden)
        hf_param = {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "up": "model.layers.0.mlp.up_proj.weight",
        }

        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert isinstance(result, dict)
        assert "gate" in result and "up" in result

        # Verify correction: pffl=3.0, pffl2=2.0 → ratio = 3/2 = 1.5
        correction = 3.0 / 2.0
        expected_gate = (sd["model.layers.0.mlp.gate_proj.weight"].float() * correction).to(
            sd["model.layers.0.mlp.gate_proj.weight"].dtype
        )
        torch.testing.assert_close(result["gate"], expected_gate)

    def test_shared_expert_fusion_missing_keys_passthrough(self, bridge):
        """Shared expert fusion is skipped if pffl/pffl2 keys are absent."""
        sd = {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(4, 8),
            "model.layers.0.mlp.up_proj.weight": torch.randn(4, 8),
        }
        hf_param = {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "up": "model.layers.0.mlp.up_proj.weight",
        }
        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert isinstance(result, dict)
        torch.testing.assert_close(result["gate"], sd["model.layers.0.mlp.gate_proj.weight"])


# ---------------------------------------------------------------------------
# maybe_modify_converted_hf_weight
# ---------------------------------------------------------------------------


class TestMaybeModifyConvertedHFWeight:
    """Tests for weight modification during Megatron → HF export."""

    def _make_ref_sd(self, layer_idx=0, hidden=8, num_experts=4):
        """Reference HF state dict (target of export)."""
        sd = {}
        prefix = f"model.layers.{layer_idx}"
        sd[f"{prefix}.router.proj.weight"] = torch.randn(num_experts, hidden)
        sd[f"{prefix}.router.scale"] = torch.ones(hidden)
        sd[f"{prefix}.pre_feedforward_layernorm_2.weight"] = torch.ones(hidden) * 2.0
        sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(16, hidden)
        sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(16, hidden)
        sd[f"{prefix}.pre_feedforward_layernorm.weight"] = torch.ones(hidden) * 3.0
        return sd

    def test_drops_synthesized_v_proj(self, bridge):
        """v_proj absent from original HF should not appear in exported weights."""
        hf_state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)}
        converted = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8),  # synthesized
        }
        result = bridge.maybe_modify_converted_hf_weight(None, converted, hf_state_dict)
        assert "model.layers.0.self_attn.v_proj.weight" not in result
        assert "model.layers.0.self_attn.q_proj.weight" in result

    def test_router_weight_unfusion(self, bridge):
        """Router weight unfusion inverts the import fusion."""
        hidden = 8
        ref_sd = self._make_ref_sd(hidden=hidden)

        # Simulate fused router weight (as it would be after import)
        factor = 1.0 * (hidden ** -0.5) / 2.0
        fused_router = (ref_sd["model.layers.0.router.proj.weight"].float() * factor).to(
            ref_sd["model.layers.0.router.proj.weight"].dtype
        )
        converted = {"model.layers.0.router.proj.weight": fused_router}

        result = bridge.maybe_modify_converted_hf_weight(None, converted, ref_sd)
        # Should recover original router weight
        torch.testing.assert_close(
            result["model.layers.0.router.proj.weight"],
            ref_sd["model.layers.0.router.proj.weight"],
            atol=1e-5, rtol=1e-5,
        )

    def test_shared_expert_gate_unfusion(self, bridge):
        """Gate/up unfusion inverts import prenorm fusion."""
        hidden = 8
        ref_sd = self._make_ref_sd(hidden=hidden)

        # Simulate fused gate weight (pffl=3, pffl2=2 → ratio=1.5)
        correction = 3.0 / 2.0
        fused_gate = (ref_sd["model.layers.0.mlp.gate_proj.weight"].float() * correction).to(
            ref_sd["model.layers.0.mlp.gate_proj.weight"].dtype
        )
        converted = {"model.layers.0.mlp.gate_proj.weight": fused_gate}

        result = bridge.maybe_modify_converted_hf_weight(None, converted, ref_sd)
        torch.testing.assert_close(
            result["model.layers.0.mlp.gate_proj.weight"],
            ref_sd["model.layers.0.mlp.gate_proj.weight"],
            atol=1e-5, rtol=1e-5,
        )

    def test_empty_hf_state_dict_passthrough(self, bridge):
        """Empty hf_state_dict is falsy → converted dict returned unchanged (early exit)."""
        converted = {"some.weight": torch.randn(4, 4)}
        result = bridge.maybe_modify_converted_hf_weight(None, converted, {})
        assert result is converted  # early return: not hf_state_dict → return as-is

    def test_none_hf_state_dict_passthrough(self, bridge):
        """Returns converted dict unchanged when hf_state_dict is None."""
        converted = {"some.weight": torch.randn(4, 4)}
        result = bridge.maybe_modify_converted_hf_weight(None, converted, None)
        assert result is converted


# ---------------------------------------------------------------------------
# mapping_registry
# ---------------------------------------------------------------------------


class TestGemma4BridgeMappingRegistry:
    def test_returns_registry(self, bridge):
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

    def test_has_mappings(self, bridge):
        assert len(bridge.mapping_registry().mappings) > 0

    def _collect_names(self, registry):
        names = []
        for m in registry.mappings:
            if hasattr(m, "megatron_param"):
                names.append(str(m.megatron_param))
            hf = getattr(m, "hf_param", None)
            if isinstance(hf, dict):
                names.extend(str(v) for v in hf.values())
            elif isinstance(hf, str):
                names.append(hf)
        return names

    def test_has_embeddings_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("embed_tokens" in n or "word_embeddings" in n for n in names)

    def test_has_final_norm_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("norm" in n for n in names)

    def test_has_qkv_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("linear_qkv" in n for n in names)

    def test_has_router_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("router" in n for n in names)

    def test_has_shared_expert_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("shared_experts" in n for n in names)

    def test_has_post_moe_layernorm(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("post_moe_layernorm" in n for n in names)

    def test_uses_causal_lm_prefix(self, bridge):
        """CausalLM bridge uses model.layers.* (not model.language_model.layers.*)."""
        names = self._collect_names(bridge.mapping_registry())
        hf_names = [n for n in names if "layers" in n]
        assert all("language_model" not in n for n in hf_names)

    def test_has_layer_scalar_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("layer_scalar" in n for n in names)
