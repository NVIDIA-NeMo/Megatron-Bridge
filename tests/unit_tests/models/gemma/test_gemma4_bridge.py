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

"""Unit tests for Gemma4Bridge (CausalLM text-only)."""

from collections import Counter
from unittest.mock import Mock

import pytest
import torch
from transformers import Gemma4TextConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma4_bridge import (
    Gemma4Bridge,
    _infer_attn_pattern,
)
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_hf_config_moe():
    """Flat Gemma4 CausalLM config (MoE: 26B-A4B)."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 62
    cfg.hidden_size = 2816
    cfg.intermediate_size = 2112
    cfg.moe_intermediate_size = 704
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 4
    cfg.head_dim = 256
    cfg.global_head_dim = 512
    cfg.num_global_key_value_heads = 2
    cfg.attention_k_eq_v = True
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
def mock_hf_config_dense():
    """Flat Gemma4 CausalLM config (Dense: enable_moe_block=False)."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 62
    cfg.hidden_size = 2816
    cfg.intermediate_size = 2112
    cfg.moe_intermediate_size = 1408
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 4
    cfg.head_dim = 256
    cfg.global_head_dim = 512
    cfg.num_global_key_value_heads = 2
    cfg.attention_k_eq_v = True
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
    cfg.enable_moe_block = False
    cfg.use_double_wide_mlp = True
    cfg.num_kv_shared_layers = 20
    cfg.num_experts = 256
    cfg.top_k_experts = 16
    cfg.layer_types = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
    cfg.final_logit_softcapping = 30.0
    return cfg


@pytest.fixture
def mock_pretrained_moe(mock_hf_config_moe):
    p = Mock(spec=PreTrainedCausalLM)
    p.config = mock_hf_config_moe
    return p


@pytest.fixture
def mock_pretrained_dense(mock_hf_config_dense):
    p = Mock(spec=PreTrainedCausalLM)
    p.config = mock_hf_config_dense
    return p


@pytest.fixture
def bridge():
    return Gemma4Bridge()


# ===========================================================================
# Registration
# ===========================================================================


class TestGemma4BridgeRegistration:
    def test_is_subclass_of_model_bridge(self):
        assert issubclass(Gemma4Bridge, MegatronModelBridge)


# ===========================================================================
# provider_bridge — MoE path
# ===========================================================================


class TestGemma4BridgeProviderBridgeMoE:
    def test_returns_moe_provider(self, bridge, mock_pretrained_moe):
        assert isinstance(bridge.provider_bridge(mock_pretrained_moe), Gemma4ModelProvider)

    def test_basic_transformer_config(self, bridge, mock_pretrained_moe):
        p = bridge.provider_bridge(mock_pretrained_moe)
        assert p.num_layers == 62
        assert p.hidden_size == 2816
        assert p.num_attention_heads == 8
        assert p.num_query_groups == 4
        assert p.kv_channels == 256
        assert p.vocab_size == 262144
        assert p.seq_length == 131072
        assert p.init_method_std == 0.02
        assert p.layernorm_epsilon == 1e-6

    def test_moe_config(self, bridge, mock_pretrained_moe):
        p = bridge.provider_bridge(mock_pretrained_moe)
        assert p.num_moe_experts == 128
        assert p.moe_router_topk == 8
        assert p.moe_ffn_hidden_size == 704
        assert p.moe_shared_expert_intermediate_size == 2112
        assert p.moe_layer_freq == 1
        assert p.moe_shared_expert_overlap is False
        assert p.moe_shared_expert_gate is False

    def test_window_size(self, bridge, mock_pretrained_moe):
        assert bridge.provider_bridge(mock_pretrained_moe).window_size == 1024

    def test_rotary_base_tuple(self, bridge, mock_pretrained_moe):
        rb = bridge.provider_bridge(mock_pretrained_moe).rotary_base
        assert isinstance(rb, tuple) and len(rb) == 2
        assert rb[0] == 10000.0
        assert rb[1] == 1000000.0

    def test_softmax_scale_is_one(self, bridge, mock_pretrained_moe):
        assert bridge.provider_bridge(mock_pretrained_moe).softmax_scale == 1.0

    def test_qk_layernorm_enabled(self, bridge, mock_pretrained_moe):
        assert bridge.provider_bridge(mock_pretrained_moe).qk_layernorm is True

    def test_global_attention_config(self, bridge, mock_pretrained_moe):
        p = bridge.provider_bridge(mock_pretrained_moe)
        assert p.global_head_dim == 512
        assert p.num_global_key_value_heads == 2
        assert p.global_rotary_percent == 0.25
        assert p.attention_k_eq_v is True

    def test_interleaved_attn_pattern(self, bridge, mock_pretrained_moe):
        assert bridge.provider_bridge(mock_pretrained_moe).interleaved_attn_pattern == (5, 1)

    def test_logit_softcapping(self, bridge, mock_pretrained_moe):
        assert bridge.provider_bridge(mock_pretrained_moe).final_logit_softcapping == 30.0

    def test_dtype_is_bf16(self, bridge, mock_pretrained_moe):
        p = bridge.provider_bridge(mock_pretrained_moe)
        assert p.bf16 is True
        assert p.params_dtype == torch.bfloat16

    def test_different_hidden_sizes(self, bridge, mock_pretrained_moe):
        for hs in [2048, 2816, 4096]:
            mock_pretrained_moe.config.hidden_size = hs
            assert bridge.provider_bridge(mock_pretrained_moe).hidden_size == hs

    def test_different_layer_counts(self, bridge, mock_pretrained_moe):
        for nl in [32, 46, 62]:
            mock_pretrained_moe.config.num_hidden_layers = nl
            assert bridge.provider_bridge(mock_pretrained_moe).num_layers == nl

    def test_vocab_size_variants(self, bridge, mock_pretrained_moe):
        for vs in [256000, 262144, 300000]:
            mock_pretrained_moe.config.vocab_size = vs
            assert bridge.provider_bridge(mock_pretrained_moe).vocab_size == vs


# ===========================================================================
# provider_bridge — Dense path
# ===========================================================================


class TestGemma4BridgeProviderBridgeDense:
    def test_returns_dense_provider(self, bridge, mock_pretrained_dense):
        assert isinstance(bridge.provider_bridge(mock_pretrained_dense), Gemma4DenseProvider)

    def test_basic_config_preserved(self, bridge, mock_pretrained_dense):
        p = bridge.provider_bridge(mock_pretrained_dense)
        assert p.num_layers == 62
        assert p.hidden_size == 2816
        assert p.num_attention_heads == 8
        assert p.num_query_groups == 4
        assert p.vocab_size == 262144
        assert p.num_moe_experts is None

    def test_does_not_return_moe_provider(self, bridge, mock_pretrained_dense):
        assert not isinstance(bridge.provider_bridge(mock_pretrained_dense), Gemma4ModelProvider)

    def test_logit_softcapping(self, bridge, mock_pretrained_dense):
        """Dense providers must preserve HF's final-logit softcap configuration."""
        assert bridge.provider_bridge(mock_pretrained_dense).final_logit_softcapping == 30.0

    def test_missing_logit_softcapping_disables_it(self, bridge, mock_pretrained_dense):
        del mock_pretrained_dense.config.final_logit_softcapping
        assert bridge.provider_bridge(mock_pretrained_dense).final_logit_softcapping is None


@pytest.mark.parametrize("provider_cls", [Gemma4DenseProvider, Gemma4ModelProvider])
def test_megatron_to_hf_config_preserves_final_logit_softcapping(provider_cls):
    provider = provider_cls(final_logit_softcapping=17.0)
    assert Gemma4Bridge.megatron_to_hf_config(provider)["final_logit_softcapping"] == 17.0


class TestGemma4DenseArchitectureConfig:
    def test_double_wide_mlp_config(self, bridge, mock_pretrained_dense):
        """E2B shared-KV layers require twice the base FFN width."""
        provider = bridge.provider_bridge(mock_pretrained_dense)

        assert provider.use_double_wide_mlp is True
        assert provider.num_kv_shared_layers == 20

    def test_dense_attention_config(self, bridge, mock_pretrained_dense):
        provider = bridge.provider_bridge(mock_pretrained_dense)

        assert provider.window_size == (1023, 0)
        assert provider.attention_k_eq_v is True


class TestGemma4BridgeConfigExport:
    def test_dense_architecture_fields(self):
        provider = Gemma4DenseProvider(
            window_size=(1023, 0),
            attention_k_eq_v=True,
            num_kv_shared_layers=20,
            use_double_wide_mlp=True,
        )

        config = Gemma4Bridge.megatron_to_hf_config(provider)

        assert config["enable_moe_block"] is False
        assert config["sliding_window"] == 1024
        assert config["attention_k_eq_v"] is True
        assert config["dtype"] == "bfloat16"
        assert config["global_head_dim"] == provider.global_kv_channels
        assert config["layer_types"][:6] == ["sliding_attention"] * 5 + ["full_attention"]
        assert config["num_global_key_value_heads"] == provider.num_global_query_groups
        assert config["num_kv_shared_layers"] == 20
        assert config["rope_parameters"]["full_attention"]["partial_rotary_factor"] == 0.25
        assert config["use_double_wide_mlp"] is True

    def test_dense_architecture_fields_after_checkpoint_config_reload(self):
        """Checkpoint YAML reloads MCore's window tuple as a list."""
        provider = Gemma4DenseProvider(window_size=[511, 0])

        config = Gemma4Bridge.megatron_to_hf_config(provider)

        assert config["sliding_window"] == 512
        Gemma4TextConfig(**config)

    def test_moe_architecture_fields(self):
        provider = Gemma4ModelProvider(
            attention_k_eq_v=True,
            num_layers=6,
            num_moe_experts=128,
            moe_router_topk=8,
            moe_ffn_hidden_size=704,
            moe_shared_expert_intermediate_size=2112,
        )

        config = Gemma4Bridge.megatron_to_hf_config(provider)

        assert config["enable_moe_block"] is True
        assert config["attention_k_eq_v"] is True
        assert config["dtype"] == "bfloat16"
        assert config["layer_types"][:6] == ["sliding_attention"] * 5 + ["full_attention"]
        assert config["num_global_key_value_heads"] == provider.num_global_key_value_heads
        assert config["num_experts"] == 128
        assert config["top_k_experts"] == 8
        assert config["moe_intermediate_size"] == 704
        assert config["intermediate_size"] == 2112

    def test_moe_architecture_fields_after_checkpoint_config_reload(self):
        """Checkpoint YAML reloads the interleaved attention tuple as a list."""
        provider = Gemma4ModelProvider(
            interleaved_attn_pattern=[5, 1],
            num_layers=30,
            vocab_size=262_144,
        )

        config = Gemma4Bridge.megatron_to_hf_config(provider)

        assert len(config["layer_types"]) == provider.num_layers
        assert config["layer_types"][:6] == ["sliding_attention"] * 5 + ["full_attention"]
        Gemma4TextConfig(**config)

    def test_moe_rope_fields_after_checkpoint_config_reload(self):
        """Checkpoint YAML reloads the dual rotary-base tuple as a list."""
        provider = Gemma4ModelProvider(
            rotary_base=[10_000, 1_000_000],
            vocab_size=262_144,
        )

        config = Gemma4Bridge.megatron_to_hf_config(provider)

        rope_parameters = config["rope_parameters"]
        assert rope_parameters["sliding_attention"]["rope_theta"] == 10_000
        assert rope_parameters["full_attention"]["rope_theta"] == 1_000_000
        Gemma4TextConfig(**config)


# ===========================================================================
# _infer_attn_pattern helper
# ===========================================================================


class TestInferAttnPattern:
    def test_5_sliding_1_global(self):
        lt = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
        assert _infer_attn_pattern(lt) == (5, 1)

    def test_all_sliding(self):
        assert _infer_attn_pattern(["sliding_attention"] * 8) == (8, 0)

    def test_single_sliding_then_global(self):
        assert _infer_attn_pattern(["sliding_attention", "full_attention", "sliding_attention"]) == (1, 1)

    def test_consecutive_global_layers(self):
        lt = ["sliding_attention"] * 3 + ["full_attention", "full_attention"]
        assert _infer_attn_pattern(lt) == (3, 2)

    def test_global_at_start(self):
        assert _infer_attn_pattern(["full_attention"] + ["sliding_attention"] * 5) == (0, 1)


# ===========================================================================
# maybe_modify_loaded_hf_weight
# ===========================================================================


class TestMaybeModifyLoadedHFWeight:
    def _make_sd(self, layer_idx=0, hidden=8, num_experts=4):
        p = f"model.layers.{layer_idx}"
        return {
            f"{p}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{p}.self_attn.k_proj.weight": torch.randn(hidden // 2, hidden),
            f"{p}.router.proj.weight": torch.randn(num_experts, hidden),
            f"{p}.router.scale": torch.ones(hidden),
            f"{p}.pre_feedforward_layernorm_2.weight": torch.ones(hidden) * 2.0,
            f"{p}.mlp.gate_proj.weight": torch.randn(16, hidden),
            f"{p}.mlp.up_proj.weight": torch.randn(16, hidden),
            f"{p}.pre_feedforward_layernorm.weight": torch.ones(hidden) * 3.0,
        }

    def test_kv_synthesis_when_both_absent(self, bridge):
        sd = self._make_sd()
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }
        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert isinstance(result, dict)
        torch.testing.assert_close(result["v"], result["k"])

    def test_kv_synthesis_uses_dense_provider_head_metadata(self, bridge, mock_pretrained_dense):
        bridge.provider_bridge(mock_pretrained_dense)
        q_weight = torch.randn(16, 8)
        sd = {"model.layers.0.self_attn.q_proj.weight": q_weight}
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }

        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)

        # q_weight has 8 query heads and global K/V uses 2 heads in the fixture.
        assert result["k"].shape == (4, 8)
        assert result["v"].shape == (4, 8)

    def test_kv_synthesis_uses_hf_config_without_provider_bridge(self, bridge, mock_hf_config_dense):
        bridge.hf_config = mock_hf_config_dense
        mock_hf_config_dense.num_attention_heads = 6
        mock_hf_config_dense.num_key_value_heads = 5
        mock_hf_config_dense.num_global_key_value_heads = 3
        mock_hf_config_dense.layer_types = ["full_attention"]
        q_weight = torch.randn(24, 8)
        sd = {"model.layers.0.self_attn.q_proj.weight": q_weight}
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }

        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)

        assert result["k"].shape == (12, 8)
        assert result["v"].shape == (12, 8)

    def test_kv_passthrough_when_v_present(self, bridge):
        sd = self._make_sd()
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(4, 8)
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }
        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert result is not None

    def test_router_weight_passthrough(self, bridge):
        sd = self._make_sd(hidden=8)
        hf_param = "model.layers.0.router.proj.weight"

        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)

        torch.testing.assert_close(result, sd[hf_param], rtol=0, atol=0)

    def test_router_fusion_missing_keys_passthrough(self, bridge):
        sd = {"model.layers.0.router.proj.weight": torch.randn(4, 8)}
        result = bridge.maybe_modify_loaded_hf_weight("model.layers.0.router.proj.weight", sd)
        torch.testing.assert_close(result, sd["model.layers.0.router.proj.weight"])

    def test_shared_expert_weights_passthrough(self, bridge):
        sd = self._make_sd(hidden=8)
        hf_param = {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "up": "model.layers.0.mlp.up_proj.weight",
        }

        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)

        torch.testing.assert_close(result["gate"], sd[hf_param["gate"]], rtol=0, atol=0)
        torch.testing.assert_close(result["up"], sd[hf_param["up"]], rtol=0, atol=0)

    def test_shared_expert_fusion_missing_keys_passthrough(self, bridge):
        sd = {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(4, 8),
            "model.layers.0.mlp.up_proj.weight": torch.randn(4, 8),
        }
        hf_param = {"gate": "model.layers.0.mlp.gate_proj.weight", "up": "model.layers.0.mlp.up_proj.weight"}
        result = bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        torch.testing.assert_close(result["gate"], sd["model.layers.0.mlp.gate_proj.weight"])


# ===========================================================================
# maybe_modify_converted_hf_weight
# ===========================================================================


class TestMaybeModifyConvertedHFWeight:
    def _make_ref_sd(self, layer_idx=0, hidden=8, num_experts=4):
        p = f"model.layers.{layer_idx}"
        return {
            f"{p}.router.proj.weight": torch.randn(num_experts, hidden),
            f"{p}.router.scale": torch.ones(hidden),
            f"{p}.pre_feedforward_layernorm_2.weight": torch.ones(hidden) * 2.0,
            f"{p}.mlp.gate_proj.weight": torch.randn(16, hidden),
            f"{p}.mlp.up_proj.weight": torch.randn(16, hidden),
            f"{p}.pre_feedforward_layernorm.weight": torch.ones(hidden) * 3.0,
        }

    def test_drops_keys_absent_from_hf_sd(self, bridge):
        hf_sd = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)}
        converted = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8),
        }
        result = bridge.maybe_modify_converted_hf_weight(None, converted, hf_sd)
        assert "model.layers.0.self_attn.v_proj.weight" not in result
        assert "model.layers.0.self_attn.q_proj.weight" in result

    def test_router_weight_export_passthrough(self, bridge):
        ref_sd = self._make_ref_sd(hidden=8)
        converted = {"model.layers.0.router.proj.weight": torch.randn(4, 8)}

        result = bridge.maybe_modify_converted_hf_weight(None, converted, ref_sd)

        torch.testing.assert_close(
            result["model.layers.0.router.proj.weight"],
            converted["model.layers.0.router.proj.weight"],
            rtol=0,
            atol=0,
        )

    def test_shared_expert_weight_export_passthrough(self, bridge):
        ref_sd = self._make_ref_sd(hidden=8)
        converted = {"model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8)}

        result = bridge.maybe_modify_converted_hf_weight(None, converted, ref_sd)

        torch.testing.assert_close(
            result["model.layers.0.mlp.gate_proj.weight"],
            converted["model.layers.0.mlp.gate_proj.weight"],
            rtol=0,
            atol=0,
        )

    def test_empty_hf_state_dict_passthrough(self, bridge):
        converted = {"some.weight": torch.randn(4, 4)}
        result = bridge.maybe_modify_converted_hf_weight(None, converted, {})
        assert result is converted

    def test_none_hf_state_dict_passthrough(self, bridge):
        converted = {"some.weight": torch.randn(4, 4)}
        result = bridge.maybe_modify_converted_hf_weight(None, converted, None)
        assert result is converted


# ===========================================================================
# mapping_registry
# ===========================================================================


class TestGemma4BridgeMappingRegistry:
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

    def _collect_hf_targets(self, registry):
        targets = []
        for m in registry.mappings:
            hf = getattr(m, "hf_param", None)
            if isinstance(hf, dict):
                targets.extend(str(v) for v in hf.values())
            elif isinstance(hf, str):
                targets.append(hf)
        return targets

    def test_returns_registry(self, bridge):
        assert isinstance(bridge.mapping_registry(), MegatronMappingRegistry)

    def test_has_mappings(self, bridge):
        assert len(bridge.mapping_registry().mappings) > 0

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

    def test_has_direct_pre_shared_expert_norm_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert "decoder.layers.*.pre_shared_expert_layernorm.weight" in names
        assert "model.layers.*.pre_feedforward_layernorm.weight" in names

    def test_has_post_moe_layernorm(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("post_moe_layernorm" in n for n in names)

    def test_selects_dense_registry_from_hf_config_without_provider_bridge(self, bridge, mock_hf_config_dense):
        bridge.hf_config = mock_hf_config_dense

        names = self._collect_names(bridge.mapping_registry())

        assert "per_layer_embedding.weight" in names
        assert "decoder.layers.*.mlp.router.weight" not in names

    def test_selects_dense_registry_when_enable_moe_block_missing(self, bridge):
        bridge.hf_config = Mock(spec=[])

        names = self._collect_names(bridge.mapping_registry())

        assert "per_layer_embedding.weight" in names
        assert "decoder.layers.*.mlp.router.weight" not in names

    def test_selects_moe_registry_from_hf_config_without_provider_bridge(self, bridge, mock_hf_config_moe):
        bridge.hf_config = mock_hf_config_moe

        names = self._collect_names(bridge.mapping_registry())

        assert "decoder.layers.*.mlp.router.weight" in names
        assert "per_layer_embedding.weight" not in names

    def test_has_layer_scalar_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("layer_scalar" in n for n in names)

    def test_uses_causal_lm_prefix(self, bridge):
        """CausalLM bridge uses model.layers.* (not model.language_model.layers.*)."""
        names = self._collect_names(bridge.mapping_registry())
        hf_layer_names = [n for n in names if "layers" in n]
        assert all("language_model" not in n for n in hf_layer_names)

    def test_moe_registry_has_no_duplicate_non_layernorm_hf_targets(self, bridge):
        targets = self._collect_hf_targets(bridge.mapping_registry())
        duplicates = {
            name: count for name, count in Counter(targets).items() if count > 1 and "input_layernorm" not in name
        }
        assert duplicates == {}

    def test_moe_registry_does_not_map_plain_mlp_params(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert "decoder.layers.*.mlp.linear_fc1.weight" not in names
        assert "decoder.layers.*.mlp.linear_fc2.weight" not in names
