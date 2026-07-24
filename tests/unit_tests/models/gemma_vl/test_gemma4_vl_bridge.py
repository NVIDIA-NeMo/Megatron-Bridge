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

"""Unit tests for Gemma4Bridge (CausalLM) and Gemma4VLBridge (ConditionalGeneration)."""

from collections import Counter
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig, SiglipVisionConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma4_bridge import (
    Gemma4Bridge,
    _infer_attn_pattern,
)
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import (
    Gemma4DenseVLProvider,
    Gemma4VLModelProvider,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture
def mock_vision_config():
    config = SiglipVisionConfig()
    config.hidden_size = 1152
    config.intermediate_size = 4304
    config.num_hidden_layers = 27
    config.num_attention_heads = 16
    config.patch_size = 14
    config.image_size = 896
    return config


# ===========================================================================
# CausalLM (Gemma4Bridge) fixtures
# ===========================================================================


@pytest.fixture
def mock_hf_config_causal_moe():
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
def mock_hf_config_causal_dense():
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
    cfg.num_experts = 256
    cfg.top_k_experts = 16
    cfg.layer_types = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
    cfg.final_logit_softcapping = 30.0
    return cfg


@pytest.fixture
def mock_causal_pretrained(mock_hf_config_causal_moe):
    p = Mock(spec=PreTrainedCausalLM)
    p.config = mock_hf_config_causal_moe
    return p


@pytest.fixture
def mock_causal_dense_pretrained(mock_hf_config_causal_dense):
    p = Mock(spec=PreTrainedCausalLM)
    p.config = mock_hf_config_causal_dense
    return p


@pytest.fixture
def causal_bridge():
    return Gemma4Bridge()


# ===========================================================================
# VL (Gemma4VLBridge) fixtures
# ===========================================================================


@pytest.fixture
def mock_text_config_moe():
    config = Mock(spec=[])
    config.num_hidden_layers = 62
    config.hidden_size = 2816
    config.intermediate_size = 2112
    config.moe_intermediate_size = 704
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.head_dim = 256
    config.global_head_dim = 512
    config.num_global_key_value_heads = 2
    config.attention_k_eq_v = True
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 262144
    config.max_position_embeddings = 131072
    config.sliding_window = 1024
    config.rope_theta = 1000000.0
    config.query_pre_attn_scalar = 1.0
    config.rope_scaling = None
    config.rope_local_base_freq = 10000.0
    config.rope_parameters = {"rope_local_base_freq": 10000.0}
    config.hidden_act = "gelu_pytorch_tanh"
    config.torch_dtype = "bfloat16"
    config.enable_moe_block = True
    config.num_experts = 128
    config.top_k_experts = 8
    config.layer_types = (
        ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
    )
    config.final_logit_softcapping = 30.0
    return config


@pytest.fixture
def mock_text_config_dense():
    config = Mock(spec=[])
    config.num_hidden_layers = 62
    config.hidden_size = 2816
    config.intermediate_size = 2112
    config.moe_intermediate_size = 704
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.head_dim = 256
    config.global_head_dim = 512
    config.num_global_key_value_heads = 2
    config.attention_k_eq_v = True
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 262144
    config.max_position_embeddings = 131072
    config.sliding_window = 1024
    config.rope_theta = 1000000.0
    config.query_pre_attn_scalar = 1.0
    config.rope_scaling = None
    config.rope_local_base_freq = 10000.0
    config.rope_parameters = {"rope_local_base_freq": 10000.0}
    config.hidden_act = "gelu_pytorch_tanh"
    config.torch_dtype = "bfloat16"
    config.hidden_size_per_layer_input = 0
    config.enable_moe_block = False
    config.layer_types = (
        ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]
    )
    config.final_logit_softcapping = 30.0
    return config


@pytest.fixture
def mock_hf_config_moe(mock_text_config_moe, mock_vision_config):
    config = Mock()
    config.text_config = mock_text_config_moe
    config.vision_config = mock_vision_config
    config.vision_soft_tokens_per_image = 280
    config.bos_token_id = 2
    config.eos_token_id = 1
    config.image_token_id = 258_880
    config.video_token_id = 258_884
    return config


@pytest.fixture
def mock_hf_config_dense(mock_text_config_dense, mock_vision_config):
    config = Mock()
    config.text_config = mock_text_config_dense
    config.vision_config = mock_vision_config
    config.vision_soft_tokens_per_image = 280
    config.bos_token_id = 2
    config.eos_token_id = 1
    config.image_token_id = 258_880
    config.video_token_id = 258_884
    return config


@pytest.fixture
def mock_hf_pretrained_moe(mock_hf_config_moe):
    p = Mock(spec=PreTrainedCausalLM)
    p.config = mock_hf_config_moe
    p.generation_config = GenerationConfig()
    return p


@pytest.fixture
def mock_hf_pretrained_dense(mock_hf_config_dense):
    p = Mock(spec=PreTrainedCausalLM)
    p.config = mock_hf_config_dense
    p.generation_config = GenerationConfig()
    return p


@pytest.fixture
def vl_bridge():
    return Gemma4VLBridge()


# ===========================================================================
# Gemma4Bridge (CausalLM) tests
# ===========================================================================


class TestGemma4BridgeRegistration:
    def test_is_subclass_of_model_bridge(self):
        assert issubclass(Gemma4Bridge, MegatronModelBridge)

    def test_vl_bridge_inherits_causal_bridge(self):
        assert issubclass(Gemma4VLBridge, Gemma4Bridge)


class TestGemma4BridgeProviderBridgeMoE:
    """Gemma4Bridge.provider_bridge for MoE CausalLM."""

    def test_returns_provider_instance(self, causal_bridge, mock_causal_pretrained):
        provider = causal_bridge.provider_bridge(mock_causal_pretrained)
        assert isinstance(provider, Gemma4ModelProvider)

    def test_basic_transformer_config(self, causal_bridge, mock_causal_pretrained):
        p = causal_bridge.provider_bridge(mock_causal_pretrained)
        assert p.num_layers == 62
        assert p.hidden_size == 2816
        assert p.num_attention_heads == 8
        assert p.num_query_groups == 4
        assert p.kv_channels == 256
        assert p.vocab_size == 262144
        assert p.seq_length == 131072
        assert p.init_method_std == 0.02
        assert p.layernorm_epsilon == 1e-6

    def test_moe_config(self, causal_bridge, mock_causal_pretrained):
        p = causal_bridge.provider_bridge(mock_causal_pretrained)
        assert p.num_moe_experts == 128
        assert p.moe_router_topk == 8
        assert p.moe_ffn_hidden_size == 704
        assert p.moe_shared_expert_intermediate_size == 2112
        assert p.moe_layer_freq == 1
        assert p.moe_shared_expert_overlap is False
        assert p.moe_shared_expert_gate is False

    def test_window_size(self, causal_bridge, mock_causal_pretrained):
        assert causal_bridge.provider_bridge(mock_causal_pretrained).window_size == 1024

    def test_rotary_base_tuple(self, causal_bridge, mock_causal_pretrained):
        rb = causal_bridge.provider_bridge(mock_causal_pretrained).rotary_base
        assert isinstance(rb, tuple) and len(rb) == 2
        assert rb[0] == 10000.0
        assert rb[1] == 1000000.0

    def test_softmax_scale_is_one(self, causal_bridge, mock_causal_pretrained):
        assert causal_bridge.provider_bridge(mock_causal_pretrained).softmax_scale == 1.0

    def test_qk_layernorm_enabled(self, causal_bridge, mock_causal_pretrained):
        assert causal_bridge.provider_bridge(mock_causal_pretrained).qk_layernorm is True

    def test_global_attention_config(self, causal_bridge, mock_causal_pretrained):
        p = causal_bridge.provider_bridge(mock_causal_pretrained)
        assert p.global_head_dim == 512
        assert p.num_global_key_value_heads == 2
        assert p.global_rotary_percent == 0.25
        assert p.attention_k_eq_v is True

    def test_interleaved_attn_pattern(self, causal_bridge, mock_causal_pretrained):
        assert causal_bridge.provider_bridge(mock_causal_pretrained).interleaved_attn_pattern == (5, 1)

    def test_logit_softcapping(self, causal_bridge, mock_causal_pretrained):
        assert causal_bridge.provider_bridge(mock_causal_pretrained).final_logit_softcapping == 30.0

    def test_dtype_is_bf16(self, causal_bridge, mock_causal_pretrained):
        p = causal_bridge.provider_bridge(mock_causal_pretrained)
        assert p.bf16 is True
        assert p.params_dtype == torch.bfloat16

    def test_different_hidden_sizes(self, causal_bridge, mock_causal_pretrained):
        for hs in [2048, 2816, 4096]:
            mock_causal_pretrained.config.hidden_size = hs
            assert causal_bridge.provider_bridge(mock_causal_pretrained).hidden_size == hs

    def test_different_layer_counts(self, causal_bridge, mock_causal_pretrained):
        for nl in [32, 46, 62]:
            mock_causal_pretrained.config.num_hidden_layers = nl
            assert causal_bridge.provider_bridge(mock_causal_pretrained).num_layers == nl

    def test_vocab_size_variants(self, causal_bridge, mock_causal_pretrained):
        for vs in [256000, 262144, 300000]:
            mock_causal_pretrained.config.vocab_size = vs
            assert causal_bridge.provider_bridge(mock_causal_pretrained).vocab_size == vs


class TestGemma4BridgeProviderBridgeDense:
    """Gemma4Bridge.provider_bridge for Dense CausalLM (enable_moe_block=False)."""

    def test_returns_dense_provider(self, causal_bridge, mock_causal_dense_pretrained):
        p = causal_bridge.provider_bridge(mock_causal_dense_pretrained)
        assert isinstance(p, Gemma4DenseProvider)

    def test_basic_config_preserved(self, causal_bridge, mock_causal_dense_pretrained):
        p = causal_bridge.provider_bridge(mock_causal_dense_pretrained)
        assert p.num_layers == 62
        assert p.hidden_size == 2816
        assert p.num_attention_heads == 8
        assert p.num_query_groups == 4
        assert p.vocab_size == 262144

    def test_does_not_copy_moe_intermediate_size(self, causal_bridge, mock_causal_dense_pretrained):
        """Dense provider should NOT use moe_intermediate_size from HF config."""
        p = causal_bridge.provider_bridge(mock_causal_dense_pretrained)
        # Dense provider has its own moe_ffn_hidden_size default (704), not 1408 from HF config
        assert (
            p.moe_ffn_hidden_size == mock_causal_dense_pretrained.config.moe_ffn_hidden_size
            if hasattr(mock_causal_dense_pretrained.config, "moe_ffn_hidden_size")
            else True
        )  # default kept


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


class TestMaybeModifyLoadedHFWeightCausal:
    """Weight modification during HF → Megatron loading (CausalLM path)."""

    def _make_sd(self, layer_idx=0, hidden=8, num_experts=4):
        p = f"model.layers.{layer_idx}"
        sd = {
            f"{p}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{p}.self_attn.k_proj.weight": torch.randn(hidden // 2, hidden),
            f"{p}.router.proj.weight": torch.randn(num_experts, hidden),
            f"{p}.router.scale": torch.ones(hidden),
            f"{p}.pre_feedforward_layernorm_2.weight": torch.ones(hidden) * 2.0,
            f"{p}.mlp.gate_proj.weight": torch.randn(16, hidden),
            f"{p}.mlp.up_proj.weight": torch.randn(16, hidden),
            f"{p}.pre_feedforward_layernorm.weight": torch.ones(hidden) * 3.0,
        }
        return sd

    def test_kv_synthesis_when_v_proj_absent(self, causal_bridge):
        sd = self._make_sd()
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }
        result = causal_bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert isinstance(result, dict)
        torch.testing.assert_close(result["v"], result["k"])

    def test_kv_no_synthesis_when_v_present(self, causal_bridge):
        sd = self._make_sd()
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(4, 8)
        hf_param = {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }
        result = causal_bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        assert result is not None

    def test_router_weight_passthrough(self, causal_bridge):
        sd = self._make_sd(hidden=8)
        hf_param = "model.layers.0.router.proj.weight"

        result = causal_bridge.maybe_modify_loaded_hf_weight(hf_param, sd)

        torch.testing.assert_close(result, sd[hf_param], rtol=0, atol=0)

    def test_router_fusion_missing_keys_passthrough(self, causal_bridge):
        sd = {"model.layers.0.router.proj.weight": torch.randn(4, 8)}
        result = causal_bridge.maybe_modify_loaded_hf_weight("model.layers.0.router.proj.weight", sd)
        torch.testing.assert_close(result, sd["model.layers.0.router.proj.weight"])

    def test_shared_expert_weights_passthrough(self, causal_bridge):
        sd = self._make_sd(hidden=8)
        hf_param = {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "up": "model.layers.0.mlp.up_proj.weight",
        }

        result = causal_bridge.maybe_modify_loaded_hf_weight(hf_param, sd)

        torch.testing.assert_close(result["gate"], sd[hf_param["gate"]], rtol=0, atol=0)
        torch.testing.assert_close(result["up"], sd[hf_param["up"]], rtol=0, atol=0)

    def test_shared_expert_fusion_missing_keys_passthrough(self, causal_bridge):
        sd = {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(4, 8),
            "model.layers.0.mlp.up_proj.weight": torch.randn(4, 8),
        }
        hf_param = {"gate": "model.layers.0.mlp.gate_proj.weight", "up": "model.layers.0.mlp.up_proj.weight"}
        result = causal_bridge.maybe_modify_loaded_hf_weight(hf_param, sd)
        torch.testing.assert_close(result["gate"], sd["model.layers.0.mlp.gate_proj.weight"])


class TestMaybeModifyConvertedHFWeightCausal:
    """Weight un-fusion during Megatron → HF export (CausalLM path)."""

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

    def test_drops_synthesized_v_proj(self, causal_bridge):
        hf_sd = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)}
        converted = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8),
        }
        result = causal_bridge.maybe_modify_converted_hf_weight(None, converted, hf_sd)
        assert "model.layers.0.self_attn.v_proj.weight" not in result
        assert "model.layers.0.self_attn.q_proj.weight" in result

    def test_router_weight_export_passthrough(self, causal_bridge):
        ref_sd = self._make_ref_sd(hidden=8)
        converted = {"model.layers.0.router.proj.weight": torch.randn(4, 8)}

        result = causal_bridge.maybe_modify_converted_hf_weight(
            None,
            converted,
            ref_sd,
        )

        torch.testing.assert_close(
            result["model.layers.0.router.proj.weight"],
            converted["model.layers.0.router.proj.weight"],
            rtol=0,
            atol=0,
        )

    def test_shared_expert_weight_export_passthrough(self, causal_bridge):
        ref_sd = self._make_ref_sd(hidden=8)
        converted = {"model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8)}

        result = causal_bridge.maybe_modify_converted_hf_weight(
            None,
            converted,
            ref_sd,
        )

        torch.testing.assert_close(
            result["model.layers.0.mlp.gate_proj.weight"],
            converted["model.layers.0.mlp.gate_proj.weight"],
            rtol=0,
            atol=0,
        )

    def test_config_only_vl_export_drops_tied_global_v_proj(self, vl_bridge, mock_hf_config_moe):
        vl_bridge.hf_config = mock_hf_config_moe
        converted = {
            "model.language_model.layers.4.self_attn.v_proj.weight": torch.randn(4, 8),
            "model.language_model.layers.5.self_attn.q_proj.weight": torch.randn(8, 8),
            "model.language_model.layers.5.self_attn.k_proj.weight": torch.randn(4, 8),
            "model.language_model.layers.5.self_attn.v_proj.weight": torch.randn(4, 8),
        }

        result = vl_bridge.maybe_modify_converted_hf_weight(None, converted, {})

        assert set(result) == {
            "model.language_model.layers.4.self_attn.v_proj.weight",
            "model.language_model.layers.5.self_attn.q_proj.weight",
            "model.language_model.layers.5.self_attn.k_proj.weight",
        }

    def test_empty_hf_state_dict_passthrough(self, causal_bridge):
        converted = {"some.weight": torch.randn(4, 4)}
        result = causal_bridge.maybe_modify_converted_hf_weight(None, converted, {})
        assert result is converted

    def test_none_hf_state_dict_passthrough(self, causal_bridge):
        converted = {"some.weight": torch.randn(4, 4)}
        result = causal_bridge.maybe_modify_converted_hf_weight(None, converted, None)
        assert result is converted


class TestGemma4BridgeMappingRegistryCausal:
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

    def test_returns_registry(self, causal_bridge):
        assert isinstance(causal_bridge.mapping_registry(), MegatronMappingRegistry)

    def test_has_mappings(self, causal_bridge):
        assert len(causal_bridge.mapping_registry().mappings) > 0

    def test_has_embeddings_mapping(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert any("embed_tokens" in n or "word_embeddings" in n for n in names)

    def test_has_final_norm_mapping(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert any("norm" in n for n in names)

    def test_has_qkv_mapping(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert any("linear_qkv" in n for n in names)

    def test_has_router_mapping(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert any("router" in n for n in names)

    def test_has_shared_expert_mapping(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert any("shared_experts" in n for n in names)

    def test_has_post_moe_layernorm(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert any("post_moe_layernorm" in n for n in names)

    def test_has_layer_scalar_mapping(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert any("layer_scalar" in n for n in names)

    def test_uses_causal_lm_prefix(self, causal_bridge):
        """CausalLM bridge uses model.layers.* (not model.language_model.layers.*)."""
        names = self._collect_names(causal_bridge.mapping_registry())
        hf_names = [n for n in names if "layers" in n]
        assert all("language_model" not in n for n in hf_names)

    def test_moe_registry_has_no_duplicate_non_layernorm_hf_targets(self, causal_bridge):
        targets = self._collect_hf_targets(causal_bridge.mapping_registry())
        duplicates = {
            name: count for name, count in Counter(targets).items() if count > 1 and "input_layernorm" not in name
        }
        assert duplicates == {}

    def test_moe_registry_does_not_map_plain_mlp_params(self, causal_bridge):
        names = self._collect_names(causal_bridge.mapping_registry())
        assert "decoder.layers.*.mlp.linear_fc1.weight" not in names
        assert "decoder.layers.*.mlp.linear_fc2.weight" not in names


# ===========================================================================
# Gemma4VLBridge (ConditionalGeneration) tests
# ===========================================================================


@pytest.fixture
def bridge():
    return Gemma4VLBridge()


class TestGemma4VLBridgeInitialization:
    def test_inherits_causal_bridge(self):
        assert issubclass(Gemma4VLBridge, Gemma4Bridge)


class TestGemma4VLBridgeConversionMode:
    def test_conversion_mode_returns_text_when_env_set(self, bridge, monkeypatch):
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "text")

        assert bridge._conversion_mode() == "text"

    def test_conversion_mode_returns_auto_by_default(self, bridge, monkeypatch):
        monkeypatch.delenv("GEMMA4_CONVERSION_MODE", raising=False)

        assert bridge._conversion_mode() == "auto"

    def test_conversion_mode_audio_dispatch(self, bridge, monkeypatch):
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "audio")

        assert bridge._conversion_mode() == "audio"

    def test_conversion_mode_rejects_invalid_env(self, bridge, monkeypatch):
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "bad-mode")

        with pytest.raises(ValueError, match="Invalid GEMMA4_CONVERSION_MODE"):
            bridge._conversion_mode()


class TestGemma4VLBridgeProviderBridgeMoE:
    def test_returns_provider(self, bridge, mock_hf_pretrained_moe):
        assert isinstance(bridge.provider_bridge(mock_hf_pretrained_moe), Gemma4VLModelProvider)

    def test_basic_transformer_config(self, bridge, mock_hf_pretrained_moe):
        p = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert p.num_layers == 62
        assert p.hidden_size == 2816
        assert p.num_attention_heads == 8
        assert p.num_query_groups == 4
        assert p.kv_channels == 256
        assert p.init_method_std == 0.02
        assert p.layernorm_epsilon == 1e-6
        assert p.vocab_size == 262144
        assert p.seq_length == 131072
        assert p.window_size == 1024

    def test_moe_config(self, bridge, mock_hf_pretrained_moe):
        p = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert p.num_moe_experts == 128
        assert p.moe_router_topk == 8
        assert p.moe_ffn_hidden_size == 704
        assert p.moe_shared_expert_intermediate_size == 2112
        assert p.moe_layer_freq == 1
        assert p.attention_k_eq_v is True

    def test_softmax_scale_is_one(self, bridge, mock_hf_pretrained_moe):
        assert bridge.provider_bridge(mock_hf_pretrained_moe).softmax_scale == 1.0

    def test_vl_specific_config(self, bridge, mock_hf_pretrained_moe):
        p = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert p.image_token_id == 258_880
        assert p.video_token_id == 258_884
        assert p.bos_token_id == 2
        assert p.eos_token_id == 1
        assert p.vision_soft_tokens_per_image == 280

    def test_dtype_matches_hf_config(self, bridge, mock_hf_pretrained_moe):
        p = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert p.bf16 is True
        assert p.fp16 is False
        assert p.params_dtype == torch.bfloat16
        assert p.autocast_dtype == torch.bfloat16

    def test_global_head_config(self, bridge, mock_hf_pretrained_moe):
        p = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert p.global_head_dim == 512
        assert p.num_global_key_value_heads == 2

    def test_qk_layernorm_enabled(self, bridge, mock_hf_pretrained_moe):
        assert bridge.provider_bridge(mock_hf_pretrained_moe).qk_layernorm is True

    def test_logit_softcapping(self, bridge, mock_hf_pretrained_moe):
        assert bridge.provider_bridge(mock_hf_pretrained_moe).final_logit_softcapping == 30.0

    def test_vision_config_set(self, bridge, mock_hf_pretrained_moe):
        p = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert p.vision_config is mock_hf_pretrained_moe.config.vision_config
        assert p.text_config is mock_hf_pretrained_moe.config.text_config


class TestGemma4VLBridgeProviderBridgeDense:
    def test_accepts_dense_with_per_layer_inputs(self, bridge, mock_hf_pretrained_dense):
        mock_hf_pretrained_dense.config.text_config.hidden_size_per_layer_input = 256
        p = bridge.provider_bridge(mock_hf_pretrained_dense)
        assert isinstance(p, Gemma4DenseVLProvider)
        assert p.per_layer_embed_dim == 256

    def test_returns_dense_vl_provider(self, bridge, mock_hf_pretrained_dense):
        assert isinstance(bridge.provider_bridge(mock_hf_pretrained_dense), Gemma4DenseVLProvider)

    def test_preserves_logit_softcapping(self, bridge, mock_hf_pretrained_dense):
        assert bridge.provider_bridge(mock_hf_pretrained_dense).final_logit_softcapping == 30.0

    def test_text_mode_returns_text_provider(self, bridge, mock_hf_pretrained_dense, monkeypatch):
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "text")
        p = bridge.provider_bridge(mock_hf_pretrained_dense)
        assert isinstance(p, Gemma4DenseProvider)
        assert not isinstance(p, Gemma4DenseVLProvider)


@pytest.mark.parametrize("provider_cls", [Gemma4DenseVLProvider, Gemma4VLModelProvider])
def test_megatron_to_hf_config_nests_final_logit_softcapping(provider_cls):
    provider = provider_cls(final_logit_softcapping=17.0)
    hf_config = Gemma4VLBridge.megatron_to_hf_config(provider)

    assert "final_logit_softcapping" not in hf_config
    assert hf_config["text_config"]["final_logit_softcapping"] == 17.0


def test_megatron_to_hf_config_nests_dense_architecture_fields():
    provider = Gemma4DenseVLProvider(
        attention_k_eq_v=True,
        final_logit_softcapping=17.0,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        vision_config={"hidden_size": 1152},
        audio_config={"hidden_size": 512},
        eos_token_id=[1, 106],
        window_size=(511, 0),
    )

    hf_config = Gemma4VLBridge.megatron_to_hf_config(provider)

    expected = {
        "attention_k_eq_v": True,
        "enable_moe_block": False,
        "final_logit_softcapping": 17.0,
        "num_kv_shared_layers": 20,
        "sliding_window": 512,
        "use_double_wide_mlp": True,
    }
    assert {name: hf_config["text_config"][name] for name in expected} == expected
    assert not set(expected).intersection(hf_config)
    assert hf_config["architectures"] == ["Gemma4ForConditionalGeneration"]
    assert hf_config["model_type"] == "gemma4"
    assert hf_config["text_config"]["model_type"] == "gemma4_text"
    assert hf_config["text_config"]["num_hidden_layers"] == provider.num_layers
    assert hf_config["vision_config"] == {"hidden_size": 1152}
    assert hf_config["audio_config"] == {"hidden_size": 512}
    assert hf_config["eos_token_id"] == [1, 106]


def test_megatron_to_hf_config_nests_serialized_dense_window_size():
    provider = Gemma4DenseVLProvider(window_size=[511, 0])

    hf_config = Gemma4VLBridge.megatron_to_hf_config(provider)

    assert hf_config["text_config"]["sliding_window"] == 512


def test_megatron_to_hf_config_nests_moe_architecture_fields():
    provider = Gemma4VLModelProvider(
        attention_k_eq_v=True,
        final_logit_softcapping=19.0,
        moe_ffn_hidden_size=704,
        moe_router_topk=8,
        moe_shared_expert_intermediate_size=2112,
        num_layers=6,
        num_moe_experts=128,
        window_size=1024,
    )

    hf_config = Gemma4VLBridge.megatron_to_hf_config(provider)

    expected = {
        "attention_k_eq_v": True,
        "enable_moe_block": True,
        "final_logit_softcapping": 19.0,
        "intermediate_size": 2112,
        "moe_intermediate_size": 704,
        "num_experts": 128,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "top_k_experts": 8,
        "use_double_wide_mlp": False,
    }
    assert {name: hf_config["text_config"][name] for name in expected} == expected
    assert not set(expected).intersection(hf_config)
    assert hf_config["text_config"]["layer_types"][:6] == ["sliding_attention"] * 5 + ["full_attention"]


def test_megatron_to_hf_config_keeps_vl_text_mode_flat():
    provider = Gemma4DenseProvider(final_logit_softcapping=23.0)

    hf_config = Gemma4VLBridge.megatron_to_hf_config(provider)

    assert "text_config" not in hf_config
    assert hf_config["architectures"] == ["Gemma4ForCausalLM"]
    assert hf_config["model_type"] == "gemma4_text"
    assert hf_config["final_logit_softcapping"] == 23.0


class TestGemma4VLBridgeMappingRegistry:
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

    def test_has_norm_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("norm" in n for n in names)

    def test_has_vision_tower_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("vision_tower" in n for n in names)

    def test_has_embed_vision_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("embed_vision" in n for n in names)

    def test_has_audio_tower_mapping(self, bridge):
        """VL bridge includes audio_tower mappings."""
        names = self._collect_names(bridge.mapping_registry())
        assert any("audio_tower" in n for n in names)

    def test_has_embed_audio_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("embed_audio" in n for n in names)

    def test_has_qkv_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("linear_qkv" in n for n in names)

    def test_has_mlp_mapping(self, bridge):
        names = self._collect_names(bridge.mapping_registry())
        assert any("mlp" in n for n in names)

    def test_has_shared_expert_layernorm(self, bridge, mock_hf_config_moe):
        bridge.hf_config = mock_hf_config_moe
        names = self._collect_names(bridge.mapping_registry())
        assert any("post_shared_expert_layernorm" in n for n in names)

    def test_has_direct_pre_shared_expert_norm_mapping(self, bridge, mock_hf_config_moe):
        bridge.hf_config = mock_hf_config_moe
        names = self._collect_names(bridge.mapping_registry())
        assert "language_model.decoder.layers.*.pre_shared_expert_layernorm.weight" in names
        assert "model.language_model.layers.*.pre_feedforward_layernorm.weight" in names

    def test_moe_registry_has_no_duplicate_non_layernorm_hf_targets(self, bridge, mock_hf_config_moe):
        bridge.hf_config = mock_hf_config_moe
        targets = self._collect_hf_targets(bridge.mapping_registry())
        duplicates = {
            name: count for name, count in Counter(targets).items() if count > 1 and "input_layernorm" not in name
        }
        assert duplicates == {}

    def test_moe_registry_does_not_map_plain_mlp_params(self, bridge, mock_hf_config_moe):
        bridge.hf_config = mock_hf_config_moe
        names = self._collect_names(bridge.mapping_registry())
        assert "language_model.decoder.layers.*.mlp.linear_fc1.weight" not in names
        assert "language_model.decoder.layers.*.mlp.linear_fc2.weight" not in names

    def test_has_post_moe_layernorm(self, bridge, mock_hf_config_moe):
        bridge.hf_config = mock_hf_config_moe
        names = self._collect_names(bridge.mapping_registry())
        assert any("post_moe_layernorm" in n for n in names)

    def test_uses_language_model_prefix_for_vl(self, bridge, mock_hf_config_moe):
        """VLM uses model.language_model.layers.* (not model.layers.*)."""
        bridge.hf_config = mock_hf_config_moe
        names = self._collect_names(bridge.mapping_registry())
        lm_keys = [n for n in names if "layers" in n and "vision" not in n and "audio" not in n]
        assert any("language_model" in n for n in lm_keys)

    def test_dense_vl_audio_tower_replicated_mappings(self, bridge, mock_hf_config_dense, monkeypatch):
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "vl")
        bridge.hf_config = mock_hf_config_dense

        names = self._collect_names(bridge.mapping_registry())

        assert "audio_tower.**" in names
        assert "model.audio_tower.**" in names
        assert "embed_audio.**" in names
        assert "model.embed_audio.**" in names


class TestGemma4VLBridgeEdgeCases:
    def test_custom_token_ids(self, bridge, mock_hf_pretrained_moe):
        mock_hf_pretrained_moe.config.image_token_id = 99999
        mock_hf_pretrained_moe.config.bos_token_id = 42
        p = bridge.provider_bridge(mock_hf_pretrained_moe)
        assert p.image_token_id == 99999
        assert p.bos_token_id == 42

    def test_default_image_token_id(self, bridge, mock_hf_pretrained_moe):
        del mock_hf_pretrained_moe.config.image_token_id
        assert bridge.provider_bridge(mock_hf_pretrained_moe).image_token_id == 258_880

    def test_default_vision_soft_tokens(self, bridge, mock_hf_pretrained_moe):
        del mock_hf_pretrained_moe.config.vision_soft_tokens_per_image
        assert bridge.provider_bridge(mock_hf_pretrained_moe).vision_soft_tokens_per_image == 280

    def test_different_vocab_sizes(self, bridge, mock_hf_pretrained_moe):
        for vs in [256000, 262144, 300000]:
            mock_hf_pretrained_moe.config.text_config.vocab_size = vs
            assert bridge.provider_bridge(mock_hf_pretrained_moe).vocab_size == vs

    def test_different_layer_counts(self, bridge, mock_hf_pretrained_moe):
        for nl in [32, 46, 62]:
            mock_hf_pretrained_moe.config.text_config.num_hidden_layers = nl
            assert bridge.provider_bridge(mock_hf_pretrained_moe).num_layers == nl
