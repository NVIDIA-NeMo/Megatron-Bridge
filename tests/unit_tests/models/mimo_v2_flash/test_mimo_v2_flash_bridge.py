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

"""Unit tests for MiMo-V2-Flash bridge."""

import pytest
from unittest.mock import Mock

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashBridge
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import MiMoV2FlashModelProvider


def _make_mock_config():
    """Create a mock HF config matching MiMo-V2-Flash architecture."""
    config = Mock()
    config.num_hidden_layers = 48
    config.hidden_size = 4096
    config.intermediate_size = 16384
    config.num_attention_heads = 64
    config.num_key_value_heads = 4
    config.head_dim = 192
    config.v_head_dim = 128
    config.vocab_size = 152576
    config.max_position_embeddings = 262144
    config.rope_theta = 5000000
    config.rms_norm_eps = 1e-5
    config.layernorm_epsilon = 1e-5
    config.tie_word_embeddings = False
    config.attention_bias = False
    config.mlp_bias = False
    config.hidden_act = "silu"
    config.partial_rotary_factor = 0.334
    config.torch_dtype = "bfloat16"
    config.rope_scaling = None
    config.use_qk_norm = False

    # Hybrid attention
    config.hybrid_layer_pattern = [0] + [1, 1, 1, 1, 0] * 9 + [1, 1]
    config.sliding_window_size = 128
    config.sliding_window = 128
    config.swa_rope_theta = 10000
    config.swa_num_key_value_heads = 8

    # MoE
    config.moe_layer_freq = [0] + [1] * 47
    config.n_routed_experts = 256
    config.moe_intermediate_size = 2048
    config.num_experts_per_tok = 8
    config.scoring_func = "sigmoid"
    config.n_shared_experts = None

    # Attention sink
    config.add_swa_attention_sink_bias = True
    config.add_full_attention_sink_bias = False
    config.attention_value_scale = 0.707

    return config


def _make_mock_pretrained(config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    return pretrained


class TestMiMoV2FlashBridgeProviderBridge:
    @pytest.fixture
    def bridge(self):
        return MiMoV2FlashBridge()

    @pytest.fixture
    def mock_pretrained(self):
        return _make_mock_pretrained(_make_mock_config())

    def test_provider_type(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, MiMoV2FlashModelProvider)

    def test_core_config_mapping(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_layers == 48
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 64
        assert provider.kv_channels == 192

    def test_hybrid_attention_pattern(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.hybrid_attention_pattern is not None
        assert len(provider.hybrid_attention_pattern) == 48
        # Layer 0 is full attention (0)
        assert provider.hybrid_attention_pattern[0] == 0

    def test_dual_rope_bases(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider.rotary_base, tuple)
        swa_theta, full_theta = provider.rotary_base
        assert swa_theta == 10000
        assert full_theta == 5000000

    def test_per_layer_kv_heads(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.full_attn_num_query_groups == 4
        assert provider.swa_num_query_groups == 8
        assert provider.num_query_groups == 4  # base = full attention value

    def test_moe_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_moe_experts == 256
        assert provider.moe_ffn_hidden_size == 2048
        assert provider.moe_router_topk == 8
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True

    def test_moe_layer_freq(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.moe_layer_freq[0] == 0  # Layer 0 is dense
        assert all(f == 1 for f in provider.moe_layer_freq[1:])  # Rest are MoE

    def test_v_head_dim(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.v_head_dim == 128

    def test_partial_rotary(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.rotary_percent == pytest.approx(0.334)

    def test_no_tie_embeddings(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.share_embeddings_and_output_weights is False

    def test_architecture_defaults(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.hidden_dropout == 0.0


class TestMiMoV2FlashBridgeMappingRegistry:
    @pytest.fixture
    def bridge(self):
        return MiMoV2FlashBridge()

    def test_has_embedding_mapping(self, bridge):
        registry = bridge.mapping_registry()
        hf_params = set()
        for m in registry.mappings:
            if hasattr(m, "hf_param"):
                hf_params.add(m.hf_param)
        assert "model.embed_tokens.weight" in hf_params

    def test_has_output_layer_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("output_layer" in p for p in megatron_params)

    def test_has_qkv_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("linear_qkv.weight" in p for p in megatron_params)

    def test_has_moe_router_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("mlp.router.weight" in p for p in megatron_params)

    def test_has_moe_expert_bias_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("router.expert_bias" in p for p in megatron_params)

    def test_has_expert_fc1_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("experts.linear_fc1" in p for p in megatron_params)

    def test_has_expert_fc2_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("experts.linear_fc2" in p for p in megatron_params)

    def test_has_both_layernorm_paths(self, bridge):
        """Verify both TE fused and non-TE layernorm mappings exist."""
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        # TE fused path
        assert any("linear_qkv.layer_norm_weight" in p for p in megatron_params)
        # Non-TE path
        assert any(
            p.endswith("input_layernorm.weight") and "decoder.layers" in p
            for p in megatron_params
        )


class TestMiMoV2FlashBridgeMegatronToHfConfig:
    @pytest.fixture
    def bridge(self):
        return MiMoV2FlashBridge()

    @pytest.fixture
    def provider(self, bridge):
        pretrained = _make_mock_pretrained(_make_mock_config())
        return bridge.provider_bridge(pretrained)

    def test_roundtrip_hybrid_pattern(self, provider):
        hf_cfg = MiMoV2FlashBridge.megatron_to_hf_config(provider)
        assert "hybrid_layer_pattern" in hf_cfg
        assert len(hf_cfg["hybrid_layer_pattern"]) == 48

    def test_roundtrip_dual_rope(self, provider):
        hf_cfg = MiMoV2FlashBridge.megatron_to_hf_config(provider)
        assert hf_cfg["swa_rope_theta"] == 10000
        assert hf_cfg["rope_theta"] == 5000000

    def test_roundtrip_kv_heads(self, provider):
        hf_cfg = MiMoV2FlashBridge.megatron_to_hf_config(provider)
        assert hf_cfg["num_key_value_heads"] == 4
        assert hf_cfg["swa_num_key_value_heads"] == 8

    def test_roundtrip_moe(self, provider):
        hf_cfg = MiMoV2FlashBridge.megatron_to_hf_config(provider)
        assert hf_cfg["n_routed_experts"] == 256
        assert hf_cfg["moe_intermediate_size"] == 2048
