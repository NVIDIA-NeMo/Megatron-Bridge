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

"""Unit tests for ERNIE 4.5 text-only MoE bridge."""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.ernie.ernie_45_bridge import (
    Ernie45Bridge,
    _PPSafeAutoMapping,
    _PPSafeGatedMLPMapping,
    _PPSafeReplicatedMapping,
    _SqueezeBiasMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider


@pytest.fixture
def ernie45_config():
    """Minimal ERNIE 4.5 MoE config mock."""
    config = Mock(spec=[])
    config.hidden_size = 2560
    config.intermediate_size = 6912
    config.num_attention_heads = 20
    config.num_key_value_heads = 4
    config.num_hidden_layers = 28
    config.max_position_embeddings = 32768
    config.rms_norm_eps = 1e-05
    config.rope_theta = 500000.0
    config.tie_word_embeddings = False
    config.torch_dtype = "bfloat16"
    config.vocab_size = 189440
    config.initializer_range = 0.02
    config.model_type = "ernie4_5_moe"
    config.architectures = ["Ernie4_5_MoeForCausalLM"]
    # MoE fields
    config.moe_num_experts = 64
    config.moe_k = 6
    config.moe_intermediate_size = 1408
    config.moe_num_shared_experts = 2
    config.router_aux_loss_coef = 0.001
    config.mlp_layer_types = ["dense"] + ["moe"] * 27
    return config


@pytest.fixture
def ernie45_config_list_format():
    """Config with list-format MoE fields."""
    config = Mock(spec=[])
    config.hidden_size = 2560
    config.intermediate_size = 6912
    config.num_attention_heads = 20
    config.num_key_value_heads = 4
    config.num_hidden_layers = 28
    config.max_position_embeddings = 32768
    config.rms_norm_eps = 1e-05
    config.rope_theta = 500000.0
    config.tie_word_embeddings = False
    config.torch_dtype = "bfloat16"
    config.vocab_size = 189440
    config.initializer_range = 0.02
    config.model_type = "ernie4_5_moe"
    config.architectures = ["Ernie4_5_MoeForCausalLM"]
    # MoE fields as lists
    config.moe_num_experts = [64]
    config.moe_k = 6
    config.moe_intermediate_size = [1408]
    config.moe_num_shared_experts = 2
    config.router_aux_loss_coef = 0.001
    config.moe_layer_start_index = [1]
    # No mlp_layer_types -- should derive from moe_layer_start_index
    del config.mlp_layer_types
    return config


@pytest.fixture
def mock_pretrained(ernie45_config):
    """Mock PreTrainedCausalLM for ERNIE 4.5."""
    pretrained = Mock()
    pretrained.config = ernie45_config
    return pretrained


class TestErnie45BridgeRegistration:
    """Test bridge class and registration."""

    def test_is_subclass_of_megatron_model_bridge(self):
        assert issubclass(Ernie45Bridge, MegatronModelBridge)

    def test_bridge_instantiation(self):
        bridge = Ernie45Bridge()
        assert bridge is not None


class TestErnie45BridgeGetNumExperts:
    """Test _get_num_experts static method."""

    def test_int_input(self):
        config = Mock(spec=[])
        config.moe_num_experts = 64
        assert Ernie45Bridge._get_num_experts(config) == 64

    def test_single_element_list(self):
        config = Mock(spec=[])
        config.moe_num_experts = [64]
        assert Ernie45Bridge._get_num_experts(config) == 64

    def test_dual_pool_list_returns_first(self):
        config = Mock(spec=[])
        config.moe_num_experts = [64, 32]
        assert Ernie45Bridge._get_num_experts(config) == 64

    def test_default_when_missing(self):
        # Mock with spec=[] means arbitrary attrs return Mock; use a simple object
        class EmptyConfig:
            pass

        cfg = EmptyConfig()
        assert Ernie45Bridge._get_num_experts(cfg) == 64


class TestErnie45BridgeProviderBridge:
    """Test provider_bridge method."""

    def test_returns_gpt_model_provider(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, GPTModelProvider)

    def test_basic_dimensions(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_layers == 28
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 20

    def test_vocabulary(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.vocab_size == 189440
        assert provider.share_embeddings_and_output_weights is False

    def test_normalization(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.normalization == "RMSNorm"
        assert provider.layernorm_epsilon == 1e-05

    def test_rope_config(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == 500000.0
        assert provider.rotary_interleaved is True

    def test_moe_basic(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 6
        assert provider.moe_ffn_hidden_size == 1408

    def test_moe_router_settings(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_router_dtype == "fp32"

    def test_shared_experts(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        # shared_expert_intermediate_size = moe_ffn_hidden_size * moe_num_shared_experts
        assert provider.moe_shared_expert_intermediate_size == 1408 * 2

    def test_moe_layer_freq_from_mlp_layer_types(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        expected = [0] + [1] * 27
        assert provider.moe_layer_freq == expected

    def test_moe_layer_freq_from_moe_layer_start_index(self, ernie45_config_list_format):
        pretrained = Mock()
        pretrained.config = ernie45_config_list_format
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(pretrained)
        expected = [0] + [1] * 27
        assert provider.moe_layer_freq == expected

    def test_mlp_config(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False

    def test_moe_intermediate_size_from_list(self, ernie45_config_list_format):
        pretrained = Mock()
        pretrained.config = ernie45_config_list_format
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(pretrained)
        assert provider.moe_ffn_hidden_size == 1408

    def test_moe_aux_loss_coeff(self, mock_pretrained):
        bridge = Ernie45Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.moe_aux_loss_coeff == 0.001


class TestErnie45BridgeMappingRegistry:
    """Test mapping_registry method."""

    def test_returns_mapping_registry(self):
        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

    def test_has_mappings(self):
        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        assert len(registry.mappings) > 0

    def test_has_embedding_mappings(self):
        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        hf_params = [m.hf_param for m in registry.mappings if hasattr(m, "hf_param") and isinstance(m.hf_param, str)]
        assert "model.embed_tokens.weight" in hf_params
        assert "lm_head.weight" in hf_params

    def test_has_qkv_mapping(self):
        from megatron.bridge.models.conversion.param_mapping import QKVMapping

        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        qkv_mappings = [m for m in registry.mappings if isinstance(m, QKVMapping)]
        assert len(qkv_mappings) > 0

        qkv = qkv_mappings[0]
        assert "q" in qkv.hf_param
        assert "k" in qkv.hf_param
        assert "v" in qkv.hf_param

    def test_has_router_mapping(self):
        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        megatron_params = [
            m.megatron_param
            for m in registry.mappings
            if hasattr(m, "megatron_param") and isinstance(m.megatron_param, str)
        ]
        assert "decoder.layers.*.mlp.router.weight" in megatron_params

    def test_has_expert_bias_mapping(self):
        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        megatron_params = [
            m.megatron_param
            for m in registry.mappings
            if hasattr(m, "megatron_param") and isinstance(m.megatron_param, str)
        ]
        assert "decoder.layers.*.mlp.router.expert_bias" in megatron_params

    def test_has_shared_expert_mappings(self):
        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        megatron_params = [
            m.megatron_param
            for m in registry.mappings
            if hasattr(m, "megatron_param") and isinstance(m.megatron_param, str)
        ]
        shared = [p for p in megatron_params if "shared_experts" in p]
        assert len(shared) > 0

    def test_has_expert_fc1_mapping(self):
        bridge = Ernie45Bridge()
        registry = bridge.mapping_registry()
        megatron_params = [
            m.megatron_param
            for m in registry.mappings
            if hasattr(m, "megatron_param") and isinstance(m.megatron_param, str)
        ]
        expert_fc1 = [p for p in megatron_params if "experts" in p and "linear_fc1" in p]
        assert len(expert_fc1) > 0


class TestSqueezeBiasMapping:
    """Test _SqueezeBiasMapping hf_to_megatron / megatron_to_hf."""

    def _make_mock_module(self):
        """Create a mock megatron_module with valid device attributes."""
        mock_module = Mock()
        mock_module.weight = torch.nn.Parameter(torch.zeros(1))
        return mock_module

    def test_squeeze_2d_to_1d(self):
        mapping = _SqueezeBiasMapping(
            megatron_param="decoder.layers.*.mlp.router.expert_bias",
            hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
        )
        hf_weights = torch.randn(1, 64)
        result = mapping.hf_to_megatron(hf_weights, self._make_mock_module())
        assert result.shape == (64,)

    def test_already_1d_passthrough(self):
        mapping = _SqueezeBiasMapping(
            megatron_param="decoder.layers.*.mlp.router.expert_bias",
            hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
        )
        hf_weights = torch.randn(64)
        result = mapping.hf_to_megatron(hf_weights, self._make_mock_module())
        assert result.shape == (64,)

    def test_unsqueeze_on_export(self):
        mapping = _SqueezeBiasMapping(
            megatron_param="decoder.layers.*.mlp.router.expert_bias",
            hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
        )
        megatron_weights = torch.randn(64)
        # megatron_to_hf returns a dict; patch super() to return the weights
        with patch.object(
            _PPSafeReplicatedMapping,
            "megatron_to_hf",
            return_value={"model.layers.0.mlp.moe_statics.e_score_correction_bias": megatron_weights},
        ):
            result = mapping.megatron_to_hf(megatron_weights, Mock())
        # Result values should be [1, 64]
        for v in result.values():
            assert v.shape == (1, 64)


class TestPPSafeMappings:
    """Test PP-safe mapping variants."""

    def test_pp_safe_auto_mapping_exists(self):
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        assert issubclass(_PPSafeAutoMapping, AutoMapping)
        mapping = _PPSafeAutoMapping(
            megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
            hf_param="model.layers.*.mlp.shared_experts.down_proj.weight",
        )
        assert mapping is not None

    def test_pp_safe_gated_mlp_mapping_exists(self):
        mapping = _PPSafeGatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            up="model.layers.*.mlp.experts.*.up_proj.weight",
        )
        assert mapping is not None

    def test_pp_safe_replicated_mapping_exists(self):
        mapping = _PPSafeReplicatedMapping(
            megatron_param="decoder.layers.*.mlp.router.weight",
            hf_param="model.layers.*.mlp.gate.weight",
        )
        assert mapping is not None
