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

from unittest.mock import Mock

import pytest

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.glm_vl.glm_45v_bridge import GLM45VBridge
from megatron.bridge.models.glm_vl.glm_45v_provider import GLM45VModelProvider
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


@pytest.fixture
def mock_text_config():
    """Create a mock text config for GLM-4.5V."""
    config = Mock()
    config.attention_bias = True
    config.head_dim = 128
    config.hidden_size = 4096
    config.rope_theta = 10000.0
    config.partial_rotary_factor = 0.5
    config.initializer_range = 0.02
    config.intermediate_size = 13696
    config.max_position_embeddings = 8192
    config.moe_intermediate_size = 1536
    config.num_attention_heads = 32
    config.n_routed_experts = 64
    config.routed_scaling_factor = 1.0
    config.num_experts_per_tok = 6
    config.num_hidden_layers = 46
    config.num_key_value_heads = 4
    config.rms_norm_eps = 1e-5
    config.use_qk_norm = True
    config.vocab_size = 151552
    config.first_k_dense_replace = 3
    # Token IDs
    config.eos_token_id = 151329
    config.image_start_token_id = 151339
    config.image_end_token_id = 151340
    config.video_start_token_id = 151341
    config.video_end_token_id = 151342
    config.image_token_id = 151363
    config.video_token_id = 151364
    return config


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for GLM-4.5V."""
    config = Mock()
    config.hidden_size = 1152
    config.intermediate_size = 4304
    config.num_hidden_layers = 27
    config.num_attention_heads = 16
    config.patch_size = 14
    config.image_size = 896
    return config


@pytest.fixture
def mock_hf_config(mock_text_config, mock_vision_config):
    """Create a mock HF config for GLM-4.5V."""
    config = Mock()
    config.text_config = mock_text_config
    config.vision_config = mock_vision_config
    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    return pretrained


@pytest.fixture
def glm_45v_bridge():
    """Create a GLM45VBridge instance."""
    return GLM45VBridge()


class TestGLM45VBridgeInitialization:
    """Test GLM45VBridge initialization and basic functionality."""

    def test_bridge_initialization(self, glm_45v_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(glm_45v_bridge, GLM45VBridge)

    def test_bridge_has_required_methods(self, glm_45v_bridge):
        """Test that bridge has required methods."""
        assert hasattr(glm_45v_bridge, "provider_bridge")
        assert callable(glm_45v_bridge.provider_bridge)

        assert hasattr(glm_45v_bridge, "mapping_registry")
        assert callable(glm_45v_bridge.mapping_registry)

        assert hasattr(glm_45v_bridge, "get_hf_tokenizer_kwargs")
        assert callable(glm_45v_bridge.get_hf_tokenizer_kwargs)


class TestGLM45VBridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, GLM45VModelProvider)

        # Check basic transformer config
        assert provider.num_layers == 46
        assert provider.hidden_size == 4096
        assert provider.vocab_size == 151552

    def test_provider_bridge_moe_config(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct MoE configuration."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # Check MoE specific configuration
        assert provider.num_moe_experts == 64
        assert provider.moe_router_topk == 6
        assert provider.moe_ffn_hidden_size == 1536
        assert provider.moe_shared_expert_intermediate_size == 1536

    def test_provider_bridge_attention_config(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct attention configuration."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # Check attention config
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 4
        assert provider.add_qkv_bias is True
        assert provider.qk_layernorm is True
        assert provider.kv_channels == 128

    def test_provider_bridge_rotary_config(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct rotary configuration."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # Check rotary base configuration
        assert provider.rotary_base == 10000.0
        assert provider.rotary_percent == 0.5

    def test_provider_bridge_vision_config(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge sets vision_config attribute."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # Should set vision_config
        assert provider.vision_config is mock_hf_pretrained.config.vision_config

    def test_provider_bridge_token_ids(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge sets VL-specific token IDs."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # Check token IDs
        assert provider.eos_token_id == 151329
        assert provider.image_start_token_id == 151339
        assert provider.image_end_token_id == 151340
        assert provider.video_start_token_id == 151341
        assert provider.video_end_token_id == 151342
        assert provider.image_token_id == 151363
        assert provider.video_token_id == 151364

    def test_provider_bridge_moe_layer_freq(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge computes correct moe_layer_freq."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # first_k_dense_replace=3 means first 3 layers are dense, rest are MoE
        # Total 46 layers: [0,0,0] + [1]*43
        expected_moe_layer_freq = [0] * 3 + [1] * 43
        assert provider.moe_layer_freq == expected_moe_layer_freq

    def test_provider_bridge_no_mtp_layers(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge sets mtp_num_layers to 0 for VL models."""
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.mtp_num_layers == 0

    def test_provider_bridge_with_custom_vocab_size(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom vocabulary size."""
        mock_hf_pretrained.config.text_config.vocab_size = 200000
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.vocab_size == 200000

    def test_provider_bridge_with_custom_num_layers(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with different layer counts."""
        mock_hf_pretrained.config.text_config.num_hidden_layers = 32
        mock_hf_pretrained.config.text_config.first_k_dense_replace = 2
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.num_layers == 32
        # 2 dense + 30 MoE
        expected_moe_layer_freq = [0] * 2 + [1] * 30
        assert provider.moe_layer_freq == expected_moe_layer_freq


class TestGLM45VBridgeTokenizerKwargs:
    """Test get_hf_tokenizer_kwargs method."""

    def test_tokenizer_kwargs_use_fast(self, glm_45v_bridge):
        """Test get_hf_tokenizer_kwargs returns use_fast=True."""
        kwargs = glm_45v_bridge.get_hf_tokenizer_kwargs()

        assert isinstance(kwargs, dict)
        assert kwargs.get("use_fast") is True


class TestGLM45VBridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_correct_type(self, glm_45v_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = glm_45v_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_required_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains all required parameter mappings."""
        registry = glm_45v_bridge.mapping_registry()

        # Extract mappings - registry should contain mappings for common parameters
        mappings = registry.mappings
        assert len(mappings) > 0

        # Check that we have mappings for embeddings, output layer, layernorms
        mapping_names = []
        for mapping in mappings:
            # Collect Megatron param pattern
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            # Collect HF param pattern(s)
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain word embeddings mapping
        has_embeddings = any("embed_tokens" in name or "word_embeddings" in name for name in mapping_names)
        assert has_embeddings, "Should contain embeddings mapping"

        # Should contain norm layer mapping
        has_norm = any("norm" in name for name in mapping_names)
        assert has_norm, "Should contain norm layer mapping"

    def test_mapping_registry_visual_params(self, glm_45v_bridge):
        """Test mapping_registry handles visual tower parameters correctly."""
        registry = glm_45v_bridge.mapping_registry()

        # Should contain visual parameter mappings
        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        has_visual = any("visual" in name for name in mapping_names)
        assert has_visual, "Should contain visual parameter mappings"

    def test_mapping_registry_qkv_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = glm_45v_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain QKV mappings
        has_qkv = any("linear_qkv" in name for name in mapping_names)
        assert has_qkv, "Should contain QKV mappings"

    def test_mapping_registry_mlp_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = glm_45v_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain MLP mappings
        has_mlp = any("mlp" in name for name in mapping_names)
        assert has_mlp, "Should contain MLP mappings"

    def test_mapping_registry_moe_expert_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains MoE expert parameter mappings."""
        registry = glm_45v_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain expert mappings
        has_experts = any("experts" in name for name in mapping_names)
        assert has_experts, "Should contain MoE expert mappings"

    def test_mapping_registry_shared_expert_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains shared expert parameter mappings."""
        registry = glm_45v_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain shared expert mappings
        has_shared_experts = any("shared_experts" in name for name in mapping_names)
        assert has_shared_experts, "Should contain shared expert mappings"

    def test_mapping_registry_router_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains router parameter mappings."""
        registry = glm_45v_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain router mappings
        has_router = any("router" in name or "gate" in name for name in mapping_names)
        assert has_router, "Should contain router/gate mappings"

    def test_mapping_registry_attention_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains attention parameter mappings."""
        registry = glm_45v_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain attention mappings
        has_attention = any("self_attn" in name or "self_attention" in name for name in mapping_names)
        assert has_attention, "Should contain attention mappings"

    def test_mapping_registry_qk_norm_mappings(self, glm_45v_bridge):
        """Test mapping_registry contains Q/K normalization mappings."""
        registry = glm_45v_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain Q/K norm mappings
        has_q_norm = any("q_norm" in name or "q_layernorm" in name for name in mapping_names)
        has_k_norm = any("k_norm" in name or "k_layernorm" in name for name in mapping_names)
        assert has_q_norm, "Should contain Q norm mappings"
        assert has_k_norm, "Should contain K norm mappings"


class TestGLM45VBridgeEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_bridge_with_minimal_config(self, glm_45v_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedVLM)
        minimal_config = Mock()

        # Create minimal text config
        text_config = Mock()
        text_config.attention_bias = True
        text_config.head_dim = 128
        text_config.hidden_size = 4096
        text_config.rope_theta = 10000.0
        text_config.partial_rotary_factor = 0.5
        text_config.initializer_range = 0.02
        text_config.intermediate_size = 13696
        text_config.max_position_embeddings = 8192
        text_config.moe_intermediate_size = 1536
        text_config.num_attention_heads = 32
        text_config.n_routed_experts = 64
        text_config.routed_scaling_factor = 1.0
        text_config.num_experts_per_tok = 6
        text_config.num_hidden_layers = 46
        text_config.num_key_value_heads = 4
        text_config.rms_norm_eps = 1e-5
        text_config.use_qk_norm = True
        text_config.vocab_size = 151552
        text_config.first_k_dense_replace = 3
        # Use defaults for token IDs by not providing them

        # Create minimal vision config
        vision_config = Mock()

        minimal_config.text_config = text_config
        minimal_config.vision_config = vision_config
        minimal_pretrained.config = minimal_config

        provider = glm_45v_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, GLM45VModelProvider)
        assert provider.num_layers == 46
        assert provider.hidden_size == 4096

    def test_provider_bridge_with_different_hidden_sizes(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with different hidden sizes."""
        test_hidden_sizes = [2048, 4096, 8192]

        for hidden_size in test_hidden_sizes:
            mock_hf_pretrained.config.text_config.hidden_size = hidden_size
            provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.hidden_size == hidden_size

    def test_provider_bridge_with_different_num_experts(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with different numbers of MoE experts."""
        test_num_experts = [8, 32, 64, 128]

        for num_experts in test_num_experts:
            mock_hf_pretrained.config.text_config.n_routed_experts = num_experts
            provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.num_moe_experts == num_experts

    def test_provider_bridge_with_different_topk(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with different MoE topk values."""
        test_topk = [2, 4, 6, 8]

        for topk in test_topk:
            mock_hf_pretrained.config.text_config.num_experts_per_tok = topk
            provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.moe_router_topk == topk

    def test_provider_bridge_with_all_dense_layers(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge when all layers are dense."""
        mock_hf_pretrained.config.text_config.first_k_dense_replace = 46  # All layers dense
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # All layers should be dense (moe_layer_freq all zeros)
        expected_moe_layer_freq = [0] * 46
        assert provider.moe_layer_freq == expected_moe_layer_freq

    def test_provider_bridge_with_all_moe_layers(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge when all layers are MoE."""
        mock_hf_pretrained.config.text_config.first_k_dense_replace = 0  # All layers MoE
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        # All layers should be MoE (moe_layer_freq all ones)
        expected_moe_layer_freq = [1] * 46
        assert provider.moe_layer_freq == expected_moe_layer_freq


class TestGLM45VBridgeCompatibility:
    """Test compatibility with different HF model configurations."""

    def test_provider_bridge_with_group_query_attention(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with group query attention."""
        mock_hf_pretrained.config.text_config.num_attention_heads = 32
        mock_hf_pretrained.config.text_config.num_key_value_heads = 4

        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 4

    def test_provider_bridge_with_different_kv_heads(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with different num_key_value_heads."""
        test_kv_heads = [1, 4, 8, 16]

        for kv_heads in test_kv_heads:
            mock_hf_pretrained.config.text_config.num_key_value_heads = kv_heads
            provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.num_query_groups == kv_heads

    def test_provider_bridge_with_qk_norm_disabled(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with QK normalization disabled."""
        mock_hf_pretrained.config.text_config.use_qk_norm = False
        provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.qk_layernorm is False

    def test_provider_bridge_with_different_rope_theta(self, glm_45v_bridge, mock_hf_pretrained):
        """Test provider_bridge with different RoPE theta values."""
        test_rope_theta = [10000.0, 100000.0, 1000000.0]

        for rope_theta in test_rope_theta:
            mock_hf_pretrained.config.text_config.rope_theta = rope_theta
            provider = glm_45v_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.rotary_base == rope_theta
