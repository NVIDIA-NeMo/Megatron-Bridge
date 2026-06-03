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
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.ministral3.ministral3_bridge import Ministral3Bridge
from megatron.bridge.models.ministral3.ministral3_provider import Ministral3ModelProvider


@pytest.fixture
def mock_text_config():
    """Create a mock text config for Ministral3."""
    config = Mock()
    config.num_hidden_layers = 26
    config.hidden_size = 3072
    config.intermediate_size = 9216
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.vocab_size = 131072
    config.rope_parameters = {
        "rope_theta": 1000000,
        "original_max_position_embeddings": 16384,
        "llama_4_scaling_beta": 0.5,
    }
    config.tie_word_embeddings = True
    return config


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for Ministral3."""
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
    """Create a mock HF config for Ministral3."""
    config = Mock()
    config.text_config = mock_text_config
    config.vision_config = mock_vision_config

    # VL-specific token IDs
    config.image_token_id = 10

    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    return pretrained


@pytest.fixture
def ministral3_bridge():
    """Create a Ministral3Bridge instance."""
    return Ministral3Bridge()


class TestMinistral3BridgeInitialization:
    """Test Ministral3Bridge initialization and basic functionality."""

    def test_bridge_initialization(self, ministral3_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(ministral3_bridge, Ministral3Bridge)

    def test_bridge_has_required_methods(self, ministral3_bridge):
        """Test that bridge has required methods."""
        assert hasattr(ministral3_bridge, "provider_bridge")
        assert callable(ministral3_bridge.provider_bridge)

        assert hasattr(ministral3_bridge, "mapping_registry")
        assert callable(ministral3_bridge.mapping_registry)


class TestMinistral3BridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Ministral3ModelProvider)

        # Check basic transformer config
        assert provider.num_layers == 26
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 9216
        assert provider.vocab_size == 131072

    def test_provider_bridge_rotary_config(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct rotary configuration."""
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        # Check rotary base configuration
        assert provider.rotary_base == 1000000

    def test_provider_bridge_hf_config_attribute(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge sets hf_config attribute."""
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        # Should set hf_config
        assert provider.hf_config is mock_hf_pretrained.config

    def test_provider_bridge_tie_word_embeddings(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge handles tie_word_embeddings correctly."""
        # Test with tie_word_embeddings=True
        mock_hf_pretrained.config.text_config.tie_word_embeddings = True
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.share_embeddings_and_output_weights is True

        # Test with tie_word_embeddings=False
        mock_hf_pretrained.config.text_config.tie_word_embeddings = False
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.share_embeddings_and_output_weights is False

    def test_provider_bridge_with_custom_vocab_size(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom vocabulary size."""
        mock_hf_pretrained.config.text_config.vocab_size = 150000
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.vocab_size == 150000

    def test_provider_bridge_with_custom_rope_theta(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom RoPE theta."""
        mock_hf_pretrained.config.text_config.rope_parameters["rope_theta"] = 500000
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.rotary_base == 500000

    def test_provider_bridge_with_different_layer_counts(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different layer counts."""
        test_layer_counts = [26, 34, 40]

        for num_layers in test_layer_counts:
            mock_hf_pretrained.config.text_config.num_hidden_layers = num_layers
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.num_layers == num_layers


class TestMinistral3BridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_correct_type(self, ministral3_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = ministral3_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_required_mappings(self, ministral3_bridge):
        """Test mapping_registry contains all required parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

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

    def test_mapping_registry_vision_tower_params(self, ministral3_bridge):
        """Test mapping_registry handles vision tower parameters correctly."""
        registry = ministral3_bridge.mapping_registry()

        # Should contain vision tower parameter mappings
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

        has_vision_tower = any("vision_tower" in name for name in mapping_names)
        assert has_vision_tower, "Should contain vision tower parameter mappings"

    def test_mapping_registry_multimodal_projector_params(self, ministral3_bridge):
        """Test mapping_registry handles multimodal projector parameters correctly."""
        registry = ministral3_bridge.mapping_registry()

        # Should contain multimodal projector parameter mappings
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

        has_projector = any("multi_modal_projector" in name for name in mapping_names)
        assert has_projector, "Should contain multimodal projector parameter mappings"

    def test_mapping_registry_qkv_mappings(self, ministral3_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

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

    def test_mapping_registry_mlp_mappings(self, ministral3_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

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

    def test_mapping_registry_attention_mappings(self, ministral3_bridge):
        """Test mapping_registry contains attention parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

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


class TestMinistral3BridgeEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_bridge_with_minimal_config(self, ministral3_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedVLM)
        minimal_config = Mock()

        # Create minimal text config
        text_config = Mock()
        text_config.num_hidden_layers = 26
        text_config.hidden_size = 3072
        text_config.intermediate_size = 9216
        text_config.vocab_size = 131072
        text_config.rope_parameters = {"rope_theta": 1000000}
        text_config.tie_word_embeddings = False

        minimal_config.text_config = text_config
        minimal_pretrained.config = minimal_config

        provider = ministral3_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, Ministral3ModelProvider)
        assert provider.num_layers == 26
        assert provider.hidden_size == 3072

    def test_provider_bridge_with_different_hidden_sizes(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different hidden sizes."""
        test_hidden_sizes = [3072, 4096, 5120]

        for hidden_size in test_hidden_sizes:
            mock_hf_pretrained.config.text_config.hidden_size = hidden_size
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.hidden_size == hidden_size

    def test_provider_bridge_with_different_ffn_sizes(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different FFN intermediate sizes."""
        test_ffn_sizes = [9216, 14336, 16384]

        for ffn_size in test_ffn_sizes:
            mock_hf_pretrained.config.text_config.intermediate_size = ffn_size
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.ffn_hidden_size == ffn_size


class TestMinistral3BridgeCompatibility:
    """Test compatibility with different HF model configurations."""

    def test_provider_bridge_with_group_query_attention(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with group query attention (default for Ministral3)."""
        mock_hf_pretrained.config.text_config.num_attention_heads = 32
        mock_hf_pretrained.config.text_config.num_key_value_heads = 8

        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        # Ministral3 uses GQA by default
        assert provider.num_attention_heads == 32
        # num_query_groups should be set from provider defaults

    def test_provider_bridge_with_different_vocab_sizes(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different vocabulary sizes."""
        test_vocab_sizes = [100000, 131072, 150000]

        for vocab_size in test_vocab_sizes:
            mock_hf_pretrained.config.text_config.vocab_size = vocab_size
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.vocab_size == vocab_size

    def test_provider_bridge_3b_config(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with 3B model configuration."""
        mock_hf_pretrained.config.text_config.num_hidden_layers = 26
        mock_hf_pretrained.config.text_config.hidden_size = 3072
        mock_hf_pretrained.config.text_config.intermediate_size = 9216

        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_layers == 26
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 9216

    def test_provider_bridge_8b_config(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with 8B model configuration."""
        mock_hf_pretrained.config.text_config.num_hidden_layers = 34
        mock_hf_pretrained.config.text_config.hidden_size = 4096
        mock_hf_pretrained.config.text_config.intermediate_size = 14336

        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_layers == 34
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 14336

    def test_provider_bridge_14b_config(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with 14B model configuration."""
        mock_hf_pretrained.config.text_config.num_hidden_layers = 40
        mock_hf_pretrained.config.text_config.hidden_size = 5120
        mock_hf_pretrained.config.text_config.intermediate_size = 16384
        mock_hf_pretrained.config.text_config.rope_parameters["rope_theta"] = 1000000000.0

        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_layers == 40
        assert provider.hidden_size == 5120
        assert provider.ffn_hidden_size == 16384
        assert provider.rotary_base == 1000000000.0
