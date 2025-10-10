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

import math
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig, SiglipVisionConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.gemma_vl.gemma3_vl_bridge import Gemma3VLBridge
from megatron.bridge.models.gemma_vl.gemma3_vl_provider import Gemma3VLModelProvider
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


@pytest.fixture
def mock_text_config():
    """Create a mock text config for Gemma3 VL."""
    config = Mock()
    config.num_hidden_layers = 28
    config.hidden_size = 2560
    config.intermediate_size = 15360
    config.num_attention_heads = 10
    config.num_key_value_heads = 10
    config.head_dim = 256
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 262144
    config.max_position_embeddings = 131072
    config.sliding_window = 512
    config.rope_local_base_freq = 10000
    config.rope_theta = 1000000.0
    config.query_pre_attn_scalar = 256
    config.rope_scaling = None
    return config


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for Gemma3 VL."""
    config = SiglipVisionConfig()
    config.hidden_size = 1152
    config.intermediate_size = 4304
    config.num_hidden_layers = 27
    config.num_attention_heads = 16
    config.patch_size = 14
    config.image_size = 896
    return config


@pytest.fixture
def mock_hf_config(mock_text_config, mock_vision_config):
    """Create a mock HF config for Gemma3 VL."""
    config = Mock()
    config.text_config = mock_text_config
    config.vision_config = mock_vision_config
    config.mm_tokens_per_image = 256

    # VL-specific token IDs
    config.bos_token_id = 2
    config.eos_token_id = 1
    config.vision_start_token_id = 255999
    config.vision_end_token_id = 256000
    config.image_token_id = 262144

    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    pretrained.generation_config = GenerationConfig()
    return pretrained


@pytest.fixture
def gemma3_vl_bridge():
    """Create a Gemma3VLBridge instance."""
    return Gemma3VLBridge()


class TestGemma3VLBridgeInitialization:
    """Test Gemma3VLBridge initialization and basic functionality."""

    def test_bridge_initialization(self, gemma3_vl_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(gemma3_vl_bridge, Gemma3VLBridge)

    def test_bridge_has_required_methods(self, gemma3_vl_bridge):
        """Test that bridge has required methods."""
        assert hasattr(gemma3_vl_bridge, "provider_bridge")
        assert callable(gemma3_vl_bridge.provider_bridge)

        assert hasattr(gemma3_vl_bridge, "mapping_registry")
        assert callable(gemma3_vl_bridge.mapping_registry)


class TestGemma3VLBridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Gemma3VLModelProvider)

        # Check basic transformer config
        assert provider.num_layers == 28
        assert provider.hidden_size == 2560
        assert provider.ffn_hidden_size == 15360
        assert provider.num_attention_heads == 10
        assert provider.num_query_groups == 10
        assert provider.kv_channels == 256
        assert provider.init_method_std == 0.02
        assert provider.layernorm_epsilon == 1e-6
        assert provider.vocab_size == 262144
        assert provider.seq_length == 131072
        assert provider.window_size == 512

    def test_provider_bridge_rotary_config(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct rotary configuration."""
        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Check rotary base configuration (tuple of local and global)
        assert provider.rotary_base == (10000, 1000000.0)
        assert provider.rope_scaling_factor == 1.0

    def test_provider_bridge_softmax_scale(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge calculates correct softmax scale."""
        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Should be 1.0 / sqrt(query_pre_attn_scalar)
        expected_scale = 1.0 / math.sqrt(256)
        assert provider.softmax_scale == expected_scale

    def test_provider_bridge_vl_specific_config(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct VL-specific configuration."""
        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Check VL-specific token IDs
        assert provider.bos_token_id == 2
        assert provider.eos_token_id == 1
        assert provider.vision_start_token_id == 255999
        assert provider.vision_end_token_id == 256000
        assert provider.image_token_id == 262144

        # Check vision config
        assert isinstance(provider.vision_config, SiglipVisionConfig)
        assert provider.mm_tokens_per_image == 256

    def test_provider_bridge_vision_projector_config(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge configures vision projector correctly."""
        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Check vision projector configuration
        assert provider.vision_projector_config.input_size == 1152  # vision_config.hidden_size
        assert provider.vision_projector_config.hidden_size == 2560  # text_config.hidden_size

    def test_provider_bridge_with_custom_token_ids(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom token IDs."""
        # Modify mock config with custom token IDs
        mock_hf_pretrained.config.bos_token_id = 100
        mock_hf_pretrained.config.eos_token_id = 101
        mock_hf_pretrained.config.vision_start_token_id = 102
        mock_hf_pretrained.config.image_token_id = 103

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.vision_start_token_id == 102
        assert provider.image_token_id == 103

    def test_provider_bridge_with_missing_token_ids(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with missing token IDs uses defaults."""
        # Remove some token IDs from config
        delattr(mock_hf_pretrained.config, "vision_start_token_id")
        delattr(mock_hf_pretrained.config, "image_token_id")

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Should use defaults
        assert provider.vision_start_token_id == 255999
        assert provider.image_token_id == 151655  # Default from bridge

    def test_provider_bridge_with_rope_scaling(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with RoPE scaling configuration."""
        # Add rope scaling to config
        mock_hf_pretrained.config.text_config.rope_scaling = {"factor": 2.0}

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.rope_scaling_factor == 2.0

    @patch.object(Gemma3VLBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_handling(self, mock_dtype_from_hf, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles dtype correctly."""
        mock_dtype_from_hf.return_value = torch.float16

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    @patch.object(Gemma3VLBridge, "dtype_from_hf")
    def test_provider_bridge_bfloat16_handling(self, mock_dtype_from_hf, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles bfloat16 correctly."""
        mock_dtype_from_hf.return_value = torch.bfloat16

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    @patch.object(Gemma3VLBridge, "dtype_from_hf")
    def test_provider_bridge_float32_handling(self, mock_dtype_from_hf, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles float32 correctly."""
        mock_dtype_from_hf.return_value = torch.float32

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is False
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float32


class TestGemma3VLBridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_correct_type(self, gemma3_vl_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = gemma3_vl_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_required_mappings(self, gemma3_vl_bridge):
        """Test mapping_registry contains all required parameter mappings."""
        registry = gemma3_vl_bridge.mapping_registry()

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

    def test_mapping_registry_vision_tower_params(self, gemma3_vl_bridge):
        """Test mapping_registry handles vision tower parameters correctly."""
        registry = gemma3_vl_bridge.mapping_registry()

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

    def test_mapping_registry_multimodal_projector_params(self, gemma3_vl_bridge):
        """Test mapping_registry handles multimodal projector parameters correctly."""
        registry = gemma3_vl_bridge.mapping_registry()

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

    def test_mapping_registry_qkv_mappings(self, gemma3_vl_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = gemma3_vl_bridge.mapping_registry()

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

    def test_mapping_registry_mlp_mappings(self, gemma3_vl_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = gemma3_vl_bridge.mapping_registry()

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

    def test_mapping_registry_attention_mappings(self, gemma3_vl_bridge):
        """Test mapping_registry contains attention parameter mappings."""
        registry = gemma3_vl_bridge.mapping_registry()

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


class TestGemma3VLBridgeEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_bridge_with_minimal_config(self, gemma3_vl_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedVLM)
        minimal_config = Mock()

        # Create minimal text config
        text_config = Mock()
        text_config.num_hidden_layers = 18
        text_config.hidden_size = 2048
        text_config.intermediate_size = 8192
        text_config.num_attention_heads = 8
        text_config.num_key_value_heads = 8
        text_config.head_dim = 256
        text_config.initializer_range = 0.02
        text_config.rms_norm_eps = 1e-6
        text_config.vocab_size = 262144
        text_config.max_position_embeddings = 32768
        text_config.sliding_window = 512
        text_config.rope_local_base_freq = 10000
        text_config.rope_theta = 1000000.0
        text_config.query_pre_attn_scalar = 256
        text_config.rope_scaling = None

        # Create minimal vision config
        vision_config = SiglipVisionConfig()

        minimal_config.text_config = text_config
        minimal_config.vision_config = vision_config
        minimal_config.mm_tokens_per_image = 256

        minimal_pretrained.config = minimal_config

        provider = gemma3_vl_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, Gemma3VLModelProvider)
        assert provider.num_layers == 18
        assert provider.hidden_size == 2048

    def test_provider_bridge_with_different_vocab_sizes(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different vocabulary sizes."""
        test_vocab_sizes = [256000, 262144, 300000]

        for vocab_size in test_vocab_sizes:
            mock_hf_pretrained.config.text_config.vocab_size = vocab_size
            provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.vocab_size == vocab_size

    def test_provider_bridge_with_different_sequence_lengths(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different sequence lengths."""
        test_seq_lengths = [8192, 32768, 131072]

        for seq_length in test_seq_lengths:
            mock_hf_pretrained.config.text_config.max_position_embeddings = seq_length
            provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.seq_length == seq_length

    def test_provider_bridge_with_different_window_sizes(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different sliding window sizes."""
        test_window_sizes = [256, 512, 1024, 4096]

        for window_size in test_window_sizes:
            mock_hf_pretrained.config.text_config.sliding_window = window_size
            provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.window_size == window_size


class TestGemma3VLBridgeCompatibility:
    """Test compatibility with different HF model configurations."""

    def test_provider_bridge_with_group_query_attention(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with group query attention."""
        mock_hf_pretrained.config.text_config.num_attention_heads = 32
        mock_hf_pretrained.config.text_config.num_key_value_heads = 8

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8

    def test_provider_bridge_with_different_rope_values(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different RoPE values."""
        test_local_freq = 5000
        test_theta = 500000.0

        mock_hf_pretrained.config.text_config.rope_local_base_freq = test_local_freq
        mock_hf_pretrained.config.text_config.rope_theta = test_theta

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.rotary_base == (test_local_freq, test_theta)

    def test_provider_bridge_with_different_vision_configs(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different vision config sizes."""
        # Create custom vision config
        custom_vision_config = SiglipVisionConfig()
        custom_vision_config.hidden_size = 768
        custom_vision_config.intermediate_size = 3072
        custom_vision_config.num_hidden_layers = 12

        mock_hf_pretrained.config.vision_config = custom_vision_config

        provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.vision_config.hidden_size == 768
        assert provider.vision_config.intermediate_size == 3072
        assert provider.vision_config.num_hidden_layers == 12
        # Vision projector should be updated with new input size
        assert provider.vision_projector_config.input_size == 768

    def test_provider_bridge_with_different_mm_tokens(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different multimodal tokens per image."""
        test_tokens = [64, 128, 256, 512, 1024]

        for tokens in test_tokens:
            mock_hf_pretrained.config.mm_tokens_per_image = tokens
            provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.mm_tokens_per_image == tokens

    def test_provider_bridge_with_different_head_dims(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different head dimensions."""
        test_head_dims = [128, 256, 512]

        for head_dim in test_head_dims:
            mock_hf_pretrained.config.text_config.head_dim = head_dim
            mock_hf_pretrained.config.text_config.query_pre_attn_scalar = head_dim

            provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)

            assert provider.kv_channels == head_dim
            expected_scale = 1.0 / math.sqrt(head_dim)
            assert provider.softmax_scale == expected_scale

    def test_provider_bridge_with_different_layer_counts(self, gemma3_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different layer counts."""
        test_layer_counts = [12, 18, 24, 28, 32]

        for num_layers in test_layer_counts:
            mock_hf_pretrained.config.text_config.num_hidden_layers = num_layers
            provider = gemma3_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.num_layers == num_layers
