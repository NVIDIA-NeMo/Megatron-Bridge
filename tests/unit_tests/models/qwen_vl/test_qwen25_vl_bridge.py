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

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.qwen25_vl_bridge import Qwen25VLBridge
from megatron.bridge.models.qwen_vl.qwen_vl_provider import Qwen25VLModelProvider


@pytest.fixture
def mock_hf_config():
    """Create a mock HF config for Qwen2.5-VL."""
    config = Mock()
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 151936
    config.max_position_embeddings = 4096
    config.rope_theta = 1000000.0
    config.tie_word_embeddings = False

    # VL-specific configuration
    config.vision_config = Qwen2_5_VLVisionConfig()
    config.bos_token_id = 151643
    config.eos_token_id = 151645
    config.vision_start_token_id = 151652
    config.vision_end_token_id = 151653
    config.vision_token_id = 151654
    config.image_token_id = 151655
    config.video_token_id = 151656

    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    pretrained.generation_config = GenerationConfig()
    return pretrained


@pytest.fixture
def qwen25_vl_bridge():
    """Create a Qwen25VLBridge instance."""
    return Qwen25VLBridge()


class TestQwen25VLBridgeInitialization:
    """Test Qwen25VLBridge initialization and basic functionality."""

    def test_bridge_initialization(self, qwen25_vl_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(qwen25_vl_bridge, Qwen25VLBridge)

    def test_bridge_has_required_methods(self, qwen25_vl_bridge):
        """Test that bridge has required methods."""
        assert hasattr(qwen25_vl_bridge, "provider_bridge")
        assert callable(qwen25_vl_bridge.provider_bridge)

        assert hasattr(qwen25_vl_bridge, "mapping_registry")
        assert callable(qwen25_vl_bridge.mapping_registry)


class TestQwen25VLBridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Qwen25VLModelProvider)

        # Check basic transformer config
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 11008
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 32
        assert provider.init_method_std == 0.02
        assert provider.layernorm_epsilon == 1e-6
        assert provider.vocab_size == 151936
        assert provider.seq_length == 4096
        assert provider.rotary_base == 1000000.0
        assert provider.share_embeddings_and_output_weights is False

    def test_provider_bridge_vl_specific_config(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct VL-specific configuration."""
        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Check VL-specific token IDs
        assert provider.bos_token_id == 151643
        assert provider.eos_token_id == 151645
        assert provider.vision_start_token_id == 151652
        assert provider.vision_end_token_id == 151653
        assert provider.vision_token_id == 151654
        assert provider.image_token_id == 151655
        assert provider.video_token_id == 151656

        # Check vision config
        assert isinstance(provider.vision_config, Qwen2_5_VLVisionConfig)
        assert provider.add_qkv_bias is True

    def test_provider_bridge_with_custom_token_ids(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom token IDs."""
        # Modify mock config with custom token IDs
        mock_hf_pretrained.config.bos_token_id = 100
        mock_hf_pretrained.config.eos_token_id = 101
        mock_hf_pretrained.config.vision_start_token_id = 102

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.vision_start_token_id == 102

    def test_provider_bridge_with_missing_token_ids(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with missing token IDs uses defaults."""
        # Remove some token IDs from config
        delattr(mock_hf_pretrained.config, "vision_start_token_id")
        delattr(mock_hf_pretrained.config, "image_token_id")

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Should use defaults
        assert provider.vision_start_token_id == 151652
        assert provider.image_token_id == 151655

    @patch.object(Qwen25VLBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_handling(self, mock_dtype_from_hf, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles dtype correctly."""
        mock_dtype_from_hf.return_value = torch.float16

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    @patch.object(Qwen25VLBridge, "dtype_from_hf")
    def test_provider_bridge_bfloat16_handling(self, mock_dtype_from_hf, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles bfloat16 correctly."""
        mock_dtype_from_hf.return_value = torch.bfloat16

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    @patch.object(Qwen25VLBridge, "make_vocab_size_divisible_by")
    def test_provider_bridge_vocab_size_divisibility(self, mock_divisible, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles vocab size divisibility."""
        mock_divisible.return_value = 128

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        mock_divisible.assert_called_once_with(151936)
        assert provider.make_vocab_size_divisible_by == 128

    def test_provider_bridge_with_tied_embeddings(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with tied embeddings."""
        mock_hf_pretrained.config.tie_word_embeddings = True

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.share_embeddings_and_output_weights is True

    def test_provider_bridge_generation_config(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge includes generation config."""
        custom_gen_config = GenerationConfig(max_length=2048, temperature=0.8)
        mock_hf_pretrained.generation_config = custom_gen_config

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.generation_config is custom_gen_config


class TestQwen25VLBridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_correct_type(self, qwen25_vl_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = qwen25_vl_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_required_mappings(self, qwen25_vl_bridge):
        """Test mapping_registry contains all required parameter mappings."""
        registry = qwen25_vl_bridge.mapping_registry()

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

        # Should contain output layer mapping
        has_output = any("lm_head" in name or "output_layer" in name for name in mapping_names)
        assert has_output, "Should contain output layer mapping"

    def test_mapping_registry_visual_params(self, qwen25_vl_bridge):
        """Test mapping_registry handles visual parameters correctly."""
        registry = qwen25_vl_bridge.mapping_registry()

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

    def test_mapping_registry_qkv_mappings(self, qwen25_vl_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = qwen25_vl_bridge.mapping_registry()

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

    def test_mapping_registry_mlp_mappings(self, qwen25_vl_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = qwen25_vl_bridge.mapping_registry()

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


class TestQwen25VLBridgeEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_bridge_with_minimal_config(self, qwen25_vl_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedVLM)
        minimal_config = Mock()

        # Set only required fields
        minimal_config.num_hidden_layers = 24
        minimal_config.hidden_size = 2048
        minimal_config.intermediate_size = 5504
        minimal_config.num_attention_heads = 16
        minimal_config.num_key_value_heads = 16
        minimal_config.initializer_range = 0.02
        minimal_config.rms_norm_eps = 1e-6
        minimal_config.vocab_size = 151936
        minimal_config.max_position_embeddings = 4096
        minimal_config.rope_theta = 1000000.0
        minimal_config.vision_config = Qwen2_5_VLVisionConfig()

        minimal_pretrained.config = minimal_config
        minimal_pretrained.generation_config = GenerationConfig()

        provider = qwen25_vl_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, Qwen25VLModelProvider)
        assert provider.num_layers == 24
        assert provider.hidden_size == 2048

    def test_provider_bridge_with_different_vocab_sizes(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different vocabulary sizes."""
        test_vocab_sizes = [32000, 151936, 152064]

        for vocab_size in test_vocab_sizes:
            mock_hf_pretrained.config.vocab_size = vocab_size
            provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.vocab_size == vocab_size

    def test_provider_bridge_with_different_sequence_lengths(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different sequence lengths."""
        test_seq_lengths = [2048, 4096, 8192, 32768]

        for seq_length in test_seq_lengths:
            mock_hf_pretrained.config.max_position_embeddings = seq_length
            provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.seq_length == seq_length


class TestQwen25VLBridgeCompatibility:
    """Test compatibility with different HF model configurations."""

    def test_provider_bridge_with_group_query_attention(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with group query attention."""
        mock_hf_pretrained.config.num_attention_heads = 32
        mock_hf_pretrained.config.num_key_value_heads = 8

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8

    def test_provider_bridge_with_different_rope_theta(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different RoPE theta values."""
        test_rope_values = [10000.0, 500000.0, 1000000.0]

        for rope_theta in test_rope_values:
            mock_hf_pretrained.config.rope_theta = rope_theta
            provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.rotary_base == rope_theta

    def test_provider_bridge_vision_config_types(self, qwen25_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with different vision config types."""
        # Test with custom vision config
        custom_vision_config = Qwen2_5_VLVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24)
        mock_hf_pretrained.config.vision_config = custom_vision_config

        provider = qwen25_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.vision_config.hidden_size == 1024
        assert provider.vision_config.intermediate_size == 4096
        assert provider.vision_config.num_hidden_layers == 24
