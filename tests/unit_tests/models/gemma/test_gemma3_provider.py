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

from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.activations import openai_gelu
from megatron.bridge.models.gemma.gemma3_provider import (
    Gemma3ModelProvider,
    Gemma3ModelProvider1B,
    Gemma3ModelProvider4B,
    Gemma3ModelProvider12B,
    Gemma3ModelProvider27B,
)


class TestGemma3ModelProvider:
    """Test cases for base Gemma3ModelProvider class."""

    def test_gemma3_model_provider_initialization(self):
        """Test Gemma3ModelProvider can be initialized with default values."""
        provider = Gemma3ModelProvider(
            num_layers=26,
            hidden_size=1152,
            num_attention_heads=4,
        )

        # Check required transformer config fields
        assert provider.num_layers == 26
        assert provider.hidden_size == 1152
        assert provider.num_attention_heads == 4

        # Check Gemma3-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == openai_gelu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.seq_length == 131_072
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is True
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.attention_backend == AttnBackend.flash

        # Check Gemma3-specific parameters
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == (10_000, 1_000_000)
        assert provider.window_size == 512
        assert provider.interleaved_attn_pattern == (5, 1)
        assert provider.vocab_size == 262_208
        assert provider.gradient_accumulation_fusion is False
        assert provider.is_vision_language is False
        assert provider.flash_decode is False
        assert provider.scatter_embedding_sequence_parallel is True

    def test_gemma3_model_provider_provide_basic_functionality(self):
        """Test basic provide method functionality with model customization."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = Gemma3ModelProvider(
            num_layers=26,
            hidden_size=1152,
            num_attention_heads=4,
        )

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            with patch(
                "megatron.bridge.models.gemma.gemma3_provider.Gemma3LanguageModelEmbedding"
            ) as mock_embedding_cls:
                with patch("megatron.bridge.models.gemma.gemma3_provider.Gemma3RotaryEmbedding") as mock_rope_cls:
                    result = provider.provide()

                    # Verify that parent provide was called
                    assert result == mock_model

                    # Verify that the rotary_base was temporarily adjusted and restored
                    assert provider.rotary_base == (10_000, 1_000_000)

                    # Verify that custom embedding and rope were created
                    mock_embedding_cls.assert_called_once()
                    mock_rope_cls.assert_called_once()


class TestGemma3ModelProvider1B:
    """Test cases for Gemma3ModelProvider1B class."""

    def test_gemma3_1b_configuration(self):
        """Test that Gemma3ModelProvider1B has correct configuration values."""
        provider = Gemma3ModelProvider1B()

        # Test 1B specific values
        assert provider.num_layers == 26
        assert provider.hidden_size == 1152
        assert provider.num_attention_heads == 4
        assert provider.num_query_groups == 1
        assert provider.kv_channels == 256
        assert provider.ffn_hidden_size == 6912
        assert provider.window_size == 512
        assert provider.rope_scaling_factor == 1.0
        assert provider.seq_length == 32768
        assert provider.vocab_size == 262_144

    def test_gemma3_1b_inheritance(self):
        """Test that Gemma3ModelProvider1B properly inherits from Gemma3ModelProvider."""
        provider = Gemma3ModelProvider1B()
        assert isinstance(provider, Gemma3ModelProvider)


class TestGemma3ModelProvider4B:
    """Test cases for Gemma3ModelProvider4B class."""

    def test_gemma3_4b_configuration(self):
        """Test that Gemma3ModelProvider4B has correct configuration values."""
        provider = Gemma3ModelProvider4B()

        # Test 4B specific values
        assert provider.num_layers == 34
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 4
        assert provider.kv_channels == 256
        assert provider.ffn_hidden_size == 10240
        assert provider.window_size == 1024
        assert provider.rope_scaling_factor == 8.0

    def test_gemma3_4b_inheritance(self):
        """Test that Gemma3ModelProvider4B properly inherits from Gemma3ModelProvider."""
        provider = Gemma3ModelProvider4B()
        assert isinstance(provider, Gemma3ModelProvider)


class TestGemma3ModelProvider12B:
    """Test cases for Gemma3ModelProvider12B class."""

    def test_gemma3_12b_configuration(self):
        """Test that Gemma3ModelProvider12B has correct configuration values."""
        provider = Gemma3ModelProvider12B()

        # Test 12B specific values
        assert provider.num_layers == 48
        assert provider.hidden_size == 3840
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 8
        assert provider.kv_channels == 256
        assert provider.ffn_hidden_size == 15360
        assert provider.window_size == 1024
        assert provider.rope_scaling_factor == 8.0

    def test_gemma3_12b_inheritance(self):
        """Test that Gemma3ModelProvider12B properly inherits from Gemma3ModelProvider."""
        provider = Gemma3ModelProvider12B()
        assert isinstance(provider, Gemma3ModelProvider)


class TestGemma3ModelProvider27B:
    """Test cases for Gemma3ModelProvider27B class."""

    def test_gemma3_27b_configuration(self):
        """Test that Gemma3ModelProvider27B has correct configuration values."""
        provider = Gemma3ModelProvider27B()

        # Test 27B specific values
        assert provider.num_layers == 62
        assert provider.hidden_size == 5376
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 16
        assert provider.kv_channels == 128
        assert provider.ffn_hidden_size == 21504
        assert provider.window_size == 1024
        assert provider.rope_scaling_factor == 8.0

        # Test special softmax_scale for 27B
        assert provider.softmax_scale == 1.0 / math.sqrt(168)

    def test_gemma3_27b_inheritance(self):
        """Test that Gemma3ModelProvider27B properly inherits from Gemma3ModelProvider."""
        provider = Gemma3ModelProvider27B()
        assert isinstance(provider, Gemma3ModelProvider)


class TestGemma3ModelProviderIntegration:
    """Integration tests for Gemma3 model providers."""

    def test_all_providers_have_provide_method(self):
        """Test that all provider classes have the provide method."""
        providers = [
            Gemma3ModelProvider1B(),
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
            Gemma3ModelProvider27B(),
        ]

        for provider in providers:
            assert hasattr(provider, "provide")
            assert callable(getattr(provider, "provide"))
