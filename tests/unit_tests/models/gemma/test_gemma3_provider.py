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
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gemma.gemma3_provider import (
    Gemma3LanguageModelEmbedding,
    Gemma3ModelProvider,
    Gemma3ModelProvider1B,
    Gemma3ModelProvider4B,
    Gemma3ModelProvider12B,
    Gemma3ModelProvider27B,
    Gemma3RotaryEmbedding,
    Gemma3SelfAttention,
    Gemma3TEDotProductAttention,
    TERowParallelLinearLayerNorm,
    _is_local_attn_layer,
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
        assert provider.seq_length == 131_072
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == (10_000, 1_000_000)  # (local, global)
        assert provider.share_embeddings_and_output_weights is True

        # Check normalization settings
        assert provider.normalization == "RMSNorm"
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.layernorm_epsilon == 1e-6

        # Check attention settings
        assert provider.window_size == 512
        assert provider.interleaved_attn_pattern == (5, 1)
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.rope_scaling_factor == 1.0
        assert provider.attention_backend == AttnBackend.flash

        # Check MLP settings
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False

        # Check other settings
        assert provider.is_vision_language is False
        assert provider.flash_decode is False
        assert provider.gradient_accumulation_fusion is False
        assert provider.scatter_embedding_sequence_parallel is True

        # Check data type settings
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.autocast_dtype == torch.bfloat16

    @patch("megatron.bridge.models.gemma.gemma3_provider.Gemma3LanguageModelEmbedding")
    @patch("megatron.bridge.models.gemma.gemma3_provider.Gemma3RotaryEmbedding")
    def test_gemma3_provider_provide_method(self, mock_rotary_embedding, mock_language_embedding):
        """Test that provide method creates and configures the model correctly."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()
        mock_model.setup_embeddings_and_output_layer = Mock()

        provider = Gemma3ModelProvider(
            num_layers=26,
            hidden_size=1152,
            num_attention_heads=4,
            kv_channels=256,
            vocab_size=262144,
            seq_length=32768,
        )

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            result = provider.provide(pre_process=True, post_process=True, vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that custom embedding was created
            mock_language_embedding.assert_called_once_with(
                config=provider,
                vocab_size=provider.vocab_size,
                max_sequence_length=provider.seq_length,
                position_embedding_type=provider.position_embedding_type,
                scatter_to_sequence_parallel=provider.scatter_embedding_sequence_parallel,
            )

            # Verify that custom rotary embedding was created
            mock_rotary_embedding.assert_called_once()
            rotary_call_args = mock_rotary_embedding.call_args[1]
            assert rotary_call_args["kv_channels"] == provider.kv_channels
            assert rotary_call_args["rotary_base"] == 1_000_000  # global base
            assert rotary_call_args["rope_scaling"] is False
            assert rotary_call_args["rope_scaling_factor"] == provider.rope_scaling_factor
            assert rotary_call_args["rotary_base_local"] == 10_000

            # Verify setup method was called
            mock_model.setup_embeddings_and_output_layer.assert_called_once()

    def test_gemma3_provider_rotary_base_handling(self):
        """Test that rotary base is handled correctly during provide."""
        provider = Gemma3ModelProvider(
            num_layers=26,
            hidden_size=1152,
            num_attention_heads=4,
            vocab_size=262144,  # Add required vocab_size
            kv_channels=256,  # Add required kv_channels
        )

        # Check initial rotary_base
        assert provider.rotary_base == (10_000, 1_000_000)

        with patch.object(provider.__class__.__bases__[0], "provide") as mock_super_provide:
            mock_model = Mock()
            mock_model.embedding = Mock()
            mock_model.setup_embeddings_and_output_layer = Mock()
            mock_super_provide.return_value = mock_model

            with (
                patch("megatron.bridge.models.gemma.gemma3_provider.Gemma3LanguageModelEmbedding"),
                patch("megatron.bridge.models.gemma.gemma3_provider.Gemma3RotaryEmbedding"),
            ):
                provider.provide()

                # Verify rotary_base was temporarily set to local value during super().provide()
                # and then restored to tuple
                assert provider.rotary_base == (10_000, 1_000_000)

    def test_gemma3_provider_without_embedding(self):
        """Test provide method when model doesn't have embedding attribute."""
        mock_model = Mock()
        # Remove embedding attribute
        del mock_model.embedding
        mock_model.setup_embeddings_and_output_layer = Mock()

        provider = Gemma3ModelProvider(
            num_layers=26,
            hidden_size=1152,
            num_attention_heads=4,
            vocab_size=262144,  # Add required vocab_size
            kv_channels=256,  # Add required kv_channels
        )

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            with (
                patch("megatron.bridge.models.gemma.gemma3_provider.Gemma3LanguageModelEmbedding") as mock_embedding,
                patch("megatron.bridge.models.gemma.gemma3_provider.Gemma3RotaryEmbedding"),
            ):
                provider.provide()

                # Verify that custom embedding was NOT created
                mock_embedding.assert_not_called()
                # But setup method should still be called
                mock_model.setup_embeddings_and_output_layer.assert_called_once()


class TestGemma3ModelProvider1B:
    """Test cases for Gemma3ModelProvider1B class."""

    def test_gemma3_1b_configuration(self):
        """Test that Gemma3ModelProvider1B has correct configuration values."""
        provider = Gemma3ModelProvider1B()

        # Test 1B specific values
        assert provider.is_vision_language is False
        assert provider.num_layers == 26
        assert provider.hidden_size == 1152
        assert provider.num_attention_heads == 4
        assert provider.num_query_groups == 1
        assert provider.kv_channels == 256
        assert provider.ffn_hidden_size == 6912
        assert provider.window_size == 512
        assert provider.rope_scaling_factor == 1.0  # no rope scaling
        assert provider.seq_length == 32768
        assert provider.bf16 is True
        assert provider.vocab_size == 262_144

        # Test inherited Gemma3 defaults
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.rotary_base == (10_000, 1_000_000)
        assert provider.interleaved_attn_pattern == (5, 1)

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
        assert provider.is_vision_language is True  # VL model
        assert provider.num_layers == 34
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 4
        assert provider.kv_channels == 256
        assert provider.ffn_hidden_size == 10240
        assert provider.window_size == 1024
        assert provider.rope_scaling_factor == 8.0
        assert provider.vocab_size == 262_208

        # Test inherited Gemma3 defaults
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True

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
        assert provider.is_vision_language is True  # VL model
        assert provider.num_layers == 48
        assert provider.hidden_size == 3840
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 8
        assert provider.kv_channels == 256
        assert provider.ffn_hidden_size == 15360
        assert provider.window_size == 1024
        assert provider.rope_scaling_factor == 8.0
        assert provider.vocab_size == 262_208

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
        assert provider.is_vision_language is True  # VL model
        assert provider.num_layers == 62
        assert provider.hidden_size == 5376
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 16
        assert provider.kv_channels == 128  # Different from other sizes
        assert provider.softmax_scale == 1.0 / math.sqrt(168)  # Special for 27B
        assert provider.ffn_hidden_size == 21504
        assert provider.window_size == 1024
        assert provider.rope_scaling_factor == 8.0
        assert provider.vocab_size == 262_208

    def test_gemma3_27b_softmax_scale_calculation(self):
        """Test that 27B model has correct softmax scale calculation."""
        provider = Gemma3ModelProvider27B()

        # Verify the softmax scale calculation: (5376 // 32)^(-0.5) = 168^(-0.5)
        expected_scale = 1.0 / math.sqrt(168)
        assert abs(provider.softmax_scale - expected_scale) < 1e-10

    def test_gemma3_27b_inheritance(self):
        """Test that Gemma3ModelProvider27B properly inherits from Gemma3ModelProvider."""
        provider = Gemma3ModelProvider27B()
        assert isinstance(provider, Gemma3ModelProvider)


class TestGemma3UtilityFunctions:
    """Test cases for Gemma3 utility functions."""

    def test_is_local_attn_layer(self):
        """Test the _is_local_attn_layer function."""
        # Test with default pattern (5, 1) - sum = 6
        pattern = (5, 1)

        # Layer 0: 0 % 6 = 0, should be global (False)
        assert _is_local_attn_layer(0, pattern) is False

        # Layer 1: 1 % 6 = 1, should be local (True)
        assert _is_local_attn_layer(1, pattern) is True

        # Layer 5: 5 % 6 = 5, should be local (True)
        assert _is_local_attn_layer(5, pattern) is True

        # Layer 6: 6 % 6 = 0, should be global (False)
        assert _is_local_attn_layer(6, pattern) is False

        # Layer 12: 12 % 6 = 0, should be global (False)
        assert _is_local_attn_layer(12, pattern) is False

    def test_is_local_attn_layer_different_pattern(self):
        """Test _is_local_attn_layer with different pattern."""
        # Test with pattern (3, 2) - sum = 5
        pattern = (3, 2)

        # Layer 0: 0 % 5 = 0, should be global (False)
        assert _is_local_attn_layer(0, pattern) is False

        # Layer 1-3: should be local (True)
        for i in range(1, 4):
            assert _is_local_attn_layer(i, pattern) is True

        # Layer 4: 4 % 5 = 4, should be local (True)
        assert _is_local_attn_layer(4, pattern) is True

        # Layer 5: 5 % 5 = 0, should be global (False)
        assert _is_local_attn_layer(5, pattern) is False


class TestGemma3CustomComponents:
    """Test cases for Gemma3-specific custom components."""

    def test_gemma3_self_attention_initialization(self):
        """Test Gemma3SelfAttention can be initialized."""
        # This is a basic test since the actual functionality requires complex mocking
        # The main logic is in the forward method which switches rope embeddings
        from megatron.core.transformer.attention import SelfAttention

        assert issubclass(Gemma3SelfAttention, SelfAttention)  # Check it inherits from SelfAttention

    def test_gemma3_te_dot_product_attention_initialization(self):
        """Test Gemma3TEDotProductAttention can be initialized."""
        # This component modifies window_size based on layer number
        # Check it's a class that exists and can be imported
        assert Gemma3TEDotProductAttention is not None
        assert callable(Gemma3TEDotProductAttention)

    def test_gemma3_language_model_embedding_forward(self):
        """Test Gemma3LanguageModelEmbedding forward method."""
        # Create a simple test without complex initialization
        mock_config = Mock()
        mock_config.hidden_size = 3

        # Create a minimal embedding instance
        embedding = Gemma3LanguageModelEmbedding.__new__(Gemma3LanguageModelEmbedding)
        embedding.config = mock_config

        # Mock the parent forward method
        with patch.object(
            Gemma3LanguageModelEmbedding.__bases__[0], "forward", return_value=torch.tensor([[1.0, 2.0, 3.0]])
        ):
            result = embedding.forward(input_ids=torch.tensor([[1]]), position_ids=torch.tensor([[0]]))

            # Should apply scaling: embeddings * sqrt(hidden_size)
            expected_scale = math.sqrt(3)
            expected = torch.tensor([[1.0, 2.0, 3.0]]) * expected_scale
            assert torch.allclose(result, expected)

    def test_gemma3_rotary_embedding_initialization(self):
        """Test Gemma3RotaryEmbedding initialization."""
        # Test that rope_scaling must be False
        with pytest.raises(AssertionError):
            with patch("megatron.bridge.models.gemma.gemma3_provider.RotaryEmbedding"):
                Gemma3RotaryEmbedding(
                    rope_scaling=True,  # Should cause assertion error
                    kv_channels=256,
                    rotary_percent=1.0,  # Add required parameter
                )

        # Test successful initialization with proper mocking
        with patch("megatron.bridge.models.gemma.gemma3_provider.RotaryEmbedding") as mock_rotary_embedding:
            mock_rotary_embedding.return_value = Mock()

            Gemma3RotaryEmbedding(
                rope_scaling=False,
                rope_scaling_factor=8.0,
                rotary_base=1_000_000,
                rotary_base_local=10_000,
                kv_channels=256,
                rotary_percent=1.0,  # Add required parameter
            )

            # Verify that RotaryEmbedding was called for local rope
            assert mock_rotary_embedding.call_count >= 1

    def test_te_row_parallel_linear_layer_norm(self):
        """Test TERowParallelLinearLayerNorm initialization and forward."""
        # Test that the class exists and can be imported
        assert TERowParallelLinearLayerNorm is not None
        assert callable(TERowParallelLinearLayerNorm)

        # Test forward method logic with minimal mocking
        layer = TERowParallelLinearLayerNorm.__new__(TERowParallelLinearLayerNorm)
        layer.post_layernorm = Mock(return_value=torch.randn(2, 512))

        # Mock the super().forward method
        with patch.object(
            TERowParallelLinearLayerNorm.__bases__[0], "forward", return_value=(torch.randn(2, 512), None)
        ) as mock_super_forward:
            x = torch.randn(2, 1024)
            output, bias = layer.forward(x)

            # Verify super().forward was called
            mock_super_forward.assert_called_once_with(x)

            # Verify post_layernorm was called
            layer.post_layernorm.assert_called_once()


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

    def test_vision_language_configuration(self):
        """Test that VL models are configured correctly."""
        # 1B is not VL
        provider_1b = Gemma3ModelProvider1B()
        assert provider_1b.is_vision_language is False

        # 4B, 12B, 27B are VL models
        vl_providers = [
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
            Gemma3ModelProvider27B(),
        ]

        for provider in vl_providers:
            assert provider.is_vision_language is True

    def test_rope_scaling_configuration(self):
        """Test rope scaling configuration across different model sizes."""
        # 1B has no rope scaling
        provider_1b = Gemma3ModelProvider1B()
        assert provider_1b.rope_scaling_factor == 1.0

        # Larger models have rope scaling
        scaled_providers = [
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
            Gemma3ModelProvider27B(),
        ]

        for provider in scaled_providers:
            assert provider.rope_scaling_factor == 8.0

    def test_window_size_configuration(self):
        """Test window size configuration across different model sizes."""
        # 1B has smaller window
        provider_1b = Gemma3ModelProvider1B()
        assert provider_1b.window_size == 512

        # Larger models have bigger window
        larger_providers = [
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
            Gemma3ModelProvider27B(),
        ]

        for provider in larger_providers:
            assert provider.window_size == 1024

    def test_kv_channels_configuration(self):
        """Test kv_channels configuration across different model sizes."""
        # Most models use 256
        standard_providers = [
            Gemma3ModelProvider1B(),
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
        ]

        for provider in standard_providers:
            assert provider.kv_channels == 256

        # 27B uses different kv_channels
        provider_27b = Gemma3ModelProvider27B()
        assert provider_27b.kv_channels == 128

    def test_vocab_size_configuration(self):
        """Test vocabulary size configuration across different model sizes."""
        # 1B has different vocab size
        provider_1b = Gemma3ModelProvider1B()
        assert provider_1b.vocab_size == 262_144

        # Larger models have same vocab size
        larger_providers = [
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
            Gemma3ModelProvider27B(),
        ]

        for provider in larger_providers:
            assert provider.vocab_size == 262_208

    def test_all_providers_inherit_correctly(self):
        """Test that all provider variants inherit from base Gemma3ModelProvider."""
        providers = [
            Gemma3ModelProvider1B(),
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
            Gemma3ModelProvider27B(),
        ]

        for provider in providers:
            assert isinstance(provider, Gemma3ModelProvider)

    def test_softmax_scale_configuration(self):
        """Test features unique to the 27B model."""
        provider_27b = Gemma3ModelProvider27B()

        # 27B has unique softmax_scale
        assert hasattr(provider_27b, "softmax_scale")
        expected_scale = 1.0 / math.sqrt(168)
        assert abs(provider_27b.softmax_scale - expected_scale) < 1e-10

        # Other models have this attribute set to 1.0 / math.sqrt(256)
        other_providers = [
            Gemma3ModelProvider1B(),
            Gemma3ModelProvider4B(),
            Gemma3ModelProvider12B(),
        ]

        for provider in other_providers:
            assert hasattr(provider, "softmax_scale")
            expected_scale = 1.0 / math.sqrt(256)
            assert abs(provider.softmax_scale - expected_scale) < 1e-10
