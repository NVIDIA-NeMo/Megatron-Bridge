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

from megatron.core.activations import fast_gelu
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gemma.gemma_provider import (
    CodeGemmaModelProvider2B,
    CodeGemmaModelProvider7B,
    GemmaModelProvider,
    GemmaModelProvider2B,
    GemmaModelProvider7B,
)


class TestGemmaModelProvider:
    """Test cases for base GemmaModelProvider class."""

    def test_gemma_model_provider_initialization(self):
        """Test GemmaModelProvider can be initialized with default values."""
        provider = GemmaModelProvider(
            num_layers=18,
            hidden_size=2048,
            num_attention_heads=8,
        )

        # Check required transformer config fields
        assert provider.num_layers == 18
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 8

        # Check Gemma-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.seq_length == 8192
        assert provider.kv_channels == 256
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is True
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.attention_backend == AttnBackend.flash

    @patch("megatron.bridge.models.gemma.gemma_provider.parallel_state")
    @patch("megatron.bridge.models.gemma.modules.extend_instance")
    def test_gemma_model_provider_provide_with_embedding_scaling(self, mock_extend_instance, mock_parallel_state):
        """Test that provide method applies embedding scaling when appropriate."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = GemmaModelProvider(
            num_layers=18,
            hidden_size=2048,
            num_attention_heads=8,
        )

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            # Test case: First pipeline stage
            mock_parallel_state.is_pipeline_first_stage.return_value = True

            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that is_pipeline_first_stage was called with correct parameters
            mock_parallel_state.is_pipeline_first_stage.assert_called_once_with(
                ignore_virtual=False,
                vp_stage=0,
            )

            # Verify that extend_instance was called with embedding scaling mixin
            mock_extend_instance.assert_called_once()
            args = mock_extend_instance.call_args[0]
            assert args[0] == mock_model.embedding  # First arg should be the embedding
            # Second arg should be the EmbeddingScalingMixin class

    @patch("megatron.bridge.models.gemma.gemma_provider.parallel_state")
    @patch("megatron.bridge.models.gemma.modules.extend_instance")
    def test_gemma_model_provider_provide_no_embedding_scaling(self, mock_extend_instance, mock_parallel_state):
        """Test that provide method doesn't apply embedding scaling when not first stage."""
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = GemmaModelProvider(
            num_layers=18,
            hidden_size=2048,
            num_attention_heads=8,
        )

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            # Test case: Not first pipeline stage
            mock_parallel_state.is_pipeline_first_stage.return_value = False

            result = provider.provide(vp_stage=1)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that is_pipeline_first_stage was called with correct parameters
            mock_parallel_state.is_pipeline_first_stage.assert_called_once_with(
                ignore_virtual=False,
                vp_stage=1,
            )

            # Verify that extend_instance was NOT called
            mock_extend_instance.assert_not_called()

    @patch("megatron.bridge.models.gemma.gemma_provider.parallel_state")
    @patch("megatron.bridge.models.gemma.modules.extend_instance")
    def test_gemma_model_provider_provide_virtual_pipeline_none(self, mock_extend_instance, mock_parallel_state):
        """Test provide method when vp_stage is None (no virtual pipeline)."""
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = GemmaModelProvider(
            num_layers=18,
            hidden_size=2048,
            num_attention_heads=8,
        )

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            # Test case: No virtual pipeline (vp_stage=None)
            mock_parallel_state.is_pipeline_first_stage.return_value = True

            _ = provider.provide(vp_stage=None)

            # Verify that is_pipeline_first_stage was called with vp_stage=None
            mock_parallel_state.is_pipeline_first_stage.assert_called_once_with(
                ignore_virtual=False,
                vp_stage=None,
            )

            # Verify that extend_instance was called since it's first stage
            mock_extend_instance.assert_called_once()


class TestGemmaModelProvider2B:
    """Test cases for GemmaModelProvider2B class."""

    def test_gemma_2b_configuration(self):
        """Test that GemmaModelProvider2B has correct configuration values."""
        provider = GemmaModelProvider2B()

        # Test 2B specific values
        assert provider.num_layers == 18
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 1
        assert provider.ffn_hidden_size == 16384

        # Test inherited Gemma defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.attention_backend == AttnBackend.flash

    def test_gemma_2b_inheritance(self):
        """Test that GemmaModelProvider2B properly inherits from GemmaModelProvider."""
        provider = GemmaModelProvider2B()
        assert isinstance(provider, GemmaModelProvider)


class TestGemmaModelProvider7B:
    """Test cases for GemmaModelProvider7B class."""

    def test_gemma_7b_configuration(self):
        """Test that GemmaModelProvider7B has correct configuration values."""
        provider = GemmaModelProvider7B()

        # Test 7B specific values
        assert provider.num_layers == 28
        assert provider.hidden_size == 3072
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 16
        assert provider.ffn_hidden_size == 24576

        # Test inherited Gemma defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.attention_backend == AttnBackend.flash

    def test_gemma_7b_inheritance(self):
        """Test that GemmaModelProvider7B properly inherits from GemmaModelProvider."""
        provider = GemmaModelProvider7B()
        assert isinstance(provider, GemmaModelProvider)


class TestCodeGemmaModelProviders:
    """Test cases for Code Gemma model provider classes."""

    def test_code_gemma_2b_configuration(self):
        """Test that CodeGemmaModelProvider2B has correct 2B configuration values."""
        provider = CodeGemmaModelProvider2B()

        # Test 2B specific values
        assert provider.num_layers == 18
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 1
        assert provider.ffn_hidden_size == 16384

        # Test inherited Gemma defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.attention_backend == AttnBackend.flash

    def test_code_gemma_7b_configuration(self):
        """Test that CodeGemmaModelProvider7B has correct 7B configuration values."""
        provider = CodeGemmaModelProvider7B()

        # Test 7B specific values
        assert provider.num_layers == 28
        assert provider.hidden_size == 3072
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 16
        assert provider.ffn_hidden_size == 24576

        # Test inherited Gemma defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.attention_backend == AttnBackend.flash

    def test_code_gemma_inheritance_chain(self):
        """Test the inheritance chain for Code Gemma providers."""
        provider_2b = CodeGemmaModelProvider2B()
        provider_7b = CodeGemmaModelProvider7B()

        # Check inheritance chain - both should inherit directly from GemmaModelProvider
        assert isinstance(provider_2b, GemmaModelProvider)
        assert isinstance(provider_7b, GemmaModelProvider)


class TestGemmaModelProviderIntegration:
    """Integration tests for Gemma model providers."""

    def test_all_providers_have_provide_method(self):
        """Test that all provider classes have the provide method."""
        providers = [
            GemmaModelProvider2B(),
            GemmaModelProvider7B(),
            CodeGemmaModelProvider2B(),
            CodeGemmaModelProvider7B(),
        ]

        for provider in providers:
            assert hasattr(provider, "provide")
            assert callable(getattr(provider, "provide"))
