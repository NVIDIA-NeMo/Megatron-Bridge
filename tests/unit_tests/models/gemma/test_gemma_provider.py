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
    GemmaModelProvider,
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

    @patch("megatron.bridge.models.gemma.gemma_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma_provider.is_vp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.modules.extend_instance")
    def test_gemma_model_provider_provide_with_embedding_scaling(self, mock_extend_instance, *_):
        """Test that provide method applies embedding scaling when appropriate."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = GemmaModelProvider(
            num_layers=18,
            hidden_size=2048,
            num_attention_heads=8,
        )

        # Attach minimal pg_collection required by provider
        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was called with embedding scaling mixin
            mock_extend_instance.assert_called_once()
            args = mock_extend_instance.call_args[0]
            assert args[0] == mock_model.embedding  # First arg should be the embedding
            # Second arg should be the EmbeddingScalingMixin class

    @patch("megatron.bridge.models.gemma.gemma_provider.is_pp_first_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma_provider.is_vp_first_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.modules.extend_instance")
    def test_gemma_model_provider_provide_no_embedding_scaling(self, mock_extend_instance, *_):
        """Test that provide method doesn't apply embedding scaling when not first stage."""
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = GemmaModelProvider(
            num_layers=18,
            hidden_size=2048,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            result = provider.provide(vp_stage=1)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was NOT called
            mock_extend_instance.assert_not_called()

    @patch("megatron.bridge.models.gemma.gemma_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma_provider.is_vp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.modules.extend_instance")
    def test_gemma_model_provider_provide_virtual_pipeline_none(self, mock_extend_instance, *_):
        """Test provide method when vp_stage is None (no virtual pipeline)."""
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = GemmaModelProvider(
            num_layers=18,
            hidden_size=2048,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            _ = provider.provide(vp_stage=None)

            # Verify that extend_instance was called since it's first stage
            mock_extend_instance.assert_called_once()


class TestGemmaModelProviderIntegration:
    """Integration tests for Gemma model providers."""

    def test_provider_accepts_explicit_architecture_values(self):
        """Test that architecture values can be supplied without size subclasses."""
        providers = [
            GemmaModelProvider(
                num_layers=18,
                hidden_size=2048,
                num_attention_heads=8,
                num_query_groups=1,
                ffn_hidden_size=16384,
            ),
            GemmaModelProvider(
                num_layers=28,
                hidden_size=3072,
                num_attention_heads=16,
                num_query_groups=16,
                ffn_hidden_size=24576,
            ),
        ]

        for provider in providers:
            assert isinstance(provider, GemmaModelProvider)
            assert hasattr(provider, "provide")
            assert callable(getattr(provider, "provide"))
            assert provider.normalization == "RMSNorm"
            assert provider.activation_func == fast_gelu
            assert provider.gated_linear_unit is True
            assert provider.attention_backend == AttnBackend.flash
