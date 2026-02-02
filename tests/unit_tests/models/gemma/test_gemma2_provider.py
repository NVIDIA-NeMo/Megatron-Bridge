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

from megatron.bridge.models.gemma.gemma2_provider import (
    Gemma2ModelProvider,
    Gemma2ModelProvider2B,
    Gemma2ModelProvider9B,
    Gemma2ModelProvider27B,
)


class TestGemma2ModelProvider:
    """Test cases for base Gemma2ModelProvider class."""

    def test_gemma2_model_provider_initialization(self):
        """Test Gemma2ModelProvider can be initialized with default values."""
        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        # Check required transformer config fields
        assert provider.num_layers == 26
        assert provider.hidden_size == 2304
        assert provider.num_attention_heads == 8

        # Check Gemma2-specific defaults
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

        # Check Gemma2-specific parameters
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == 10000
        assert provider.window_size == (4096, 0)
        assert provider.vocab_size == 256000
        assert provider.gradient_accumulation_fusion is False
        assert provider.query_pre_attn_scalar == 224
        assert provider.attn_logit_softcapping == 50.0
        assert provider.final_logit_softcapping == 30.0

    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_last_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_last_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.extend_instance")
    def test_gemma2_provider_provide_with_embedding_scaling(self, mock_extend_instance, *_):
        """Test that provide method applies embedding scaling when appropriate."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was called for embedding scaling
            assert mock_extend_instance.call_count == 1
            args = mock_extend_instance.call_args_list[0][0]
            assert args[0] == mock_model.embedding

    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_first_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_first_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.extend_instance")
    def test_gemma2_provider_provide_with_output_layer_scaling(self, mock_extend_instance, *_):
        """Test that provide method applies output layer modifications when appropriate."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()
        mock_model.output_layer = Mock()

        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            # Use vp_stage=0 to satisfy vp_size None assertion in helpers
            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was called for output layer modifications
            assert mock_extend_instance.call_count == 1
            args = mock_extend_instance.call_args_list[0][0]
            assert args[0] == mock_model.output_layer

    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.extend_instance")
    def test_gemma2_provider_provide_both_stages(self, mock_extend_instance, *_):
        """Test provide method when model is both first and last stage."""
        mock_model = Mock()
        mock_model.embedding = Mock()
        mock_model.output_layer = Mock()

        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was called twice (embedding + output layer)
            assert mock_extend_instance.call_count == 2


class TestGemma2ModelProvider2B:
    """Test cases for Gemma2ModelProvider2B class."""

    def test_gemma2_2b_configuration(self):
        """Test that Gemma2ModelProvider2B has correct configuration values."""
        provider = Gemma2ModelProvider2B()

        # Test 2B specific values
        assert provider.num_layers == 26
        assert provider.hidden_size == 2304
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 4
        assert provider.ffn_hidden_size == 9216
        assert provider.query_pre_attn_scalar == 256

        # Test inherited Gemma2 defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.window_size == (4096, 0)
        assert provider.attn_logit_softcapping == 50.0
        assert provider.final_logit_softcapping == 30.0

    def test_gemma2_2b_inheritance(self):
        """Test that Gemma2ModelProvider2B properly inherits from Gemma2ModelProvider."""
        provider = Gemma2ModelProvider2B()
        assert isinstance(provider, Gemma2ModelProvider)


class TestGemma2ModelProvider9B:
    """Test cases for Gemma2ModelProvider9B class."""

    def test_gemma2_9b_configuration(self):
        """Test that Gemma2ModelProvider9B has correct configuration values."""
        provider = Gemma2ModelProvider9B()

        # Test 9B specific values
        assert provider.num_layers == 42
        assert provider.hidden_size == 3584
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 14336
        assert provider.query_pre_attn_scalar == 256

        # Test inherited Gemma2 defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True

    def test_gemma2_9b_inheritance(self):
        """Test that Gemma2ModelProvider9B properly inherits from Gemma2ModelProvider."""
        provider = Gemma2ModelProvider9B()
        assert isinstance(provider, Gemma2ModelProvider)


class TestGemma2ModelProvider27B:
    """Test cases for Gemma2ModelProvider27B class."""

    def test_gemma2_27b_configuration(self):
        """Test that Gemma2ModelProvider27B has correct configuration values."""
        provider = Gemma2ModelProvider27B()

        # Test 27B specific values
        assert provider.num_layers == 46
        assert provider.hidden_size == 4608
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 16
        assert provider.ffn_hidden_size == 36864
        assert provider.query_pre_attn_scalar == 144

    def test_gemma2_27b_inheritance(self):
        """Test that Gemma2ModelProvider27B properly inherits from Gemma2ModelProvider."""
        provider = Gemma2ModelProvider27B()
        assert isinstance(provider, Gemma2ModelProvider)


class TestGemma2ModelProviderIntegration:
    """Integration tests for Gemma2 model providers."""

    def test_all_providers_have_provide_method(self):
        """Test that all provider classes have the provide method."""
        providers = [
            Gemma2ModelProvider2B(),
            Gemma2ModelProvider9B(),
            Gemma2ModelProvider27B(),
        ]

        for provider in providers:
            assert hasattr(provider, "provide")
            assert callable(getattr(provider, "provide"))
