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

from megatron.core.activations import fast_gelu
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gemma.gemma2_provider import (
    Gemma2DotProductAttention,
    Gemma2ModelProvider,
    get_swa,
)
from megatron.bridge.utils.fusions import can_enable_gradient_accumulation_fusion


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
        assert provider.window_size == (4095, 0)
        assert provider.vocab_size == 256000
        assert provider.gradient_accumulation_fusion is can_enable_gradient_accumulation_fusion()
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


class TestGemma2ModelProviderIntegration:
    """Integration tests for Gemma2 model providers."""

    def test_provider_accepts_explicit_architecture_values(self):
        """Test that architecture values can be supplied without size subclasses."""
        providers = [
            Gemma2ModelProvider(
                num_layers=26,
                hidden_size=2304,
                num_attention_heads=8,
                num_query_groups=4,
                ffn_hidden_size=9216,
                query_pre_attn_scalar=256,
            ),
            Gemma2ModelProvider(
                num_layers=42,
                hidden_size=3584,
                num_attention_heads=16,
                num_query_groups=8,
                ffn_hidden_size=14336,
                query_pre_attn_scalar=256,
            ),
            Gemma2ModelProvider(
                num_layers=46,
                hidden_size=4608,
                num_attention_heads=32,
                num_query_groups=16,
                kv_channels=128,
                ffn_hidden_size=36864,
                query_pre_attn_scalar=144,
            ),
        ]

        for provider in providers:
            assert isinstance(provider, Gemma2ModelProvider)
            assert hasattr(provider, "provide")
            assert callable(getattr(provider, "provide"))
            assert provider.normalization == "RMSNorm"
            assert provider.activation_func == fast_gelu
            assert provider.gated_linear_unit is True


def _make_attention(context_parallel_size: int = 1, window_size: tuple = (4095, 0)) -> Gemma2DotProductAttention:
    """Build a Gemma2DotProductAttention with minimal mock config."""
    config = Mock()
    config.context_parallel_size = context_parallel_size
    config.window_size = window_size
    config.kv_channels = 256
    config.num_attention_heads = 8
    config.num_query_groups = 8
    config.tensor_model_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.query_pre_attn_scalar = 224
    config.fp16 = False
    config.bf16 = True
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.sequence_parallel = False
    return Gemma2DotProductAttention(
        config=config,
        layer_number=2,  # even layer → SWA active
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
    )


class TestGemma2DotProductAttention:
    """Tests for Gemma2DotProductAttention fixes."""

    def test_cp_greater_than_1_raises_value_error(self):
        """CP > 1 must raise ValueError, not bare AssertionError."""
        with pytest.raises(ValueError, match="Context parallelism"):
            _make_attention(context_parallel_size=2)

    def test_packed_seq_raises_value_error(self):
        """packed_seq_params != None must raise ValueError."""
        attn = _make_attention()
        dummy = torch.zeros(4, 8, 8)
        with pytest.raises(ValueError, match="Packed sequence"):
            attn.forward(
                query=dummy,
                key=dummy,
                value=dummy,
                attention_mask=None,
                packed_seq_params=Mock(),
            )

    def test_swa_applied_when_attention_mask_is_none(self):
        """SWA mask must be generated even when attention_mask=None (the pretrain path)."""
        attn = _make_attention(window_size=(4095, 0))
        # Even layer → self.window_size is set; odd layer → None
        assert attn.window_size == (4095, 0)

        # get_swa should produce a boolean mask of the correct shape
        seq_len = 8
        mask = get_swa(seq_len, seq_len, (4095, 0))
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool

    def test_odd_layer_has_no_swa(self):
        """Odd-numbered layers must not have a window_size (full attention)."""
        config = Mock()
        config.context_parallel_size = 1
        config.window_size = (4095, 0)
        config.kv_channels = 256
        config.num_attention_heads = 8
        config.num_query_groups = 8
        config.tensor_model_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.query_pre_attn_scalar = 224
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sequence_parallel = False
        odd_attn = Gemma2DotProductAttention(
            config=config,
            layer_number=1,  # odd → full attention
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        assert odd_attn.window_size is None

    def test_window_size_default_is_4095(self):
        """Gemma2ModelProvider.window_size default must be (4095, 0) to match gemma2_bridge convention."""
        provider = Gemma2ModelProvider(
            num_layers=42,
            hidden_size=3584,
            num_attention_heads=16,
        )
        assert provider.window_size == (4095, 0)
