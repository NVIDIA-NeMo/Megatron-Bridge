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

import torch.nn.functional as F

from megatron.bridge.models.starcoder.starcoder2_provider import (
    Starcoder2ModelProvider,
    Starcoder2ModelProvider3B,
    Starcoder2ModelProvider7B,
    Starcoder2ModelProvider15B,
)


class TestStarcoder2ModelProvider:
    """Test cases for Starcoder2ModelProvider class."""

    def test_starcoder2_model_provider_defaults(self):
        """Test Starcoder2ModelProvider has correct default values."""
        provider = Starcoder2ModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
        )

        # Check required transformer config fields
        assert provider.num_layers == 12
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 12

        # Check Starcoder2-specific defaults + transformer config post init
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func == F.gelu
        assert provider.add_bias_linear is True
        assert provider.seq_length == 16384
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_percent == 1.0
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.init_method_std == 0.01
        assert provider.share_embeddings_and_output_weights is False
        assert provider.kv_channels == 64
        assert provider.num_query_groups == 12
        assert provider.window_size is None
        assert provider.attention_softmax_in_fp32 is True
        assert provider.bias_activation_fusion is True
        assert provider.bias_dropout_fusion is True
        assert provider.layernorm_epsilon == 1e-5


class TestStarcoder2ModelProvider3B:
    """Test cases for Starcoder2ModelProvider3B class."""

    def test_starcoder2_3b_defaults(self):
        """Test Starcoder2ModelProvider3B has correct default values for 3B model."""
        provider = Starcoder2ModelProvider3B()

        # Check 3B-specific configuration
        assert provider.num_layers == 30
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 12288
        assert provider.num_query_groups == 2
        assert provider.num_attention_heads == 24
        assert provider.init_method_std == 0.018042
        assert provider.rotary_base == 999999.4420358813

        # Check inherited Starcoder2 defaults
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func == F.gelu
        assert provider.add_bias_linear is True
        assert provider.seq_length == 16384
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_percent == 1.0
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.window_size is None
        assert provider.attention_softmax_in_fp32 is True
        assert provider.bias_activation_fusion is True
        assert provider.bias_dropout_fusion is True
        assert provider.layernorm_epsilon == 1e-5


class TestStarcoder2ModelProvider7B:
    """Test cases for Starcoder2ModelProvider7B class."""

    def test_starcoder2_7b_defaults(self):
        """Test Starcoder2ModelProvider7B has correct default values for 7B model."""
        provider = Starcoder2ModelProvider7B()

        # Check 7B-specific configuration
        assert provider.num_layers == 32
        assert provider.hidden_size == 4608
        assert provider.ffn_hidden_size == 18432
        assert provider.num_query_groups == 4
        assert provider.num_attention_heads == 36
        assert provider.init_method_std == 0.018042
        assert provider.rotary_base == 1_000_000

        # Check inherited Starcoder2 defaults
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func == F.gelu
        assert provider.add_bias_linear is True
        assert provider.seq_length == 16384
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_percent == 1.0
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.kv_channels is 128
        assert provider.window_size is None
        assert provider.attention_softmax_in_fp32 is True
        assert provider.bias_activation_fusion is True
        assert provider.bias_dropout_fusion is True
        assert provider.layernorm_epsilon == 1e-5


class TestStarcoder2ModelProvider15B:
    """Test cases for Starcoder2ModelProvider15B class."""

    def test_starcoder2_15b_defaults(self):
        """Test Starcoder2ModelProvider15B has correct default values for 15B model."""
        provider = Starcoder2ModelProvider15B()

        # Check 15B-specific configuration
        assert provider.num_layers == 40
        assert provider.hidden_size == 6144
        assert provider.ffn_hidden_size == 24576
        assert provider.num_query_groups == 4
        assert provider.num_attention_heads == 48
        assert provider.init_method_std == 0.01275
        assert provider.rotary_base == 100_000

        # Check inherited Starcoder2 defaults
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func == F.gelu
        assert provider.add_bias_linear is True
        assert provider.seq_length == 16384
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_percent == 1.0
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.kv_channels == 128
        assert provider.window_size is None
        assert provider.attention_softmax_in_fp32 is True
        assert provider.bias_activation_fusion is True
        assert provider.bias_dropout_fusion is True
        assert provider.layernorm_epsilon == 1e-5


class TestStarcoder2ProviderInheritance:
    """Test inheritance relationships between Starcoder2 providers."""

    def test_starcoder2_models_inherit_from_base(self):
        """Test Starcoder2 providers inherit from Starcoder2ModelProvider."""
        assert issubclass(Starcoder2ModelProvider3B, Starcoder2ModelProvider)
        assert issubclass(Starcoder2ModelProvider7B, Starcoder2ModelProvider)
        assert issubclass(Starcoder2ModelProvider15B, Starcoder2ModelProvider)

    def test_starcoder2_models_inherit_from_gpt(self):
        """Test Starcoder2 providers inherit from GPTModelProvider."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        assert issubclass(Starcoder2ModelProvider, GPTModelProvider)
        assert issubclass(Starcoder2ModelProvider3B, GPTModelProvider)
        assert issubclass(Starcoder2ModelProvider7B, GPTModelProvider)
        assert issubclass(Starcoder2ModelProvider15B, GPTModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with all Starcoder2 providers
        providers = [
            Starcoder2ModelProvider3B(),
            Starcoder2ModelProvider7B(),
            Starcoder2ModelProvider15B(),
        ]

        for provider in providers:
            # The provide method should be inherited from GPTModelProvider
            assert hasattr(provider, "provide")
            assert callable(provider.provide)
