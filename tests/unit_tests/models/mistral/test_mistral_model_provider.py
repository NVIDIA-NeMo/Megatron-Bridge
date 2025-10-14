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

from megatron.bridge.models.mistral import (
    MistralModelProvider,
    MistralSmall3ModelProvider24B,
)


class TestMistralModelProvider:
    """Test cases for base MistralModelProvider class."""

    def test_mistral_model_provider_initialization(self):
        """Test MistralModelProvider can be initialized with default values."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

        # Check Mistral-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False  # Different from Mistral
        assert provider.qk_layernorm is False  # Mistral specific feature
        assert provider.num_query_groups == 8  # Default for Mistral
        assert provider.seq_length == 32768  # Extended context for Mistral
        assert provider.init_method_std == 0.02
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.vocab_size == 32768
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5
        assert provider.rotary_base == 1000000.0
        assert provider.position_embedding_type == "rope"

    def test_mistral_model_provider_with_custom_rope(self):
        """Test MistralModelProvider with custom RoPE configuration."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
            rotary_percent=0.5,
        )

        assert provider.rotary_base == 500000.0
        assert provider.rotary_percent == 0.5

    def test_mistral_model_provider_ffn_hidden_size(self):
        """Test MistralModelProvider FFN hidden size calculation."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
        )

        assert provider.ffn_hidden_size == 11008

    def test_mistral_model_provider_group_query_attention(self):
        """Test MistralModelProvider with group query attention."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=16,
        )

        assert provider.num_query_groups == 16

    def test_mistral_model_provider_custom_vocab_size(self):
        """Test MistralModelProvider with custom vocabulary size."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
        )

        assert provider.vocab_size == 32000

    def test_mistral_model_provider_custom_sequence_length(self):
        """Test MistralModelProvider with custom sequence length."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=65536,
        )

        assert provider.seq_length == 65536

    def test_mistral_model_provider_qk_layernorm_feature(self):
        """Test MistralModelProvider QK layernorm feature."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            qk_layernorm=True,  # Override default
        )

        assert provider.qk_layernorm is True


class TestMistralSmall3ModelProvider24B:
    """Test cases for MistralSmall3ModelProvider24B class."""

    def test_mistral_small3_24b_default_configuration(self):
        """Test MistralSmall3ModelProvider24B model has correct default configuration."""
        provider = MistralSmall3ModelProvider24B()

        # Check Mistral Small3 24B specific configuration
        assert provider.num_layers == 40
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 32
        assert provider.ffn_hidden_size == 32768
        assert provider.share_embeddings_and_output_weights is False

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.vocab_size == 131072
        assert provider.seq_length == 32768
        assert provider.qk_layernorm is False
        assert provider.add_qkv_bias is False

    def test_mistral_small3_24b_override_configuration(self):
        """Test MistralSmall3ModelProvider24B model with overridden configuration."""
        provider = MistralSmall3ModelProvider24B(
            seq_length=32768,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 32768
        assert provider.hidden_dropout == 0.1

        # Check defaults remain
        assert provider.num_layers == 40
        assert provider.hidden_size == 5120


class TestMistralProviderInheritance:
    """Test inheritance relationships between Mistral providers."""

    def test_mistral_models_inherit_from_base(self):
        """Test Mistral providers inherit from MistralModelProvider."""
        assert issubclass(MistralModelProvider, MistralModelProvider)
        assert issubclass(MistralSmall3ModelProvider24B, MistralModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Mistral
        provider = MistralModelProvider()

        # The provide method should be inherited from GPTModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)


class TestMistralProviderEdgeCases:
    """Test edge cases and error conditions."""

    def test_valid_num_query_groups(self):
        """Test that valid num_query_groups configuration works."""
        # num_attention_heads must be divisible by num_query_groups
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,  # 32 divisible by 8
        )
        assert provider.num_query_groups == 8

    def test_vocabulary_size_divisibility(self):
        """Test vocabulary size divisibility configuration."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32768,
            make_vocab_size_divisible_by=128,
        )

        # The actual vocab size should be adjusted if needed
        assert provider.make_vocab_size_divisible_by == 128

    def test_seq_length_override(self):
        """Test sequence length configuration."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=32768,  # Very long context
        )

        assert provider.seq_length == 32768

    def test_rotary_base_configuration(self):
        """Test rotary base configuration."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=1000000.0,
        )

        assert provider.rotary_base == 1000000.0

    def test_layernorm_epsilon_override(self):
        """Test layernorm epsilon configuration."""
        provider = MistralModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            layernorm_epsilon=1e-5,
        )

        assert provider.layernorm_epsilon == 1e-5


class TestMistralProviderQueryGroupsConsistency:
    """Test cases to verify query groups consistency across Mistral models."""

    def test_mistral_model_provider_num_query_groups(self):
        """Test that MistralModelProvider has correct num_query_groups."""
        provider = MistralModelProvider()
        # Uses default from base class
        assert provider.num_query_groups == 8

    def test_mistral_small3_24b_num_query_groups(self):
        """Test that MistralSmall3ModelProvider24B has correct num_query_groups."""
        provider = MistralSmall3ModelProvider24B()
        # Uses default from base class
        assert provider.num_query_groups == 8
