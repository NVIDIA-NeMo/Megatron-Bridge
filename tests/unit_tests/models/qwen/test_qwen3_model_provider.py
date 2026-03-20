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

from megatron.bridge.models.qwen import (
    Qwen3ModelProvider,
    Qwen3ModelProvider1P7B,
    Qwen3ModelProvider4B,
    Qwen3ModelProvider8B,
    Qwen3ModelProvider14B,
    Qwen3ModelProvider32B,
    Qwen3ModelProvider600M,
)


class TestQwen3ModelProvider:
    """Test cases for base Qwen3ModelProvider class."""

    def test_qwen3_model_provider_initialization(self):
        """Test Qwen3ModelProvider can be initialized with default values."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

        # Check Qwen3-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False  # Different from Qwen2
        assert provider.qk_layernorm is True  # Qwen3 specific feature
        assert provider.kv_channels == 128  # Qwen3 specific
        assert provider.num_query_groups == 8  # Default for Qwen3
        assert provider.seq_length == 40960  # Extended context for Qwen3
        assert provider.init_method_std == 0.02
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.vocab_size == 151936
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == 1000000.0
        assert provider.position_embedding_type == "rope"

    def test_qwen3_model_provider_with_custom_rope(self):
        """Test Qwen3ModelProvider with custom RoPE configuration."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
            rotary_percent=0.5,
        )

        assert provider.rotary_base == 500000.0
        assert provider.rotary_percent == 0.5

    def test_qwen3_model_provider_ffn_hidden_size(self):
        """Test Qwen3ModelProvider FFN hidden size calculation."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
        )

        assert provider.ffn_hidden_size == 11008

    def test_qwen3_model_provider_group_query_attention(self):
        """Test Qwen3ModelProvider with group query attention."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=16,
        )

        assert provider.num_query_groups == 16

    def test_qwen3_model_provider_custom_vocab_size(self):
        """Test Qwen3ModelProvider with custom vocabulary size."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
        )

        assert provider.vocab_size == 32000

    def test_qwen3_model_provider_custom_sequence_length(self):
        """Test Qwen3ModelProvider with custom sequence length."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=65536,
        )

        assert provider.seq_length == 65536

    def test_qwen3_model_provider_qk_layernorm_feature(self):
        """Test Qwen3ModelProvider QK layernorm feature."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            qk_layernorm=False,  # Override default
        )

        assert provider.qk_layernorm is False

    def test_qwen3_model_provider_kv_channels(self):
        """Test Qwen3ModelProvider KV channels configuration."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            kv_channels=256,
        )

        assert provider.kv_channels == 256


class TestQwen3ModelProvider600M:
    """Test cases for Qwen3ModelProvider600M class."""

    def test_qwen3_600m_default_configuration(self):
        """Test Qwen3 600M model has correct default configuration."""
        provider = Qwen3ModelProvider600M()

        # Check Qwen3 600M specific configuration
        assert provider.num_layers == 28
        assert provider.hidden_size == 1024
        assert provider.num_attention_heads == 16
        assert provider.ffn_hidden_size == 3072
        assert provider.share_embeddings_and_output_weights is True

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.vocab_size == 151936
        assert provider.seq_length == 40960
        assert provider.qk_layernorm is True
        assert provider.add_qkv_bias is False

    def test_qwen3_600m_override_configuration(self):
        """Test Qwen3 600M model with overridden configuration."""
        provider = Qwen3ModelProvider600M(
            seq_length=32768,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 32768
        assert provider.hidden_dropout == 0.1

        # Check defaults remain
        assert provider.num_layers == 28
        assert provider.hidden_size == 1024


class TestQwen3ModelProvider1P7B:
    """Test cases for Qwen3ModelProvider1P7B class."""

    def test_qwen3_1p7b_default_configuration(self):
        """Test Qwen3 1.7B model has correct default configuration."""
        provider = Qwen3ModelProvider1P7B()

        # Check Qwen3 1.7B specific configuration
        assert provider.num_layers == 28
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 16
        assert provider.ffn_hidden_size == 6144
        assert provider.share_embeddings_and_output_weights is True

        # Check inherited defaults
        assert provider.vocab_size == 151936
        assert provider.seq_length == 40960
        assert provider.qk_layernorm is True


class TestQwen3ModelProvider4B:
    """Test cases for Qwen3ModelProvider4B class."""

    def test_qwen3_4b_default_configuration(self):
        """Test Qwen3 4B model has correct default configuration."""
        provider = Qwen3ModelProvider4B()

        # Check Qwen3 4B specific configuration
        assert provider.num_layers == 36
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 32
        assert provider.ffn_hidden_size == 9728
        assert provider.share_embeddings_and_output_weights is True

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.vocab_size == 151936


class TestQwen3ModelProvider8B:
    """Test cases for Qwen3ModelProvider8B class."""

    def test_qwen3_8b_default_configuration(self):
        """Test Qwen3 8B model has correct default configuration."""
        provider = Qwen3ModelProvider8B()

        # Check Qwen3 8B specific configuration
        assert provider.num_layers == 36
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.ffn_hidden_size == 12288

        # Check inherited defaults
        assert provider.seq_length == 40960
        assert provider.normalization == "RMSNorm"
        assert provider.qk_layernorm is True
        # Note: share_embeddings_and_output_weights not explicitly set, so uses default False


class TestQwen3ModelProvider14B:
    """Test cases for Qwen3ModelProvider14B class."""

    def test_qwen3_14b_default_configuration(self):
        """Test Qwen3 14B model has correct default configuration."""
        provider = Qwen3ModelProvider14B()

        # Check Qwen3 14B specific configuration
        assert provider.num_layers == 40
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 40
        assert provider.ffn_hidden_size == 17408

        # Check inherited defaults
        assert provider.seq_length == 40960
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.qk_layernorm is True


class TestQwen3ModelProvider32B:
    """Test cases for Qwen3ModelProvider32B class."""

    def test_qwen3_32b_default_configuration(self):
        """Test Qwen3 32B model has correct default configuration."""
        provider = Qwen3ModelProvider32B()

        # Check Qwen3 32B specific configuration
        assert provider.num_layers == 64
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 64
        assert provider.ffn_hidden_size == 25600

        # Check inherited defaults
        assert provider.seq_length == 40960
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.qk_layernorm is True


class TestQwen3ProviderInheritance:
    """Test inheritance relationships between Qwen3 providers."""

    def test_qwen3_models_inherit_from_base(self):
        """Test Qwen3 providers inherit from Qwen3ModelProvider."""
        assert issubclass(Qwen3ModelProvider600M, Qwen3ModelProvider)
        assert issubclass(Qwen3ModelProvider1P7B, Qwen3ModelProvider)
        assert issubclass(Qwen3ModelProvider4B, Qwen3ModelProvider)
        assert issubclass(Qwen3ModelProvider8B, Qwen3ModelProvider)
        assert issubclass(Qwen3ModelProvider14B, Qwen3ModelProvider)
        assert issubclass(Qwen3ModelProvider32B, Qwen3ModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Qwen3 8B
        provider = Qwen3ModelProvider8B()

        # The provide method should be inherited from GPTModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)


class TestQwen3ProviderEdgeCases:
    """Test edge cases and error conditions."""

    def test_valid_num_query_groups(self):
        """Test that valid num_query_groups configuration works."""
        # num_attention_heads must be divisible by num_query_groups
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=16,  # 32 divisible by 16
        )
        assert provider.num_query_groups == 16

    def test_vocabulary_size_divisibility(self):
        """Test vocabulary size divisibility configuration."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=151936,
            make_vocab_size_divisible_by=128,
        )

        # The actual vocab size should be adjusted if needed
        assert provider.make_vocab_size_divisible_by == 128

    def test_seq_length_override(self):
        """Test sequence length configuration."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=131072,  # Very long context
        )

        assert provider.seq_length == 131072

    def test_rotary_base_configuration(self):
        """Test rotary base configuration."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
        )

        assert provider.rotary_base == 500000.0

    def test_layernorm_epsilon_override(self):
        """Test layernorm epsilon configuration."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            layernorm_epsilon=1e-5,
        )

        assert provider.layernorm_epsilon == 1e-5

    def test_qk_layernorm_disabled(self):
        """Test that QK layernorm can be disabled."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            qk_layernorm=False,
        )

        assert provider.qk_layernorm is False

    def test_kv_channels_override(self):
        """Test KV channels configuration override."""
        provider = Qwen3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            kv_channels=256,
        )

        assert provider.kv_channels == 256


class TestQwen3ProviderQueryGroupsConsistency:
    """Test cases to verify query groups consistency across Qwen3 models."""

    def test_qwen3_600m_num_query_groups(self):
        """Test that Qwen3 600M has correct num_query_groups."""
        provider = Qwen3ModelProvider600M()
        # Uses default from base class
        assert provider.num_query_groups == 8

    def test_qwen3_1p7b_num_query_groups(self):
        """Test that Qwen3 1.7B has correct num_query_groups."""
        provider = Qwen3ModelProvider1P7B()
        # Uses default from base class
        assert provider.num_query_groups == 8

    def test_qwen3_4b_num_query_groups(self):
        """Test that Qwen3 4B has correct num_query_groups."""
        provider = Qwen3ModelProvider4B()
        # Uses default from base class
        assert provider.num_query_groups == 8

    def test_qwen3_8b_num_query_groups(self):
        """Test that Qwen3 8B has correct num_query_groups."""
        provider = Qwen3ModelProvider8B()
        # Uses default from base class
        assert provider.num_query_groups == 8

    def test_qwen3_14b_num_query_groups(self):
        """Test that Qwen3 14B has correct num_query_groups."""
        provider = Qwen3ModelProvider14B()
        # Uses default from base class
        assert provider.num_query_groups == 8

    def test_qwen3_32b_num_query_groups(self):
        """Test that Qwen3 32B has correct num_query_groups."""
        provider = Qwen3ModelProvider32B()
        # Uses default from base class
        assert provider.num_query_groups == 8


class TestQwen3ProviderVocabularyConsistency:
    """Test cases to verify vocabulary size consistency across Qwen3 models."""

    def test_qwen3_models_vocab_size(self):
        """Test that all Qwen3 models have consistent vocab size."""
        providers = [
            Qwen3ModelProvider600M(),
            Qwen3ModelProvider1P7B(),
            Qwen3ModelProvider4B(),
            Qwen3ModelProvider8B(),
            Qwen3ModelProvider14B(),
            Qwen3ModelProvider32B(),
        ]

        # All Qwen3 models use the same vocabulary size
        for provider in providers:
            assert provider.vocab_size == 151936


class TestQwen3ProviderContextLengthConsistency:
    """Test cases to verify context length consistency across Qwen3 models."""

    def test_qwen3_models_context_length(self):
        """Test that all Qwen3 models have extended context length."""
        providers = [
            Qwen3ModelProvider600M(),
            Qwen3ModelProvider1P7B(),
            Qwen3ModelProvider4B(),
            Qwen3ModelProvider8B(),
            Qwen3ModelProvider14B(),
            Qwen3ModelProvider32B(),
        ]

        # All Qwen3 models use extended context length
        for provider in providers:
            assert provider.seq_length == 40960


class TestQwen3ProviderEmbeddingSharing:
    """Test cases to verify embedding sharing behavior across Qwen3 models."""

    def test_qwen3_small_models_tied_embeddings(self):
        """Test that smaller Qwen3 models have tied embeddings."""
        providers_with_tied_embeddings = [
            Qwen3ModelProvider600M(),
            Qwen3ModelProvider1P7B(),
            Qwen3ModelProvider4B(),
        ]

        for provider in providers_with_tied_embeddings:
            assert provider.share_embeddings_and_output_weights is True

    def test_qwen3_large_models_untied_embeddings(self):
        """Test that larger Qwen3 models don't have tied embeddings by default."""
        providers_with_untied_embeddings = [
            Qwen3ModelProvider8B(),
            Qwen3ModelProvider14B(),
            Qwen3ModelProvider32B(),
        ]

        for provider in providers_with_untied_embeddings:
            # These use the default from base class
            assert provider.share_embeddings_and_output_weights is False
