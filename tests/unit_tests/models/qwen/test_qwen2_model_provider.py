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

from megatron.hub.models.qwen import (
    Qwen2ModelProvider,
    Qwen2ModelProvider1P5B,
    Qwen2ModelProvider7B,
    Qwen2ModelProvider72B,
    Qwen2ModelProvider500M,
    Qwen25ModelProvider1P5B,
    Qwen25ModelProvider3B,
    Qwen25ModelProvider7B,
    Qwen25ModelProvider14B,
    Qwen25ModelProvider32B,
    Qwen25ModelProvider72B,
    Qwen25ModelProvider500M,
)


class TestQwen2ModelProvider:
    """Test cases for base Qwen2ModelProvider class."""

    def test_qwen2_model_provider_initialization(self):
        """Test Qwen2ModelProvider can be initialized with default values."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

        # Check Qwen2-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is True
        assert provider.seq_length == 4096
        assert provider.init_method_std == 0.02
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.vocab_size == 151936
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == 1000000.0
        assert provider.position_embedding_type == "rope"

    def test_qwen2_model_provider_with_custom_rope(self):
        """Test Qwen2ModelProvider with custom RoPE configuration."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
            rotary_percent=0.5,
        )

        assert provider.rotary_base == 500000.0
        assert provider.rotary_percent == 0.5

    def test_qwen2_model_provider_ffn_hidden_size(self):
        """Test Qwen2ModelProvider FFN hidden size calculation."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
        )

        assert provider.ffn_hidden_size == 11008

    def test_qwen2_model_provider_group_query_attention(self):
        """Test Qwen2ModelProvider with group query attention."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
        )

        assert provider.num_query_groups == 8

    def test_qwen2_model_provider_custom_vocab_size(self):
        """Test Qwen2ModelProvider with custom vocabulary size."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
        )

        assert provider.vocab_size == 32000

    def test_qwen2_model_provider_custom_sequence_length(self):
        """Test Qwen2ModelProvider with custom sequence length."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
        )

        assert provider.seq_length == 8192


class TestQwen2ModelProvider500M:
    """Test cases for Qwen2ModelProvider500M class."""

    def test_qwen2_500m_default_configuration(self):
        """Test Qwen2 500M model has correct default configuration."""
        provider = Qwen2ModelProvider500M()

        # Check Qwen2 500M specific configuration
        assert provider.num_layers == 24
        assert provider.hidden_size == 896
        assert provider.num_attention_heads == 14
        assert provider.num_query_groups == 2
        assert provider.ffn_hidden_size == 4864

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.vocab_size == 151936
        assert provider.seq_length == 4096
        assert provider.add_qkv_bias is True

    def test_qwen2_500m_override_configuration(self):
        """Test Qwen2 500M model with overridden configuration."""
        provider = Qwen2ModelProvider500M(
            seq_length=8192,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 8192
        assert provider.hidden_dropout == 0.1

        # Check defaults remain
        assert provider.num_layers == 24
        assert provider.hidden_size == 896


class TestQwen25ModelProvider500M:
    """Test cases for Qwen25ModelProvider500M class."""

    def test_qwen25_500m_default_configuration(self):
        """Test Qwen2.5 500M model has correct default configuration."""
        provider = Qwen25ModelProvider500M()

        # Check Qwen2.5 500M specific configuration
        assert provider.num_layers == 24
        assert provider.hidden_size == 896
        assert provider.num_attention_heads == 14
        assert provider.num_query_groups == 2
        assert provider.ffn_hidden_size == 4864
        assert provider.seq_length == 32768  # Extended context for 2.5

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.vocab_size == 151936

    def test_qwen25_500m_extended_context(self):
        """Test Qwen2.5 500M has extended context length."""
        provider = Qwen25ModelProvider500M()

        # Qwen2.5 500M supports 32k context window
        assert provider.seq_length == 32768


class TestQwen2ModelProvider1P5B:
    """Test cases for Qwen2ModelProvider1P5B class."""

    def test_qwen2_1p5b_default_configuration(self):
        """Test Qwen2 1.5B model has correct default configuration."""
        provider = Qwen2ModelProvider1P5B()

        # Check Qwen2 1.5B specific configuration
        assert provider.num_layers == 28
        assert provider.hidden_size == 1536
        assert provider.num_attention_heads == 12
        assert provider.num_query_groups == 2
        assert provider.ffn_hidden_size == 8960

        # Check inherited defaults
        assert provider.seq_length == 4096
        assert provider.vocab_size == 151936


class TestQwen25ModelProvider1P5B:
    """Test cases for Qwen25ModelProvider1P5B class."""

    def test_qwen25_1p5b_default_configuration(self):
        """Test Qwen2.5 1.5B model has correct default configuration."""
        provider = Qwen25ModelProvider1P5B()

        # Check Qwen2.5 1.5B specific configuration
        assert provider.num_layers == 28
        assert provider.hidden_size == 1536
        assert provider.num_attention_heads == 12
        assert provider.num_query_groups == 2
        assert provider.ffn_hidden_size == 8960
        assert provider.seq_length == 131072  # Extended context for 2.5

        # Check inherited defaults
        assert provider.vocab_size == 151936


class TestQwen25ModelProvider3B:
    """Test cases for Qwen25ModelProvider3B class."""

    def test_qwen25_3b_default_configuration(self):
        """Test Qwen2.5 3B model has correct default configuration."""
        provider = Qwen25ModelProvider3B()

        # Check Qwen2.5 3B specific configuration
        assert provider.num_layers == 36
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 16
        assert provider.num_query_groups == 2
        assert provider.ffn_hidden_size == 11008
        assert provider.vocab_size == 151936
        assert provider.share_embeddings_and_output_weights is True

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu


class TestQwen2ModelProvider7B:
    """Test cases for Qwen2ModelProvider7B class."""

    def test_qwen2_7b_default_configuration(self):
        """Test Qwen2 7B model has correct default configuration."""
        provider = Qwen2ModelProvider7B()

        # Check Qwen2 7B specific configuration
        assert provider.num_layers == 28
        assert provider.hidden_size == 3584
        assert provider.num_attention_heads == 28
        assert provider.num_query_groups == 4
        assert provider.ffn_hidden_size == 18944
        assert provider.vocab_size == 152064

        # Check inherited defaults
        assert provider.seq_length == 4096
        assert provider.normalization == "RMSNorm"


class TestQwen25ModelProvider7B:
    """Test cases for Qwen25ModelProvider7B class."""

    def test_qwen25_7b_default_configuration(self):
        """Test Qwen2.5 7B model has correct default configuration."""
        provider = Qwen25ModelProvider7B()

        # Check Qwen2.5 7B specific configuration
        assert provider.num_layers == 28
        assert provider.hidden_size == 3584
        assert provider.num_attention_heads == 28
        assert provider.num_query_groups == 4
        assert provider.ffn_hidden_size == 18944
        assert provider.vocab_size == 152064
        assert provider.seq_length == 131072  # Extended context for 2.5

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"


class TestQwen25ModelProvider14B:
    """Test cases for Qwen25ModelProvider14B class."""

    def test_qwen25_14b_default_configuration(self):
        """Test Qwen2.5 14B model has correct default configuration."""
        provider = Qwen25ModelProvider14B()

        # Check Qwen2.5 14B specific configuration
        assert provider.num_layers == 48
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 40
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 13824
        assert provider.vocab_size == 152064
        assert provider.layernorm_epsilon == 1e-5
        assert provider.seq_length == 131072

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu


class TestQwen25ModelProvider32B:
    """Test cases for Qwen25ModelProvider32B class."""

    def test_qwen25_32b_default_configuration(self):
        """Test Qwen2.5 32B model has correct default configuration."""
        provider = Qwen25ModelProvider32B()

        # Check Qwen2.5 32B specific configuration
        assert provider.num_layers == 64
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 40
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 27648
        assert provider.vocab_size == 152064
        assert provider.layernorm_epsilon == 1e-5
        assert provider.seq_length == 131072

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu


class TestQwen2ModelProvider72B:
    """Test cases for Qwen2ModelProvider72B class."""

    def test_qwen2_72b_default_configuration(self):
        """Test Qwen2 72B model has correct default configuration."""
        provider = Qwen2ModelProvider72B()

        # Check Qwen2 72B specific configuration
        assert provider.num_layers == 80
        assert provider.hidden_size == 8192
        assert provider.num_attention_heads == 64
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 29568
        assert provider.vocab_size == 152064
        assert provider.layernorm_epsilon == 1e-5

        # Check inherited defaults
        assert provider.seq_length == 4096
        assert provider.normalization == "RMSNorm"


class TestQwen25ModelProvider72B:
    """Test cases for Qwen25ModelProvider72B class."""

    def test_qwen25_72b_default_configuration(self):
        """Test Qwen2.5 72B model has correct default configuration."""
        provider = Qwen25ModelProvider72B()

        # Check Qwen2.5 72B specific configuration
        assert provider.num_layers == 80
        assert provider.hidden_size == 8192
        assert provider.num_attention_heads == 64
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 29568
        assert provider.vocab_size == 152064
        assert provider.layernorm_epsilon == 1e-5
        assert provider.seq_length == 131072  # Extended context for 2.5

        # Check inherited defaults
        assert provider.normalization == "RMSNorm"


class TestQwen2VsQwen25Differences:
    """Test cases to verify differences between Qwen2 and Qwen2.5 models."""

    def test_qwen2_vs_qwen25_500m_context_length(self):
        """Test that Qwen2.5 500M has extended context compared to Qwen2."""
        qwen2_provider = Qwen2ModelProvider500M()
        qwen25_provider = Qwen25ModelProvider500M()

        assert qwen2_provider.seq_length == 4096
        assert qwen25_provider.seq_length == 32768

    def test_qwen2_vs_qwen25_1p5b_context_length(self):
        """Test that Qwen2.5 1.5B has extended context compared to Qwen2."""
        qwen2_provider = Qwen2ModelProvider1P5B()
        qwen25_provider = Qwen25ModelProvider1P5B()

        assert qwen2_provider.seq_length == 4096
        assert qwen25_provider.seq_length == 131072

    def test_qwen2_vs_qwen25_7b_context_length(self):
        """Test that Qwen2.5 7B has extended context compared to Qwen2."""
        qwen2_provider = Qwen2ModelProvider7B()
        qwen25_provider = Qwen25ModelProvider7B()

        assert qwen2_provider.seq_length == 4096
        assert qwen25_provider.seq_length == 131072

    def test_qwen2_vs_qwen25_72b_context_length(self):
        """Test that Qwen2.5 72B has extended context compared to Qwen2."""
        qwen2_provider = Qwen2ModelProvider72B()
        qwen25_provider = Qwen25ModelProvider72B()

        assert qwen2_provider.seq_length == 4096
        assert qwen25_provider.seq_length == 131072

    def test_qwen25_3b_unique_features(self):
        """Test that Qwen2.5 3B has unique features not in Qwen2."""
        provider = Qwen25ModelProvider3B()

        # Qwen2.5 3B has tied embeddings
        assert provider.share_embeddings_and_output_weights is True

        # But other models don't by default
        base_provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert base_provider.share_embeddings_and_output_weights is False


class TestQwen2ProviderInheritance:
    """Test inheritance relationships between Qwen2 providers."""

    def test_qwen2_models_inherit_from_base(self):
        """Test Qwen2 providers inherit from Qwen2ModelProvider."""
        assert issubclass(Qwen2ModelProvider500M, Qwen2ModelProvider)
        assert issubclass(Qwen2ModelProvider1P5B, Qwen2ModelProvider)
        assert issubclass(Qwen2ModelProvider7B, Qwen2ModelProvider)
        assert issubclass(Qwen2ModelProvider72B, Qwen2ModelProvider)

    def test_qwen25_models_inherit_from_qwen2(self):
        """Test Qwen2.5 providers inherit from corresponding Qwen2 providers."""
        assert issubclass(Qwen25ModelProvider500M, Qwen2ModelProvider500M)
        assert issubclass(Qwen25ModelProvider1P5B, Qwen2ModelProvider1P5B)
        assert issubclass(Qwen25ModelProvider7B, Qwen2ModelProvider7B)
        assert issubclass(Qwen25ModelProvider72B, Qwen2ModelProvider72B)

    def test_qwen25_unique_models_inherit_from_base(self):
        """Test Qwen2.5 unique models inherit from base Qwen2ModelProvider."""
        assert issubclass(Qwen25ModelProvider3B, Qwen2ModelProvider)
        assert issubclass(Qwen25ModelProvider14B, Qwen2ModelProvider)
        assert issubclass(Qwen25ModelProvider32B, Qwen2ModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Qwen2.5 7B
        provider = Qwen25ModelProvider7B()

        # The provide method should be inherited from GPTModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)


class TestQwen2ProviderEdgeCases:
    """Test edge cases and error conditions."""

    def test_valid_num_query_groups(self):
        """Test that valid num_query_groups configuration works."""
        # num_attention_heads must be divisible by num_query_groups
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,  # 32 divisible by 8
        )
        assert provider.num_query_groups == 8

    def test_vocabulary_size_divisibility(self):
        """Test vocabulary size divisibility configuration."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=152064,
            make_vocab_size_divisible_by=128,
        )

        # The actual vocab size should be adjusted if needed
        assert provider.make_vocab_size_divisible_by == 128

    def test_seq_length_override(self):
        """Test sequence length configuration."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
        )

        assert provider.seq_length == 8192

    def test_rotary_base_configuration(self):
        """Test rotary base configuration."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000.0,
        )

        assert provider.rotary_base == 500000.0

    def test_layernorm_epsilon_override(self):
        """Test layernorm epsilon configuration."""
        provider = Qwen2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            layernorm_epsilon=1e-5,
        )

        assert provider.layernorm_epsilon == 1e-5


class TestQwen2ProviderQueryGroupsConsistency:
    """Test cases to verify query groups consistency across models."""

    def test_qwen2_500m_num_query_groups(self):
        """Test that Qwen2 500M has correct num_query_groups."""
        provider = Qwen2ModelProvider500M()
        assert provider.num_query_groups == 2

    def test_qwen2_1p5b_num_query_groups(self):
        """Test that Qwen2 1.5B has correct num_query_groups."""
        provider = Qwen2ModelProvider1P5B()
        assert provider.num_query_groups == 2

    def test_qwen25_3b_num_query_groups(self):
        """Test that Qwen2.5 3B has correct num_query_groups."""
        provider = Qwen25ModelProvider3B()
        assert provider.num_query_groups == 2

    def test_qwen2_7b_num_query_groups(self):
        """Test that Qwen2 7B has correct num_query_groups."""
        provider = Qwen2ModelProvider7B()
        assert provider.num_query_groups == 4

    def test_qwen25_14b_num_query_groups(self):
        """Test that Qwen2.5 14B has correct num_query_groups."""
        provider = Qwen25ModelProvider14B()
        assert provider.num_query_groups == 8

    def test_qwen25_32b_num_query_groups(self):
        """Test that Qwen2.5 32B has correct num_query_groups."""
        provider = Qwen25ModelProvider32B()
        assert provider.num_query_groups == 8

    def test_qwen2_72b_num_query_groups(self):
        """Test that Qwen2 72B has correct num_query_groups."""
        provider = Qwen2ModelProvider72B()
        assert provider.num_query_groups == 8


class TestQwen2ProviderVocabularyConsistency:
    """Test cases to verify vocabulary size consistency across models."""

    def test_qwen2_base_models_vocab_size(self):
        """Test that base Qwen2 models have 151936 vocab size."""
        providers = [
            Qwen2ModelProvider500M(),
            Qwen2ModelProvider1P5B(),
            Qwen25ModelProvider500M(),
            Qwen25ModelProvider1P5B(),
            Qwen25ModelProvider3B(),
        ]

        for provider in providers:
            assert provider.vocab_size == 151936

    def test_qwen2_large_models_vocab_size(self):
        """Test that large Qwen2 models have 152064 vocab size."""
        providers = [
            Qwen2ModelProvider7B(),
            Qwen2ModelProvider72B(),
            Qwen25ModelProvider7B(),
            Qwen25ModelProvider14B(),
            Qwen25ModelProvider32B(),
            Qwen25ModelProvider72B(),
        ]

        for provider in providers:
            assert provider.vocab_size == 152064
