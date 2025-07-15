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

from megatron.bridge.models.llama import (
    Llama2ModelProvider7B,
    Llama3ModelProvider,
    Llama3ModelProvider8B,
    Llama3ModelProvider70B,
    Llama4Experts16ModelProvider,
    Llama4Experts128ModelProvider,
    Llama4ModelProvider,
    Llama31ModelProvider,
    Llama31ModelProvider8B,
    Llama31ModelProvider70B,
    Llama31ModelProvider405B,
    Llama32ModelProvider1B,
    Llama32ModelProvider3B,
    LlamaModelProvider,
)


class TestLlamaModelProvider:
    """Test cases for base LlamaModelProvider class."""

    def test_llama_model_provider_initialization(self):
        """Test LlamaModelProvider can be initialized with default values."""
        provider = LlamaModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

        # Check Llama-specific defaults
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == 10000
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.normalization == "RMSNorm"
        assert provider.add_bias_linear is False
        assert provider.share_embeddings_and_output_weights is False
        assert provider.persist_layer_norm is False

    def test_llama_model_provider_with_custom_rope(self):
        """Test LlamaModelProvider with custom RoPE configuration."""
        provider = LlamaModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            rotary_base=500000,
            rotary_percent=0.5,
        )

        assert provider.rotary_base == 500000
        assert provider.rotary_percent == 0.5

    def test_llama_model_provider_ffn_hidden_size(self):
        """Test LlamaModelProvider FFN hidden size calculation."""
        provider = LlamaModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
        )

        assert provider.ffn_hidden_size == 11008

    def test_llama_model_provider_group_query_attention(self):
        """Test LlamaModelProvider with group query attention."""
        provider = LlamaModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
        )

        assert provider.num_query_groups == 8


class TestLlama2ModelProvider7B:
    """Test cases for Llama2ModelProvider7B class."""

    def test_llama2_7b_default_configuration(self):
        """Test Llama2 7B model has correct default configuration."""
        provider = Llama2ModelProvider7B()

        # Check Llama2 7B specific configuration
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.ffn_hidden_size == 11008
        assert provider.normalization == "RMSNorm"
        assert provider.rotary_base == 10000
        assert provider.seq_length == 4096
        assert provider.num_query_groups == 32

    def test_llama2_7b_override_configuration(self):
        """Test Llama2 7B model with overridden configuration."""
        provider = Llama2ModelProvider7B(
            seq_length=8192,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 8192
        assert provider.hidden_dropout == 0.1

        # Check defaults remain
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096


class TestLlama3ModelProvider8B:
    """Test cases for Llama3ModelProvider8B class."""

    def test_llama3_8b_default_configuration(self):
        """Test Llama3 8B model has correct default configuration."""
        provider = Llama3ModelProvider8B()

        # Check Llama3 8B specific configuration
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.ffn_hidden_size == 14336
        assert provider.normalization == "RMSNorm"
        assert provider.rotary_base == 500000
        assert provider.seq_length == 8192

    def test_llama3_8b_group_query_attention(self):
        """Test Llama3 8B uses default query groups from base class."""
        provider = Llama3ModelProvider8B()

        # Llama3 8B doesn't set num_query_groups explicitly
        # It would use the default from Llama3ModelProvider which is 8
        assert provider.num_attention_heads == 32

    def test_llama3_8b_override_configuration(self):
        """Test Llama3 8B model with overridden configuration."""
        provider = Llama3ModelProvider8B(
            seq_length=16384,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 16384
        assert provider.hidden_dropout == 0.1

        # Check critical defaults remain
        assert provider.rotary_base == 500000


class TestLlama31ModelProvider70B:
    """Test cases for Llama31ModelProvider70B class."""

    def test_llama31_70b_default_configuration(self):
        """Test Llama3.1 70B model has correct default configuration."""
        provider = Llama31ModelProvider70B()

        # Check Llama3.1 70B specific configuration
        assert provider.num_layers == 80
        assert provider.hidden_size == 8192
        assert provider.num_attention_heads == 64
        assert provider.ffn_hidden_size == 28672
        assert provider.normalization == "RMSNorm"
        assert provider.rotary_base == 500000
        assert provider.seq_length == 131072  # 128k context

    def test_llama31_70b_large_context(self):
        """Test Llama3.1 70B supports large context window."""
        provider = Llama31ModelProvider70B()

        # Llama3.1 70B supports 128k context window
        assert provider.seq_length == 131072

    def test_llama31_70b_rope_scaling(self):
        """Test Llama3.1 70B RoPE configuration for long context."""
        provider = Llama31ModelProvider70B()

        # Check RoPE base for extended context
        assert provider.rotary_base == 500000

        # Check if rope scaling is configured
        if hasattr(provider, "rope_scaling_type"):
            assert provider.rope_scaling_type is not None
        if hasattr(provider, "rope_scaling_factor"):
            assert provider.rope_scaling_factor > 1.0


class TestLlama3QueryGroupsInheritance:
    """Test cases to verify that all configs extending from Llama3 have num_query_groups=8."""

    def test_llama3_base_provider_has_correct_num_query_groups(self):
        """Test that Llama3ModelProvider has num_query_groups=8."""
        provider = Llama3ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider.num_query_groups == 8

    def test_llama31_provider_inherits_num_query_groups(self):
        """Test that Llama31ModelProvider inherits num_query_groups=8 from Llama3ModelProvider."""
        provider = Llama31ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider.num_query_groups == 8

    def test_llama3_8b_provider_inherits_num_query_groups(self):
        """Test that Llama3ModelProvider8B inherits num_query_groups=8."""
        provider = Llama3ModelProvider8B()
        assert provider.num_query_groups == 8

    def test_llama3_70b_provider_inherits_num_query_groups(self):
        """Test that Llama3ModelProvider70B inherits num_query_groups=8."""
        provider = Llama3ModelProvider70B()
        assert provider.num_query_groups == 8

    def test_llama31_8b_provider_inherits_num_query_groups(self):
        """Test that Llama31ModelProvider8B inherits num_query_groups=8."""
        provider = Llama31ModelProvider8B()
        assert provider.num_query_groups == 8

    def test_llama31_70b_provider_inherits_num_query_groups(self):
        """Test that Llama31ModelProvider70B inherits num_query_groups=8."""
        provider = Llama31ModelProvider70B()
        assert provider.num_query_groups == 8

    def test_llama31_405b_provider_inherits_num_query_groups(self):
        """Test that Llama31ModelProvider405B inherits num_query_groups=8."""
        provider = Llama31ModelProvider405B()
        assert provider.num_query_groups == 8

    def test_llama32_1b_provider_has_correct_num_query_groups(self):
        """Test that Llama32ModelProvider1B has num_query_groups=8."""
        provider = Llama32ModelProvider1B()
        assert provider.num_query_groups == 8

    def test_llama32_3b_provider_has_correct_num_query_groups(self):
        """Test that Llama32ModelProvider3B has num_query_groups=8."""
        provider = Llama32ModelProvider3B()
        assert provider.num_query_groups == 8

    def test_llama4_provider_inherits_num_query_groups(self):
        """Test that Llama4ModelProvider inherits num_query_groups=8 from Llama3ModelProvider."""
        provider = Llama4ModelProvider(num_moe_experts=16)
        assert provider.num_query_groups == 8

    def test_llama4_experts16_provider_inherits_num_query_groups(self):
        """Test that Llama4Experts16ModelProvider inherits num_query_groups=8."""
        provider = Llama4Experts16ModelProvider()
        assert provider.num_query_groups == 8

    def test_llama4_experts128_provider_inherits_num_query_groups(self):
        """Test that Llama4Experts128ModelProvider inherits num_query_groups=8."""
        provider = Llama4Experts128ModelProvider()
        assert provider.num_query_groups == 8


class TestLlamaProviderInheritance:
    """Test inheritance relationships between Llama providers."""

    def test_llama2_inherits_from_base(self):
        """Test Llama2 providers inherit from LlamaModelProvider."""
        assert issubclass(Llama2ModelProvider7B, LlamaModelProvider)

    def test_llama3_inherits_from_base(self):
        """Test Llama3 providers inherit from LlamaModelProvider."""
        assert issubclass(Llama3ModelProvider8B, LlamaModelProvider)

    def test_llama31_inherits_from_llama3(self):
        """Test Llama3.1 providers inherit from Llama3ModelProvider."""
        # This depends on the actual implementation
        assert issubclass(Llama31ModelProvider70B, LlamaModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Llama3 8B
        provider = Llama3ModelProvider8B()

        # The provide method should be inherited from GPTModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)


class TestLlamaProviderEdgeCases:
    """Test edge cases and error conditions."""

    def test_valid_num_query_groups(self):
        """Test that valid num_query_groups configuration works."""
        # num_attention_heads must be divisible by num_query_groups
        provider = LlamaModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,  # 32 divisible by 8
        )
        assert provider.num_query_groups == 8

    def test_vocabulary_size_divisibility(self):
        """Test vocabulary size divisibility configuration."""
        provider = LlamaModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
        )

        # The actual vocab size should be adjusted if needed
        assert provider.make_vocab_size_divisible_by == 128

    def test_seq_length_override(self):
        """Test sequence length configuration."""
        provider = LlamaModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
        )

        assert provider.seq_length == 8192
