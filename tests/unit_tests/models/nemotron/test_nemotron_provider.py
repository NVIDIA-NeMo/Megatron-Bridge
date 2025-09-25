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

import pytest

from megatron.bridge.models.nemotron.nemotron_provider import (
    Nemotron3ModelProvider4B,
    Nemotron3ModelProvider8B,
    Nemotron3ModelProvider22B,
    Nemotron4ModelProvider15B,
    Nemotron4ModelProvider340B,
    NemotronModelProvider,
    squared_relu,
)


@pytest.mark.unit
class TestNemotronModelProvider:
    """Test cases for base NemotronModelProvider class."""

    def test_nemotron_model_provider_initialization(self):
        """Test NemotronModelProvider can be initialized with default values."""
        provider = NemotronModelProvider()

        # Check Nemotron-specific defaults
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func is squared_relu
        assert provider.position_embedding_type == "rope"
        assert provider.share_embeddings_and_output_weights is False
        assert provider.add_bias_linear is False
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.rotary_percent == 0.5
        assert provider.bias_dropout_add_fusion is False
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.cross_entropy_loss_fusion is True


class TestNemotronSpecificProviders:
    """Test cases for specific Nemotron model provider configurations."""

    def test_nemotron3_4b_config(self):
        """Test Nemotron3 4B provider configuration matches HF model specs."""
        provider = Nemotron3ModelProvider4B()

        assert provider.hidden_size == 3072
        assert provider.num_layers == 32
        assert provider.num_attention_heads == 24
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 9216
        assert provider.kv_channels == 128
        assert provider.seq_length == 4096
        assert provider.init_method_std == 0.0134

    def test_nemotron3_8b_config(self):
        """Test Nemotron3 8B provider configuration matches HF model specs."""
        provider = Nemotron3ModelProvider8B()

        assert provider.hidden_size == 4096
        assert provider.num_layers == 32
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups is None
        assert provider.ffn_hidden_size == 16384
        assert provider.kv_channels is None
        assert provider.seq_length == 4096
        assert provider.init_method_std == 0.010

    def test_nemotron3_22b_config(self):
        """Test Nemotron3 22B provider configuration."""
        provider = Nemotron3ModelProvider22B()

        assert provider.hidden_size == 6144
        assert provider.num_layers == 40
        assert provider.num_attention_heads == 48
        assert provider.num_query_groups is None
        assert provider.ffn_hidden_size == 24576
        assert provider.kv_channels is None
        assert provider.seq_length == 4096
        assert provider.init_method_std == 0.008

    def test_nemotron4_15b_config(self):
        """Test Nemotron4 15B provider configuration."""
        provider = Nemotron4ModelProvider15B()

        assert provider.hidden_size == 6144
        assert provider.num_layers == 32
        assert provider.num_attention_heads == 48
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 24576
        assert provider.kv_channels is None
        assert provider.seq_length == 4096
        assert provider.init_method_std == 0.0134

    def test_nemotron4_340b_config(self):
        """Test Nemotron4 340B provider configuration."""
        provider = Nemotron4ModelProvider340B()

        # Should match nvidia/Nemotron-4-340B-Base/Instruct (if available)
        assert provider.hidden_size == 18432
        assert provider.num_layers == 96
        assert provider.num_attention_heads == 96
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 73728
        assert provider.kv_channels is None
        assert provider.seq_length == 4096
        assert provider.init_method_std == 0.0063

    def test_all_providers_have_nemotron_defaults(self):
        """Test that all specific providers inherit Nemotron-specific defaults."""
        providers = [
            Nemotron3ModelProvider4B(),
            Nemotron3ModelProvider8B(),
            Nemotron3ModelProvider22B(),
            Nemotron4ModelProvider15B(),
            Nemotron4ModelProvider340B(),
        ]

        for provider in providers:
            # Check Nemotron-specific defaults
            assert provider.normalization == "LayerNorm"
            assert provider.position_embedding_type == "rope"
            assert provider.share_embeddings_and_output_weights is False
            assert provider.add_bias_linear is False
            assert provider.hidden_dropout == 0.0
            assert provider.attention_dropout == 0.0
            assert provider.rotary_percent == 0.5
            assert provider.layernorm_zero_centered_gamma is True
            assert provider.cross_entropy_loss_fusion is True
