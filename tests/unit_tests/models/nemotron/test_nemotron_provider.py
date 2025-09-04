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

from megatron.bridge.models.nemotron import (
    Nemotron3ModelProvider4B,
    Nemotron3ModelProvider8B,
    Nemotron3ModelProvider22B,
    Nemotron4ModelProvider15B,
    Nemotron4ModelProvider340B,
    NemotronModelProvider,
)
from megatron.bridge.models.nemotron.nemotron_provider import squared_relu


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

        # Check Nemotron3Config4B default values (base class defaults)
        assert provider.num_layers == 32
        assert provider.seq_length == 4096
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 9216
        assert provider.num_attention_heads == 24
        assert provider.num_query_groups == 8
        assert provider.kv_channels == 128
        assert provider.init_method_std == 0.0134


@pytest.mark.unit
class TestNemotron3ModelProvider4B:
    """Test cases for Nemotron3ModelProvider4B class."""

    def test_nemotron3_4b_default_configuration(self):
        """Test Nemotron3 4B model has correct default configuration."""
        provider = Nemotron3ModelProvider4B()

        # Check Nemotron3 4B specific configuration
        assert provider.num_layers == 32
        assert provider.seq_length == 4096
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 9216
        assert provider.num_attention_heads == 24
        assert provider.num_query_groups == 8
        assert provider.kv_channels == 128
        assert provider.init_method_std == 0.0134

        # Check inherited Nemotron defaults
        assert provider.activation_func is squared_relu
        assert provider.normalization == "LayerNorm"
        assert provider.position_embedding_type == "rope"


@pytest.mark.unit
class TestNemotron3ModelProvider8B:
    """Test cases for Nemotron3ModelProvider8B class."""

    def test_nemotron3_8b_default_configuration(self):
        """Test Nemotron3 8B model has correct default configuration."""
        provider = Nemotron3ModelProvider8B()

        # Check Nemotron3 8B specific configuration
        assert provider.num_layers == 32
        assert provider.seq_length == 4096
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 16384
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 32  # Full attention: None -> num_attention_heads in __post_init__
        assert provider.kv_channels == 128  # None -> hidden_size // num_attention_heads in __post_init__
        assert provider.init_method_std == 0.01

        # Check inherited Nemotron defaults
        assert provider.activation_func is squared_relu
        assert provider.normalization == "LayerNorm"


@pytest.mark.unit
class TestNemotron3ModelProvider22B:
    """Test cases for Nemotron3ModelProvider22B class."""

    def test_nemotron3_22b_default_configuration(self):
        """Test Nemotron3 22B model has correct default configuration."""
        provider = Nemotron3ModelProvider22B()

        # Check Nemotron3 22B specific configuration
        assert provider.num_layers == 40
        assert provider.seq_length == 4096
        assert provider.hidden_size == 6144
        assert provider.ffn_hidden_size == 24576
        assert provider.num_attention_heads == 48
        assert provider.num_query_groups == 48  # Full attention: None -> num_attention_heads in __post_init__
        assert provider.kv_channels == 128  # None -> hidden_size // num_attention_heads in __post_init__
        assert provider.init_method_std == 0.008

        # Check inherited Nemotron defaults
        assert provider.activation_func is squared_relu
        assert provider.normalization == "LayerNorm"


@pytest.mark.unit
class TestNemotron4ModelProvider15B:
    """Test cases for Nemotron4ModelProvider15B class."""

    def test_nemotron4_15b_default_configuration(self):
        """Test Nemotron4 15B model has correct default configuration."""
        provider = Nemotron4ModelProvider15B()

        # Check Nemotron4 15B specific configuration
        assert provider.num_layers == 32
        assert provider.seq_length == 4096
        assert provider.hidden_size == 6144
        assert provider.ffn_hidden_size == 24576
        assert provider.num_attention_heads == 48
        assert provider.num_query_groups == 8  # Uses GQA
        assert provider.kv_channels == 128  # None -> hidden_size // num_attention_heads in __post_init__
        assert provider.init_method_std == 0.0134

        # Check inherited Nemotron defaults
        assert provider.activation_func is squared_relu
        assert provider.normalization == "LayerNorm"


@pytest.mark.unit
class TestNemotron4ModelProvider340B:
    """Test cases for Nemotron4ModelProvider340B class."""

    def test_nemotron4_340b_default_configuration(self):
        """Test Nemotron4 340B model has correct default configuration."""
        provider = Nemotron4ModelProvider340B()

        # Check Nemotron4 340B specific configuration
        assert provider.num_layers == 96
        assert provider.seq_length == 4096
        assert provider.hidden_size == 18432
        assert provider.ffn_hidden_size == 73728
        assert provider.num_attention_heads == 96
        assert provider.num_query_groups == 8  # Uses GQA
        assert provider.kv_channels == 192  # None -> hidden_size // num_attention_heads in __post_init__
        assert provider.init_method_std == 0.0063

        # Check inherited Nemotron defaults
        assert provider.activation_func is squared_relu
        assert provider.normalization == "LayerNorm"


@pytest.mark.unit
class TestNemotronProviderInheritance:
    """Test inheritance relationships between Nemotron providers."""

    def test_nemotron3_4b_inherits_from_base(self):
        """Test Nemotron3 4B provider inherits from NemotronModelProvider."""
        assert issubclass(Nemotron3ModelProvider4B, NemotronModelProvider)

    def test_nemotron3_8b_inherits_from_base(self):
        """Test Nemotron3 8B provider inherits from NemotronModelProvider."""
        assert issubclass(Nemotron3ModelProvider8B, NemotronModelProvider)

    def test_nemotron3_22b_inherits_from_base(self):
        """Test Nemotron3 22B provider inherits from NemotronModelProvider."""
        assert issubclass(Nemotron3ModelProvider22B, NemotronModelProvider)

    def test_nemotron4_15b_inherits_from_base(self):
        """Test Nemotron4 15B provider inherits from NemotronModelProvider."""
        assert issubclass(Nemotron4ModelProvider15B, NemotronModelProvider)

    def test_nemotron4_340b_inherits_from_base(self):
        """Test Nemotron4 340B provider inherits from NemotronModelProvider."""
        assert issubclass(Nemotron4ModelProvider340B, NemotronModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Nemotron3 4B
        provider = Nemotron3ModelProvider4B()

        # The provide method should be inherited from GPTModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)
