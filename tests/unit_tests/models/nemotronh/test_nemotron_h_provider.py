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

import torch
import torch.nn.functional as F

from megatron.bridge.models.nemotronh.nemotron_h_provider import (
    NemotronHModel4BProvider,
    NemotronHModel8BProvider,
    NemotronHModel47BProvider,
    NemotronHModel56BProvider,
    NemotronHModelProvider,
    NemotronNano9Bv2Provider,
    NemotronNano12Bv2Provider,
)


class TestNemotronHModelProvider:
    """Test cases for base NemotronHModelProvider class."""

    def test_nemotron_h_model_provider_initialization(self):
        """Test NemotronHModelProvider can be initialized with default values."""
        provider = NemotronHModelProvider(
            num_layers=52,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 52
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

        # Check Nemotron-H specific defaults
        assert provider.seq_length == 8192
        assert provider.mamba_num_groups == 8
        assert provider.mamba_head_dim == 64
        assert provider.num_query_groups == 8
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.masked_softmax_fusion is True
        assert provider.apply_query_key_layer_scaling is False
        assert provider.persist_layer_norm is True
        assert provider.attention_softmax_in_fp32 is False
        assert provider.first_last_layers_bf16 is True
        assert provider.is_hybrid_model is True

    def test_nemotron_h_custom_activation_function(self):
        """Test NemotronHModelProvider with custom activation function."""

        def custom_activation(x):
            return torch.pow(F.relu(x), 2)

        provider = NemotronHModelProvider(
            num_layers=52,
            hidden_size=4096,
            num_attention_heads=32,
            activation_func=custom_activation,
        )

        # Test that the activation function is set correctly
        test_input = torch.tensor([1.0, -1.0, 2.0])
        expected_output = torch.pow(F.relu(test_input), 2)
        actual_output = provider.activation_func(test_input)

        assert torch.allclose(actual_output, expected_output)

    def test_nemotron_h_mamba_configuration(self):
        """Test NemotronHModelProvider Mamba-specific configuration."""
        provider = NemotronHModelProvider(
            num_layers=52,
            hidden_size=4096,
            num_attention_heads=32,
            mamba_num_groups=16,
            mamba_head_dim=128,
        )

        assert provider.mamba_num_groups == 16
        assert provider.mamba_head_dim == 128


class TestNemotronHModel4BProvider:
    """Test cases for NemotronHModel4BProvider class."""

    def test_nemotron_h_4b_default_configuration(self):
        """Test Nemotron-H 4B model has correct default configuration."""
        provider = NemotronHModel4BProvider()

        # Check Nemotron-H 4B specific configuration
        assert provider.num_layers == 52
        assert provider.hidden_size == 3072
        assert provider.num_attention_heads == 32
        assert provider.mamba_num_heads == 112
        assert provider.kv_channels == 128
        assert provider.mamba_state_dim == 128
        assert provider.ffn_hidden_size == 12288
        assert provider.hybrid_override_pattern == "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
        assert provider.use_mamba_mem_eff_path is False

    def test_nemotron_h_4b_override_configuration(self):
        """Test Nemotron-H 4B model with overridden configuration."""
        provider = NemotronHModel4BProvider(
            seq_length=16384,
            hidden_dropout=0.1,
            use_mamba_mem_eff_path=True,
        )

        # Check overridden values
        assert provider.seq_length == 16384
        assert provider.hidden_dropout == 0.1
        assert provider.use_mamba_mem_eff_path is True

        # Check defaults remain
        assert provider.num_layers == 52
        assert provider.hidden_size == 3072
        assert provider.mamba_num_heads == 112


class TestNemotronHModel8BProvider:
    """Test cases for NemotronHModel8BProvider class."""

    def test_nemotron_h_8b_default_configuration(self):
        """Test Nemotron-H 8B model has correct default configuration."""
        provider = NemotronHModel8BProvider()

        # Check Nemotron-H 8B specific configuration
        assert provider.num_layers == 52
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.mamba_state_dim == 128
        assert provider.ffn_hidden_size == 21504
        assert provider.hybrid_override_pattern == "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"

    def test_nemotron_h_8b_override_configuration(self):
        """Test Nemotron-H 8B model with overridden configuration."""
        provider = NemotronHModel8BProvider(
            seq_length=32768,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 32768
        assert provider.hidden_dropout == 0.1

        # Check critical defaults remain
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 21504


class TestNemotronHModel47BProvider:
    """Test cases for NemotronHModel47BProvider class."""

    def test_nemotron_h_47b_default_configuration(self):
        """Test Nemotron-H 47B model has correct default configuration."""
        provider = NemotronHModel47BProvider()

        # Check Nemotron-H 47B specific configuration
        assert provider.num_layers == 98
        assert provider.hidden_size == 8192
        assert provider.num_attention_heads == 64
        assert provider.mamba_state_dim == 256
        assert provider.ffn_hidden_size == 30720
        assert (
            "M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M-"
            in provider.hybrid_override_pattern
        )

    def test_nemotron_h_47b_override_configuration(self):
        """Test Nemotron-H 47B model with overridden configuration."""
        provider = NemotronHModel47BProvider(
            seq_length=65536,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 65536
        assert provider.hidden_dropout == 0.1

        # Check critical defaults remain
        assert provider.num_layers == 98
        assert provider.hidden_size == 8192


class TestNemotronHModel56BProvider:
    """Test cases for NemotronHModel56BProvider class."""

    def test_nemotron_h_56b_default_configuration(self):
        """Test Nemotron-H 56B model has correct default configuration."""
        provider = NemotronHModel56BProvider()

        # Check Nemotron-H 56B specific configuration
        assert provider.num_layers == 118
        assert provider.hidden_size == 8192
        assert provider.num_attention_heads == 64
        assert provider.mamba_state_dim == 256
        assert provider.ffn_hidden_size == 32768
        assert (
            "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
            in provider.hybrid_override_pattern
        )

    def test_nemotron_h_56b_override_configuration(self):
        """Test Nemotron-H 56B model with overridden configuration."""
        provider = NemotronHModel56BProvider(
            seq_length=131072,  # 128k context
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 131072
        assert provider.hidden_dropout == 0.1

        # Check critical defaults remain
        assert provider.num_layers == 118
        assert provider.hidden_size == 8192


class TestNemotronHProviderInheritance:
    """Test inheritance relationships between Nemotron-H providers."""

    def test_nemotron_h_4b_inherits_from_base(self):
        """Test Nemotron-H 4B provider inherits from NemotronHModelProvider."""
        assert issubclass(NemotronHModel4BProvider, NemotronHModelProvider)

    def test_nemotron_h_8b_inherits_from_base(self):
        """Test Nemotron-H 8B provider inherits from NemotronHModelProvider."""
        assert issubclass(NemotronHModel8BProvider, NemotronHModelProvider)

    def test_nemotron_h_47b_inherits_from_base(self):
        """Test Nemotron-H 47B provider inherits from NemotronHModelProvider."""
        assert issubclass(NemotronHModel47BProvider, NemotronHModelProvider)

    def test_nemotron_h_56b_inherits_from_base(self):
        """Test Nemotron-H 56B provider inherits from NemotronHModelProvider."""
        assert issubclass(NemotronHModel56BProvider, NemotronHModelProvider)

    def test_nemotron_nano_9b_v2_inherits_from_base(self):
        """Test Nemotron Nano v2 9B provider inherits from NemotronHModelProvider."""
        assert issubclass(NemotronNano9Bv2Provider, NemotronHModelProvider)

    def test_nemotron_nano_12b_v2_inherits_from_base(self):
        """Test Nemotron Nano v2 12B provider inherits from NemotronHModelProvider."""
        assert issubclass(NemotronNano12Bv2Provider, NemotronHModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Nemotron-H 4B
        providers = [
            NemotronHModel4BProvider(),
            NemotronHModel8BProvider(),
            NemotronHModel47BProvider(),
            NemotronHModel56BProvider(),
            NemotronNano9Bv2Provider(),
            NemotronNano12Bv2Provider(),
        ]

        for provider in providers:
            # The provide method should be inherited from MambaModelProvider
            assert hasattr(provider, "provide")
            assert callable(provider.provide)


class TestNemotronNano9Bv2Provider:
    """Test cases for NemotronNano9Bv2Provider class."""

    def test_nemotron_nano_9b_v2_default_configuration(self):
        """Test Nemotron Nano v2 9B model has correct default configuration."""
        provider = NemotronNano9Bv2Provider()
        provider.finalize()

        # Check Nemotron Nano v2 9B specific configuration
        assert provider.num_layers == 56
        assert provider.hidden_size == 4480
        assert provider.num_attention_heads == 40
        assert provider.mamba_num_heads == 128
        assert provider.kv_channels == 112
        assert provider.mamba_state_dim == 128
        assert provider.ffn_hidden_size == 15680
        assert provider.mamba_head_dim == 80
        assert provider.hybrid_override_pattern == "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"

    def test_nemotron_nano_9b_v2_override_configuration(self):
        """Test Nemotron Nano v2 9B model with overridden configuration."""
        provider = NemotronNano9Bv2Provider(
            seq_length=16384,
            hidden_dropout=0.1,
            mamba_head_dim=96,
        )

        # Check overridden values
        assert provider.seq_length == 16384
        assert provider.hidden_dropout == 0.1
        assert provider.mamba_head_dim == 96

        # Check critical defaults remain
        assert provider.num_layers == 56
        assert provider.hidden_size == 4480
        assert provider.mamba_num_heads == 128
        assert provider.ffn_hidden_size == 15680


class TestNemotronNano12Bv2Provider:
    """Test cases for NemotronNano12Bv2Provider class."""

    def test_nemotron_nano_12b_v2_default_configuration(self):
        """Test Nemotron Nano v2 12B model has correct default configuration."""
        provider = NemotronNano12Bv2Provider()
        provider.finalize()

        # Check Nemotron Nano v2 12B specific configuration
        assert provider.num_layers == 62
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 40
        assert provider.mamba_num_heads == 128
        assert provider.kv_channels == 128
        assert provider.mamba_state_dim == 128
        assert provider.ffn_hidden_size == 20480
        assert provider.mamba_head_dim == 80
        assert provider.hybrid_override_pattern == "M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-"

    def test_nemotron_nano_12b_v2_override_configuration(self):
        """Test Nemotron Nano v2 12B model with overridden configuration."""
        provider = NemotronNano12Bv2Provider(
            seq_length=32768,
            hidden_dropout=0.1,
            mamba_head_dim=96,
        )

        # Check overridden values
        assert provider.seq_length == 32768
        assert provider.hidden_dropout == 0.1
        assert provider.mamba_head_dim == 96

        # Check critical defaults remain
        assert provider.num_layers == 62
        assert provider.hidden_size == 5120
        assert provider.mamba_num_heads == 128
        assert provider.ffn_hidden_size == 20480


class TestHybridPatterns:
    """Test hybrid override patterns of Nemotron-H providers."""

    def test_hybrid_patterns_contain_mamba_and_attention(self):
        """Test that all hybrid patterns contain both Mamba and Attention layers."""
        providers = [
            NemotronHModel4BProvider(),
            NemotronHModel8BProvider(),
            NemotronHModel47BProvider(),
            NemotronHModel56BProvider(),
            NemotronNano9Bv2Provider(),
            NemotronNano12Bv2Provider(),
        ]

        for provider in providers:
            pattern = provider.hybrid_override_pattern
            assert "M" in pattern  # Mamba layers
            assert "*" in pattern  # Attention layers
            assert len(pattern) > 0
