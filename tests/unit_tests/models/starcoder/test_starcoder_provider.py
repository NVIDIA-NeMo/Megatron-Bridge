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

from megatron.bridge.models.starcoder.starcoder_provider import (
    StarcoderConfig15B,
    StarcoderModelProvider,
)


class TestStarcoderModelProvider:
    """Test cases for StarcoderModelProvider class."""

    def test_starcoder_model_provider_defaults(self):
        """Test StarcoderModelProvider has correct default values."""
        provider = StarcoderModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
        )

        # Check required transformer config fields
        assert provider.num_layers == 12
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 12

        # Check Starcoder-specific defaults
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func == F.gelu
        assert provider.add_bias_linear is True
        assert provider.seq_length == 8192
        assert provider.position_embedding_type == "learned_absolute"
        assert provider.hidden_dropout == 0.2
        assert provider.attention_dropout == 0.2
        assert provider.init_method_std == 0.01
        assert provider.layernorm_epsilon == 1e-5
        assert provider.share_embeddings_and_output_weights is False
        assert provider.kv_channels == 64
        assert provider.num_query_groups == 1
        assert provider.attention_softmax_in_fp32 is True
        assert provider.bias_activation_fusion is True
        assert provider.bias_dropout_fusion is True


class TestStarcoderConfig15B:
    """Test cases for StarcoderConfig15B class."""

    def test_starcoder_config_15b_defaults(self):
        """Test StarcoderConfig15B has correct default values for 15B model."""
        provider = StarcoderConfig15B()

        # Check 15B-specific configuration
        assert provider.num_layers == 40
        assert provider.hidden_size == 6144
        assert provider.ffn_hidden_size == 24576
        assert provider.num_attention_heads == 48
        assert provider.init_method_std == 0.02

        # Check inherited Starcoder defaults
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func == F.gelu
        assert provider.add_bias_linear is True
        assert provider.seq_length == 8192
        assert provider.position_embedding_type == "learned_absolute"
        assert provider.hidden_dropout == 0.2
        assert provider.attention_dropout == 0.2
        assert provider.layernorm_epsilon == 1e-5
        assert provider.share_embeddings_and_output_weights is False
        assert provider.kv_channels == 128
        assert provider.num_query_groups == 1
        assert provider.attention_softmax_in_fp32 is True
        assert provider.bias_activation_fusion is True
        assert provider.bias_dropout_fusion is True


class TestStarcoderProviderInheritance:
    """Test inheritance relationships between Starcoder providers."""

    def test_starcoder_models_inherit_from_gpt(self):
        """Test Starcoder providers inherit from GPTModelProvider."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        assert issubclass(StarcoderModelProvider, GPTModelProvider)
        assert issubclass(StarcoderConfig15B, StarcoderModelProvider)
        assert issubclass(StarcoderConfig15B, GPTModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        provider = StarcoderConfig15B()
        assert hasattr(provider, "provide")
        assert callable(provider.provide)
