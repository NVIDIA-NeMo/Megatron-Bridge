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


from megatron.bridge.models.ministral3.ministral3_provider import (
    Ministral3ModelProvider,
    Ministral3ModelProvider3B,
    Ministral3ModelProvider8B,
    Ministral3ModelProvider14B,
)


class TestMinistral3ModelProvider:
    """Test cases for Ministral3ModelProvider base class."""

    def test_ministral3_model_provider_initialization(self):
        """Test Ministral3ModelProvider can be initialized with default values."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 26
        assert provider.hidden_size == 3072
        assert provider.num_attention_heads == 32

        # Check Ministral3-inherited defaults from Mistral
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "yarn"
        assert provider.add_bias_linear is False
        assert provider.seq_length == 32768  # Default
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5
        assert provider.rotary_base == 1000000
        assert provider.num_query_groups == 8

    def test_ministral3_vl_specific_defaults(self):
        """Test Ministral3ModelProvider VL-specific default configuration."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        # Check VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.image_token_id == 10

        # Check freeze options defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_ministral3_yarn_rope_defaults(self):
        """Test Ministral3ModelProvider YARN RoPE default configuration."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        # Check YARN RoPE defaults
        assert provider.yarn_rotary_scaling_factor == 16.0
        assert provider.yarn_original_max_position_embeddings == 16384
        assert provider.yarn_beta_fast == 32.0
        assert provider.yarn_beta_slow == 1.0
        assert provider.yarn_correction_range_round_to_int is False
        assert provider.yarn_mscale == 1.0
        assert provider.yarn_mscale_all_dim == 1.0

    def test_ministral3_custom_image_token_id(self):
        """Test Ministral3ModelProvider with custom image token ID."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
            image_token_id=100,
        )

        assert provider.image_token_id == 100

    def test_ministral3_freeze_options(self):
        """Test Ministral3ModelProvider with freeze options."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True

    def test_ministral3_provide_method_exists(self):
        """Test that provide method exists and is callable."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_ministral3_provide_language_model_method_exists(self):
        """Test that provide_language_model method exists and is callable."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)


class TestMinistral3ModelProvider3B:
    """Test cases for Ministral3ModelProvider3B."""

    def test_ministral3_3b_initialization(self):
        """Test Ministral3ModelProvider3B can be initialized with correct defaults."""
        provider = Ministral3ModelProvider3B()

        # Check 3B specific configuration
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 9216
        assert provider.num_layers == 26
        assert provider.share_embeddings_and_output_weights is True


class TestMinistral3ModelProvider8B:
    """Test cases for Ministral3ModelProvider8B."""

    def test_ministral3_8b_initialization(self):
        """Test Ministral3ModelProvider8B can be initialized with correct defaults."""
        provider = Ministral3ModelProvider8B()

        # Check 8B specific configuration
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 14336
        assert provider.num_layers == 34


class TestMinistral3ModelProvider14B:
    """Test cases for Ministral3ModelProvider14B."""

    def test_ministral3_14b_initialization(self):
        """Test Ministral3ModelProvider14B can be initialized with correct defaults."""
        provider = Ministral3ModelProvider14B()

        # Check 14B specific configuration
        assert provider.hidden_size == 5120
        assert provider.ffn_hidden_size == 16384
        assert provider.num_layers == 40
        assert provider.rotary_base == 1000000000.0
