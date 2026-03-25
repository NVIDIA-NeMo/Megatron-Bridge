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

from megatron.bridge.models.qwen_audio import Qwen2AudioModelProvider


class TestQwen2AudioModelProvider:
    """Test cases for Qwen2AudioModelProvider class."""

    def test_initialization(self):
        """Test Qwen2AudioModelProvider can be initialized with default values."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

    def test_audio_specific_defaults(self):
        """Test Qwen2AudioModelProvider audio-specific default configuration."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Audio-language models shouldn't scatter embeddings
        assert provider.scatter_embedding_sequence_parallel is False

        # HF config defaults to None
        assert provider.hf_config is None

        # Audio-specific token ID
        assert provider.audio_token_id == 151646

        # Token IDs
        assert provider.bos_token_id == 151643
        assert provider.eos_token_id == 151645
        assert provider.pad_token_id == 151643

        # Freeze options defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_audio_model is False
        assert provider.freeze_audio_projection is False

    def test_custom_token_ids(self):
        """Test Qwen2AudioModelProvider with custom token IDs."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            audio_token_id=200,
            bos_token_id=201,
            eos_token_id=202,
            pad_token_id=203,
        )

        assert provider.audio_token_id == 200
        assert provider.bos_token_id == 201
        assert provider.eos_token_id == 202
        assert provider.pad_token_id == 203

    def test_freeze_options(self):
        """Test Qwen2AudioModelProvider with freeze options."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            freeze_language_model=True,
            freeze_audio_model=True,
            freeze_audio_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_audio_model is True
        assert provider.freeze_audio_projection is True

    def test_custom_hf_config(self):
        """Test Qwen2AudioModelProvider with custom hf_config."""
        dummy_config = {"text_config": {}, "audio_config": {}}
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            hf_config=dummy_config,
        )

        assert provider.hf_config is dummy_config

    def test_provide_method_exists(self):
        """Test that provide method exists and is callable."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_provide_language_model_method_exists(self):
        """Test that provide_language_model method exists and is callable."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)

    def test_inherit_from_qwen2_provider(self):
        """Test that Qwen2AudioModelProvider inherits Qwen2 configurations correctly."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            seq_length=8192,
            vocab_size=152064,
            rotary_base=500000.0,
        )

        # Check that inherited configurations work
        assert provider.seq_length == 8192
        assert provider.vocab_size == 152064
        assert provider.rotary_base == 500000.0

        # Qwen2 defaults should be inherited
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_qkv_bias is True
        assert provider.add_bias_linear is False

        # Audio-specific overrides should still work
        assert provider.scatter_embedding_sequence_parallel is False

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal valid configuration
        provider = Qwen2AudioModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=1,
        )

        assert provider.num_layers == 1
        assert provider.hidden_size == 64
        assert provider.num_attention_heads == 1
        assert provider.scatter_embedding_sequence_parallel is False

        # Test with large configuration
        provider_large = Qwen2AudioModelProvider(
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_query_groups=8,
        )

        assert provider_large.num_layers == 80
        assert provider_large.hidden_size == 8192
        assert provider_large.num_attention_heads == 64
        assert provider_large.num_query_groups == 8


class TestQwen2AudioModelProviderInheritance:
    """Test inheritance relationships for Qwen2AudioModelProvider."""

    def test_inherits_from_gpt_provider(self):
        """Test that Qwen2AudioModelProvider inherits from GPTModelProvider."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        assert issubclass(Qwen2AudioModelProvider, GPTModelProvider)

    def test_inherits_from_qwen2_provider(self):
        """Test that Qwen2AudioModelProvider inherits from Qwen2ModelProvider."""
        from megatron.bridge.models.qwen.qwen_provider import Qwen2ModelProvider

        assert issubclass(Qwen2AudioModelProvider, Qwen2ModelProvider)

    def test_provider_method_inheritance(self):
        """Test that inherited methods work correctly."""
        provider = Qwen2AudioModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Should inherit all Qwen2ModelProvider methods
        assert hasattr(provider, "provide")
        assert hasattr(provider, "provide_language_model")

        # Audio-specific fields should also exist
        assert hasattr(provider, "freeze_language_model")
        assert hasattr(provider, "freeze_audio_model")
        assert hasattr(provider, "freeze_audio_projection")
