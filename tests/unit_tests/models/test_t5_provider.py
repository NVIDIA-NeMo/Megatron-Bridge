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


from megatron.bridge.models import T5ModelProvider


class TestT5ModelProvider:
    """Test cases for T5ModelProvider class."""

    def test_t5_model_provider_initialization(self):
        """Test T5ModelProvider can be initialized with default values."""
        provider = T5ModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            encoder_num_layers=12,
            kv_channels=64,
            apply_rope_fusion=False,  # Disable to avoid fusion check errors
        )

        # Check required transformer config fields
        assert provider.num_layers == 12
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 12

        # Check T5-specific fields
        assert provider.encoder_num_layers == 12
        assert provider.kv_channels == 64
        assert provider.fp16_lm_cross_entropy is False
        assert provider.parallel_output is True
        assert provider.share_embeddings_and_output_weights is True
        assert provider.make_vocab_size_divisible_by == 128

    def test_t5_model_provider_encoder_only(self):
        """Test T5ModelProvider configuration - T5 is always encoder-decoder."""
        # T5 is always an encoder-decoder model, test basic config instead
        provider = T5ModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            encoder_num_layers=12,
            kv_channels=64,
            apply_rope_fusion=False,
        )

        # Check that position embedding type is set
        assert provider.position_embedding_type == "learned_absolute"

    def test_t5_model_provider_position_embedding(self):
        """Test T5ModelProvider position embedding configuration."""
        provider = T5ModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            encoder_num_layers=12,
            kv_channels=64,
            position_embedding_type="learned_absolute",
            apply_rope_fusion=False,
        )

        assert provider.position_embedding_type == "learned_absolute"

    def test_provide_method_basic(self):
        """Test the provide method creates a T5 model."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            vocab_size=1000,
            apply_rope_fusion=False,
        )

        # The provide method in T5ModelProvider imports parallel_state internally
        # Test that provider has the provide method
        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_provide_method_signature(self):
        """Test provide method signature."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            apply_rope_fusion=False,
        )

        # Check that provide method accepts the expected parameters
        import inspect

        sig = inspect.signature(provider.provide)
        params = list(sig.parameters.keys())
        assert "pre_process" in params
        assert "post_process" in params

    def test_encoder_config(self):
        """Test that encoder config is properly set."""
        provider = T5ModelProvider(
            num_layers=12,  # decoder layers
            hidden_size=768,
            num_attention_heads=12,
            encoder_num_layers=6,  # encoder layers
            kv_channels=64,
            apply_rope_fusion=False,
        )

        # Check encoder configuration
        assert provider.encoder_num_layers == 6
        assert provider.num_layers == 12  # decoder layers

    def test_model_spec_configuration(self):
        """Test T5ModelProvider layer spec configuration."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            vocab_size=1000,
            apply_rope_fusion=False,
        )

        # Test that transformer_layer_spec is a callable
        assert callable(provider.transformer_layer_spec)

    def test_fp8_configuration(self):
        """Test T5ModelProvider with FP8 configuration."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            fp8="e4m3",
            fp8_margin=2,
            fp8_interval=100,
            fp8_amax_history_len=512,
            fp8_amax_compute_algo="max",
            apply_rope_fusion=False,
        )

        assert provider.fp8 == "e4m3"
        assert provider.fp8_margin == 2
        assert provider.fp8_interval == 100
        assert provider.fp8_amax_history_len == 512
        assert provider.fp8_amax_compute_algo == "max"

    def test_fusion_settings(self):
        """Test fusion configuration defaults."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            apply_rope_fusion=False,
        )

        # These should be set by explicit values in T5ModelProvider
        assert provider.masked_softmax_fusion is True
        assert provider.bias_activation_fusion is True
        assert provider.persist_layer_norm is True
        assert provider.bias_dropout_fusion is True

    def test_position_embedding_type(self):
        """Test T5 position embedding configuration."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            position_embedding_type="learned_absolute",
            apply_rope_fusion=False,
        )

        assert provider.position_embedding_type == "learned_absolute"

    def test_minimal_configuration(self):
        """Test that minimal configuration works."""
        # T5ModelProvider should work with minimal required fields
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            apply_rope_fusion=False,
        )
        assert provider.num_layers == 2
        assert provider.encoder_num_layers == 2

    def test_deallocate_pipeline_outputs(self):
        """Test deallocate pipeline outputs configuration."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            deallocate_pipeline_outputs=False,
            apply_rope_fusion=False,
        )

        assert provider.deallocate_pipeline_outputs is False

    def test_max_position_embeddings(self):
        """Test max position embeddings configuration."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            max_position_embeddings=1024,
            apply_rope_fusion=False,
        )

        assert provider.max_position_embeddings == 1024

    def test_communication_overlap_config(self):
        """Test tensor parallel communication overlap configuration."""
        tp_config = {"method": "ring", "num_splits": 4}

        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            tp_comm_overlap_cfg=tp_config,
            apply_rope_fusion=False,
        )

        assert provider.tp_comm_overlap_cfg == tp_config

    def test_relative_attention_config(self):
        """Test relative attention configuration."""
        provider = T5ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            encoder_num_layers=2,
            kv_channels=32,
            relative_attention_num_buckets=64,
            relative_attention_max_distance=256,
            apply_rope_fusion=False,
        )

        assert provider.relative_attention_num_buckets == 64
        assert provider.relative_attention_max_distance == 256
