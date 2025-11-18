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

import math

import torch
from transformers import SiglipVisionConfig

from megatron.bridge.models.gemma_vl.gemma3_vl_provider import Gemma3VLModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma3_vl import Gemma3VLMultimodalProjectorConfig


class TestGemma3VLModelProvider:
    """Test cases for Gemma3VLModelProvider class."""

    def test_gemma3_vl_model_provider_initialization(self):
        """Test Gemma3VLModelProvider can be initialized with default values."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        # Check required transformer config fields
        assert provider.num_layers == 28
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 10

        # Check Gemma3-inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.seq_length == 131_072  # Gemma3 default
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is True
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == (10_000, 1_000_000)  # Gemma3 tuple format
        assert provider.window_size == 512
        assert provider.interleaved_attn_pattern == (5, 1)
        assert provider.rope_scaling_factor == 1.0
        assert provider.softmax_scale == 1.0 / math.sqrt(256)

    def test_gemma3_vl_vl_specific_defaults(self):
        """Test Gemma3VLModelProvider VL-specific default configuration."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        # Check VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert isinstance(provider.vision_config, SiglipVisionConfig)
        assert isinstance(provider.vision_projector_config, Gemma3VLMultimodalProjectorConfig)
        assert provider.mm_tokens_per_image == 256

        # Check token IDs
        assert provider.bos_token_id == 2
        assert provider.eos_token_id == 1
        assert provider.vision_start_token_id == 255999
        assert provider.vision_end_token_id == 256000
        assert provider.image_token_id == 262144

        # Check freeze options defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_gemma3_vl_custom_vision_config(self):
        """Test Gemma3VLModelProvider with custom vision configuration."""
        custom_vision_config = SiglipVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=448,
            patch_size=14,
        )

        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            vision_config=custom_vision_config,
        )

        assert provider.vision_config.hidden_size == 1024
        assert provider.vision_config.intermediate_size == 4096
        assert provider.vision_config.num_hidden_layers == 24
        assert provider.vision_config.num_attention_heads == 16
        assert provider.vision_config.image_size == 448
        assert provider.vision_config.patch_size == 14

    def test_gemma3_vl_custom_vision_projector_config(self):
        """Test Gemma3VLModelProvider with custom vision projector configuration."""
        custom_projector_config = Gemma3VLMultimodalProjectorConfig(
            input_size=768,
            hidden_size=2048,
            tokens_per_image=512,
        )

        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            vision_projector_config=custom_projector_config,
        )

        assert provider.vision_projector_config.input_size == 768
        assert provider.vision_projector_config.hidden_size == 2048
        assert provider.vision_projector_config.tokens_per_image == 512

    def test_gemma3_vl_custom_token_ids(self):
        """Test Gemma3VLModelProvider with custom token IDs."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            bos_token_id=100,
            eos_token_id=101,
            vision_start_token_id=102,
            vision_end_token_id=103,
            image_token_id=104,
        )

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.vision_start_token_id == 102
        assert provider.vision_end_token_id == 103
        assert provider.image_token_id == 104

    def test_gemma3_vl_freeze_options(self):
        """Test Gemma3VLModelProvider with freeze options."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True

    def test_gemma3_vl_custom_rotary_configuration(self):
        """Test Gemma3VLModelProvider with custom rotary configuration."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            rotary_base=(5000, 500000.0),
            rope_scaling_factor=2.0,
        )

        assert provider.rotary_base == (5000, 500000.0)
        assert provider.rope_scaling_factor == 2.0

    def test_gemma3_vl_custom_window_size(self):
        """Test Gemma3VLModelProvider with custom window size."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            window_size=1024,
        )

        assert provider.window_size == 1024

    def test_gemma3_vl_custom_interleaved_attention_pattern(self):
        """Test Gemma3VLModelProvider with custom interleaved attention pattern."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            interleaved_attn_pattern=(3, 2),
        )

        assert provider.interleaved_attn_pattern == (3, 2)

    def test_gemma3_vl_inherit_from_gemma3_provider(self):
        """Test that Gemma3VLModelProvider inherits Gemma3 configurations correctly."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            seq_length=65536,
            vocab_size=262144,
            softmax_scale=1.0 / math.sqrt(128),
        )

        # Check that inherited configurations work
        assert provider.seq_length == 65536
        assert provider.vocab_size == 262144
        assert provider.softmax_scale == 1.0 / math.sqrt(128)

        # VL-specific overrides should still work
        assert provider.scatter_embedding_sequence_parallel is False
        assert isinstance(provider.vision_config, SiglipVisionConfig)

    def test_gemma3_vl_provide_method_exists(self):
        """Test that provide method exists and is callable."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_gemma3_vl_provide_language_model_method_exists(self):
        """Test that provide_language_model method exists and is callable."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)

    def test_gemma3_vl_model_provider_ffn_hidden_size(self):
        """Test Gemma3VLModelProvider FFN hidden size calculation."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            ffn_hidden_size=15360,
        )

        assert provider.ffn_hidden_size == 15360

    def test_gemma3_vl_model_provider_group_query_attention(self):
        """Test Gemma3VLModelProvider with group query attention."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            num_query_groups=5,
        )

        assert provider.num_query_groups == 5

    def test_gemma3_vl_vision_config_default_type(self):
        """Test that default vision config is of correct type."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        assert isinstance(provider.vision_config, SiglipVisionConfig)

    def test_gemma3_vl_vision_projector_config_default_type(self):
        """Test that default vision projector config is of correct type."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        assert isinstance(provider.vision_projector_config, Gemma3VLMultimodalProjectorConfig)

    def test_gemma3_vl_model_provider_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal valid configuration
        provider = Gemma3VLModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=1,
        )

        assert provider.num_layers == 1
        assert provider.hidden_size == 64
        assert provider.num_attention_heads == 1
        assert provider.position_embedding_type == "rope"

        # Test with large configuration
        provider_large = Gemma3VLModelProvider(
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_query_groups=8,
        )

        assert provider_large.num_layers == 80
        assert provider_large.hidden_size == 8192
        assert provider_large.num_attention_heads == 64
        assert provider_large.num_query_groups == 8

    def test_gemma3_vl_custom_mm_tokens_per_image(self):
        """Test Gemma3VLModelProvider with custom multimodal tokens per image."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            mm_tokens_per_image=512,
        )

        assert provider.mm_tokens_per_image == 512

    def test_gemma3_vl_precision_configuration(self):
        """Test Gemma3VLModelProvider precision configuration."""
        # Test default precision (should inherit Gemma3 defaults)
        provider_default = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        assert provider_default.bf16 is True
        assert provider_default.fp16 is False
        assert provider_default.params_dtype == torch.bfloat16

        # Test custom precision
        provider_fp16 = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            fp16=True,
            bf16=False,
            params_dtype=torch.float16,
        )

        assert provider_fp16.fp16 is True
        assert provider_fp16.bf16 is False
        assert provider_fp16.params_dtype == torch.float16


class TestGemma3VLModelProviderInheritance:
    """Test inheritance relationships for Gemma3VLModelProvider."""

    def test_gemma3_vl_inherits_from_gemma3_provider(self):
        """Test that Gemma3VLModelProvider inherits from Gemma3ModelProvider."""
        from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider

        assert issubclass(Gemma3VLModelProvider, Gemma3ModelProvider)

    def test_gemma3_vl_provider_method_inheritance(self):
        """Test that inherited methods work correctly."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        # Should inherit all Gemma3ModelProvider methods
        assert hasattr(provider, "provide")
        assert hasattr(provider, "provide_language_model")

        # VL-specific attributes should also exist
        assert hasattr(provider, "freeze_language_model")
        assert hasattr(provider, "freeze_vision_model")
        assert hasattr(provider, "freeze_vision_projection")
        assert hasattr(provider, "vision_config")
        assert hasattr(provider, "vision_projector_config")


class TestGemma3VLModelProviderSpecificConfiguration:
    """Test Gemma3VL-specific configuration scenarios."""

    def test_scatter_embedding_sequence_parallel_override(self):
        """Test that scatter_embedding_sequence_parallel can be overridden."""
        # Default should be False for VL models
        provider_default = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )
        assert provider_default.scatter_embedding_sequence_parallel is False

        # Should be able to override
        provider_override = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            scatter_embedding_sequence_parallel=True,
        )
        assert provider_override.scatter_embedding_sequence_parallel is True

    def test_position_embedding_rope_requirement(self):
        """Test that position embedding type is rope for Gemma3 VL models."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
        )

        # Should be rope for Gemma3 models
        assert provider.position_embedding_type == "rope"

        # Should have rotary_base configured as tuple
        assert hasattr(provider, "rotary_base")
        assert isinstance(provider.rotary_base, tuple)
        assert len(provider.rotary_base) == 2

    def test_vision_config_customization(self):
        """Test vision config can be customized properly."""
        custom_config = SiglipVisionConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=32,
            image_size=896,
            patch_size=14,
        )

        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            vision_config=custom_config,
        )

        assert provider.vision_config.hidden_size == 2048
        assert provider.vision_config.intermediate_size == 8192
        assert provider.vision_config.num_hidden_layers == 32
        assert provider.vision_config.num_attention_heads == 32
        assert provider.vision_config.image_size == 896
        assert provider.vision_config.patch_size == 14

    def test_vision_projector_config_customization(self):
        """Test vision projector config can be customized properly."""
        custom_projector_config = Gemma3VLMultimodalProjectorConfig(
            input_size=2048,
            hidden_size=4096,
            image_size=448,
            patch_dim=16,
            tokens_per_image=512,
        )

        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            vision_projector_config=custom_projector_config,
        )

        assert provider.vision_projector_config.input_size == 2048
        assert provider.vision_projector_config.hidden_size == 4096
        assert provider.vision_projector_config.image_size == 448
        assert provider.vision_projector_config.patch_dim == 16
        assert provider.vision_projector_config.tokens_per_image == 512

    def test_gemma3_specific_attention_configuration(self):
        """Test Gemma3-specific attention configurations."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            window_size=1024,
            interleaved_attn_pattern=(3, 2),
            softmax_scale=1.0 / math.sqrt(128),
        )

        # Check Gemma3-specific attention settings
        assert provider.window_size == 1024
        assert provider.interleaved_attn_pattern == (3, 2)
        assert provider.softmax_scale == 1.0 / math.sqrt(128)

    def test_gemma3_rotary_embedding_configuration(self):
        """Test Gemma3-specific rotary embedding configuration."""
        provider = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            rotary_base=(5000, 500000.0),
            rope_scaling_factor=1.5,
        )

        # Check Gemma3-specific rotary settings
        assert provider.rotary_base == (5000, 500000.0)
        assert provider.rope_scaling_factor == 1.5

    def test_gemma3_vl_model_size_variants(self):
        """Test different Gemma3 VL model size configurations."""
        # Test 1B-like configuration
        provider_1b = Gemma3VLModelProvider(
            num_layers=26,
            hidden_size=1152,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=256,
            ffn_hidden_size=6912,
            vocab_size=262144,
        )

        assert provider_1b.num_layers == 26
        assert provider_1b.hidden_size == 1152
        assert provider_1b.num_attention_heads == 4
        assert provider_1b.num_query_groups == 1

        # Test 4B-like configuration
        provider_4b = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            num_query_groups=10,
            kv_channels=256,
            ffn_hidden_size=15360,
            vocab_size=262144,
        )

        assert provider_4b.num_layers == 28
        assert provider_4b.hidden_size == 2560
        assert provider_4b.num_attention_heads == 10
        assert provider_4b.num_query_groups == 10

    def test_gemma3_vl_freeze_combinations(self):
        """Test different combinations of freeze options."""
        # Test freezing only language model
        provider_freeze_lang = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=False,
        )

        assert provider_freeze_lang.freeze_language_model is True
        assert provider_freeze_lang.freeze_vision_model is False
        assert provider_freeze_lang.freeze_vision_projection is False

        # Test freezing only vision components
        provider_freeze_vision = Gemma3VLModelProvider(
            num_layers=28,
            hidden_size=2560,
            num_attention_heads=10,
            freeze_language_model=False,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider_freeze_vision.freeze_language_model is False
        assert provider_freeze_vision.freeze_vision_model is True
        assert provider_freeze_vision.freeze_vision_projection is True
