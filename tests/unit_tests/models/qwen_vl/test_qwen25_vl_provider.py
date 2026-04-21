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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models.qwen_vl import Qwen25VLModelProvider


class TestQwen25VLModelProvider:
    """Test cases for Qwen25VLModelProvider class."""

    def test_qwen25_vl_model_provider_initialization(self):
        """Test Qwen25VLModelProvider can be initialized with default values."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

        # Check Qwen2-inherited defaults
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

    def test_qwen25_vl_vl_specific_defaults(self):
        """Test Qwen25VLModelProvider VL-specific default configuration."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Check VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.position_embedding_type == "mrope"
        assert provider.mrope_section == [16, 24, 24]
        assert isinstance(provider.vision_config, Qwen2_5_VLVisionConfig)

        # Check token IDs
        assert provider.bos_token_id == 151643
        assert provider.eos_token_id == 151645
        assert provider.vision_start_token_id == 151652
        assert provider.vision_end_token_id == 151653
        assert provider.vision_token_id == 151654
        assert provider.image_token_id == 151655
        assert provider.video_token_id == 151656

        # Check freeze options defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_qwen25_vl_custom_vision_config(self):
        """Test Qwen25VLModelProvider with custom vision configuration."""
        custom_vision_config = Qwen2_5_VLVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
        )

        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vision_config=custom_vision_config,
        )

        assert provider.vision_config.hidden_size == 1024
        assert provider.vision_config.intermediate_size == 4096
        assert provider.vision_config.num_hidden_layers == 24
        assert provider.vision_config.num_attention_heads == 16

    def test_qwen25_vl_custom_token_ids(self):
        """Test Qwen25VLModelProvider with custom token IDs."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            bos_token_id=100,
            eos_token_id=101,
            vision_start_token_id=102,
            vision_end_token_id=103,
            vision_token_id=104,
            image_token_id=105,
            video_token_id=106,
        )

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.vision_start_token_id == 102
        assert provider.vision_end_token_id == 103
        assert provider.vision_token_id == 104
        assert provider.image_token_id == 105
        assert provider.video_token_id == 106

    def test_qwen25_vl_freeze_options(self):
        """Test Qwen25VLModelProvider with freeze options."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True

    def test_qwen25_vl_custom_mrope_configuration(self):
        """Test Qwen25VLModelProvider with custom mRoPE configuration."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            mrope_section=[8, 16, 16],
        )

        assert provider.mrope_section == [8, 16, 16]
        assert provider.position_embedding_type == "mrope"

    def test_qwen25_vl_inherit_from_qwen2_provider(self):
        """Test that Qwen25VLModelProvider inherits Qwen2 configurations correctly."""
        provider = Qwen25VLModelProvider(
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

        # VL-specific overrides should still work
        assert provider.position_embedding_type == "mrope"
        assert provider.scatter_embedding_sequence_parallel is False

    def test_qwen25_vl_provide_method_exists(self):
        """Test that provide method exists and is callable."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_qwen25_vl_provide_language_model_method_exists(self):
        """Test that provide_language_model method exists and is callable."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)

    def test_qwen25_vl_model_provider_ffn_hidden_size(self):
        """Test Qwen25VLModelProvider FFN hidden size calculation."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
        )

        assert provider.ffn_hidden_size == 11008

    def test_qwen25_vl_model_provider_group_query_attention(self):
        """Test Qwen25VLModelProvider with group query attention."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
        )

        assert provider.num_query_groups == 8

    def test_qwen25_vl_vision_config_default_type(self):
        """Test that default vision config is of correct type."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert isinstance(provider.vision_config, Qwen2_5_VLVisionConfig)

    def test_qwen25_vl_model_provider_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal valid configuration
        provider = Qwen25VLModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=1,
        )

        assert provider.num_layers == 1
        assert provider.hidden_size == 64
        assert provider.num_attention_heads == 1
        assert provider.position_embedding_type == "mrope"

        # Test with large configuration
        provider_large = Qwen25VLModelProvider(
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_query_groups=8,
        )

        assert provider_large.num_layers == 80
        assert provider_large.hidden_size == 8192
        assert provider_large.num_attention_heads == 64
        assert provider_large.num_query_groups == 8


class TestQwen25VLModelProviderInheritance:
    """Test inheritance relationships for Qwen25VLModelProvider."""

    def test_qwen25_vl_inherits_from_qwen2_provider(self):
        """Test that Qwen25VLModelProvider inherits from Qwen2ModelProvider."""
        from megatron.bridge.models import Qwen2ModelProvider

        assert issubclass(Qwen25VLModelProvider, Qwen2ModelProvider)

    def test_qwen25_vl_provider_method_inheritance(self):
        """Test that inherited methods work correctly."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Should inherit all Qwen2ModelProvider methods
        assert hasattr(provider, "provide")
        assert hasattr(provider, "provide_language_model")

        # VL-specific methods should also exist
        assert hasattr(provider, "freeze_language_model")
        assert hasattr(provider, "freeze_vision_model")
        assert hasattr(provider, "freeze_vision_projection")


class TestQwen25VLModelProviderSpecificConfiguration:
    """Test Qwen25VL-specific configuration scenarios."""

    def test_scatter_embedding_sequence_parallel_override(self):
        """Test that scatter_embedding_sequence_parallel can be overridden."""
        # Default should be False for VL models
        provider_default = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider_default.scatter_embedding_sequence_parallel is False

        # Should be able to override
        provider_override = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            scatter_embedding_sequence_parallel=True,
        )
        assert provider_override.scatter_embedding_sequence_parallel is True

    def test_position_embedding_mrope_requirement(self):
        """Test that position embedding type is always mrope for VL models."""
        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        # Should always be mrope for VL models
        assert provider.position_embedding_type == "mrope"

        # Should have mrope_section configured
        assert hasattr(provider, "mrope_section")
        assert isinstance(provider.mrope_section, list)
        assert len(provider.mrope_section) == 3

    def test_vision_config_customization(self):
        """Test vision config can be customized properly."""
        custom_config = Qwen2_5_VLVisionConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=32,
            image_size=448,
            patch_size=14,
        )

        provider = Qwen25VLModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vision_config=custom_config,
        )

        assert provider.vision_config.hidden_size == 2048
        assert provider.vision_config.intermediate_size == 8192
        assert provider.vision_config.num_hidden_layers == 32
        assert provider.vision_config.num_attention_heads == 32
        assert provider.vision_config.image_size == 448
        assert provider.vision_config.patch_size == 14
