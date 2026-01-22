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


from megatron.bridge.models.mimo import LlavaMimoProvider


class TestLlavaMimoProviderInitialization:
    """Test cases for LlavaMimoProvider initialization."""

    def test_llava_provider_initialization_with_defaults(self):
        """Test LlavaMimoProvider has correct default configuration."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        # Check provider-level defaults
        assert provider.vocab_size == 32256
        assert provider.image_special_token_id == 32000
        assert provider.vision_projector_input_size == 1024  # CLIP ViT-L/14

        # Check language config was created
        assert provider.language_config is not None
        assert provider.language_config.num_layers == 32
        assert provider.language_config.hidden_size == 4096
        assert provider.language_config.num_attention_heads == 32
        assert provider.language_config.num_query_groups == 32
        assert provider.language_config.ffn_hidden_size == 11008
        assert provider.language_config.normalization == "RMSNorm"
        assert provider.language_config.gated_linear_unit is True
        assert provider.language_config.add_bias_linear is False

        # Check vision projection config was created
        assert provider.vision_projection_config is not None
        assert provider.vision_projection_config.num_layers == 2  # 2-layer MLP

    def test_llava_provider_post_init_creates_specs(self):
        """Test that __post_init__ creates MIMO specs."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        # Check that specs are created
        assert provider.language_model_spec is not None
        assert "images" in provider.modality_submodules_spec
        assert provider.special_token_ids == {"images": 32000}

    def test_llava_provider_language_model_spec(self):
        """Test that language model spec is configured correctly."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        assert provider.language_model_spec is not None
        assert provider.language_model_spec.module is not None

        # Check language model params
        params = provider.language_model_spec.params
        assert "config" in params
        assert "transformer_layer_spec" in params
        assert params["vocab_size"] == 32256
        assert params["max_sequence_length"] == 4096

    def test_llava_provider_vision_submodule_spec(self):
        """Test that vision submodule spec is configured correctly."""
        from unittest.mock import Mock

        # Need to provide vision_encoder_module since it's required
        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        assert "images" in provider.modality_submodules_spec
        vision_spec = provider.modality_submodules_spec["images"]

        assert vision_spec is not None
        assert "encoders" in vision_spec.submodules
        assert "input_projections" in vision_spec.submodules
        assert "clip_encoder" in vision_spec.submodules["encoders"]

    def test_llava_provider_custom_configuration(self):
        """Test LlavaMimoProvider with custom configuration."""
        from unittest.mock import Mock

        from megatron.bridge.models.transformer_config import TransformerConfig

        custom_lang_config = TransformerConfig(
            num_layers=16,
            hidden_size=2048,
            num_attention_heads=16,
        )

        provider = LlavaMimoProvider(
            language_config=custom_lang_config,
            vocab_size=50000,
            image_special_token_id=40000,
            vision_encoder_module=Mock,
        )

        assert provider.language_config.num_layers == 16
        assert provider.language_config.hidden_size == 2048
        assert provider.language_config.num_attention_heads == 16
        assert provider.vocab_size == 50000
        assert provider.special_token_ids == {"images": 40000}

    def test_llava_provider_custom_vision_encoder(self):
        """Test LlavaMimoProvider with custom vision encoder."""
        from unittest.mock import Mock

        custom_encoder = Mock
        custom_params = {"pretrained": True}

        provider = LlavaMimoProvider(
            vision_encoder_module=custom_encoder,
            vision_encoder_params=custom_params,
            vision_projector_input_size=768,
        )

        assert provider.vision_encoder_module == custom_encoder
        assert provider.vision_encoder_params == custom_params
        assert provider.vision_projector_input_size == 768


class TestLlavaMimoProviderMethods:
    """Test cases for LlavaMimoProvider methods."""

    def test_llava_provider_has_provide_method(self):
        """Test that LlavaMimoProvider has provide method."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_llava_provider_language_config_creation(self):
        """Test language config is created properly."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        assert provider.language_config.num_layers == 32
        assert provider.language_config.hidden_size == 4096
        assert provider.language_config.num_attention_heads == 32
        assert provider.language_config.normalization == "RMSNorm"

    def test_llava_provider_vision_projection_config_creation(self):
        """Test vision projection config is created properly."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        assert provider.vision_projection_config.num_layers == 2  # 2-layer MLP
        assert provider.vision_projection_config.hidden_size == 4096  # Matches language model

    def test_llava_provider_vision_projection_spec_creation(self):
        """Test vision projection spec is created properly."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)
        projection_spec = provider._get_vision_projection_spec()

        assert projection_spec is not None
        assert projection_spec.params["projector_type"] == "mlp"
        assert projection_spec.params["input_size"] == 1024


class TestLlavaMimoProviderInheritance:
    """Test inheritance relationships."""

    def test_llava_provider_inherits_from_mimo_provider(self):
        """Test that LlavaMimoProvider inherits from MimoModelProvider."""
        from megatron.bridge.models.mimo import MimoModelProvider

        assert issubclass(LlavaMimoProvider, MimoModelProvider)

    def test_llava_provider_inherits_model_provider_mixin(self):
        """Test that LlavaMimoProvider inherits from ModelProviderMixin."""
        from megatron.bridge.models.model_provider import ModelProviderMixin

        assert issubclass(LlavaMimoProvider, ModelProviderMixin)

    def test_llava_provider_has_inherited_methods(self):
        """Test that LlavaMimoProvider has inherited methods."""
        from unittest.mock import Mock

        provider = LlavaMimoProvider(vision_encoder_module=Mock)

        # From ModelProviderMixin
        assert hasattr(provider, "provide_distributed_model")
        assert hasattr(provider, "initialize_model_parallel")

        # From MimoModelProvider
        assert hasattr(provider, "freeze_language_model")
        assert hasattr(provider, "freeze_modality_encoders")


class TestLlavaMimoProviderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_llava_provider_with_minimal_config(self):
        """Test LlavaMimoProvider with minimal configuration."""
        from unittest.mock import Mock

        from megatron.bridge.models.transformer_config import TransformerConfig

        minimal_config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=1,
        )

        provider = LlavaMimoProvider(language_config=minimal_config, vision_encoder_module=Mock)

        assert provider.language_config.num_layers == 1
        assert provider.language_config.hidden_size == 64
        assert provider.language_model_spec is not None

    def test_llava_provider_precision_configuration(self):
        """Test LlavaMimoProvider with precision settings."""
        from unittest.mock import Mock

        import torch

        from megatron.bridge.models.transformer_config import TransformerConfig

        lang_config = TransformerConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        provider = LlavaMimoProvider(language_config=lang_config, vision_encoder_module=Mock)

        # Check that precision is set in config
        assert provider.language_config.bf16 is True
        assert provider.language_config.params_dtype == torch.bfloat16

    def test_llava_provider_with_different_vocab_sizes(self):
        """Test LlavaMimoProvider with different vocab sizes."""
        from unittest.mock import Mock

        provider_small = LlavaMimoProvider(vocab_size=10000, vision_encoder_module=Mock)
        provider_large = LlavaMimoProvider(vocab_size=100000, vision_encoder_module=Mock)

        assert provider_small.vocab_size == 10000
        assert provider_large.vocab_size == 100000

        # Check specs reflect the vocab size
        assert provider_small.language_model_spec.params["vocab_size"] == 10000
        assert provider_large.language_model_spec.params["vocab_size"] == 100000

    def test_llava_provider_requires_vision_encoder(self):
        """Test that LlavaMimoProvider raises error without vision encoder."""
        import pytest

        with pytest.raises(ValueError, match="vision_encoder_module must be provided"):
            _ = LlavaMimoProvider()
            # Error is raised in __post_init__ when building specs
