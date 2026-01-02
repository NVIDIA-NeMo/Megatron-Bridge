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

from unittest.mock import Mock

import pytest

from megatron.bridge.models.glm_vl.glm_45v_provider import GLM45VModelProvider


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for GLM-4.5V."""
    config = Mock()
    config.hidden_size = 1152
    config.intermediate_size = 4304
    config.num_hidden_layers = 27
    config.num_attention_heads = 16
    config.patch_size = 14
    config.image_size = 896
    return config


class TestGLM45VModelProvider:
    """Test cases for GLM45VModelProvider base class."""

    def test_glm_45v_model_provider_initialization(self, mock_vision_config):
        """Test GLM45VModelProvider can be initialized with default values."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        # Check inherited defaults from GLM45AirModelProvider106B
        # These are expected values for GLM-4.5 Air 106B base model
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False

    def test_glm_45v_vl_specific_defaults(self, mock_vision_config):
        """Test GLM45VModelProvider VL-specific default configuration."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        # Check VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.position_embedding_type == "mrope"
        assert provider.mrope_section == [8, 12, 12]

        # Check freeze options defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_glm_45v_token_ids_defaults(self, mock_vision_config):
        """Test GLM45VModelProvider token ID defaults."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        # Check token ID defaults
        assert provider.eos_token_id == 151329
        assert provider.image_start_token_id == 151339
        assert provider.image_end_token_id == 151340
        assert provider.video_start_token_id == 151341
        assert provider.video_end_token_id == 151342
        assert provider.image_token_id == 151363
        assert provider.video_token_id == 151364

    def test_glm_45v_custom_token_ids(self, mock_vision_config):
        """Test GLM45VModelProvider with custom token IDs."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
            eos_token_id=100,
            image_start_token_id=101,
            image_end_token_id=102,
            video_start_token_id=103,
            video_end_token_id=104,
            image_token_id=105,
            video_token_id=106,
        )

        assert provider.eos_token_id == 100
        assert provider.image_start_token_id == 101
        assert provider.image_end_token_id == 102
        assert provider.video_start_token_id == 103
        assert provider.video_end_token_id == 104
        assert provider.image_token_id == 105
        assert provider.video_token_id == 106

    def test_glm_45v_vision_config(self, mock_vision_config):
        """Test GLM45VModelProvider stores vision config."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        assert provider.vision_config is mock_vision_config

    def test_glm_45v_freeze_options(self, mock_vision_config):
        """Test GLM45VModelProvider with freeze options."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True

    def test_glm_45v_freeze_language_only(self, mock_vision_config):
        """Test GLM45VModelProvider with only language model frozen."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=False,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_glm_45v_freeze_vision_only(self, mock_vision_config):
        """Test GLM45VModelProvider with only vision model frozen."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
            freeze_language_model=False,
            freeze_vision_model=True,
            freeze_vision_projection=False,
        )

        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is False

    def test_glm_45v_freeze_projection_only(self, mock_vision_config):
        """Test GLM45VModelProvider with only projection frozen."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is True

    def test_glm_45v_provide_method_exists(self, mock_vision_config):
        """Test that provide method exists and is callable."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_glm_45v_provide_language_model_method_exists(self, mock_vision_config):
        """Test that provide_language_model method exists and is callable."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)

    def test_glm_45v_mrope_section(self, mock_vision_config):
        """Test GLM45VModelProvider MRoPE section configuration."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        # MRoPE section should be [8, 12, 12] for GLM-4.5V
        assert provider.mrope_section == [8, 12, 12]

    def test_glm_45v_custom_mrope_section(self, mock_vision_config):
        """Test GLM45VModelProvider with custom MRoPE section."""
        custom_mrope = [4, 16, 12]
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
            mrope_section=custom_mrope,
        )

        assert provider.mrope_section == custom_mrope

    def test_glm_45v_scatter_embedding_disabled(self, mock_vision_config):
        """Test GLM45VModelProvider has scatter_embedding_sequence_parallel disabled."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        # VL models shouldn't scatter embeddings across sequence parallel regions
        assert provider.scatter_embedding_sequence_parallel is False


class TestGLM45VModelProviderEdgeCases:
    """Test edge cases for GLM45VModelProvider."""

    def test_glm_45v_with_none_vision_config(self):
        """Test GLM45VModelProvider can be initialized without vision_config."""
        # The default factory should create a default config
        provider = GLM45VModelProvider()

        assert provider.vision_config is not None

    def test_glm_45v_position_embedding_type(self, mock_vision_config):
        """Test GLM45VModelProvider uses mrope position embedding."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        assert provider.position_embedding_type == "mrope"

    def test_glm_45v_different_vision_configs(self):
        """Test GLM45VModelProvider with different vision configs."""
        # Test with different vision hidden sizes
        for hidden_size in [768, 1024, 1152, 1536]:
            vision_config = Mock()
            vision_config.hidden_size = hidden_size

            provider = GLM45VModelProvider(
                vision_config=vision_config,
            )

            assert provider.vision_config.hidden_size == hidden_size


class TestGLM45VModelProviderInheritance:
    """Test inheritance behavior from GLM45AirModelProvider106B."""

    def test_glm_45v_inherits_from_air_provider(self, mock_vision_config):
        """Test GLM45VModelProvider inherits from GLM45AirModelProvider106B."""
        from megatron.bridge.models import GLM45AirModelProvider106B

        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        assert isinstance(provider, GLM45AirModelProvider106B)

    def test_glm_45v_overrides_position_embedding(self, mock_vision_config):
        """Test GLM45VModelProvider overrides position embedding type."""
        # VL provider should use mrope
        vl_provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        assert vl_provider.position_embedding_type == "mrope"

    def test_glm_45v_overrides_scatter_embedding(self, mock_vision_config):
        """Test GLM45VModelProvider overrides scatter_embedding_sequence_parallel."""
        provider = GLM45VModelProvider(
            vision_config=mock_vision_config,
        )

        # VL models should have scatter_embedding_sequence_parallel=False
        assert provider.scatter_embedding_sequence_parallel is False
