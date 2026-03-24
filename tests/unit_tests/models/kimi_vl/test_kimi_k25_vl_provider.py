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

from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for Kimi K2.5 VL."""
    config = Mock()
    config.hidden_size = 1152
    config.merge_kernel_size = (2, 2)
    return config


class TestKimiK25VLModelProvider:
    """Test cases for KimiK25VLModelProvider."""

    def test_vl_specific_defaults(self, mock_vision_config):
        """Test VL-specific default configuration."""
        provider = KimiK25VLModelProvider(vision_config=mock_vision_config)

        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_token_id_defaults(self, mock_vision_config):
        """Test token ID defaults."""
        provider = KimiK25VLModelProvider(vision_config=mock_vision_config)

        assert provider.media_placeholder_token_id == 163605
        assert provider.bos_token_id == 163584
        assert provider.eos_token_id == 163585
        assert provider.pad_token_id == 163839
        assert provider.image_token_id == 163605
        assert provider.ignore_index == -100

    def test_custom_token_ids(self, mock_vision_config):
        """Test provider with custom token IDs."""
        provider = KimiK25VLModelProvider(
            vision_config=mock_vision_config,
            media_placeholder_token_id=100,
            bos_token_id=101,
            eos_token_id=102,
            pad_token_id=103,
            image_token_id=104,
            ignore_index=-200,
        )

        assert provider.media_placeholder_token_id == 100
        assert provider.bos_token_id == 101
        assert provider.eos_token_id == 102
        assert provider.pad_token_id == 103
        assert provider.image_token_id == 104
        assert provider.ignore_index == -200

    def test_vision_config_stored(self, mock_vision_config):
        """Test vision config is stored."""
        provider = KimiK25VLModelProvider(vision_config=mock_vision_config)
        assert provider.vision_config is mock_vision_config

    def test_freeze_all(self, mock_vision_config):
        """Test freeze all options."""
        provider = KimiK25VLModelProvider(
            vision_config=mock_vision_config,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True

    def test_freeze_language_only(self, mock_vision_config):
        """Test freeze language model only."""
        provider = KimiK25VLModelProvider(
            vision_config=mock_vision_config,
            freeze_language_model=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_provide_methods_exist(self, mock_vision_config):
        """Test that provide methods exist and are callable."""
        provider = KimiK25VLModelProvider(vision_config=mock_vision_config)

        assert hasattr(provider, "provide")
        assert callable(provider.provide)
        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)


class TestKimiK25VLModelProviderInheritance:
    """Test inheritance behavior."""

    def test_inherits_from_mla_provider(self, mock_vision_config):
        """Test KimiK25VLModelProvider inherits from MLAModelProvider."""
        from megatron.bridge.models.mla_provider import MLAModelProvider

        provider = KimiK25VLModelProvider(vision_config=mock_vision_config)
        assert isinstance(provider, MLAModelProvider)

    def test_accepts_language_model_kwargs(self, mock_vision_config):
        """Test that language model params can be passed as kwargs (from provider_bridge)."""
        provider = KimiK25VLModelProvider(
            vision_config=mock_vision_config,
            num_layers=61,
            hidden_size=7168,
            num_moe_experts=384,
        )
        assert provider.num_layers == 61
        assert provider.hidden_size == 7168
        assert provider.num_moe_experts == 384
