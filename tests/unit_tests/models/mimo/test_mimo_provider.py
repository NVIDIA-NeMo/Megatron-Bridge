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

import pytest
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.mimo import MimoModelProvider


class TestMimoModelProviderInitialization:
    """Test cases for MimoModelProvider initialization."""

    def test_mimo_provider_initialization_with_defaults(self):
        """Test MimoModelProvider can be initialized with default values."""
        provider = MimoModelProvider()

        # Check MIMO-specific defaults
        assert provider.language_model_spec is None
        assert provider.modality_submodules_spec == {}
        assert provider.special_token_ids == {}
        assert provider.freeze_language_model is False
        assert provider.freeze_modality_encoders == {}
        assert provider.freeze_modality_projections == {}

    def test_mimo_provider_initialization_with_specs(self):
        """Test MimoModelProvider with custom specs."""
        from unittest.mock import Mock

        language_spec = ModuleSpec(module=Mock, params={})
        modality_spec = {"images": ModuleSpec(module=Mock, params={})}

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=modality_spec,
            special_token_ids={"images": 32000},
        )

        assert provider.language_model_spec == language_spec
        assert provider.modality_submodules_spec == modality_spec
        assert provider.special_token_ids == {"images": 32000}

    def test_mimo_provider_freeze_options(self):
        """Test MimoModelProvider with freeze options."""
        provider = MimoModelProvider(
            freeze_language_model=True,
            freeze_modality_encoders={"images": True},
            freeze_modality_projections={"images": False},
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_modality_encoders == {"images": True}
        assert provider.freeze_modality_projections == {"images": False}


class TestMimoModelProviderProvideMethod:
    """Test cases for MimoModelProvider.provide() method."""

    def test_provide_raises_error_without_language_model_spec(self):
        """Test that provide() validates language_model_spec is set."""
        provider = MimoModelProvider(
            language_model_spec=None,
        )

        with pytest.raises(ValueError, match="language_model_spec must be configured"):
            provider.provide()

    def test_provide_method_exists(self):
        """Test that provide method is callable."""
        provider = MimoModelProvider()

        assert hasattr(provider, "provide")
        assert callable(provider.provide)


class TestMimoModelProviderInheritance:
    """Test inheritance and method availability."""

    def test_mimo_provider_inherits_model_provider_mixin(self):
        """Test that MimoModelProvider inherits from ModelProviderMixin."""
        from megatron.bridge.models.model_provider import ModelProviderMixin

        assert issubclass(MimoModelProvider, ModelProviderMixin)

    def test_mimo_provider_has_provide_distributed_model(self):
        """Test that MimoModelProvider has provide_distributed_model method."""
        provider = MimoModelProvider()

        assert hasattr(provider, "provide_distributed_model")
        assert callable(provider.provide_distributed_model)

    def test_mimo_provider_has_from_hf_pretrained(self):
        """Test that MimoModelProvider has from_hf_pretrained classmethod."""
        assert hasattr(MimoModelProvider, "from_hf_pretrained")
        assert callable(MimoModelProvider.from_hf_pretrained)
