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

"""Unit tests for the custom dataset provider protocol."""

from dataclasses import dataclass
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider, FinetuningDatasetConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


class TestDatasetBuildContext:
    """Test the DatasetBuildContext dataclass."""

    def test_context_creation(self):
        """Test creating a DatasetBuildContext with all fields."""
        tokenizer = MagicMock(spec=MegatronTokenizer)
        context = DatasetBuildContext(train_samples=1000, valid_samples=100, test_samples=50, tokenizer=tokenizer)

        assert context.train_samples == 1000
        assert context.valid_samples == 100
        assert context.test_samples == 50
        assert context.tokenizer is tokenizer

    def test_context_creation_without_tokenizer(self):
        """Test creating a DatasetBuildContext without tokenizer."""
        context = DatasetBuildContext(train_samples=1000, valid_samples=100, test_samples=50)

        assert context.train_samples == 1000
        assert context.valid_samples == 100
        assert context.test_samples == 50
        assert context.tokenizer is None

    def test_context_is_frozen(self):
        """Test that DatasetBuildContext is frozen (immutable)."""
        context = DatasetBuildContext(train_samples=1000, valid_samples=100, test_samples=50)

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.7+
            context.train_samples = 2000


class TestDatasetProvider:
    """Test the DatasetProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that DatasetProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DatasetProvider()

    def test_concrete_implementation_works(self):
        """Test that concrete implementations of DatasetProvider work correctly."""

        @dataclass
        class MockDatasetProvider(DatasetProvider):
            """Mock implementation for testing."""

            seq_length: int = 512
            seed: int = 1234

            def build_datasets(
                self, context: DatasetBuildContext
            ) -> Tuple[Any | None, Any | None, Any | None]:
                # Mock dataset objects
                train_ds = MagicMock()
                train_ds.name = "train"
                train_ds.samples = context.train_samples

                valid_ds = MagicMock()
                valid_ds.name = "valid"
                valid_ds.samples = context.valid_samples

                test_ds = MagicMock()
                test_ds.name = "test"
                test_ds.samples = context.test_samples

                return train_ds, valid_ds, test_ds

        # Test instantiation
        provider = MockDatasetProvider(seq_length=1024, seed=5678)
        assert provider.seq_length == 1024
        assert provider.seed == 5678

        # Test inherited DataloaderConfig fields
        assert hasattr(provider, "dataloader_type")
        assert hasattr(provider, "num_workers")
        assert hasattr(provider, "data_sharding")
        assert hasattr(provider, "pin_memory")
        assert hasattr(provider, "persistent_workers")

        # Test build_datasets method
        context = DatasetBuildContext(train_samples=1000, valid_samples=100, test_samples=50)

        train_ds, valid_ds, test_ds = provider.build_datasets(context)

        assert train_ds.name == "train"
        assert train_ds.samples == 1000
        assert valid_ds.name == "valid"
        assert valid_ds.samples == 100
        assert test_ds.name == "test"
        assert test_ds.samples == 50

    def test_build_datasets_with_none_returns(self):
        """Test that build_datasets can return None for unused splits."""

        @dataclass
        class TrainOnlyDatasetProvider(DatasetProvider):
            """Provider that only creates training data."""

            seq_length: int = 512

            def build_datasets(
                self, context: DatasetBuildContext
            ) -> Tuple[Any | None, Any | None, Any | None]:
                train_ds = MagicMock()
                train_ds.name = "train"
                return train_ds, None, None

        provider = TrainOnlyDatasetProvider()
        context = DatasetBuildContext(train_samples=1000, valid_samples=0, test_samples=0)

        train_ds, valid_ds, test_ds = provider.build_datasets(context)

        assert train_ds is not None
        assert train_ds.name == "train"
        assert valid_ds is None
        assert test_ds is None

    def test_must_implement_build_datasets(self):
        """Test that concrete classes must implement build_datasets method."""

        @dataclass
        class IncompleteProvider(DatasetProvider):
            """Provider that doesn't implement build_datasets."""

            seq_length: int = 512
            # Missing build_datasets implementation

        with pytest.raises(TypeError):
            IncompleteProvider()


class TestDatasetProviderIntegration:
    """Test integration with the dataset provider system."""

    def test_get_dataset_provider_with_protocol(self):
        """Test that get_dataset_provider correctly handles DatasetProvider instances."""

        @dataclass
        class CustomDatasetProvider(DatasetProvider):
            """Custom provider for testing."""

            seq_length: int = 512
            custom_field: str = "test"

            def build_datasets(
                self, context: DatasetBuildContext
            ) -> Tuple[Any | None, Any | None, Any | None]:
                train_ds = MagicMock()
                train_ds.context = context
                train_ds.config = self
                return train_ds, None, None

        provider_config = CustomDatasetProvider(seq_length=1024, custom_field="integration_test")
        provider_func = get_dataset_provider(provider_config)

        # Test that we get a callable
        assert callable(provider_func)

        # Test calling the provider function with the expected signature
        tokenizer = MagicMock(spec=MegatronTokenizer)
        train_val_test_num_samples = [1000, 100, 50]

        train_ds, valid_ds, test_ds = provider_func(train_val_test_num_samples, provider_config, tokenizer=tokenizer)

        # Verify the results
        assert train_ds is not None
        assert valid_ds is None
        assert test_ds is None

        # Verify the context was created correctly
        assert train_ds.context.train_samples == 1000
        assert train_ds.context.valid_samples == 100
        assert train_ds.context.test_samples == 50
        assert train_ds.context.tokenizer is tokenizer

        # Verify the config was passed correctly
        assert train_ds.config.seq_length == 1024
        assert train_ds.config.custom_field == "integration_test"

    def test_get_dataset_provider_fallback_to_registry(self):
        """Test that get_dataset_provider falls back to registry for non-protocol configs."""

        # Create a mock FinetuningDatasetConfig
        finetuning_config = MagicMock(spec=FinetuningDatasetConfig)

        # Mock the registry to return a specific function
        mock_provider = MagicMock()

        with patch("megatron.bridge.data.utils._REGISTRY", {type(finetuning_config): mock_provider}):
            provider_func = get_dataset_provider(finetuning_config)
            assert provider_func is mock_provider

    def test_protocol_adapter_signature(self):
        """Test that the protocol adapter has the correct signature for legacy compatibility."""

        @dataclass
        class TestProvider(DatasetProvider):
            seq_length: int = 512

            def build_datasets(
                self, context: DatasetBuildContext
            ) -> Tuple[Any | None, Any | None, Any | None]:
                # Store the context for verification
                self._last_context = context
                return MagicMock(), MagicMock(), MagicMock()

        provider_config = TestProvider()
        provider_func = get_dataset_provider(provider_config)

        # Test calling with positional arguments (legacy style)
        train_val_test_num_samples = [500, 50, 25]
        tokenizer = MagicMock(spec=MegatronTokenizer)

        result = provider_func(train_val_test_num_samples, provider_config, tokenizer)

        # Verify the adapter correctly created the context
        assert provider_config._last_context.train_samples == 500
        assert provider_config._last_context.valid_samples == 50
        assert provider_config._last_context.test_samples == 25
        assert provider_config._last_context.tokenizer is tokenizer

        # Verify return type
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_protocol_adapter_without_tokenizer(self):
        """Test protocol adapter works without tokenizer parameter."""

        @dataclass
        class TestProvider(DatasetProvider):
            seq_length: int = 512

            def build_datasets(
                self, context: DatasetBuildContext
            ) -> Tuple[Any | None, Any | None, Any | None]:
                self._last_context = context
                return MagicMock(), MagicMock(), MagicMock()

        provider_config = TestProvider()
        provider_func = get_dataset_provider(provider_config)

        # Test calling without tokenizer
        train_val_test_num_samples = [500, 50, 25]
        _ = provider_func(train_val_test_num_samples, provider_config)

        # Verify the adapter correctly handled missing tokenizer
        assert provider_config._last_context.train_samples == 500
        assert provider_config._last_context.valid_samples == 50
        assert provider_config._last_context.test_samples == 25
        assert provider_config._last_context.tokenizer is None
