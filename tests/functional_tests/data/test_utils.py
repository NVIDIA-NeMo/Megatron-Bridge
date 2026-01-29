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
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

from megatron.bridge.data.utils import (
    finetuning_train_valid_test_datasets_provider,
    get_dataset_provider,
    pretrain_train_valid_test_datasets_provider,
)
from megatron.bridge.training.config import (
    DatasetBuildContext,
    DatasetProvider,
    FinetuningDatasetConfig,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer, build_tokenizer


class TestDataUtils:
    def test_pretrain_train_valid_test_datasets_provider(self, ensure_test_data):
        # Build tokenizer
        tokenizer = build_tokenizer(
            tokenizer_config=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072),
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        data_path = f"{ensure_test_data}/datasets/train/test_text_document"
        # Configure dataset
        dataset_config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=8192,
            split="950,45,5",
            tokenizer=tokenizer,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            blend=[[data_path, data_path], [0.3, 0.7]],
        )

        # Get datasets
        train_ds, valid_ds, test_ds = pretrain_train_valid_test_datasets_provider(
            train_val_test_num_samples=[1000, 100, 10],
            dataset_config=dataset_config,
        )

        assert train_ds.weights == [0.3, 0.7]
        assert (train_ds.size, valid_ds.size, test_ds.size) == (1000, 100, 10)

    def test_finetuning_train_valid_test_datasets_provider(self, ensure_test_data):
        # Configure dataset
        data_path = ensure_test_data
        dataset_config = FinetuningDatasetConfig(
            dataset_root=f"{data_path}/datasets/finetune_train",
            seq_length=8192,
        )

        # Build tokenizer
        tokenizer = build_tokenizer(
            tokenizer_config=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072),
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        # Get datasets
        train_ds, valid_ds, test_ds = finetuning_train_valid_test_datasets_provider(
            train_val_test_num_samples=[1000, 100, 10],
            dataset_config=dataset_config,
            tokenizer=tokenizer,
        )

        assert (valid_ds, test_ds) == (None, None)

        # Configure dataset
        data_path = ensure_test_data
        dataset_config = FinetuningDatasetConfig(
            dataset_root=f"{data_path}/datasets/finetune",
            seq_length=8192,
        )

        # Get datasets
        train_ds, valid_ds, test_ds = finetuning_train_valid_test_datasets_provider(
            train_val_test_num_samples=[1000, 100, 10],
            dataset_config=dataset_config,
            tokenizer=tokenizer,
        )

        assert (valid_ds, test_ds) != (None, None)
        assert train_ds.max_seq_length == 8192


class TestProtocolAdapterBehavior:
    """Test the behavior of the protocol adapter function."""

    def test_adapter_preserves_legacy_signature(self):
        """Test that the protocol adapter maintains the legacy function signature."""

        @dataclass
        class TestProvider(DatasetProvider):
            seq_length: int = 512

            def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
                # Return mock datasets with context information
                train_ds = MagicMock()
                train_ds.context_info = {"train_samples": context.train_samples, "tokenizer": context.tokenizer}
                return train_ds, None, None

        provider_config = TestProvider()
        adapter_func = get_dataset_provider(provider_config)

        # Test legacy call pattern: (samples_list, config, tokenizer=None)
        tokenizer = MagicMock(spec=MegatronTokenizer)
        samples = [1000, 100, 50]

        # Call with positional args
        result1 = adapter_func(samples, provider_config, tokenizer)

        # Call with keyword args
        result2 = adapter_func(samples, provider_config, tokenizer=tokenizer)

        # Both should work and produce same results
        assert result1[0].context_info["train_samples"] == 1000
        assert result1[0].context_info["tokenizer"] is tokenizer
        assert result2[0].context_info["train_samples"] == 1000
        assert result2[0].context_info["tokenizer"] is tokenizer

    def test_adapter_handles_missing_tokenizer(self):
        """Test that adapter handles cases where tokenizer is not provided."""

        @dataclass
        class TestProvider(DatasetProvider):
            seq_length: int = 512

            def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
                train_ds = MagicMock()
                train_ds.has_tokenizer = context.tokenizer is not None
                return train_ds, None, None

        provider_config = TestProvider()
        adapter_func = get_dataset_provider(provider_config)

        # Call without tokenizer
        samples = [1000, 100, 50]
        result = adapter_func(samples, provider_config)

        assert result[0].has_tokenizer is False

    def test_adapter_error_handling(self):
        """Test that adapter properly propagates errors from build_datasets."""

        @dataclass
        class ErrorProvider(DatasetProvider):
            seq_length: int = 512

            def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
                raise ValueError("Custom dataset error")

        provider_config = ErrorProvider()
        adapter_func = get_dataset_provider(provider_config)

        with pytest.raises(ValueError, match="Custom dataset error"):
            adapter_func([1000, 100, 50], provider_config)
