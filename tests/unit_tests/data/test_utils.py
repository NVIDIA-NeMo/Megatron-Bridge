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

import json
import os
import subprocess

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

from megatron.bridge.data.utils import (
    finetuning_train_valid_test_datasets_provider,
    pretrain_train_valid_test_datasets_provider,
)
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


DATA_PATH = "/home/data/datasets/train/test_text_document"


class TestDataUtils:
    def test_pretrain_train_valid_test_datasets_provider(self):
        # Build tokenizer
        tokenizer = build_tokenizer(
            tokenizer_config=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072),
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        # Configure dataset
        dataset_config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=8192,
            split="950,45,5",
            tokenizer=tokenizer,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            blend=[[DATA_PATH, DATA_PATH], [0.3, 0.7]],
        )

        # Get datasets
        train_ds, valid_ds, test_ds = pretrain_train_valid_test_datasets_provider(
            train_val_test_num_samples=[1000, 100, 10],
            dataset_config=dataset_config,
        )

        assert train_ds.weights == [0.3, 0.7]
        assert (train_ds.size, valid_ds.size, test_ds.size) == (1000, 100, 10)

    def test_finetuning_train_valid_test_datasets_provider(self):
        # Configure dataset
        dataset_config = FinetuningDatasetConfig(
            dataset_root="/opt/data/datasets/finetune_train",
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
        dataset_config = FinetuningDatasetConfig(
            dataset_root="/opt/data/datasets/finetune",
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
