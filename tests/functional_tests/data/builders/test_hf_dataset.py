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

import os
from pathlib import PosixPath

import pytest
from datasets import load_dataset

from megatron.bridge.data.builders.hf_dataset import HFDatasetBuilder, preprocess_and_split_data
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


def process_example_fn(example, tokenizer):
    return {
        "input": f"question: {example['question']} context: {example['passage']}",
        "output": int(example["answer"]),
        "original_answers": int(example["answer"]),
    }


def get_tokenizer(ensure_test_data):
    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=f"{ensure_test_data}/tokenizers/huggingface",
    )
    tokenizer = build_tokenizer(
        tokenizer_config=tokenizer_config,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )

    return tokenizer


class TestDataHFDataset:
    def test_preprocess_and_split_data_split_val_from_train(self, ensure_test_data):
        path = f"{ensure_test_data}/datasets/hf"
        os.makedirs(path, exist_ok=True)
        path = PosixPath(path)
        preprocess_and_split_data(
            dset=load_dataset("boolq"),
            dataset_name="boolq",
            dataset_root=path,
            process_example_fn=process_example_fn,
            tokenizer=get_tokenizer(ensure_test_data),
            val_proportion=0.1,
            delete_raw=True,
            do_validation=True,
            do_test=True,
        )

        assert os.path.exists(path / "training.jsonl")
        assert os.path.exists(path / "validation.jsonl")
        assert os.path.exists(path / "test.jsonl")

    def test_preprocess_and_split_data(self, ensure_test_data):
        path = f"{ensure_test_data}/datasets/hf"
        os.makedirs(path, exist_ok=True)
        path = PosixPath(path)
        preprocess_and_split_data(
            dset=load_dataset("boolq"),
            dataset_name="boolq",
            dataset_root=path,
            process_example_fn=process_example_fn,
            tokenizer=get_tokenizer(ensure_test_data),
            val_proportion=0.1,
            delete_raw=True,
            do_validation=True,
            do_test=True,
            split_val_from_train=False,
            rewrite=True,
        )

        assert os.path.exists(path / "training.jsonl")
        assert os.path.exists(path / "validation.jsonl")
        assert os.path.exists(path / "test.jsonl")

    def test_hf_dataset_builder(self, ensure_test_data):
        path = f"{ensure_test_data}/datasets/hf"
        os.makedirs(path, exist_ok=True)
        path = PosixPath(path)
        builder = HFDatasetBuilder(
            dataset_name="boolq",
            dataset_root=path,
            process_example_fn=process_example_fn,
            tokenizer=get_tokenizer(ensure_test_data),
            rewrite=True,
        )

        builder.prepare_data()

        assert os.path.exists(path / "training.jsonl")
        assert os.path.exists(path / "validation.jsonl")
        assert os.path.exists(path / "test.jsonl")

    def test_hf_dataset_builder_lambda(self, ensure_test_data):
        path = f"{ensure_test_data}/datasets/hf"
        os.makedirs(path, exist_ok=True)
        path = PosixPath(path)
        builder = HFDatasetBuilder(
            dataset_name="boolq",
            dataset_root=path,
            process_example_fn=process_example_fn,
            tokenizer=get_tokenizer(ensure_test_data),
            rewrite=True,
            hf_filter_lambda="test",
            download_mode=None,
        )

        with pytest.raises(ValueError):
            builder.prepare_data()

    def test_hf_dataset_builder_with_dict(self, ensure_test_data):
        path = f"{ensure_test_data}/datasets/hf"
        os.makedirs(path, exist_ok=True)
        path = PosixPath(path)
        builder = HFDatasetBuilder(
            dataset_dict=load_dataset("boolq"),
            dataset_name="boolq",
            dataset_root=path,
            process_example_fn=process_example_fn,
            tokenizer=get_tokenizer(ensure_test_data),
            rewrite=True,
        )

        builder.prepare_data()

        assert os.path.exists(path / "training.jsonl")
        assert os.path.exists(path / "validation.jsonl")
        assert os.path.exists(path / "test.jsonl")
