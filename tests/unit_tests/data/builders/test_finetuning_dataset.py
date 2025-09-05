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

import subprocess
from pathlib import PosixPath

import pytest
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


def get_dataset(
    ensure_test_data,
    dataset_dirname="finetune",
    packed_sequence_size=1,
    packed_train_data_path=None,
    packed_val_data_path=None,
    tokenizer_name="null",
):
    path = f"{ensure_test_data}/datasets/{dataset_dirname}"
    # path = "/home/data/finetune_dataset"
    if tokenizer_name == "null":
        tokenizer_config = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072)
        tokenizer_model_name = "null"
    elif tokenizer_name == "hf":
        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=f"{ensure_test_data}/tokenizers/huggingface",
        )
        tokenizer_model_name = None
    else:
        tokenizer_config = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072)
        tokenizer_model_name = None
    tokenizer = build_tokenizer(
        tokenizer_config=tokenizer_config,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )
    packed_sequence_specs = PackedSequenceSpecs(
        packed_sequence_size=packed_sequence_size,
        tokenizer_model_name=tokenizer_model_name,
        packed_train_data_path=packed_train_data_path,
        packed_val_data_path=packed_val_data_path,
    )

    dataset = FinetuningDatasetBuilder(
        dataset_root=path,
        tokenizer=tokenizer,
        packed_sequence_specs=packed_sequence_specs,
    )

    return dataset, path


class TestDataFineTuningDataset:
    def test_extract_tokenizer_model_name(self, ensure_test_data):
        dataset, _ = get_dataset(ensure_test_data)
        tokenizer_name = dataset._extract_tokenizer_model_name()

        assert tokenizer_name == "null"

        dataset, _ = get_dataset(ensure_test_data, tokenizer_name="hf")
        tokenizer_name = dataset._extract_tokenizer_model_name()

        name = f"{ensure_test_data}/tokenizers/huggingface"
        name = name.replace("/", "--")
        assert tokenizer_name == name

        dataset, _ = get_dataset(ensure_test_data, tokenizer_name=None)
        tokenizer_name = dataset._extract_tokenizer_model_name()

        assert "unknown_tokenizer" in tokenizer_name

    def test_default_pack_path(self, ensure_test_data):
        dataset, path = get_dataset(ensure_test_data)
        default_pack_path = dataset.default_pack_path

        assert default_pack_path == PosixPath(f"{path}/packed/null")

    def test_train_path_packed(self, ensure_test_data):
        npy_path = f"{ensure_test_data}/datasets/finetune/test.npy"
        subprocess.run(["touch", npy_path])
        dataset, _ = get_dataset(ensure_test_data, packed_train_data_path=npy_path)
        train_path_packed = dataset.train_path_packed

        assert train_path_packed == PosixPath(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        train_path_packed = dataset.train_path_packed

        assert train_path_packed == PosixPath(f"{ensure_test_data}/datasets/finetune/packed/null/training_1.npy")

        dataset, _ = get_dataset(ensure_test_data, packed_sequence_size=-1)

        with pytest.raises(ValueError):
            train_path_packed = dataset.train_path_packed

    def test_validation_path_packed(self, ensure_test_data):
        npy_path = f"{ensure_test_data}/datasets/finetune/test.npy"
        subprocess.run(["touch", npy_path])
        dataset, _ = get_dataset(ensure_test_data, packed_val_data_path=npy_path)
        validation_path_packed = dataset.validation_path_packed

        assert validation_path_packed == PosixPath(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        validation_path_packed = dataset.validation_path_packed

        assert validation_path_packed == PosixPath(
            f"{ensure_test_data}/datasets/finetune/packed/null/validation_1.npy"
        )

        dataset, _ = get_dataset(ensure_test_data, packed_sequence_size=-1)
        try:
            validation_path_packed = dataset.validation_path_packed
        except ValueError:
            None

    def test_prepare_packed_data(self, ensure_test_data):
        dataset, path = get_dataset(ensure_test_data)

        with pytest.raises(KeyError):
            dataset.prepare_packed_data()

    def test_paths_packed_with_msc_url(self, ensure_test_data):
        MultiStorageClientFeature.enable()

        npy_path = f"msc://default{ensure_test_data}/datasets/finetune/test.npy"
        msc = MultiStorageClientFeature.import_package()
        msc.Path(npy_path).touch(exist_ok=True)

        # Train
        dataset, _ = get_dataset(ensure_test_data, packed_train_data_path=npy_path)
        train_path_packed = dataset.train_path_packed

        assert train_path_packed == msc.Path(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        train_path_packed = dataset.train_path_packed

        assert train_path_packed == msc.Path(f"{ensure_test_data}/datasets/finetune/packed/null/training_1.npy")

        # Validation
        dataset, _ = get_dataset(ensure_test_data, packed_val_data_path=npy_path)
        validation_path_packed = dataset.validation_path_packed

        assert validation_path_packed == msc.Path(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        validation_path_packed = dataset.validation_path_packed

        assert validation_path_packed == msc.Path(f"{ensure_test_data}/datasets/finetune/packed/null/validation_1.npy")

        dataset, _ = get_dataset(ensure_test_data, packed_sequence_size=-1)

        with pytest.raises(ValueError):
            train_path_packed = dataset.train_path_packed

        with pytest.raises(ValueError):
            validation_path_packed = dataset.validation_path_packed

    def test_build_dataset_with_msc_url(self, ensure_test_data):
        MultiStorageClientFeature.enable()

        dataset_dirname = "finetune_msc"
        jsonl_example = '{"input": "John von Neumann Von Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}\n'

        msc = MultiStorageClientFeature.import_package()
        msc.Path(f"{ensure_test_data}/datasets/{dataset_dirname}").mkdir(exist_ok=True)

        with open(f"{ensure_test_data}/datasets/{dataset_dirname}/training.jsonl", "w") as f:
            for i in range(10):
                f.write(jsonl_example)

        with open(f"{ensure_test_data}/datasets/{dataset_dirname}/validation.jsonl", "w") as f:
            for i in range(10):
                f.write(jsonl_example)

        with open(f"{ensure_test_data}/datasets/{dataset_dirname}/test.jsonl", "w") as f:
            for i in range(10):
                f.write(jsonl_example)

        dataset, _ = get_dataset(
            ensure_test_data, dataset_dirname=dataset_dirname, packed_sequence_size=2048, tokenizer_name="hf"
        )
        assert not dataset.pack_metadata.exists()

        datasets = dataset.build()
        assert dataset.pack_metadata.exists()

        assert datasets[0] is not None
        assert datasets[1] is not None
        assert datasets[2] is not None
