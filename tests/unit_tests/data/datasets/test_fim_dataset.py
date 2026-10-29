# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np


class _AlwaysFimRng:
    def binomial(self, *_args):
        return 1

    def randint(self, **_kwargs):
        return np.array([0, 0])


class _SpecialTokenAwareTokenizer:
    def __init__(self):
        self.remove_special_tokens = None

    def ids_to_text(self, _sample, remove_special_tokens=None):
        self.remove_special_tokens = remove_special_tokens
        if remove_special_tokens:
            return "SKIP payload"
        return "<bos>SKIP payload"

    def text_to_ids(self, text):
        return [len(text) + 10]


class _PlainTokenizer:
    def ids_to_text(self, _sample):
        return "payload"

    def text_to_ids(self, text):
        return [len(text) + 10]


class _Tokenizer:
    def __init__(self, library, tokenizer):
        self.library = library
        self._tokenizer = tokenizer


class _IndexedDataset:
    def __init__(self, sample):
        self.sample = sample

    def get(self, _document_id, *, offset, length):
        return self.sample[offset : offset + length]


def _load_fim_dataset_class():
    class _GPTDataset:
        pass

    stubs = {
        "megatron.core.datasets.gpt_dataset": types.SimpleNamespace(GPTDataset=_GPTDataset),
        "megatron.core.datasets.indexed_dataset": types.SimpleNamespace(IndexedDataset=object),
        "megatron.core.datasets.utils": types.SimpleNamespace(Split=object),
        "megatron.bridge.training.config": types.SimpleNamespace(GPTFIMDatasetConfig=object),
    }
    module_path = Path(__file__).parents[4] / "src" / "megatron" / "bridge" / "data" / "datasets" / "fim_dataset.py"
    spec = importlib.util.spec_from_file_location("_fim_dataset_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.GPTFIMDataset


def _build_dataset(tokenizer, no_fim_prefix):
    dataset_class = _load_fim_dataset_class()
    dataset = object.__new__(dataset_class)
    original = np.array([1, 2, 3], dtype=np.int64)

    dataset.shuffle_index = np.array([0])
    dataset.sample_index = np.array([[0, 0], [0, 2]])
    dataset.document_index = np.array([0])
    dataset.dataset = _IndexedDataset(original)
    dataset.np_rng = _AlwaysFimRng()
    dataset.fim_rate = 1.0
    dataset.fim_spm_rate = 0.0
    dataset.fragment_fim_rate = 1.0
    dataset.fim_split_sample = None
    dataset.no_fim_prefix = no_fim_prefix
    dataset.config = types.SimpleNamespace(tokenizer=tokenizer)
    dataset.prefix_tok_id = 100
    dataset.middle_tok_id = 101
    dataset.suffix_tok_id = 102
    dataset.pad_tok_id = 103
    dataset.eod_tok_id = 104

    return dataset, original


def test_no_fim_prefix_ignores_tokenizer_added_special_tokens():
    tokenizer = _Tokenizer("huggingface", _SpecialTokenAwareTokenizer())
    dataset, original = _build_dataset(tokenizer, no_fim_prefix="SKIP")

    sample, _ = dataset._query_document_sample_shuffle_indices(0)

    np.testing.assert_array_equal(sample, original)


def test_fim_remains_compatible_with_tokenizers_without_removal_option():
    tokenizer = _Tokenizer("sentencepiece", _PlainTokenizer())
    dataset, _ = _build_dataset(tokenizer, no_fim_prefix=None)

    sample, _ = dataset._query_document_sample_shuffle_indices(0)

    assert sample[0] == dataset.prefix_tok_id
