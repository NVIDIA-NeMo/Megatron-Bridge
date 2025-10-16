#!/usr/bin/env python3
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
import tempfile

import numpy as np
import pytest
import torch
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.datasets.utils import (
    _add_speaker_and_signal,
    _build_memmap_index_files,
    _deallocate_indexed_dataset_memory,
    _get_header_conversation_type_mask_role,
    _identify_start_index_of_subsequence,
    _index_file_exists,
    _index_fn,
    _JSONLMemMapDataset,
    _make_indexed_dataset_compatibility,
    _OnlineSampleMapping,
    _response_value_formater,
    build_index_from_memdata,
    handle_index,
)


class IndexedDataset:
    def __init__(self, sizes: int = None, doc_idx: int = None):
        self.sizes = sizes
        self.doc_idx = doc_idx

    def __len__(self):
        if self.sizes is None:
            return 1
        else:
            return self.sizes


class TestDataUtils:
    def test_online_sample_mapping(self):
        online = _OnlineSampleMapping(10, 10)

        assert len(online) == 10
        assert online[5] == (3, None, None)
        assert online[1:3] == [(9, None, None), (6, None, None)]
        assert np.array_equal(online.get_sample_block(0), np.array([2, 9, 6, 4, 0, 3, 1, 7, 8, 5]))

    def test_deallocate_indexed_dataset_memory(self):
        indexed_dataset = IndexedDataset(1, 1)
        _deallocate_indexed_dataset_memory(indexed_dataset)

        assert indexed_dataset.sizes == None
        assert indexed_dataset.doc_idx == None

    def test_identify_start_index_of_subsequence(self):
        subsequence = torch.tensor([1, 3])
        sequence = torch.tensor([2, 3, 1, 3])

        start_index = _identify_start_index_of_subsequence(subsequence, sequence)

        assert start_index == 2

        subsequence = torch.tensor([3, 2])
        start_index = _identify_start_index_of_subsequence(subsequence, sequence)

        assert start_index == -1

    @pytest.mark.parametrize("label", [None, "this ", 1])
    def test_response_value_formater(self, label):
        label_start = "test "
        end_signal = "function"

        if label is None:
            expected = ""
        elif label == "this ":
            expected = "test this function"
        else:
            expected = None

        try:
            response = _response_value_formater(label, label_start, end_signal)
            assert response == expected
        except ValueError:
            None

    @pytest.fixture
    def special_tokens(self):
        return {
            "turn_start": "<|turn|>",
            "end_of_turn": "<|endofturn|>",
            "label_start": "<|label|>",
            "end_of_name": "<|endname|>",
            "system_turn_start": "|<system>|",
        }

    @pytest.mark.parametrize("gtype", [None, "VALUE_TO_TEXT", "TEXT_TO_VALUE", "TEST"])
    def test_add_speaker_and_signal(self, gtype, special_tokens):
        header = "<header>"
        source = [
            {"from": "user", "value": "Hello"},
            {"from": "assistant", "value": "Hi", "label": "greeting"},
        ]
        mask_role = {"user"}

        if gtype is None:
            expected = (
                "<header><|turn|>user<|endname|>Hello<|endofturn|><|turn|>assistant<|endname|>Hi<|endofturn|><|turn|>"
            )
        elif gtype == "VALUE_TO_TEXT":
            expected = (
                "<header>"
                "<|turn|>user<|endname|>Hello<|endofturn|>"
                "<|turn|>assistant<|endname|><|label|>greeting<|endname|>Hi<|endofturn|><|turn|>"
            )
        else:
            expected = (
                "<header>"
                "<|turn|>user<|endname|>Hello<|endofturn|>"
                "<|turn|>assistant<|endname|>Hi<|endofturn|><|label|>greeting<|endname|><|turn|>"
            )

        try:
            result = _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens)
            assert result == expected
        except ValueError:
            None

    def test_index_file_exists(self):
        if_exists = _index_file_exists("test")

        assert if_exists == False

    def test_get_header_conversation_type_mask_role(self, special_tokens):
        source = {
            "system": "Simple header.",
            "conversations": [{"from": "user", "value": "Hi there"}],
        }

        header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)

        assert data_type is None
        assert mask_role == "User"
        assert "Simple header." in header
        assert "<|turn|>user<|endname|>Hi there<|endofturn|>" in conversation

        source = {
            "system": "This is a system prompt.",
            "type": "VALUE_TO_TEXT",
            "mask": {"user"},
            "conversations": [
                {"from": "user", "value": "Hello"},
                {"from": "assistant", "value": "Hi", "label": "greeting"},
            ],
        }

        header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)

        assert data_type == "VALUE_TO_TEXT"
        assert mask_role == {"user"}
        assert "This is a system prompt." in header
        assert "<|turn|>user<|endname|>Hello<|endofturn|>" in conversation
        assert "<|label|>greeting" in conversation

    def test_make_indexed_dataset_compatibility(self):
        dataset = IndexedDataset()

        dataset = _make_indexed_dataset_compatibility(dataset)

        assert np.array_equal(dataset.doc_idx, np.array([0, 1], dtype=np.int64))
        assert np.array_equal(dataset.sizes, np.array([1], dtype=np.int32))

        try:
            dataset = IndexedDataset(5, 5)
            dataset = _make_indexed_dataset_compatibility(dataset)
        except AttributeError:
            None

    @pytest.mark.parametrize("idx", [-1, -15])
    def test_handle_index(self, idx):
        dataset = IndexedDataset(5, 5)

        if idx == -1:
            expected = 4
        else:
            expected = None

        try:
            index = handle_index(dataset, idx)
        except IndexError:
            index = None

        assert expected == index

    def test_index_fn(self):
        MultiStorageClientFeature.enable()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test case 1: Simple filename with index_mapping_dir
            index_fn = _index_fn("test", "test")
            assert index_fn == "test/test.idx"

            # Test case 2: Simple filename without index_mapping_dir (None)
            index_fn = _index_fn("test", None)
            assert index_fn == "test.idx"

            # Test case 3: Relative path with index_mapping_dir
            index_fn = _index_fn("relative/path/to/data.jsonl", "mapping_dir")
            assert index_fn == "mapping_dir/relative/path/to/data.jsonl.idx"

            # Test case 4: Absolute path with index_mapping_dir (should strip leading /)
            index_fn = _index_fn("/absolute/path/to/data.jsonl", "mapping_dir")
            assert index_fn == "mapping_dir/absolute/path/to/data.jsonl.idx"

            # Test case 5: Path with leading .. (should strip)
            index_fn = _index_fn("../../path/to/data.jsonl", "mapping_dir")
            assert index_fn == "mapping_dir/path/to/data.jsonl.idx"

            # Test case 6: File with MSC URL (should strip leading /)
            index_fn = _index_fn("data.jsonl", f"msc://default{temp_dir}/index_mapping_dir")
            assert index_fn == f"msc://default{temp_dir}/index_mapping_dir/data.jsonl.idx"

    def test_jsonl_memmap_dataset(self):
        jsonl_example = '{"input": "John von Neumann Von Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}\n'

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/training.jsonl", "w") as f:
                for i in range(10):
                    f.write(jsonl_example)

            ds = _JSONLMemMapDataset(
                dataset_paths=[f"{temp_dir}/training.jsonl"],
            )

            assert len(ds) > 0
            assert ds[0] is not None
            assert os.path.exists(f"{temp_dir}/training.jsonl.idx.npy")
            assert os.path.exists(f"{temp_dir}/training.jsonl.idx.info")

    def test_jsonl_memmap_dataset_with_msc_url(self):
        jsonl_example = '{"input": "John von Neumann Von Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}\n'

        with tempfile.TemporaryDirectory() as temp_dir:
            MultiStorageClientFeature.enable()
            msc = MultiStorageClientFeature.import_package()

            with open(f"{temp_dir}/test.jsonl", "w") as f:
                for i in range(10):
                    f.write(jsonl_example)

            ds = _JSONLMemMapDataset(
                dataset_paths=[f"msc://default{temp_dir}/test.jsonl"],
            )

            assert len(ds) > 0
            assert ds[0] is not None
            assert msc.Path(f"{temp_dir}/test.jsonl.idx.npy").exists()
            assert msc.Path(f"{temp_dir}/test.jsonl.idx.info").exists()

    def test_build_memmap_index_files(self):
        jsonl_example = '{"input": "John von Neumann Von Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}\n'

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/training.jsonl", "w") as f:
                for i in range(10):
                    f.write(jsonl_example)

            assert _build_memmap_index_files(
                10,
                build_index_from_memdata,
                f"{temp_dir}/training.jsonl",
                None,
            )

            assert os.path.exists(f"{temp_dir}/training.jsonl.idx.npy")
            assert os.path.exists(f"{temp_dir}/training.jsonl.idx.info")

    def test_build_memmap_index_files_with_msc_url(self):
        jsonl_example = '{"input": "John von Neumann Von Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}\n'

        with tempfile.TemporaryDirectory() as temp_dir:
            MultiStorageClientFeature.enable()
            msc = MultiStorageClientFeature.import_package()

            with open(f"{temp_dir}/training.jsonl", "w") as f:
                for i in range(10):
                    f.write(jsonl_example)

            assert _build_memmap_index_files(
                10,
                build_index_from_memdata,
                f"msc://default{temp_dir}/training.jsonl",
                None,
            )

            assert msc.Path(f"msc://default{temp_dir}/training.jsonl.idx.npy")
            assert msc.Path(f"msc://default{temp_dir}/training.jsonl.idx.info")
