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

import numpy as np
import torch
import pytest

from megatron.bridge.data.datasets.utils import (
    _add_speaker_and_signal,
    _deallocate_indexed_dataset_memory,
    _get_header_conversation_type_mask_role,
    _identify_start_index_of_subsequence,
    _index_file_exists,
    _make_indexed_dataset_compatibility,
    _OnlineSampleMapping,
    _response_value_formater,
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
