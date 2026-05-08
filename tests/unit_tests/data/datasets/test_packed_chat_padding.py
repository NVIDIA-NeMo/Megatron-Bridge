# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Regression tests for packed sequence padding with chat datasets (#2610).

These cover the interaction between ``_chat_preprocess`` (which returns
``torch.LongTensor`` / ``torch.BoolTensor``) and ``pre_pad_dataset`` inside
``tokenize_dataset`` when ``pad_seq_to_mult > 1``. Two bugs are covered:

1. Tensor/list mismatch: padding via ``val + [pad_id] * N`` previously raised
   ``TypeError`` because torch tensors do not support list concatenation.
2. Missing ``loss_mask`` padding: padded ``input_ids`` could group rows of
   different original ``loss_mask`` lengths into the same histogram bin,
   producing an inhomogeneous-shape error in ``fill_packing_strategy``.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from megatron.bridge.data.datasets.packed_sequence import tokenize_dataset
from megatron.bridge.data.datasets.packing_utils import create_hist, fill_packing_strategy


def _make_mock_tokenizer(eod: int = 0) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.eod = eod
    tokenizer.eos_id = eod
    tokenizer._tokenizer = MagicMock()
    return tokenizer


def _make_mock_dataset(items: list[dict], pad_seq_length_to_mult: int, max_seq_length: int) -> MagicMock:
    dataset = MagicMock()
    dataset.tokenizer = _make_mock_tokenizer()
    dataset.pad_seq_length_to_mult = pad_seq_length_to_mult
    dataset.max_seq_length = max_seq_length
    dataset.__len__ = MagicMock(return_value=len(items))
    dataset.__getitem__ = MagicMock(side_effect=lambda idx: items[idx])
    return dataset


@pytest.mark.unit
class TestTokenizeDatasetChatPadding:
    """Exercise the ``pad_seq_to_mult > 1`` branch with chat-format items."""

    def _run(self, items: list[dict], pad_seq_to_mult: int, max_seq_length: int = 64) -> np.ndarray:
        with patch("megatron.bridge.data.datasets.packed_sequence.create_sft_dataset") as mock_create:
            mock_create.return_value = _make_mock_dataset(items, pad_seq_to_mult, max_seq_length)
            return tokenize_dataset(
                path=Path("dummy.jsonl"),
                tokenizer=_make_mock_tokenizer(),
                max_seq_length=max_seq_length,
                seed=0,
                dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
                pad_seq_to_mult=pad_seq_to_mult,
                num_tokenizer_workers=1,
            )

    def test_torch_tensor_inputs_do_not_raise(self):
        """Tensor inputs from `_chat_preprocess` should pad without TypeError."""
        items = [
            {
                "input_ids": torch.LongTensor([10, 11, 12]),
                "loss_mask": torch.BoolTensor([False, True, True]),
                "context_ids": torch.LongTensor([10]),
                "answer_ids": torch.LongTensor([11, 12]),
            },
        ]

        result = self._run(items, pad_seq_to_mult=8)

        assert len(result) == 1
        # 3 -> ceil to 8 -> +1 extra pad token = 9 total elements.
        assert len(result[0]["input_ids"]) == 9
        assert isinstance(result[0]["input_ids"], list)

    def test_loss_mask_is_padded_to_input_ids_length(self):
        """`loss_mask` must be padded so all bin-mates have equal length."""
        items = [
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "loss_mask": torch.BoolTensor([False, True, True]),
                "context_ids": torch.LongTensor([1]),
                "answer_ids": torch.LongTensor([2, 3]),
            },
            {
                "input_ids": torch.LongTensor([4, 5, 6, 7, 8]),
                "loss_mask": torch.BoolTensor([False, False, True, True, True]),
                "context_ids": torch.LongTensor([4, 5]),
                "answer_ids": torch.LongTensor([6, 7, 8]),
            },
        ]

        result = self._run(items, pad_seq_to_mult=8)

        # Both items round to max_length_to_pad=8 -> +1 extra = length 9.
        for row in result:
            assert len(row["input_ids"]) == len(row["loss_mask"]) == 9

    def test_loss_mask_padding_uses_false(self):
        """Padded loss_mask positions must be False so they contribute no loss."""
        items = [
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "loss_mask": torch.BoolTensor([True, True, True]),
                "context_ids": torch.LongTensor([1]),
                "answer_ids": torch.LongTensor([2, 3]),
            },
        ]

        result = self._run(items, pad_seq_to_mult=8)

        loss_mask = result[0]["loss_mask"]
        # Original 3 entries preserved; trailing positions are False.
        assert loss_mask[:3] == [True, True, True]
        assert all(v is False for v in loss_mask[3:])

    def test_context_ids_padded_with_pad_id(self):
        """`context_ids` keeps padding with `pad_id`, like `input_ids`."""
        items = [
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "loss_mask": torch.BoolTensor([False, True, True]),
                "context_ids": torch.LongTensor([1, 2]),
                "answer_ids": torch.LongTensor([3]),
            },
        ]

        result = self._run(items, pad_seq_to_mult=8)
        # eod is 0 in the mock; padded tail of context_ids must be 0.
        context_ids = result[0]["context_ids"]
        assert context_ids[:2] == [1, 2]
        assert all(v == 0 for v in context_ids[2:])

    def test_non_chat_list_inputs_still_padded(self):
        """Plain-list (non-chat) inputs continue to work unchanged."""
        items = [
            {
                "input_ids": [1, 2, 3],
                "answer_start_idx": 1,
            },
        ]

        result = self._run(items, pad_seq_to_mult=8)
        assert len(result[0]["input_ids"]) == 9
        # `loss_mask` was never present; we must not invent it.
        assert "loss_mask" not in result[0]

    def test_pad_seq_to_mult_one_skips_padding(self):
        """When pad_seq_to_mult is 1, the padding branch is bypassed entirely."""
        original = torch.LongTensor([1, 2, 3])
        items = [
            {
                "input_ids": original,
                "loss_mask": torch.BoolTensor([False, True, True]),
                "context_ids": torch.LongTensor([1]),
                "answer_ids": torch.LongTensor([2, 3]),
            },
        ]

        result = self._run(items, pad_seq_to_mult=1)

        # With pad_seq_to_mult=1, items pass through untouched (still tensors).
        assert torch.equal(result[0]["input_ids"], original)


@pytest.mark.unit
class TestPackedChatEndToEnd:
    """Verify padded chat output flows into the packing pipeline cleanly."""

    def test_create_hist_then_fill_packing_strategy_no_error(self):
        """Two original-different-length items, padded to the same bucket, must
        round-trip through `create_hist` and `fill_packing_strategy`. Before the
        fix, this raised because `np.array([x['loss_mask']...])` saw rows of
        different lengths."""
        max_seq_length = 32
        pad_seq_to_mult = 8

        items = [
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "loss_mask": torch.BoolTensor([False, True, True]),
                "context_ids": torch.LongTensor([1]),
                "answer_ids": torch.LongTensor([2, 3]),
            },
            {
                "input_ids": torch.LongTensor([4, 5, 6, 7, 8]),
                "loss_mask": torch.BoolTensor([False, False, True, True, True]),
                "context_ids": torch.LongTensor([4, 5]),
                "answer_ids": torch.LongTensor([6, 7, 8]),
            },
        ]

        with patch("megatron.bridge.data.datasets.packed_sequence.create_sft_dataset") as mock_create:
            mock_create.return_value = _make_mock_dataset(items, pad_seq_to_mult, max_seq_length)
            tokenized = tokenize_dataset(
                path=Path("dummy.jsonl"),
                tokenizer=_make_mock_tokenizer(),
                max_seq_length=max_seq_length,
                seed=0,
                dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
                pad_seq_to_mult=pad_seq_to_mult,
                num_tokenizer_workers=1,
            )

        sequences, _histogram = create_hist(tokenized, truncate_seq_len=max_seq_length)
        # Both items should land in the same bucket: input_ids length = 9, so
        # `seq_len = len(input_ids) - 1 = 8`.
        assert len(sequences[8]) == 2

        # Pack them together. This previously raised because the two items'
        # loss_mask lengths differed (3 vs 5).
        assignments = [[8, 8]]
        output = fill_packing_strategy(assignments, sequences, pack_size=max_seq_length, pad_id=0)

        assert len(output) == 1
        # `create_hist` keys items by `len(input_ids) - 1` (8) but
        # `fill_packing_strategy` writes the FULL padded input_ids (length 9)
        # for each item in the assignment, so the packed pair has length 9+9=18.
        # `loss_mask` is rolled by 1 (`x[1:] + [False]`), preserving the same length.
        assert len(output[0]["input_ids"]) == 18
        assert len(output[0]["loss_mask"]) == 18
