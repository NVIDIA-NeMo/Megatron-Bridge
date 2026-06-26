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

import torch

from megatron.bridge.data.datasets.packed_sequence import _materialize_dataset_items, _pre_pad_data_point


PAD_ID = 0


def test_pre_pad_data_point_chat_tensors_do_not_raise():
    """Chat path returns torch tensors; padding must not raise TypeError (see issue #2610)."""
    data = {
        "input_ids": torch.LongTensor([5, 6, 7]),
        "loss_mask": torch.BoolTensor([False, True, True]),
        "context_ids": torch.LongTensor([5, 6]),
    }
    # max_length_to_pad=8 -> input_ids padded to 8 - 3 + 1 = 6 extra -> length 9
    _pre_pad_data_point(data, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert isinstance(data["input_ids"], list)
    assert isinstance(data["loss_mask"], list)
    # loss_mask must end up the same length as input_ids, otherwise fill_packing_strategy's
    # np.array([...loss_mask...]) raises an inhomogeneous-shape error when samples are grouped.
    assert len(data["loss_mask"]) == len(data["input_ids"])
    # padded loss_mask positions carry 0 (no loss on pad tokens)
    assert data["loss_mask"][3:] == [0] * (len(data["loss_mask"]) - 3)
    assert data["input_ids"][3:] == [PAD_ID] * (len(data["input_ids"]) - 3)


def test_pre_pad_data_point_equalizes_loss_mask_lengths():
    """Two samples that round to the same padded input length must get equal-length loss_masks."""
    a = {"input_ids": torch.LongTensor([1, 2, 3]), "loss_mask": torch.BoolTensor([False, True, True])}
    b = {
        "input_ids": torch.LongTensor([1, 2, 3, 4, 5]),
        "loss_mask": torch.BoolTensor([False, False, True, True, True]),
    }
    # both round up to the same multiple-of-8 target
    _pre_pad_data_point(a, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)
    _pre_pad_data_point(b, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert len(a["input_ids"]) == len(b["input_ids"])
    assert len(a["loss_mask"]) == len(b["loss_mask"]) == len(a["input_ids"])


def test_pre_pad_data_point_non_chat_lists_still_work():
    """Non-chat (GPTSFTDataset) path returns plain lists without loss_mask; must be unaffected."""
    data = {"input_ids": [9, 9, 9], "context_ids": [9, 9]}
    _pre_pad_data_point(data, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert data["input_ids"] == [9, 9, 9] + [PAD_ID] * 6
    assert "loss_mask" not in data


def test_pre_pad_data_point_truncates_overlong():
    """Sequences longer than max_seq_length are truncated."""
    data = {"input_ids": list(range(20)), "loss_mask": [1] * 20}
    _pre_pad_data_point(data, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert len(data["input_ids"]) == 16
    assert len(data["loss_mask"]) == 16


def test_materialize_dataset_items_uses_serial_path_for_non_positive_workers(monkeypatch):
    """Non-positive worker counts should not create a multiprocessing pool."""

    class TinyDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index + 10

    def fail_pool(*args, **kwargs):
        raise AssertionError("Pool should not be constructed for non-positive worker counts")

    monkeypatch.setattr("megatron.bridge.data.datasets.packed_sequence.Pool", fail_pool)

    assert _materialize_dataset_items(TinyDataset(), -1).tolist() == [10, 11, 12]
    assert _materialize_dataset_items(TinyDataset(), 0).tolist() == [10, 11, 12]
