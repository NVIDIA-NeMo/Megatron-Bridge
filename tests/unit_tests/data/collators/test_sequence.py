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

import pytest
import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch


pytestmark = pytest.mark.unit


def test_prepare_sequence_batch_pads_to_efficiency_multiple():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, -100]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    prepare_sequence_batch(
        batch,
        sequence_length=10,
        pad_to_max_length=False,
        pad_to_multiple_of=4,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 0]]
    assert batch["labels"].tolist() == [[2, 3, -100, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0]]


def test_prepare_sequence_batch_pads_to_model_length_when_required():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, -100]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    prepare_sequence_batch(batch, sequence_length=6, pad_to_max_length=True)

    assert batch["input_ids"].shape == (1, 6)
    assert batch["labels"].tolist() == [[2, 3, -100, -100, -100, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0]]


def test_prepare_sequence_batch_handles_rectangular_4d_attention_mask():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, -100]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones((1, 1, 3, 2), dtype=torch.bool),
    }

    prepare_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["attention_mask"].shape == (1, 1, 4, 4)
    assert batch["attention_mask"][0, 0, :3, :2].all()
    assert not batch["attention_mask"][0, 0, 3, :].any()
    assert not batch["attention_mask"][0, 0, :, 2:].any()


def test_prepare_sequence_batch_packs_directly_with_current_metadata():
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 6, 7, 8],
            ]
        ),
        "labels": torch.tensor(
            [
                [2, 3, -100, -100, -100],
                [5, 6, 7, 8, -100],
            ]
        ),
        "loss_mask": torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
            ]
        ),
        "position_ids": torch.arange(5).unsqueeze(0).expand(2, -1).clone(),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    }

    prepare_sequence_batch(
        batch,
        sequence_length=16,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 4, 5, 6, 7, 8, 0, 0, 0]]
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 8]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3, 8]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 12]
    assert batch["cu_seqlens_kv_padded"].tolist() == [0, 4, 12]
    assert batch["max_seqlen_q"].item() == 8
    assert batch["max_seqlen_kv"].item() == 8
    assert "cu_seqlens" not in batch
    assert "cu_seqlens_unpadded" not in batch
    assert "cu_seqlens_argmin" not in batch
    assert "cu_seqlens_unpadded_argmin" not in batch


def test_prepare_sequence_batch_rejects_left_padded_direct_packing():
    batch = {
        "input_ids": torch.tensor([[0, 0, 1, 2]]),
        "labels": torch.tensor([[-100, -100, 1, 2]]),
        "loss_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="right-padded"):
        prepare_sequence_batch(
            batch,
            sequence_length=16,
            enable_in_batch_packing=True,
        )


def test_prepare_sequence_batch_omits_padded_metadata_when_alignment_is_not_requested():
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 6, 7, 8],
            ]
        ),
        "labels": torch.tensor(
            [
                [2, 3, -100, -100, -100],
                [5, 6, 7, 8, -100],
            ]
        ),
        "loss_mask": torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
            ]
        ),
        "position_ids": torch.arange(5).unsqueeze(0).expand(2, -1).clone(),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    }

    prepare_sequence_batch(
        batch,
        sequence_length=16,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=1,
    )

    assert batch["cu_seqlens_q"].tolist() == [0, 3, 8]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3, 8]
    assert batch["max_seqlen_q"].item() == 5
    assert batch["max_seqlen_kv"].item() == 5
    assert "cu_seqlens_q_padded" not in batch
    assert "cu_seqlens_kv_padded" not in batch
    assert "cu_seqlens" not in batch
    assert "cu_seqlens_unpadded" not in batch
    assert "cu_seqlens_argmin" not in batch
    assert "cu_seqlens_unpadded_argmin" not in batch


def test_prepare_sequence_batch_emits_padded_metadata_for_aligned_cp_multiple():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [5, 6, 7, 8, 9, 10, 11, 12]]),
        "position_ids": torch.arange(8).unsqueeze(0).expand(2, -1).clone(),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]),
    }

    prepare_sequence_batch(
        batch,
        sequence_length=8,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["cu_seqlens_q"].tolist() == [0, 4, 12]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 12]
    assert batch["cu_seqlens_kv_padded"].tolist() == [0, 4, 12]


def test_prepare_sequence_batch_rejects_overlength_packed_row():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "position_ids": torch.arange(5).unsqueeze(0),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
    }

    with pytest.raises(ValueError, match="exceeds configured sequence_length 4"):
        prepare_sequence_batch(
            batch,
            sequence_length=4,
            enable_in_batch_packing=True,
        )


def test_prepare_sequence_batch_rejects_empty_packed_row():
    batch = {
        "input_ids": torch.tensor([[0, 0], [1, 2]]),
        "position_ids": torch.arange(2).unsqueeze(0).expand(2, -1).clone(),
        "attention_mask": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="empty sequence row"):
        prepare_sequence_batch(
            batch,
            sequence_length=4,
            enable_in_batch_packing=True,
        )
