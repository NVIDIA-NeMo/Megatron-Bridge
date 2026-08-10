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

from megatron.bridge.training.utils.padding_utils import (
    get_padded_sequence_length,
    pad_batch_sequence_tensors,
)


def test_get_padded_sequence_length_aligns_to_multiple():
    assert get_padded_sequence_length(7, 4) == 8
    assert get_padded_sequence_length(8, 4) == 8


def test_get_padded_sequence_length_can_force_seq_length():
    assert get_padded_sequence_length(7, 4, force_to_seq_length=True, seq_length=12) == 12


def test_get_padded_sequence_length_rejects_bad_forced_seq_length_when_requested():
    with pytest.raises(ValueError, match="must be divisible"):
        get_padded_sequence_length(
            7,
            4,
            force_to_seq_length=True,
            seq_length=10,
            validate_forced_seq_length=True,
            error_context="dense context parallelism",
        )


def test_pad_batch_sequence_tensors_pads_common_raw_batch_tensors():
    tokens = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, -100]])
    loss_mask = torch.ones(1, 3)
    attention_mask = torch.ones(1, 3, dtype=torch.bool)
    position_ids = torch.arange(3).unsqueeze(0)

    tokens, labels, loss_mask, attention_mask, position_ids = pad_batch_sequence_tensors(
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        target_len=4,
    )

    assert tokens.tolist() == [[1, 2, 3, 0]]
    assert labels.tolist() == [[2, 3, -100, -100]]
    assert loss_mask.tolist() == [[1, 1, 1, 0]]
    assert attention_mask.tolist() == [[True, True, True, False]]
    assert position_ids.tolist() == [[0, 1, 2, 3]]
