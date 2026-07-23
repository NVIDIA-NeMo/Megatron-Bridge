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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.nemotron_omni.sequence_packing import pack_sequences_from_attention_mask


pytestmark = pytest.mark.unit


class _FakeProcessGroup:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def _process_groups(*, cp_size: int, cp_rank: int = 0, tp_size: int = 1):
    return SimpleNamespace(
        cp=_FakeProcessGroup(cp_size, cp_rank),
        tp=_FakeProcessGroup(tp_size),
    )


def test_cp1_packing_builds_mamba_sequence_boundaries():
    tokens = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )

    packed, packed_seq_params = pack_sequences_from_attention_mask(
        tokens,
        attention_mask,
        pg_collection=_process_groups(cp_size=1),
    )

    assert torch.equal(packed, torch.tensor([[10, 11, 12, 20, 21]]))
    assert torch.equal(packed_seq_params.cu_seqlens_q_padded, torch.tensor([0, 3, 5], dtype=torch.int32))
    assert packed_seq_params.total_tokens == 5
    assert torch.equal(packed_seq_params.seq_idx, torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32))


@pytest.mark.parametrize(
    ("cp_rank", "expected"),
    [
        (0, [10, 11, 0, 0, 20, 0]),
        (1, [12, 13, 14, 0, 21, 22]),
    ],
)
def test_cp2_packing_builds_global_mamba_sequence_boundaries(cp_rank, expected):
    tokens = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    packed, packed_seq_params = pack_sequences_from_attention_mask(
        tokens,
        attention_mask,
        pg_collection=_process_groups(cp_size=2, cp_rank=cp_rank),
    )

    assert torch.equal(packed, torch.tensor([expected]))
    assert torch.equal(packed_seq_params.cu_seqlens_q_padded, torch.tensor([0, 8, 12], dtype=torch.int32))
    assert packed_seq_params.total_tokens == 12
    assert torch.equal(
        packed_seq_params.seq_idx,
        torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32),
    )
