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

import megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention as attention_module
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
    _qwen_attention_mask_for_core_attention,
)


class _FakeTEDotProductAttention:
    pass


def test_converts_qwen_valid_mask_for_te(monkeypatch):
    monkeypatch.setattr(attention_module, "TEDotProductAttention", _FakeTEDotProductAttention)
    valid_mask = torch.tensor([[True, True, False], [False, True, True]])

    actual = _qwen_attention_mask_for_core_attention(_FakeTEDotProductAttention(), valid_mask, packed_seq_params=None)

    expected = torch.tensor([[[[False, False, True]]], [[[True, False, False]]]])
    torch.testing.assert_close(actual, expected)


def test_converts_all_valid_qwen_mask_for_te_without_host_sync(monkeypatch):
    monkeypatch.setattr(attention_module, "TEDotProductAttention", _FakeTEDotProductAttention)
    monkeypatch.setattr(
        torch.Tensor, "item", lambda _tensor: pytest.fail("attention mask must not synchronize to host")
    )
    valid_mask = torch.ones((2, 4), dtype=torch.bool)

    actual = _qwen_attention_mask_for_core_attention(_FakeTEDotProductAttention(), valid_mask, packed_seq_params=None)
    monkeypatch.undo()

    torch.testing.assert_close(actual, torch.zeros((2, 1, 1, 4), dtype=torch.bool))


@pytest.mark.parametrize(
    ("mask", "packed_seq_params"),
    [
        (torch.ones((2, 4), dtype=torch.long), None),
        (torch.ones((2, 1, 4, 4), dtype=torch.bool), None),
        (torch.ones((2, 4), dtype=torch.bool), object()),
    ],
)
def test_preserves_non_qwen_te_mask_contracts(monkeypatch, mask, packed_seq_params):
    monkeypatch.setattr(attention_module, "TEDotProductAttention", _FakeTEDotProductAttention)

    actual = _qwen_attention_mask_for_core_attention(
        _FakeTEDotProductAttention(), mask, packed_seq_params=packed_seq_params
    )

    assert actual is mask
