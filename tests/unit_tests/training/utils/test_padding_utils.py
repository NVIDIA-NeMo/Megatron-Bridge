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

"""Unit tests for padding_utils module.

The three helpers (pad_or_truncate_2d_to_len, pad_or_truncate_pos_to_len,
pad_or_truncate_attn_to_len) are pure-CPU tensor operations that pad or
truncate batch dimensions. They are easy to verify deterministically by
constructing small input tensors and inspecting output shape, dtype, and
value placement.
"""

import pytest
import torch

from megatron.bridge.training.utils.padding_utils import (
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)


class TestPadOrTruncate2dToLen:
    def test_returns_none_for_none_input(self):
        assert pad_or_truncate_2d_to_len(None, target_len=4, max_cap=8, pad_value=0) is None

    def test_pads_to_target_len_when_shorter(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = pad_or_truncate_2d_to_len(x, target_len=5, max_cap=8, pad_value=0)
        assert out.shape == (2, 5)
        assert torch.equal(out, torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]))

    def test_pads_with_custom_pad_value(self):
        x = torch.tensor([[1, 2], [3, 4]])
        out = pad_or_truncate_2d_to_len(x, target_len=4, max_cap=8, pad_value=-100)
        assert torch.equal(out, torch.tensor([[1, 2, -100, -100], [3, 4, -100, -100]]))

    def test_truncates_when_above_max_cap(self):
        x = torch.arange(20).reshape(2, 10)
        out = pad_or_truncate_2d_to_len(x, target_len=4, max_cap=6, pad_value=0)
        assert out.shape == (2, 6)
        assert torch.equal(out, x[:, :6])

    def test_passthrough_when_between_target_and_cap(self):
        x = torch.arange(12).reshape(2, 6)
        out = pad_or_truncate_2d_to_len(x, target_len=4, max_cap=8, pad_value=0)
        # current_len=6, target=4 (not shorter), max_cap=8 (not exceeded) -> identity
        assert out is x

    def test_pad_value_works_with_float_dtype(self):
        x = torch.tensor([[1.0, 2.0]])
        out = pad_or_truncate_2d_to_len(x, target_len=4, max_cap=8, pad_value=-1.5)
        assert torch.equal(out, torch.tensor([[1.0, 2.0, -1.5, -1.5]]))


class TestPadOrTruncatePosToLen:
    def test_returns_none_for_none_input(self):
        assert pad_or_truncate_pos_to_len(None, target_len=4, max_cap=8) is None

    def test_appends_monotonic_range_when_shorter(self):
        pos = torch.tensor([[0, 1, 2]])
        out = pad_or_truncate_pos_to_len(pos, target_len=6, max_cap=8)
        assert out.shape == (1, 6)
        assert torch.equal(out, torch.tensor([[0, 1, 2, 3, 4, 5]]))

    def test_broadcasts_extension_across_batch(self):
        pos = torch.tensor([[0, 1, 2], [0, 1, 2]])
        out = pad_or_truncate_pos_to_len(pos, target_len=5, max_cap=8)
        assert out.shape == (2, 5)
        # Both batch entries get the same monotonic extension.
        assert torch.equal(out[0], torch.tensor([0, 1, 2, 3, 4]))
        assert torch.equal(out[1], torch.tensor([0, 1, 2, 3, 4]))

    def test_extension_preserves_dtype(self):
        pos = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        out = pad_or_truncate_pos_to_len(pos, target_len=5, max_cap=8)
        assert out.dtype == torch.int64

    def test_truncates_when_above_max_cap(self):
        pos = torch.arange(20).reshape(2, 10)
        out = pad_or_truncate_pos_to_len(pos, target_len=4, max_cap=6)
        assert out.shape == (2, 6)
        assert torch.equal(out, pos[:, :6])

    def test_passthrough_when_between_target_and_cap(self):
        pos = torch.arange(12).reshape(2, 6)
        out = pad_or_truncate_pos_to_len(pos, target_len=4, max_cap=8)
        assert out is pos


class TestPadOrTruncateAttnToLen:
    def test_returns_none_for_none_input(self):
        assert pad_or_truncate_attn_to_len(None, target_len=4, max_cap=8) is None

    def test_pads_2d_bool_mask_with_false(self):
        mask = torch.tensor([[True, True, False]])
        out = pad_or_truncate_attn_to_len(mask, target_len=5, max_cap=8)
        assert out.shape == (1, 5)
        assert out.dtype == torch.bool
        assert torch.equal(out, torch.tensor([[True, True, False, False, False]]))

    def test_pads_2d_int_mask_with_zero(self):
        mask = torch.tensor([[1, 1, 0]])
        out = pad_or_truncate_attn_to_len(mask, target_len=5, max_cap=8)
        assert torch.equal(out, torch.tensor([[1, 1, 0, 0, 0]]))

    def test_truncates_2d_mask_when_above_max_cap(self):
        mask = torch.ones((2, 10), dtype=torch.bool)
        out = pad_or_truncate_attn_to_len(mask, target_len=4, max_cap=6)
        assert out.shape == (2, 6)

    def test_passthrough_2d_mask_when_between_target_and_cap(self):
        mask = torch.ones((2, 6), dtype=torch.bool)
        out = pad_or_truncate_attn_to_len(mask, target_len=4, max_cap=8)
        assert out is mask

    def test_pads_4d_mask_in_both_seq_dims(self):
        # (batch=1, heads=2, s1=3, s2=3)
        mask = torch.ones((1, 2, 3, 3), dtype=torch.bool)
        out = pad_or_truncate_attn_to_len(mask, target_len=5, max_cap=8)
        # padded in both s1 and s2 axes to length 5
        assert out.shape == (1, 2, 5, 5)
        # original block stays intact at top-left
        assert torch.all(out[:, :, :3, :3])
        # padded regions are False
        assert not torch.any(out[:, :, 3:, :])
        assert not torch.any(out[:, :, :, 3:])

    def test_truncates_4d_mask_when_above_max_cap(self):
        mask = torch.ones((1, 2, 10, 10), dtype=torch.bool)
        out = pad_or_truncate_attn_to_len(mask, target_len=4, max_cap=6)
        assert out.shape == (1, 2, 6, 6)

    def test_passthrough_4d_mask_when_between_target_and_cap(self):
        mask = torch.ones((1, 2, 6, 6), dtype=torch.bool)
        out = pad_or_truncate_attn_to_len(mask, target_len=4, max_cap=8)
        assert out is mask

    def test_raises_for_unsupported_rank(self):
        # 3D mask is not supported
        bad = torch.ones((1, 4, 4), dtype=torch.bool)
        with pytest.raises(ValueError, match="attention mask must be 2D or 4D"):
            pad_or_truncate_attn_to_len(bad, target_len=8, max_cap=8)
