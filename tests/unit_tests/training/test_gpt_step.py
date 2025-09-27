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

from functools import partial
from unittest.mock import patch

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.training.gpt_step import _create_loss_function, get_packed_seq_params


class TestGetPackedSeqParams:
    """Tests for the get_packed_seq_params function."""

    def test_basic_packed_seq_params_with_max_seqlen(self):
        """Test basic functionality with cu_seqlens and max_seqlen."""
        # Create test batch with packed sequence data
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 12, 20, -1, -1]], dtype=torch.int32),  # batch size 1
            "max_seqlen": torch.tensor([[15]], dtype=torch.int32),  # batch size 1
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was squeezed and padding removed (stops at first -1)
        expected_cu_seqlens = torch.tensor([0, 5, 12, 20], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen was squeezed
        expected_max_seqlen = torch.tensor(15, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)
        assert torch.equal(result.max_seqlen_kv, expected_max_seqlen)

        # Verify qkv_format is correct
        assert result.qkv_format == "thd"

    def test_packed_seq_params_without_max_seqlen(self):
        """Test functionality when max_seqlen is not provided."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 3, 8, 15, -1]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was processed correctly
        expected_cu_seqlens = torch.tensor([0, 3, 8, 15], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen is None when not provided
        assert result.max_seqlen_q is None
        assert result.max_seqlen_kv is None

        # Verify qkv_format is correct
        assert result.qkv_format == "thd"

    def test_packed_seq_params_with_cu_seqlens_argmin(self):
        """Test functionality when cu_seqlens_argmin is provided for performance."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 4, 9, 16, 22, -1, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(5),  # Index where -1 starts
            "max_seqlen": torch.tensor([[18]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was truncated using cu_seqlens_argmin
        expected_cu_seqlens = torch.tensor([0, 4, 9, 16, 22], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen was processed correctly
        expected_max_seqlen = torch.tensor(18, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)
        assert torch.equal(result.max_seqlen_kv, expected_max_seqlen)

    def test_packed_seq_params_no_padding(self):
        """Test functionality when cu_seqlens has no padding (-1 values)."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 7, 14]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[10]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # When there's no -1 padding, argmin returns 0 (index of min value)
        # So cu_seqlens[:0] returns empty tensor
        expected_cu_seqlens = torch.empty(0, dtype=torch.int32)  # Empty tensor
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

    def test_packed_seq_params_with_cu_seqlens_argmin_zero(self):
        """Test edge case when cu_seqlens_argmin is 0."""
        batch = {
            "cu_seqlens": torch.tensor([[-1, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(0),  # All are padding
        }

        result = get_packed_seq_params(batch)

        # Verify empty cu_seqlens when argmin is 0
        expected_cu_seqlens = torch.empty(0, dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

    def test_packed_seq_params_batch_dimension_removal(self):
        """Test that batch dimensions are properly squeezed."""
        # Test with different batch size dimensions
        batch = {
            "cu_seqlens": torch.tensor([[[0, 6, 12, -1]]], dtype=torch.int32),  # Shape [1, 1, 4]
            "max_seqlen": torch.tensor([[[20]]], dtype=torch.int32),  # Shape [1, 1, 1]
        }

        result = get_packed_seq_params(batch)

        # Verify dimensions were squeezed properly
        expected_cu_seqlens = torch.tensor([0, 6, 12], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)

        expected_max_seqlen = torch.tensor(20, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)

    def test_packed_seq_params_with_different_dtypes(self):
        """Test functionality with different tensor dtypes."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 10, 20, -1]], dtype=torch.int64),  # int64 instead of int32
            "max_seqlen": torch.tensor([[25]], dtype=torch.int64),
        }

        result = get_packed_seq_params(batch)

        # Function should handle different dtypes
        expected_cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.int64)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)

        expected_max_seqlen = torch.tensor(25, dtype=torch.int64)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)

    def test_packed_seq_params_all_fields_match(self):
        """Test that cu_seqlens_q/kv and max_seqlen_q/kv are identical."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 11, 18, -1]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[12]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify that q and kv parameters are identical (as expected for this function)
        assert torch.equal(result.cu_seqlens_q, result.cu_seqlens_kv)
        assert torch.equal(result.max_seqlen_q, result.max_seqlen_kv)


class TestCreateLossFunction:
    """Tests for the _create_loss_function helper function."""

    def test_create_loss_function_both_true(self):
        """Test create_loss_function with both flags as True."""
        loss_mask = torch.tensor([[1.0, 1.0, 0.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=True)

        # Verify it returns a partial function
        assert isinstance(loss_func, partial)
        assert loss_func.func.__name__ == "masked_next_token_loss"

        # Verify the partial has correct arguments
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == True
        assert loss_func.keywords["check_for_spiky_loss"] == True

    def test_create_loss_function_both_false(self):
        """Test _create_loss_function with both flags as False."""
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=False, check_for_spiky_loss=False)

        # Verify the partial has correct arguments
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == False
        assert loss_func.keywords["check_for_spiky_loss"] == False

    def test_create_loss_function_mixed_values(self):
        """Test create_loss_function with mixed flag values."""
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)

        # Verify the partial has correct mixed values
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == True
        assert loss_func.keywords["check_for_spiky_loss"] == False

    @patch("megatron.bridge.training.gpt_step.masked_next_token_loss")
    def test_create_loss_function_callable(self, mock_loss_func):
        """Test that the created loss function can be called correctly."""
        loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
        output_tensor = torch.tensor([2.5])

        # Mock return value
        expected_result = (torch.tensor(3.0), torch.tensor(2), {"lm loss": torch.tensor([3.0, 2.0])})
        mock_loss_func.return_value = expected_result

        # Create the loss function
        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)

        # Call the partial function
        result = loss_func(output_tensor)

        # Verify the underlying function was called correctly
        mock_loss_func.assert_called_once_with(
            loss_mask, output_tensor, check_for_nan_in_loss=True, check_for_spiky_loss=False
        )

        # Verify the result
        assert result == expected_result
