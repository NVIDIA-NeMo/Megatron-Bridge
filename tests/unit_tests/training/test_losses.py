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

"""Unit tests for megatron.bridge.training.losses module."""

from functools import partial
from unittest.mock import MagicMock, patch

import torch

from megatron.bridge.training.losses import (
    _DEFAULT_SPIKY_LOSS_FACTOR,
    create_masked_next_token_loss_function,
    masked_next_token_loss,
)


class TestMaskedNextTokenLoss:
    """Test cases for the masked_next_token_loss function."""

    def test_tuple_output_tensor_conversion(self):
        """Test that tuple output_tensor (lines 63-64) is correctly unpacked and converted to float."""
        # Setup: Create a tuple output_tensor with losses and loss_mask
        batch_size, seq_len = 4, 8

        # Create losses as int32 to test float conversion
        losses_tensor = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.int32)
        mask_tensor = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int32)

        # Create tuple output (as would come from LLaVA-style models)
        output_tensor = (losses_tensor, mask_tensor)

        # Create initial loss_mask (will be overridden by tuple[1])
        initial_loss_mask = torch.ones(batch_size, seq_len)

        # Mock the rerun_state_machine to avoid dependency issues
        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_rsm.return_value = MagicMock()

            # Execute
            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=initial_loss_mask,
                output_tensor=output_tensor,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            )

        # Verify: Check that losses were extracted and converted to float
        expected_losses = losses_tensor.view(-1).float()
        expected_mask = mask_tensor.view(-1).float()
        expected_loss = torch.sum(expected_losses * expected_mask)

        assert torch.isclose(loss, expected_loss), f"Loss mismatch: got {loss.item()}, expected {expected_loss.item()}"
        assert num_tokens == expected_mask.sum().to(torch.int), (
            f"Token count mismatch: got {num_tokens.item()}, expected {expected_mask.sum().item()}"
        )

    def test_tuple_output_tensor_shape_transformation(self):
        """Test that tuple tensors are correctly flattened with .view(-1)."""
        # Create 3D tensors to test view transformation
        losses_tensor = torch.randn(2, 4, 8)  # [batch, seq, hidden]
        mask_tensor = torch.ones(2, 4, 8)

        output_tensor = (losses_tensor, mask_tensor)
        initial_loss_mask = torch.ones(2, 4, 8)

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_rsm.return_value = MagicMock()

            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=initial_loss_mask,
                output_tensor=output_tensor,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            )

        # Verify flattening happened correctly
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert num_tokens.ndim == 0, f"num_tokens should be scalar, got shape {num_tokens.shape}"
        assert num_tokens.item() == 2 * 4 * 8, "All tokens should be counted"

    def test_tuple_output_tensor_dtype_conversion(self):
        """Test that tuple tensors are converted to float regardless of input dtype."""
        # Test with various dtypes
        dtypes = [torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float64]

        for dtype in dtypes:
            losses_tensor = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
            mask_tensor = torch.tensor([[1, 0], [1, 1]], dtype=dtype)

            output_tensor = (losses_tensor, mask_tensor)
            initial_loss_mask = torch.ones(2, 2)

            with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
                mock_rsm.return_value = MagicMock()

                loss, num_tokens, reporting = masked_next_token_loss(
                    loss_mask=initial_loss_mask,
                    output_tensor=output_tensor,
                    check_for_nan_in_loss=False,
                    check_for_spiky_loss=False,
                )

            # Verify conversion to float32
            assert loss.dtype == torch.float32, f"Loss should be float32, got {loss.dtype} for input {dtype}"
            expected_loss = 1.0 * 1 + 2.0 * 0 + 3.0 * 1 + 4.0 * 1
            assert torch.isclose(loss, torch.tensor(expected_loss)), f"Loss calculation incorrect for dtype {dtype}"

    def test_single_tensor_output(self):
        """Test that non-tuple output_tensor still works correctly."""
        # Create a single tensor output (standard case)
        losses_tensor = torch.randn(4, 8)
        loss_mask = torch.randint(0, 2, (4, 8)).float()

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_rsm.return_value = MagicMock()

            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=loss_mask,
                output_tensor=losses_tensor,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            )

        # Verify
        expected_loss = torch.sum(losses_tensor.view(-1).float() * loss_mask.view(-1))
        assert torch.isclose(loss, expected_loss), "Loss mismatch for single tensor output"

    def test_nan_detection(self):
        """Test that NaN values are detected when check_for_nan_in_loss=True."""
        losses_tensor = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]])
        mask_tensor = torch.ones(2, 2)

        output_tensor = (losses_tensor, mask_tensor)
        initial_loss_mask = torch.ones(2, 2)

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_state_machine = MagicMock()
            mock_rsm.return_value = mock_state_machine

            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=initial_loss_mask,
                output_tensor=output_tensor,
                check_for_nan_in_loss=True,
                check_for_spiky_loss=False,
            )

            # Verify that validate_result was called for NaN check
            assert mock_state_machine.validate_result.call_count >= 2
            # First call should be for NaN check
            first_call = mock_state_machine.validate_result.call_args_list[0]
            assert first_call[1]["rejection_func"] == torch.isnan
            assert "NaN" in first_call[1]["message"]

    def test_inf_detection(self):
        """Test that Inf values are detected when check_for_nan_in_loss=True."""
        losses_tensor = torch.tensor([[1.0, 2.0], [float("inf"), 4.0]])
        mask_tensor = torch.ones(2, 2)

        output_tensor = (losses_tensor, mask_tensor)
        initial_loss_mask = torch.ones(2, 2)

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_state_machine = MagicMock()
            mock_rsm.return_value = mock_state_machine

            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=initial_loss_mask,
                output_tensor=output_tensor,
                check_for_nan_in_loss=True,
                check_for_spiky_loss=False,
            )

            # Verify that validate_result was called for Inf check
            assert mock_state_machine.validate_result.call_count >= 2
            # Second call should be for Inf check
            second_call = mock_state_machine.validate_result.call_args_list[1]
            assert second_call[1]["rejection_func"] == torch.isinf
            assert "Inf" in second_call[1]["message"]

    def test_spiky_loss_detection(self):
        """Test that spiky loss is detected when check_for_spiky_loss=True."""
        losses_tensor = torch.ones(2, 2) * 100.0  # Artificially high loss
        mask_tensor = torch.ones(2, 2)

        output_tensor = (losses_tensor, mask_tensor)
        initial_loss_mask = torch.ones(2, 2)

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_state_machine = MagicMock()
            mock_rsm.return_value = mock_state_machine

            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=initial_loss_mask,
                output_tensor=output_tensor,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=True,
            )

            # Verify that validate_result was called for spiky loss check
            assert mock_state_machine.validate_result.call_count >= 1
            call = mock_state_machine.validate_result.call_args_list[0]
            assert "Spiky loss" in call[1]["message"]
            assert call[1]["fatal"] is False  # Spiky loss should not be fatal

    def test_reporting_metrics(self):
        """Test that reporting metrics are correctly returned."""
        losses_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask_tensor = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        output_tensor = (losses_tensor, mask_tensor)
        initial_loss_mask = torch.ones(2, 2)

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_rsm.return_value = MagicMock()

            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=initial_loss_mask,
                output_tensor=output_tensor,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            )

        # Verify reporting metrics
        assert "lm loss" in reporting
        assert reporting["lm loss"].shape == (2,)
        assert reporting["lm loss"][0] == loss
        assert reporting["lm loss"][1] == num_tokens

    def test_loss_mask_override_with_tuple(self):
        """Test that loss_mask is correctly overridden when output_tensor is a tuple."""
        # Create different masks
        initial_loss_mask = torch.ones(2, 2)
        override_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        losses_tensor = torch.ones(2, 2)

        output_tensor = (losses_tensor, override_mask)

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_rsm.return_value = MagicMock()

            loss, num_tokens, reporting = masked_next_token_loss(
                loss_mask=initial_loss_mask,
                output_tensor=output_tensor,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            )

        # Loss should use override_mask, not initial_loss_mask
        expected_loss = torch.sum(losses_tensor.view(-1) * override_mask.view(-1))
        assert torch.isclose(loss, expected_loss), "Loss should use overridden mask from tuple"
        assert num_tokens.item() == 2, "Only 2 tokens should be counted (diagonal elements)"


class TestCreateMaskedNextTokenLossFunction:
    """Test cases for create_masked_next_token_loss_function."""

    def test_returns_partial_function(self):
        """Test that the function returns a partial function."""
        loss_mask = torch.ones(4, 8)
        loss_fn = create_masked_next_token_loss_function(
            loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False
        )

        assert isinstance(loss_fn, partial), "Should return a partial function"
        assert loss_fn.func == masked_next_token_loss, "Should wrap masked_next_token_loss"

    def test_partial_function_parameters(self):
        """Test that the partial function has correct parameters."""
        loss_mask = torch.ones(4, 8)
        loss_fn = create_masked_next_token_loss_function(
            loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=True
        )

        # Check that parameters are bound correctly
        assert loss_fn.args[0] is loss_mask, "loss_mask should be first argument"
        assert loss_fn.keywords["check_for_nan_in_loss"] is True
        assert loss_fn.keywords["check_for_spiky_loss"] is True

    def test_partial_function_execution(self):
        """Test that the partial function can be executed."""
        loss_mask = torch.ones(2, 2)
        loss_fn = create_masked_next_token_loss_function(
            loss_mask=loss_mask, check_for_nan_in_loss=False, check_for_spiky_loss=False
        )

        losses_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask_tensor = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        output_tensor = (losses_tensor, mask_tensor)

        with patch("megatron.bridge.training.losses.get_rerun_state_machine") as mock_rsm:
            mock_rsm.return_value = MagicMock()

            # Execute the partial function
            loss, num_tokens, reporting = loss_fn(output_tensor)

        expected_loss = 1.0 + 3.0 + 4.0
        assert torch.isclose(loss, torch.tensor(expected_loss)), "Partial function should work correctly"


class TestConstants:
    """Test module constants."""

    def test_default_spiky_loss_factor(self):
        """Test that _DEFAULT_SPIKY_LOSS_FACTOR has expected value."""
        assert _DEFAULT_SPIKY_LOSS_FACTOR == 10.0, "_DEFAULT_SPIKY_LOSS_FACTOR should be 10.0"
        assert isinstance(_DEFAULT_SPIKY_LOSS_FACTOR, float), "_DEFAULT_SPIKY_LOSS_FACTOR should be a float"
