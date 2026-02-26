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

import unittest
from functools import partial
from unittest.mock import MagicMock, patch

import torch

from megatron.bridge.diffusion.models.dit.dit_step import DITForwardStep


class TestDITForwardStep(unittest.TestCase):
    """Unit tests for DITForwardStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dit_forward_step = DITForwardStep()

    @patch("megatron.bridge.diffusion.models.dit.dit_step.parallel_state.is_pipeline_last_stage")
    def test_forward_step_with_pipeline_stages(self, mock_is_last_stage):
        """Test forward_step method for both last and non-last pipeline stages."""
        batch_size = 4

        # Test 1: Pipeline last stage with loss_mask
        mock_is_last_stage.return_value = True

        # Create mock state
        mock_state = MagicMock()
        mock_timers = MagicMock()
        mock_state.timers = lambda x: mock_timers
        mock_state.cfg.rerun_state_machine.check_for_nan_in_loss = True
        mock_state.cfg.rerun_state_machine.check_for_spiky_loss = False
        mock_straggler_timer = MagicMock()
        mock_state.straggler_timer = mock_straggler_timer

        # Create mock model
        mock_model = MagicMock()

        # Create batch with loss_mask
        loss_mask = torch.ones(batch_size)
        batch = {"loss_mask": loss_mask}

        # Mock diffusion pipeline output
        output_batch = MagicMock()
        mock_loss = torch.randn(batch_size, 10)  # Loss with additional dimension

        self.dit_forward_step.diffusion_pipeline.training_step = MagicMock(return_value=(output_batch, mock_loss))

        # Act
        output_tensor, loss_function = self.dit_forward_step.forward_step(
            mock_state, batch, mock_model, return_schedule_plan=False
        )

        # Assert for last stage
        mock_timers.stop.assert_called_once()
        self.dit_forward_step.diffusion_pipeline.training_step.assert_called_once_with(mock_model, batch, 0)

        # Verify output tensor has the correct shape (mean over last dimension)
        self.assertEqual(output_tensor.shape, (batch_size,))
        torch.testing.assert_close(output_tensor, torch.mean(mock_loss, dim=-1))

        # Verify loss function is a partial function
        self.assertIsInstance(loss_function, partial)

        # Verify straggler timer was used as context manager
        mock_straggler_timer.__enter__.assert_called_once()
        mock_straggler_timer.__exit__.assert_called_once()

        # Test 2: NOT pipeline last stage
        mock_is_last_stage.return_value = False

        # Reset mocks for second test
        mock_timers.reset_mock()
        mock_straggler_timer.reset_mock()

        # Create new mock state
        mock_state = MagicMock()
        mock_state.timers = lambda x: mock_timers
        mock_state.cfg.rerun_state_machine.check_for_nan_in_loss = False
        mock_state.cfg.rerun_state_machine.check_for_spiky_loss = True
        mock_state.straggler_timer = mock_straggler_timer

        # Create batch without loss_mask
        batch = {}

        self.dit_forward_step.diffusion_pipeline.training_step = MagicMock(return_value=mock_loss)

        # Act
        output_tensor, loss_function = self.dit_forward_step.forward_step(
            mock_state, batch, mock_model, return_schedule_plan=False
        )

        # Assert for not last stage
        mock_timers.stop.assert_called_once()
        self.dit_forward_step.diffusion_pipeline.training_step.assert_called_once_with(mock_model, batch, 0)

        # Verify output tensor is directly returned (no mean operation)
        torch.testing.assert_close(output_tensor, mock_loss)

        # Verify loss function is a partial function
        self.assertIsInstance(loss_function, partial)

        # Verify straggler timer was used as context manager
        mock_straggler_timer.__enter__.assert_called_once()
        mock_straggler_timer.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
