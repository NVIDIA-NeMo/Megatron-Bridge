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

import torch

from megatron.bridge.training.state import TrainState


class TestTrainState:
    """Test suite for TrainState class."""

    def test_initialization_defaults(self):
        """Test that TrainState initializes with correct default values."""
        state = TrainState()

        assert state.step == 0
        assert state.consumed_train_samples == 0
        assert state.skipped_train_samples == 0
        assert state.consumed_valid_samples == 0
        assert state.floating_point_operations_so_far == 0
        assert state.do_train is False
        assert state.do_valid is False
        assert state.do_test is False

    def test_initialization_with_custom_values(self):
        """Test initialization with custom values."""
        state = TrainState(
            step=100,
            consumed_train_samples=1000,
            skipped_train_samples=50,
            consumed_valid_samples=200,
            floating_point_operations_so_far=1500,
            do_train=True,
            do_valid=True,
            do_test=False,
        )

        assert state.step == 100
        assert state.consumed_train_samples == 1000
        assert state.skipped_train_samples == 50
        assert state.consumed_valid_samples == 200
        assert state.floating_point_operations_so_far == 1500
        assert state.do_train is True
        assert state.do_valid is True
        assert state.do_test is False

    def test_state_dict_structure_and_types(self):
        """Test that state_dict returns correct structure and tensor types."""
        state = TrainState(
            step=42,
            consumed_train_samples=500,
            skipped_train_samples=25,
            consumed_valid_samples=100,
            floating_point_operations_so_far=2000,
            do_train=True,
            do_valid=False,
            do_test=True,
        )

        state_dict = state.state_dict()

        # Check all expected keys are present
        expected_keys = {
            "step",
            "consumed_train_samples",
            "skipped_train_samples",
            "consumed_valid_samples",
            "floating_point_operations_so_far",
            "do_train",
            "do_valid",
            "do_test",
        }
        assert set(state_dict.keys()) == expected_keys

        # Check tensor types
        assert state_dict["step"].dtype == torch.int32
        assert state_dict["consumed_train_samples"].dtype == torch.int32
        assert state_dict["skipped_train_samples"].dtype == torch.int32
        assert state_dict["consumed_valid_samples"].dtype == torch.int32
        assert state_dict["floating_point_operations_so_far"].dtype == torch.float64
        assert state_dict["do_train"].dtype == torch.bool
        assert state_dict["do_valid"].dtype == torch.bool
        assert state_dict["do_test"].dtype == torch.bool

        # Check values
        assert state_dict["step"].item() == 42
        assert state_dict["consumed_train_samples"].item() == 500
        assert state_dict["skipped_train_samples"].item() == 25
        assert state_dict["consumed_valid_samples"].item() == 100
        assert state_dict["floating_point_operations_so_far"].item() == 2000
        assert state_dict["do_train"].item() is True
        assert state_dict["do_valid"].item() is False
        assert state_dict["do_test"].item() is True

    def test_load_state_dict(self):
        """Test loading state from state dictionary."""
        # Create a state dict manually
        state_dict = {
            "step": torch.tensor(75, dtype=torch.int32),
            "consumed_train_samples": torch.tensor(800, dtype=torch.int32),
            "skipped_train_samples": torch.tensor(40, dtype=torch.int32),
            "consumed_valid_samples": torch.tensor(150, dtype=torch.int32),
            "floating_point_operations_so_far": torch.tensor(3000.5, dtype=torch.float64),
            "do_train": torch.tensor(False, dtype=torch.bool),
            "do_valid": torch.tensor(True, dtype=torch.bool),
            "do_test": torch.tensor(False, dtype=torch.bool),
        }

        state = TrainState()
        state.load_state_dict(state_dict)

        assert state.step == 75
        assert state.consumed_train_samples == 800
        assert state.skipped_train_samples == 40
        assert state.consumed_valid_samples == 150
        assert state.floating_point_operations_so_far == 3000.5
        assert state.do_train is False
        assert state.do_valid is True
        assert state.do_test is False

    def test_round_trip_serialization(self):
        """Test that state_dict -> load_state_dict preserves all values."""
        # Create original state with various values
        original_state = TrainState(
            step=123,
            consumed_train_samples=2500,
            skipped_train_samples=100,
            consumed_valid_samples=500,
            floating_point_operations_so_far=12345.67,
            do_train=True,
            do_valid=False,
            do_test=True,
        )

        # Serialize to state dict
        state_dict = original_state.state_dict()

        # Create new state and load from dict
        new_state = TrainState()
        new_state.load_state_dict(state_dict)

        # Verify all values are preserved
        assert new_state.step == original_state.step
        assert new_state.consumed_train_samples == original_state.consumed_train_samples
        assert new_state.skipped_train_samples == original_state.skipped_train_samples
        assert new_state.consumed_valid_samples == original_state.consumed_valid_samples
        assert new_state.floating_point_operations_so_far == original_state.floating_point_operations_so_far
        assert new_state.do_train == original_state.do_train
        assert new_state.do_valid == original_state.do_valid
        assert new_state.do_test == original_state.do_test

    def test_state_dict_with_defaults(self):
        """Test state_dict with default values."""
        state = TrainState()
        state_dict = state.state_dict()

        # All default values should be properly serialized
        assert state_dict["step"].item() == 0
        assert state_dict["consumed_train_samples"].item() == 0
        assert state_dict["skipped_train_samples"].item() == 0
        assert state_dict["consumed_valid_samples"].item() == 0
        assert state_dict["floating_point_operations_so_far"].item() == 0
        assert state_dict["do_train"].item() is False
        assert state_dict["do_valid"].item() is False
        assert state_dict["do_test"].item() is False
