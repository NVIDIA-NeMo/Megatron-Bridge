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

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop


@pytest.mark.unit
class TestApplyMoETokenDrop:
    """Test cases for the apply_moe_token_drop function."""

    @pytest.fixture
    def valid_model_provider(self):
        """Create a valid GPTModelProvider mock for testing."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = "alltoall"
        mock_provider.moe_router_load_balancing_type = "aux_loss"
        return mock_provider

    def test_apply_moe_token_drop_default_parameters(self, valid_model_provider):
        """Test apply_moe_token_drop with default parameters."""
        apply_moe_token_drop(valid_model_provider)

        assert valid_model_provider.moe_expert_capacity_factor == 1.0
        assert valid_model_provider.moe_pad_expert_input_to_capacity is True

    def test_apply_moe_token_drop_custom_capacity_factor(self, valid_model_provider):
        """Test apply_moe_token_drop with custom capacity factor."""
        apply_moe_token_drop(valid_model_provider, moe_expert_capacity_factor=2.5)

        assert valid_model_provider.moe_expert_capacity_factor == 2.5
        assert valid_model_provider.moe_pad_expert_input_to_capacity is True

    def test_apply_moe_token_drop_disable_padding(self, valid_model_provider):
        """Test apply_moe_token_drop with padding disabled."""
        apply_moe_token_drop(valid_model_provider, moe_pad_expert_input_to_capacity=False)

        assert valid_model_provider.moe_expert_capacity_factor == 1.0
        assert valid_model_provider.moe_pad_expert_input_to_capacity is False

    def test_apply_moe_token_drop_custom_parameters(self, valid_model_provider):
        """Test apply_moe_token_drop with custom parameters."""
        apply_moe_token_drop(
            valid_model_provider,
            moe_expert_capacity_factor=1.5,
            moe_pad_expert_input_to_capacity=False,
        )

        assert valid_model_provider.moe_expert_capacity_factor == 1.5
        assert valid_model_provider.moe_pad_expert_input_to_capacity is False

    def test_apply_moe_token_drop_negative_capacity_factor_with_padding_disabled(self, valid_model_provider):
        """Test apply_moe_token_drop with negative capacity factor sets to None when padding is disabled."""
        apply_moe_token_drop(
            valid_model_provider, moe_expert_capacity_factor=-1.0, moe_pad_expert_input_to_capacity=False
        )

        assert valid_model_provider.moe_expert_capacity_factor is None
        assert valid_model_provider.moe_pad_expert_input_to_capacity is False

    def test_apply_moe_token_drop_negative_capacity_factor_with_padding_raises_error(self, valid_model_provider):
        """Test apply_moe_token_drop with negative capacity factor raises ValueError when padding is enabled."""
        with pytest.raises(
            ValueError, match="moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity"
        ):
            apply_moe_token_drop(
                valid_model_provider, moe_expert_capacity_factor=-1.0, moe_pad_expert_input_to_capacity=True
            )

    def test_apply_moe_token_drop_zero_capacity_factor(self, valid_model_provider):
        """Test apply_moe_token_drop with zero capacity factor."""
        apply_moe_token_drop(valid_model_provider, moe_expert_capacity_factor=0.0)

        assert valid_model_provider.moe_expert_capacity_factor == 0.0
        assert valid_model_provider.moe_pad_expert_input_to_capacity is True

    @pytest.mark.parametrize("dispatcher_type", ["alltoall", "alltoall_seq"])
    def test_apply_moe_token_drop_valid_dispatcher_types(self, dispatcher_type):
        """Test apply_moe_token_drop with valid dispatcher types."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = dispatcher_type
        mock_provider.moe_router_load_balancing_type = "aux_loss"

        apply_moe_token_drop(mock_provider)

        assert mock_provider.moe_expert_capacity_factor == 1.0
        assert mock_provider.moe_pad_expert_input_to_capacity is True

    @pytest.mark.parametrize("load_balancing_type", ["seq_aux_loss", "aux_loss", "none"])
    def test_apply_moe_token_drop_valid_load_balancing_types(self, load_balancing_type):
        """Test apply_moe_token_drop with valid load balancing types."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = "alltoall"
        mock_provider.moe_router_load_balancing_type = load_balancing_type

        apply_moe_token_drop(mock_provider)

        assert mock_provider.moe_expert_capacity_factor == 1.0
        assert mock_provider.moe_pad_expert_input_to_capacity is True

    @pytest.mark.parametrize(
        "invalid_dispatcher",
        [
            "foo",
            "bar",
        ],
    )
    def test_apply_moe_token_drop_invalid_dispatcher_type(self, invalid_dispatcher):
        """Test apply_moe_token_drop with invalid dispatcher types raises AssertionError."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = invalid_dispatcher
        mock_provider.moe_router_load_balancing_type = "aux_loss"

        with pytest.raises(
            AssertionError, match="moe_expert_capacity_factor only works with alltoall token dispatcher"
        ):
            apply_moe_token_drop(mock_provider)

    @pytest.mark.parametrize("invalid_load_balancing", ["foo", "bar"])
    def test_apply_moe_token_drop_invalid_load_balancing_type(self, invalid_load_balancing):
        """Test apply_moe_token_drop with invalid load balancing types raises AssertionError."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = "alltoall"
        mock_provider.moe_router_load_balancing_type = invalid_load_balancing

        with pytest.raises(
            AssertionError, match="moe_expert_capacity_factor only works with aux_loss or none load balancing"
        ):
            apply_moe_token_drop(mock_provider)

    def test_apply_moe_token_drop_padding_with_none_capacity_factor_raises_error(self):
        """Test that enabling padding with None capacity factor raises ValueError."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = "alltoall"
        mock_provider.moe_router_load_balancing_type = "aux_loss"

        with pytest.raises(
            ValueError, match="moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity"
        ):
            apply_moe_token_drop(
                mock_provider,
                moe_expert_capacity_factor=-1.0,  # This becomes None
                moe_pad_expert_input_to_capacity=True,
            )

    def test_apply_moe_token_drop_padding_disabled_with_none_capacity_factor(self):
        """Test that disabling padding with None capacity factor works."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = "alltoall"
        mock_provider.moe_router_load_balancing_type = "aux_loss"

        apply_moe_token_drop(
            mock_provider,
            moe_expert_capacity_factor=-1.0,  # This becomes None
            moe_pad_expert_input_to_capacity=False,
        )

        assert mock_provider.moe_expert_capacity_factor is None
        assert mock_provider.moe_pad_expert_input_to_capacity is False

    @pytest.mark.parametrize("capacity_factor", [0.5, 1.0, 1.5, 2.0, 10.0])
    def test_apply_moe_token_drop_various_capacity_factors(self, valid_model_provider, capacity_factor):
        """Test apply_moe_token_drop with various positive capacity factors."""
        apply_moe_token_drop(valid_model_provider, moe_expert_capacity_factor=capacity_factor)

        assert valid_model_provider.moe_expert_capacity_factor == capacity_factor
        assert valid_model_provider.moe_pad_expert_input_to_capacity is True

    def test_apply_moe_token_drop_combination_alltoall_seq_with_seq_aux_loss(self):
        """Test valid combination of alltoall_seq dispatcher with seq_aux_loss."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = "alltoall_seq"
        mock_provider.moe_router_load_balancing_type = "seq_aux_loss"

        apply_moe_token_drop(mock_provider, moe_expert_capacity_factor=1.2)

        assert mock_provider.moe_expert_capacity_factor == 1.2
        assert mock_provider.moe_pad_expert_input_to_capacity is True

    def test_apply_moe_token_drop_combination_alltoall_with_none_load_balancing(self):
        """Test valid combination of alltoall dispatcher with none load balancing."""
        mock_provider = Mock()
        mock_provider.num_moe_experts = 8  # MoE model
        mock_provider.moe_token_dispatcher_type = "alltoall"
        mock_provider.moe_router_load_balancing_type = "none"

        apply_moe_token_drop(mock_provider, moe_expert_capacity_factor=0.8)

        assert mock_provider.moe_expert_capacity_factor == 0.8
        assert mock_provider.moe_pad_expert_input_to_capacity is True

    @patch("megatron.bridge.training.utils.moe_token_drop.logger")
    def test_apply_moe_token_drop_warns_for_non_moe_model_none_experts(self, mock_logger):
        """Test that apply_moe_token_drop logs warning and returns early when num_moe_experts is None."""
        # Create a mock TransformerConfig without MoE
        mock_provider = Mock()
        mock_provider.num_moe_experts = None  # Explicitly set to None (default value)

        # Store initial state before applying
        initial_capacity_factor = getattr(mock_provider, "moe_expert_capacity_factor", "UNSET")
        initial_pad_capacity = getattr(mock_provider, "moe_pad_expert_input_to_capacity", "UNSET")

        # Apply MoE token drop
        apply_moe_token_drop(mock_provider)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "MoE token drop is only applicable to MoE models" in mock_logger.warning.call_args[0][0]

        # Verify configs were NOT modified (still in initial state)
        assert getattr(mock_provider, "moe_expert_capacity_factor", "UNSET") == initial_capacity_factor
        assert getattr(mock_provider, "moe_pad_expert_input_to_capacity", "UNSET") == initial_pad_capacity

    @patch("megatron.bridge.training.utils.moe_token_drop.logger")
    def test_apply_moe_token_drop_warns_for_non_moe_model_zero_experts(self, mock_logger):
        """Test that apply_moe_token_drop logs warning and returns early when num_moe_experts is 0."""
        # Create a mock TransformerConfig without MoE
        mock_provider = Mock()
        mock_provider.num_moe_experts = 0  # Explicitly set to 0

        # Store initial state before applying
        initial_capacity_factor = getattr(mock_provider, "moe_expert_capacity_factor", "UNSET")
        initial_pad_capacity = getattr(mock_provider, "moe_pad_expert_input_to_capacity", "UNSET")

        # Apply MoE token drop
        apply_moe_token_drop(mock_provider)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "MoE token drop is only applicable to MoE models" in mock_logger.warning.call_args[0][0]

        # Verify configs were NOT modified (still in initial state)
        assert getattr(mock_provider, "moe_expert_capacity_factor", "UNSET") == initial_capacity_factor
        assert getattr(mock_provider, "moe_pad_expert_input_to_capacity", "UNSET") == initial_pad_capacity
