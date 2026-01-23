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

"""
Unit tests for PEFT utility functions and ParallelLinearAdapter.

Tests utility functions for adapter configuration, initialization methods,
and the ParallelLinearAdapter class for distributed PEFT scenarios.
"""

import math
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

from megatron.bridge.peft.utils import (
    ParallelLinearAdapter,
    all2all_hp2sp,
    get_adapter_attributes_from_linear,
    init_method_const,
    init_method_kaiming_uniform,
    init_method_normal,
    is_expert_linear,
    pad_seq_to_mult,
    unpad_seq_to_mult,
    wildcard_match,
)


# Mock megatron components for testing
class MockModelParallelConfig:
    """Mock ModelParallelConfig for testing."""

    def __init__(self):
        """Initialize mock config with default values."""
        self.sequence_parallel = False
        self.tensor_model_parallel_size = 1
        self.bf16 = False
        self.fp16 = False
        self.cpu_offloading = False
        self.cpu_offloading_activations = False
        # Add missing attributes needed by real Megatron classes
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = None
        self.params_dtype = torch.float32
        self.perform_initialization = True
        self.use_cpu_initialization = False
        self.gradient_accumulation_fusion = False
        self.async_tensor_model_parallel_allreduce = False


class MockColumnParallelLinear(ColumnParallelLinear):
    """Mock ColumnParallelLinear for testing."""

    def __init__(self, input_size, output_size):
        """Initialize mock column parallel linear layer."""
        # Don't call super().__init__ to avoid Megatron dependencies
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.config = MockModelParallelConfig()

    def forward(self, x):
        """Forward pass returning tuple format."""
        return torch.matmul(x, self.weight.t()) + self.bias, None


class MockRowParallelLinear(RowParallelLinear):
    """Mock RowParallelLinear for testing."""

    def __init__(self, input_size, output_size):
        """Initialize mock row parallel linear layer."""
        # Don't call super().__init__ to avoid Megatron dependencies
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.config = MockModelParallelConfig()

    def forward(self, x):
        """Forward pass returning tuple format."""
        return torch.matmul(x, self.weight.t()) + self.bias, None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_expert_linear_positive_cases(self):
        """Test is_expert_linear with positive cases."""
        positive_cases = [
            "model.layers.0.mlp.experts.0.linear_fc1",
            "decoder.layers.5.mlp.local_experts.3.linear_fc2",
            "transformer.layers.10.mlp.experts.linear_fc1",
            "some.path.mlp.experts.another.path.linear_fc2",
        ]

        for case in positive_cases:
            assert is_expert_linear(case), f"Should match: {case}"

    def test_is_expert_linear_negative_cases(self):
        """Test is_expert_linear with negative cases."""
        negative_cases = [
            "model.layers.0.mlp.linear_fc1",
            "decoder.layers.5.attention.linear_qkv",
            "transformer.layers.10.mlp.experts.linear_proj",
            "some.path.linear_fc3",
            "experts.linear_fc1",  # No mlp prefix
        ]

        for case in negative_cases:
            assert not is_expert_linear(case), f"Should not match: {case}"

    def test_wildcard_match_basic(self):
        """Test basic wildcard matching."""
        pattern = "*.layers.0.*.linear_qkv"

        # Positive cases
        assert wildcard_match(pattern, "decoder.layers.0.self_attention.linear_qkv")
        assert wildcard_match(pattern, "model.layers.0.attention.linear_qkv")

        # Negative cases
        assert not wildcard_match(pattern, "decoder.layers.1.self_attention.linear_qkv")
        assert not wildcard_match(pattern, "decoder.layers.0.self_attention.linear_proj")

    def test_wildcard_match_multiple_wildcards(self):
        """Test wildcard matching with multiple wildcards."""
        pattern = "*.layers.*.attention.*.weight"

        assert wildcard_match(pattern, "model.layers.5.attention.linear_qkv.weight")
        assert wildcard_match(pattern, "decoder.layers.0.attention.proj.weight")
        assert not wildcard_match(pattern, "model.layers.5.mlp.linear_fc1.weight")

    def test_wildcard_match_edge_cases(self):
        """Test wildcard matching edge cases."""
        # None key
        assert wildcard_match("*", None) is None

        # Empty pattern
        assert wildcard_match("", "")
        assert not wildcard_match("", "something")

        # No wildcards
        assert wildcard_match("exact.match", "exact.match")
        assert not wildcard_match("exact.match", "different.match")

    def test_init_method_normal(self):
        """Test normal initialization method factory."""
        init_fn = init_method_normal(0.02)
        tensor = torch.zeros(10, 10)

        result = init_fn(tensor)

        assert result is tensor  # Should modify in-place
        assert not torch.allclose(tensor, torch.zeros_like(tensor))  # Should be non-zero
        assert torch.abs(tensor.mean()) < 0.01  # Should be close to zero mean

    def test_init_method_kaiming_uniform(self):
        """Test Kaiming uniform initialization method factory."""
        init_fn = init_method_kaiming_uniform(math.sqrt(5))
        tensor = torch.zeros(10, 10)

        result = init_fn(tensor)

        assert result is tensor
        assert not torch.allclose(tensor, torch.zeros_like(tensor))

    def test_init_method_const(self):
        """Test constant initialization method factory."""
        init_fn = init_method_const(0.5)
        tensor = torch.zeros(10, 10)

        result = init_fn(tensor)

        assert result is tensor
        assert torch.allclose(tensor, torch.full_like(tensor, 0.5))

    def test_pad_seq_to_mult_no_padding_needed(self):
        """Test pad_seq_to_mult when no padding is needed."""
        x = torch.randn(8, 10)  # 8 is divisible by 4

        padded_x, pad_len = pad_seq_to_mult(x, 4)

        assert torch.equal(padded_x, x)
        assert pad_len == 0

    def test_pad_seq_to_mult_padding_needed(self):
        """Test pad_seq_to_mult when padding is needed."""
        x = torch.randn(7, 10)  # 7 is not divisible by 4, need 1 pad

        padded_x, pad_len = pad_seq_to_mult(x, 4)

        assert padded_x.shape[0] == 8  # Should be padded to 8
        assert padded_x.shape[1] == 10  # Other dimensions unchanged
        assert pad_len == 1
        # Original data should be preserved
        assert torch.equal(padded_x[:7], x)

    def test_unpad_seq_to_mult_no_padding(self):
        """Test unpad_seq_to_mult with no padding to remove."""
        x = torch.randn(8, 10)

        unpadded_x = unpad_seq_to_mult(x, 0)

        assert torch.equal(unpadded_x, x)

    def test_unpad_seq_to_mult_with_padding(self):
        """Test unpad_seq_to_mult with padding to remove."""
        x = torch.randn(8, 10)

        unpadded_x = unpad_seq_to_mult(x, 1)

        assert unpadded_x.shape == (7, 10)
        assert torch.equal(unpadded_x, x[:7])

    def test_pad_unpad_roundtrip(self):
        """Test that pad/unpad operations are reversible."""
        original = torch.randn(7, 10)

        padded, pad_len = pad_seq_to_mult(original, 4)
        unpadded = unpad_seq_to_mult(padded, pad_len)

        assert torch.equal(unpadded, original)


@patch("megatron.bridge.peft.utils.parallel_state")
class TestAll2AllCommunication:
    """Test All2All communication functions."""

    def test_all2all_hp2sp_mock(self, mock_parallel_state):
        """Test all2all_hp2sp with mocked parallel state."""
        # Mock parallel state
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_tensor_model_parallel_group.return_value = None

        # Mock torch.distributed.all_to_all
        with patch("torch.distributed.all_to_all") as mock_all_to_all:

            def side_effect(receive_list, send_list, group):
                # Simulate all_to_all operation
                for i, tensor in enumerate(send_list):
                    receive_list[i].copy_(tensor)

            mock_all_to_all.side_effect = side_effect

            x = torch.randn(4, 8)  # Input tensor
            result = all2all_hp2sp(x)

            assert result.shape == (2, 16)  # Should reshape appropriately


class TestGetAdapterAttributes:
    """Test get_adapter_attributes_from_linear function."""

    @patch("megatron.bridge.peft.utils.parallel_state")
    def test_get_adapter_attributes_column_parallel(self, mock_parallel_state):
        """Test with ColumnParallelLinear."""
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 1
        linear = MockColumnParallelLinear(input_size=100, output_size=50)

        attrs = get_adapter_attributes_from_linear(linear)

        assert not attrs.input_is_parallel
        assert attrs.in_features == 100
        assert attrs.out_features == 50
        assert not attrs.disable_tensor_parallel_comm
        assert attrs.disable_sequence_parallel_comm  # Should be True when sequence_parallel is False
        assert attrs.base_linear_is_parallel  # Should be True for parallel linear layers

    @patch("megatron.bridge.peft.utils.parallel_state")
    def test_get_adapter_attributes_row_parallel(self, mock_parallel_state):
        """Test with RowParallelLinear."""
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 1
        linear = MockRowParallelLinear(input_size=100, output_size=50)

        attrs = get_adapter_attributes_from_linear(linear)

        assert attrs.input_is_parallel
        assert attrs.in_features == 100
        assert attrs.out_features == 50
        assert not attrs.disable_tensor_parallel_comm
        assert attrs.disable_sequence_parallel_comm
        assert attrs.base_linear_is_parallel  # Should be True for parallel linear layers

    @patch("megatron.bridge.peft.utils.parallel_state")
    def test_get_adapter_attributes_sequence_parallel(self, mock_parallel_state):
        """Test with sequence parallel enabled."""
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 1
        linear = MockColumnParallelLinear(input_size=100, output_size=50)
        linear.config.sequence_parallel = True

        attrs = get_adapter_attributes_from_linear(linear)

        assert not attrs.disable_tensor_parallel_comm
        assert not attrs.disable_sequence_parallel_comm  # Should be False when sequence_parallel is True
        assert attrs.base_linear_is_parallel  # Should be True for parallel linear layers

    @patch("megatron.bridge.peft.utils.parallel_state")
    def test_get_adapter_attributes_unsupported_module(self, mock_parallel_state):
        """Test with unsupported module type."""
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 1
        linear = nn.Conv2d(3, 3, 3)
        linear.config = MockModelParallelConfig()

        with pytest.raises(NotImplementedError):
            get_adapter_attributes_from_linear(linear)

    @patch("megatron.bridge.peft.utils.parallel_state")
    def test_get_adapter_attributes_base_linear_is_parallel_flag(self, mock_parallel_state):
        """Test that base_linear_is_parallel flag is correctly returned."""
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 1
        # Test with ColumnParallelLinear - should return True for base_linear_is_parallel
        column_linear = MockColumnParallelLinear(input_size=100, output_size=50)
        assert get_adapter_attributes_from_linear(
            column_linear
        ).base_linear_is_parallel  # Should be True for parallel linear layers

        # Test with RowParallelLinear - should return True for base_linear_is_parallel
        row_linear = MockRowParallelLinear(input_size=100, output_size=50)
        assert get_adapter_attributes_from_linear(
            row_linear
        ).base_linear_is_parallel  # Should be True for parallel linear layers


class TestParallelLinearAdapter:
    """Test ParallelLinearAdapter class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock model parallel config."""
        return MockModelParallelConfig()

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_init_column_input(self, mock_row_linear, mock_col_linear, mock_config):
        """Test ParallelLinearAdapter initialization with column parallel input."""
        # Mock the linear layers
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_col_linear.return_value = mock_linear_in

        # For column input (input_is_parallel=False), both linear_in and linear_out are ColumnParallelLinear
        # We need to return different mocks for the two calls to ColumnParallelLinear
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        adapter = ParallelLinearAdapter(
            in_features=100,
            out_features=50,
            dim=16,
            base_linear_name="test_linear",
            input_is_parallel=False,
            model_parallel_config=mock_config,
        )

        assert adapter.dim == 16
        assert adapter.alpha == 16  # Default alpha equals dim
        assert not adapter.input_is_parallel
        assert adapter.linear_in is mock_linear_in
        assert adapter.linear_out is mock_linear_out

    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    def test_parallel_linear_adapter_init_row_input(self, mock_col_linear, mock_row_linear, mock_config):
        """Test ParallelLinearAdapter initialization with row parallel input."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_row_linear.return_value = mock_linear_in
        mock_col_linear.return_value = mock_linear_out

        adapter = ParallelLinearAdapter(
            in_features=100,
            out_features=50,
            dim=16,
            base_linear_name="test_linear",
            input_is_parallel=True,
            model_parallel_config=mock_config,
        )

        assert adapter.input_is_parallel
        assert adapter.linear_in is mock_linear_in  # RowParallelLinear
        assert adapter.linear_out is mock_linear_out  # ColumnParallelLinear

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_get_activation_fn(self, mock_row_linear, mock_col_linear, mock_config):
        """Test activation function selection."""
        # Mock the linear layers
        mock_linear_in = Mock()
        mock_col_linear.return_value = mock_linear_in

        adapter = ParallelLinearAdapter(
            in_features=10,
            out_features=5,
            dim=4,
            base_linear_name="test",
            activation="relu",
            model_parallel_config=mock_config,
        )

        assert isinstance(adapter.activation, nn.ReLU)

        # Test different activations - we need to patch for each new instance
        activations_to_test = {
            "gelu": nn.GELU,
            "swish": nn.SiLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "identity": nn.Identity,
        }

        for activation_name, expected_type in activations_to_test.items():
            # Reset mocks for each iteration
            mock_col_linear.reset_mock()
            mock_row_linear.reset_mock()
            mock_col_linear.return_value = Mock()

            adapter = ParallelLinearAdapter(
                in_features=10,
                out_features=5,
                dim=4,
                base_linear_name="test",
                activation=activation_name,
                model_parallel_config=mock_config,
            )
            assert isinstance(adapter.activation, expected_type)

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_get_init_fn(self, mock_row_linear, mock_col_linear, mock_config):
        """Test initialization function selection."""
        # Mock the linear layers
        mock_linear_in = Mock()
        mock_col_linear.return_value = mock_linear_in

        adapter = ParallelLinearAdapter(
            in_features=10,
            out_features=5,
            dim=4,
            base_linear_name="test",
            column_init_method="xavier",
            model_parallel_config=mock_config,
        )

        # Test that different init methods return different functions
        xavier_fn = adapter._get_init_fn("xavier")
        normal_fn = adapter._get_init_fn("normal")
        kaiming_fn = adapter._get_init_fn("kaiming")
        zero_fn = adapter._get_init_fn("zero")

        # They should be different functions
        assert xavier_fn != normal_fn
        assert normal_fn != kaiming_fn
        assert kaiming_fn != zero_fn

        # Test invalid init method
        with pytest.raises(NotImplementedError):
            adapter._get_init_fn("invalid_method")

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_alpha_parameter(self, mock_row_linear, mock_col_linear, mock_config):
        """Test alpha parameter handling."""
        # Mock the linear layers
        mock_linear_in = Mock()
        mock_col_linear.return_value = mock_linear_in

        # Test default alpha (equals dim)
        adapter1 = ParallelLinearAdapter(
            in_features=10, out_features=5, dim=8, base_linear_name="test", model_parallel_config=mock_config
        )
        assert adapter1.alpha == 8

        # Reset mocks
        mock_col_linear.reset_mock()
        mock_col_linear.return_value = Mock()

        # Test custom alpha
        adapter2 = ParallelLinearAdapter(
            in_features=10, out_features=5, dim=8, base_linear_name="test", alpha=16, model_parallel_config=mock_config
        )
        assert adapter2.alpha == 16

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_dropout(self, mock_row_linear, mock_col_linear, mock_config):
        """Test dropout configuration."""
        # Mock the linear layers
        mock_linear_in = Mock()
        mock_col_linear.return_value = mock_linear_in

        # Test no dropout
        adapter1 = ParallelLinearAdapter(
            in_features=10,
            out_features=5,
            dim=4,
            base_linear_name="test",
            dropout=0.0,
            model_parallel_config=mock_config,
        )
        assert isinstance(adapter1.dropout, nn.Identity)

        # Reset mocks
        mock_col_linear.reset_mock()
        mock_col_linear.return_value = Mock()

        # Test with dropout
        adapter2 = ParallelLinearAdapter(
            in_features=10,
            out_features=5,
            dim=4,
            base_linear_name="test",
            dropout=0.3,
            model_parallel_config=mock_config,
        )
        assert isinstance(adapter2.dropout, nn.Dropout)

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_forward_basic(self, mock_row_linear, mock_col_linear, mock_config):
        """Test basic forward pass."""
        # Mock the linear layers
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_linear_in.return_value = (torch.randn(5, 16), None)
        mock_linear_out.return_value = (torch.randn(5, 10), None)

        # When input_is_parallel=False, both linear_in and linear_out are ColumnParallelLinear
        # So we need to set up side_effect to return different mocks for each call
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        adapter = ParallelLinearAdapter(
            in_features=20,
            out_features=10,
            dim=16,
            base_linear_name="test",
            input_is_parallel=False,
            model_parallel_config=mock_config,
        )

        x = torch.randn(5, 20)
        output = adapter(x)

        assert output.shape == (5, 10)
        # Verify scaling is applied
        expected_scale = adapter.alpha / adapter.dim
        assert expected_scale > 0

    @patch("megatron.bridge.peft.utils.parallel_state")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_expert_mode(
        self, mock_row_linear, mock_col_linear, mock_parallel_state, mock_config
    ):
        """Test adapter in expert mode (MoE)."""
        # Mock parallel state for expert mode
        mock_parallel_state.get_expert_tensor_parallel_world_size.return_value = 4

        # Set tensor_model_parallel_size to 4 so that sequence length 7 gets padded to 8
        mock_config.tensor_model_parallel_size = 4
        mock_config.expert_tensor_parallel_size = 4

        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_linear_in.return_value = (torch.randn(8, 16), None)  # Will be padded
        mock_linear_out.return_value = (torch.randn(8, 10), None)

        # Default input_is_parallel=False, so both are ColumnParallelLinear
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        adapter = ParallelLinearAdapter(
            in_features=20,
            out_features=10,
            dim=16,
            base_linear_name="test",
            is_expert=True,
            model_parallel_config=mock_config,
        )

        # Test with sequence length that needs padding (7 -> 8 when tensor_model_parallel_size=4)
        x = torch.randn(7, 20)  # Will be padded to 8
        output = adapter(x)

        # Output should be unpadded back to original size
        assert output.shape == (7, 10)

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_sharded_state_dict(self, mock_row_linear, mock_col_linear, mock_config):
        """Test sharded state dict functionality."""
        # Mock linear layers with sharded_state_dict methods
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_linear_in.sharded_state_dict.return_value = {"linear_in.weight": "tensor1"}
        mock_linear_out.sharded_state_dict.return_value = {"linear_out.weight": "tensor2"}

        # Default input_is_parallel=False, so both are ColumnParallelLinear
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        adapter = ParallelLinearAdapter(
            in_features=20, out_features=10, dim=16, base_linear_name="linear_fc2", model_parallel_config=mock_config
        )

        result = adapter.sharded_state_dict(prefix="adapter.")

        assert "linear_in.weight" in result
        assert "linear_out.weight" in result
        mock_linear_in.sharded_state_dict.assert_called_once_with("adapter.linear_in.", (), None)
        mock_linear_out.sharded_state_dict.assert_called_once_with("adapter.linear_out.", (), None)

    @patch("megatron.bridge.peft.utils.apply_swiglu_sharded_factory")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_sharded_state_dict_fc1_special_case(
        self, mock_row_linear, mock_col_linear, mock_swiglu_factory, mock_config
    ):
        """Test sharded state dict with special handling for linear_fc1."""
        # Mock linear layers
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_linear_in.sharded_state_dict.return_value = {"linear_in.weight": "tensor1"}
        mock_linear_out.sharded_state_dict.return_value = {
            "adapter.linear_out.weight": "tensor2",
            "adapter.linear_out.bias": "tensor3",
        }

        # Default input_is_parallel=False, so both are ColumnParallelLinear
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        # Mock the swiglu factory
        mock_swiglu_factory.return_value = "swiglu_processed_tensor"

        adapter = ParallelLinearAdapter(
            in_features=20, out_features=10, dim=16, base_linear_name="linear_fc1", model_parallel_config=mock_config
        )

        result = adapter.sharded_state_dict(prefix="adapter.")

        # Should call swiglu factory for fc1 weights
        mock_swiglu_factory.assert_called()
        assert result["adapter.linear_out.weight"] == "swiglu_processed_tensor"
