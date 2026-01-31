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
Unit tests for DoRA PEFT components.

Tests DoRA adapters, ParallelLinearDoRAAdapter, and DoRALinear
functionality for Parameter-Efficient Fine-Tuning.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedTensor

from megatron.bridge.peft.dora_layers import DoRALinear, ParallelLinearDoRAAdapter
from megatron.bridge.peft.utils import ParallelLinearAdapter
from tests.unit_tests.peft.test_utils import MockModelParallelConfig


class MockLinearWithTupleReturn(nn.Module):
    """Mock linear module that returns tuples like Megatron layers."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.config = MockModelParallelConfig()
        self.config.sequence_parallel = True

    @property
    def weight(self):
        """Delegate weight access to the internal linear layer."""
        return self.linear.weight

    @property
    def bias(self):
        """Delegate bias access to the internal linear layer."""
        return self.linear.bias

    def forward(self, x, *args, **kwargs):
        """Return tuple format like Megatron linear layers."""
        output = self.linear(x)
        return output, None  # (output, bias)


class MockParallelLinearAdapter(nn.Module):
    """Mock adapter for testing purposes."""

    def __init__(self, in_features=10, out_features=10, dim=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dim = dim
        self.alpha = 8
        self.dropout = nn.Identity()
        self.input_is_parallel = False

    def forward(self, x):
        return self.linear(x)


class TestParallelLinearDoRAAdapter:
    """Test the ParallelLinearDoRAAdapter class."""

    @pytest.fixture
    def mock_config(self):
        """Mock model parallel config."""
        config = MockModelParallelConfig()
        config.sequence_parallel = True
        return config

    def create_dora_adapter(self, mock_config, mock_col_linear, mock_row_linear):
        """Helper method to create a DoRA adapter for testing."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        adapter = ParallelLinearDoRAAdapter(
            in_features=10,
            out_features=5,
            dim=4,
            base_linear_name="test_linear",
            activation="identity",
            norm_type=None,
            column_init_method="xavier",
            row_init_method="zero",
            gather_output=False,
            input_is_parallel=False,
            dropout=0.1,
            dropout_position="pre",
            model_parallel_config=mock_config,
            alpha=8,
            disable_sequence_parallel_comm=False,
        )
        return adapter

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_init_weight_magnitude(self, mock_row_linear, mock_col_linear, mock_config):
        """Test weight magnitude initialization."""
        dora_adapter = self.create_dora_adapter(mock_config, mock_col_linear, mock_row_linear)
        magnitude_values = torch.randn(5)
        dora_adapter.init_weight_magnitude(magnitude_values)

        assert hasattr(dora_adapter, "weight_magnitude")
        assert isinstance(dora_adapter.weight_magnitude, nn.Parameter)
        assert dora_adapter.weight_magnitude.requires_grad is True
        assert torch.equal(dora_adapter.weight_magnitude, magnitude_values)

        retrieved_magnitude = dora_adapter.get_weight_magnitude()
        assert torch.equal(retrieved_magnitude, magnitude_values)
        assert retrieved_magnitude is dora_adapter.weight_magnitude

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.dora_layers.make_sharded_tensor_for_checkpoint")
    @patch("megatron.bridge.peft.dora_layers.make_tp_sharded_tensor_for_checkpoint")
    def test_sharded_state_dict_input_parallel(
        self, mock_tp_sharded, mock_sharded, mock_row_linear, mock_col_linear, mock_config
    ):
        """Test sharded state dict when input is parallel."""
        dora_adapter = self.create_dora_adapter(mock_config, mock_col_linear, mock_row_linear)
        dora_adapter.input_is_parallel = True
        magnitude_values = torch.randn(5)
        dora_adapter.init_weight_magnitude(magnitude_values)

        # Mock the parent's sharded_state_dict
        with patch.object(ParallelLinearAdapter, "sharded_state_dict", return_value={}):
            mock_sharded.return_value = ShardedTensor.from_rank_offsets("test_tensor", magnitude_values, (0, 0, 1))

            result = dora_adapter.sharded_state_dict(prefix="test_")

            # Should use make_sharded_tensor_for_checkpoint for input_is_parallel=True
            mock_sharded.assert_called_once()
            mock_tp_sharded.assert_not_called()
            assert "test_weight_magnitude" in result

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    @patch("megatron.bridge.peft.dora_layers.make_sharded_tensor_for_checkpoint")
    @patch("megatron.bridge.peft.dora_layers.make_tp_sharded_tensor_for_checkpoint")
    def test_sharded_state_dict_not_input_parallel(
        self, mock_tp_sharded, mock_sharded, mock_row_linear, mock_col_linear, mock_config
    ):
        """Test sharded state dict when input is not parallel."""
        dora_adapter = self.create_dora_adapter(mock_config, mock_col_linear, mock_row_linear)
        dora_adapter.input_is_parallel = False
        magnitude_values = torch.randn(5)
        dora_adapter.init_weight_magnitude(magnitude_values)

        # Mock the parent's sharded_state_dict
        with patch.object(ParallelLinearAdapter, "sharded_state_dict", return_value={}):
            mock_tp_sharded.return_value = ShardedTensor.from_rank_offsets("test_tensor", magnitude_values, (0, 0, 1))

            result = dora_adapter.sharded_state_dict(prefix="test_")

            # Should use make_tp_sharded_tensor_for_checkpoint for input_is_parallel=False
            mock_tp_sharded.assert_called_once()
            mock_sharded.assert_not_called()
            assert "test_weight_magnitude" in result


class TestDoRALinear:
    """Test the DoRALinear adapter wrapper."""

    @pytest.fixture
    def mock_linear(self):
        """Create a mock linear module."""
        return MockLinearWithTupleReturn()

    @pytest.fixture
    def mock_dora_adapter(self):
        """Create a mock DoRA adapter."""
        adapter = Mock()
        adapter.dim = 4
        adapter.alpha = 8
        adapter.dropout = nn.Identity()
        adapter.input_is_parallel = False

        # Create mock linear layers with weight attributes
        # Dimensions: base layer (10,10), adapter: in(10)->dim(4)->out(10)
        mock_linear_out = Mock()
        mock_linear_out.weight = torch.randn(10, 4)  # out_features=10, dim=4
        mock_linear_in = Mock()
        mock_linear_in.weight = torch.randn(4, 10)  # dim=4, in_features=10

        adapter.linear_out = mock_linear_out
        adapter.linear_in = mock_linear_in
        adapter.get_weight_magnitude.return_value = torch.randn(10)  # Match output dimension

        # Mock the forward method and __call__ method to return tensors
        def mock_adapter_forward(x):
            return torch.randn(x.shape[0], 10)

        adapter.forward = mock_adapter_forward
        adapter.side_effect = mock_adapter_forward
        return adapter

    def create_dora_linear(self, mock_linear, mock_dora_adapter):
        """Create a DoRALinear instance."""
        with patch.object(DoRALinear, "_get_weight_norm", return_value=torch.randn(10)):
            with patch.object(mock_dora_adapter, "init_weight_magnitude"):
                return DoRALinear(mock_linear, mock_dora_adapter)

    def test_dora_linear_init(self, mock_linear, mock_dora_adapter):
        """Test DoRALinear initialization."""
        with patch.object(DoRALinear, "_get_weight_norm", return_value=torch.randn(10)):
            with patch.object(mock_dora_adapter, "init_weight_magnitude") as mock_init:
                dora_linear = DoRALinear(mock_linear, mock_dora_adapter)

                assert dora_linear.to_wrap is mock_linear
                assert dora_linear.adapter is mock_dora_adapter
                assert dora_linear.scaling == 8 / 4  # alpha / dim
                mock_init.assert_called_once()

    def test_get_weight_norm_not_input_parallel(self, mock_linear, mock_dora_adapter):
        """Test weight norm calculation when input is not parallel."""
        mock_dora_adapter.input_is_parallel = False

        with patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region") as mock_gather:
            mock_gather.return_value = torch.randn(10, 4)

            with patch.object(mock_dora_adapter, "init_weight_magnitude"):
                dora_linear = DoRALinear(mock_linear, mock_dora_adapter)
                weight_norm = dora_linear._get_weight_norm()

                assert isinstance(weight_norm, torch.Tensor)
                assert weight_norm.shape == (10,)  # Should be norm along dim=1

    def test_get_weight_norm_input_parallel(self, mock_linear, mock_dora_adapter):
        """Test weight norm calculation when input is parallel."""
        mock_dora_adapter.input_is_parallel = True

        with patch("megatron.bridge.peft.dora_layers.gather_from_tensor_model_parallel_region") as mock_gather:
            mock_gather.return_value = torch.randn(4, 10)

            with patch.object(mock_dora_adapter, "init_weight_magnitude"):
                dora_linear = DoRALinear(mock_linear, mock_dora_adapter)
                weight_norm = dora_linear._get_weight_norm()

                assert isinstance(weight_norm, torch.Tensor)
                assert weight_norm.shape == (10,)  # Should be norm along dim=1

    def test_forward_no_dropout(self, mock_linear, mock_dora_adapter):
        """Test forward pass without dropout."""
        dora_linear = self.create_dora_linear(mock_linear, mock_dora_adapter)
        x = torch.randn(2, 10)
        dora_linear.training = False

        with patch.object(dora_linear, "base_linear_forward") as mock_base_forward:
            with patch.object(dora_linear, "_get_weight_norm") as mock_weight_norm:
                mock_base_forward.return_value = (torch.randn(2, 10), None, x)
                mock_weight_norm.return_value = torch.randn(10)
                dora_linear.adapter.get_weight_magnitude.return_value = torch.randn(10)

                output, bias = dora_linear(x)

                assert isinstance(output, torch.Tensor)
                assert output.shape == (1, 2, 10)  # DoRA uses view(1, 1, -1) which broadcasts to this shape
                assert bias is None
                mock_base_forward.assert_called_once_with(x)

    def test_forward_with_dropout(self, mock_linear, mock_dora_adapter):
        """Test forward pass with dropout during training."""
        dora_linear = self.create_dora_linear(mock_linear, mock_dora_adapter)
        x = torch.randn(2, 10)
        dora_linear.training = True
        dora_linear.adapter.dropout = nn.Dropout(0.1)

        # Mock the required methods
        with patch.object(dora_linear, "base_linear_forward") as mock_base_forward:
            with patch.object(dora_linear, "_get_weight_norm") as mock_weight_norm:
                mock_base_forward.return_value = (torch.randn(2, 10), None, x)
                mock_weight_norm.return_value = torch.randn(10)
                dora_linear.adapter.get_weight_magnitude.return_value = torch.randn(10)

                output, bias = dora_linear(x)

                assert isinstance(output, torch.Tensor)
                assert output.shape == (1, 2, 10)  # DoRA uses view(1, 1, -1) which broadcasts to this shape
                assert bias is None
                # Should be called twice: once for main forward, once for dropout correction
                assert mock_base_forward.call_count == 2

    def test_forward_magnitude_scaling(self, mock_linear, mock_dora_adapter):
        """Test that magnitude scaling is applied correctly."""
        dora_linear = self.create_dora_linear(mock_linear, mock_dora_adapter)
        x = torch.randn(2, 10)
        dora_linear.training = False

        # Set up deterministic values for testing
        linear_output = torch.ones(2, 10)
        adapter_output = torch.ones(2, 10)
        weight_magnitude = torch.tensor([2.0] * 10)  # 10 dimensions
        weight_norm = torch.tensor([1.0] * 10)  # 10 dimensions

        with patch.object(dora_linear, "base_linear_forward") as mock_base_forward:
            with patch.object(dora_linear, "_get_weight_norm") as mock_weight_norm:
                mock_base_forward.return_value = (linear_output, None, x)
                mock_weight_norm.return_value = weight_norm
                dora_linear.adapter.get_weight_magnitude.return_value = weight_magnitude
                # Override both forward method and direct call behavior
                dora_linear.adapter.forward = lambda x: adapter_output
                dora_linear.adapter.side_effect = lambda x: adapter_output

                output, bias = dora_linear(x)

                # Expected: mag_norm_scale * (linear_output + adapter_output)
                # mag_norm_scale has shape (1, 1, 10) and broadcasts with (2, 10) to give (1, 2, 10)
                # mag_norm_scale = weight_magnitude / weight_norm = [2, 2, 2, ...] (10 dims)
                # linear_output + adapter_output = ones(2, 10) + ones(2, 10) = 2 * ones(2, 10)
                # result = 2 * 2 * ones(2, 10) = 4 * ones(1, 2, 10) due to broadcasting
                expected_output = torch.full((1, 2, 10), 4.0)
                assert torch.allclose(output, expected_output, atol=1e-6)

    def test_inheritance_from_adapter_wrapper(self):
        """Test that DoRALinear properly inherits from AdapterWrapper."""
        from megatron.bridge.peft.adapter_wrapper import AdapterWrapper

        assert issubclass(DoRALinear, AdapterWrapper)
