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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

from megatron.bridge.peft import utils as peft_utils
from megatron.bridge.peft.utils import (
    GroupedExpertLinearAdapter,
    ParallelLinearAdapter,
    all2all_hp2sp,
    enable_legacy_shared_expert_adapter_loading,
    get_adapter_attributes_from_linear,
    init_method_const,
    init_method_kaiming_uniform,
    init_method_normal,
    is_expert_linear,
    is_grouped_expert_linear,
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


class MockProcessGroup:
    """Small process-group stand-in with MCore-style size/rank methods."""

    def __init__(self, size: int = 1, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def make_mock_pg_collection(
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    etp_size: int = 1,
    etp_rank: int = 0,
    edp_size: int = 1,
    edp_rank: int = 0,
) -> SimpleNamespace:
    """Build the subset of ProcessGroupCollection used by PEFT tests."""

    return SimpleNamespace(
        tp=MockProcessGroup(tp_size, tp_rank),
        ep=MockProcessGroup(ep_size, ep_rank),
        expt_tp=MockProcessGroup(etp_size, etp_rank),
        expt_dp=MockProcessGroup(edp_size, edp_rank),
        dp_cp=MockProcessGroup(),
    )


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

    def test_is_grouped_expert_linear(self):
        """Grouped expert helper should exclude sequential local expert modules."""
        assert is_grouped_expert_linear("decoder.layers.0.mlp.experts.linear_fc1")
        assert not is_grouped_expert_linear("decoder.layers.0.mlp.experts.local_experts.0.linear_fc1")

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

    def test_pad_unpad_roundtrip_preserves_gradients(self):
        """Padding expert tokens must not detach adapter gradients."""
        original = torch.randn(7, 10, requires_grad=True)

        padded, pad_len = pad_seq_to_mult(original, 4)
        unpadded = unpad_seq_to_mult(padded, pad_len)
        unpadded.sum().backward()

        torch.testing.assert_close(original.grad, torch.ones_like(original))


class TestAll2AllCommunication:
    """Test All2All communication functions."""

    def test_all2all_hp2sp_mock(self):
        """Test all2all_hp2sp with an explicit tensor-parallel process group."""
        tp_group = MockProcessGroup(size=2)

        # Mock torch.distributed.all_to_all
        with patch("torch.distributed.all_to_all") as mock_all_to_all:

            def side_effect(receive_list, send_list, group):
                # Simulate all_to_all operation
                for i, tensor in enumerate(send_list):
                    receive_list[i].copy_(tensor)

            mock_all_to_all.side_effect = side_effect

            x = torch.randn(4, 8)  # Input tensor
            result = all2all_hp2sp(x, tp_group)

            assert result.shape == (2, 16)  # Should reshape appropriately


class TestGetAdapterAttributes:
    """Test get_adapter_attributes_from_linear function."""

    def test_get_adapter_attributes_column_parallel(self):
        """Test with ColumnParallelLinear."""
        linear = MockColumnParallelLinear(input_size=100, output_size=50)

        attrs = get_adapter_attributes_from_linear(linear)

        assert not attrs.input_is_parallel
        assert attrs.in_features == 100
        assert attrs.out_features == 50
        assert not attrs.disable_tensor_parallel_comm
        assert attrs.disable_sequence_parallel_comm  # Should be True when sequence_parallel is False
        assert attrs.base_linear_is_parallel  # Should be True for parallel linear layers

    def test_get_adapter_attributes_row_parallel(self):
        """Test with RowParallelLinear."""
        linear = MockRowParallelLinear(input_size=100, output_size=50)

        attrs = get_adapter_attributes_from_linear(linear)

        assert attrs.input_is_parallel
        assert attrs.in_features == 100
        assert attrs.out_features == 50
        assert not attrs.disable_tensor_parallel_comm
        assert attrs.disable_sequence_parallel_comm
        assert attrs.base_linear_is_parallel  # Should be True for parallel linear layers

    def test_get_adapter_attributes_sequence_parallel(self):
        """Test with sequence parallel enabled."""
        linear = MockColumnParallelLinear(input_size=100, output_size=50)
        linear.config.sequence_parallel = True

        attrs = get_adapter_attributes_from_linear(linear)

        assert not attrs.disable_tensor_parallel_comm
        assert not attrs.disable_sequence_parallel_comm  # Should be False when sequence_parallel is True
        assert attrs.base_linear_is_parallel  # Should be True for parallel linear layers

    def test_get_adapter_attributes_te_sequence_parallel_input_regather(self):
        """Test that input re-gather uses the local TE LayerNorm shard, including with UB overlap."""

        class FakeTELayerNormColumnParallelLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    sequence_parallel=True,
                    tensor_model_parallel_size=2,
                    tp_comm_overlap=False,
                    tp_comm_overlap_disable_qkv=False,
                )
                self.in_features = 16
                self.out_features = 12
                self.parallel_mode = "column"
                self.return_layernorm_output = False
                self.return_layernorm_output_gathered = False
                self.ub_overlap_ag = False
                self._tp_group = MockProcessGroup(size=2)

        with (
            patch.object(peft_utils, "HAVE_TE", True),
            patch.object(peft_utils, "TECL", (FakeTELayerNormColumnParallelLinear,)),
            patch.object(
                peft_utils,
                "TELayerNormColumnParallelLinear",
                FakeTELayerNormColumnParallelLinear,
            ),
            patch.object(peft_utils, "version", return_value="1.10.0"),
        ):
            baseline = FakeTELayerNormColumnParallelLinear()
            baseline_attrs = get_adapter_attributes_from_linear(baseline)
            assert baseline.return_layernorm_output
            assert baseline.return_layernorm_output_gathered
            assert baseline_attrs.disable_sequence_parallel_comm

            regather = FakeTELayerNormColumnParallelLinear()
            regather_attrs = get_adapter_attributes_from_linear(regather, sequence_parallel_input_regather=True)
            assert regather.return_layernorm_output
            assert not regather.return_layernorm_output_gathered
            assert not regather_attrs.disable_sequence_parallel_comm

            overlap_regather = FakeTELayerNormColumnParallelLinear()
            overlap_regather.ub_overlap_ag = True
            overlap_regather.config.tp_comm_overlap = True
            overlap_regather_attrs = get_adapter_attributes_from_linear(
                overlap_regather, sequence_parallel_input_regather=True
            )
            assert overlap_regather.return_layernorm_output
            assert not overlap_regather.return_layernorm_output_gathered
            assert not overlap_regather_attrs.disable_sequence_parallel_comm

    def test_get_adapter_attributes_unsupported_module(self):
        """Test with unsupported module type."""
        linear = nn.Conv2d(3, 3, 3)
        linear.config = MockModelParallelConfig()

        with pytest.raises(NotImplementedError):
            get_adapter_attributes_from_linear(linear)

    def test_get_adapter_attributes_base_linear_is_parallel_flag(self):
        """Test that base_linear_is_parallel flag is correctly returned."""
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
        assert adapter.base_linear_is_parallel is True
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

    @patch("megatron.bridge.peft.utils.gather_from_sequence_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_sequence_parallel_input_regather_uses_mcore_sp_linear(
        self,
        mock_row_linear,
        mock_col_linear,
        mock_gather,
        mock_config,
    ):
        """Eligible adapters should let MCore's linear own the SP gather."""
        mock_config.sequence_parallel = True
        mock_linear_in = Mock()
        mock_linear_in.weight = nn.Parameter(torch.ones(4, 8))
        mock_linear_in.side_effect = lambda x: (torch.cat((x[..., :4], x[..., :4]), dim=0), None)
        mock_linear_out = Mock()
        mock_linear_out.side_effect = lambda x: (x[..., :2], None)
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]

        adapter = ParallelLinearAdapter(
            in_features=8,
            out_features=2,
            dim=4,
            base_linear_name="decoder.layers.0.self_attention.linear_qkv",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=mock_config,
            disable_sequence_parallel_comm=False,
            sequence_parallel_input_regather=True,
            pg_collection=make_mock_pg_collection(tp_size=2),
        )
        local_input = torch.randn(3, 8, requires_grad=True)
        output = adapter(local_input)

        assert output.shape == (6, 2)
        mock_gather.assert_not_called()
        mock_linear_in.assert_called_once_with(local_input)
        assert mock_linear_in.sequence_parallel is True
        mock_linear_out.assert_called_once()

    @patch("megatron.bridge.peft.utils.gather_from_sequence_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_sequence_parallel_input_regather_fallback_uses_external_gather(
        self,
        mock_row_linear,
        mock_col_linear,
        mock_gather,
        mock_config,
    ):
        """A static fallback should retain the existing external gather path."""
        mock_config.sequence_parallel = True
        mock_linear_in = Mock()
        mock_linear_in.weight = nn.Parameter(torch.ones(4, 8), requires_grad=False)
        mock_linear_in.side_effect = lambda x: (x[..., :4], None)
        mock_linear_out = Mock()
        mock_linear_out.side_effect = lambda x: (x[..., :2], None)
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_gather.side_effect = lambda x, group: torch.cat((x, x), dim=0)

        adapter = ParallelLinearAdapter(
            in_features=8,
            out_features=2,
            dim=4,
            base_linear_name="decoder.layers.0.self_attention.linear_qkv",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=mock_config,
            disable_sequence_parallel_comm=False,
            sequence_parallel_input_regather=True,
            pg_collection=make_mock_pg_collection(tp_size=2),
        )
        local_input = torch.randn(3, 8, requires_grad=True)
        output = adapter(local_input)

        assert output.shape == (6, 2)
        mock_gather.assert_called_once_with(local_input, group=adapter.tp_group)
        assert mock_linear_in.sequence_parallel is False

    @patch("megatron.bridge.peft.utils.gather_from_sequence_parallel_region")
    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_sequence_parallel_input_regather_tp1_fallback(
        self,
        mock_row_linear,
        mock_col_linear,
        mock_gather,
        mock_config,
    ):
        """TP=1 should keep the existing path instead of re-enabling MCore SP."""
        mock_config.sequence_parallel = True
        mock_linear_in = Mock()
        mock_linear_in.weight = nn.Parameter(torch.ones(4, 8))
        mock_linear_in.side_effect = lambda x: (x[..., :4], None)
        mock_linear_out = Mock()
        mock_linear_out.side_effect = lambda x: (x[..., :2], None)
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_gather.side_effect = lambda x, group: x

        adapter = ParallelLinearAdapter(
            in_features=8,
            out_features=2,
            dim=4,
            base_linear_name="decoder.layers.0.self_attention.linear_qkv",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=mock_config,
            disable_sequence_parallel_comm=False,
            sequence_parallel_input_regather=True,
            pg_collection=make_mock_pg_collection(tp_size=1),
        )
        local_input = torch.randn(3, 8, requires_grad=True)
        output = adapter(local_input)

        assert output.shape == (3, 2)
        mock_gather.assert_called_once_with(local_input, group=adapter.tp_group)
        assert mock_linear_in.sequence_parallel is False

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_sequence_parallel_input_regather_eligibility_gates(
        self, mock_row_linear, mock_col_linear, mock_config
    ):
        """Only qkv/fc1 SP LoRA without overlapping recompute should be eligible."""
        mock_config.sequence_parallel = True
        mock_linear_in = Mock()
        mock_linear_in.weight = nn.Parameter(torch.ones(4, 8))
        mock_linear_out = Mock()
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        adapter = ParallelLinearAdapter(
            in_features=8,
            out_features=2,
            dim=4,
            base_linear_name="decoder.layers.0.self_attention.linear_qkv",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=mock_config,
            disable_sequence_parallel_comm=False,
            sequence_parallel_input_regather=True,
            pg_collection=make_mock_pg_collection(tp_size=2),
        )
        x = torch.randn(3, 8, requires_grad=True)

        assert adapter._sequence_parallel_input_regather_eligibility(x) == (True, None)

        mock_linear_in.weight.requires_grad_(False)
        assert not adapter._sequence_parallel_input_regather_eligibility(x)[0]
        mock_linear_in.weight.requires_grad_(True)

        adapter.base_linear_name = "decoder.layers.0.self_attention.linear_proj"
        assert not adapter._sequence_parallel_input_regather_eligibility(x)[0]
        adapter.base_linear_name = "decoder.layers.0.mlp.linear_fc1"

        adapter.input_is_parallel = True
        assert not adapter._sequence_parallel_input_regather_eligibility(x)[0]
        adapter.input_is_parallel = False

        mock_config.recompute_granularity = "selective"
        mock_config.recompute_modules = ["mlp"]
        assert not adapter._sequence_parallel_input_regather_eligibility(x)[0]

        adapter.base_linear_name = "decoder.layers.0.self_attention.linear_qkv"
        assert adapter._sequence_parallel_input_regather_eligibility(x) == (True, None)

        mock_config.recompute_granularity = "full"
        assert not adapter._sequence_parallel_input_regather_eligibility(x)[0]

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_expert_mode(self, mock_row_linear, mock_col_linear, mock_config):
        """Test adapter in expert mode (MoE)."""
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
        mock_config.gated_linear_unit = True

        adapter = ParallelLinearAdapter(
            in_features=20, out_features=10, dim=16, base_linear_name="linear_fc1", model_parallel_config=mock_config
        )

        result = adapter.sharded_state_dict(prefix="adapter.")

        # Should call swiglu factory for fc1 weights
        mock_swiglu_factory.assert_called()
        assert result["adapter.linear_out.weight"] == "swiglu_processed_tensor"

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_grouped_expert_sharded_state_dict_uses_expert_axis(
        self, mock_row_linear, mock_col_linear, mock_config
    ):
        """Shared grouped-expert adapters should add a stable checkpoint expert axis."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        linear_in_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        linear_out_weight = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
        mock_linear_in.sharded_state_dict.return_value = {
            "adapter.linear_in.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_in.weight", linear_in_weight, replica_id=(0, 0, 0)
            ),
            "adapter.linear_in._extra_state": torch.tensor([1.0]),
        }
        mock_linear_out.sharded_state_dict.return_value = {
            "adapter.linear_out.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_out.weight", linear_out_weight, replica_id=(0, 0, 0)
            ),
            "adapter.linear_out._extra_state": torch.tensor([2.0]),
        }
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config.num_moe_experts = 4
        mock_config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=1, etp_rank=0, edp_rank=3)

        adapter = ParallelLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            is_expert=True,
            model_parallel_config=mock_config,
        )
        result = adapter.sharded_state_dict(prefix="adapter.")

        assert "adapter.linear_in._extra_state" not in result
        assert "adapter.linear_out._extra_state" not in result
        factory = result["adapter.linear_in.weight"]
        assert isinstance(factory, ShardedTensorFactory)
        assert factory.replica_id == (0, 0, 3)

        built = factory.build()
        assert len(built) == 2
        assert built[0].global_shape == (4, 2, 2)
        assert built[0].global_offset == (2, 0, 0)
        assert built[1].global_offset == (3, 0, 0)

        merged = factory.merge_fn([torch.ones(2, 2), torch.full((2, 2), 3.0)])
        torch.testing.assert_close(merged, torch.full((2, 2), 2.0))

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_grouped_expert_shared_adapter_syncs_grad_across_ep(
        self, mock_row_linear, mock_col_linear, mock_config
    ):
        """Shared grouped-expert adapters must not drift across EP ranks.

        A shared expert adapter is one logical weight used by every EP rank, but
        MCore's expert DDP sync only covers expert-DP replicas. The EP grad hook
        keeps EP>1 ranks from updating that shared adapter from different local
        token subsets.
        """
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        mock_linear_in.weight = nn.Parameter(torch.ones(2, 2))
        mock_linear_out.weight = nn.Parameter(torch.ones(2, 2))
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config._pg_collection = make_mock_pg_collection(ep_size=2)

        ParallelLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            is_expert=True,
            model_parallel_config=mock_config,
        )

        ep_group = mock_config._pg_collection.ep

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_reduce") as mock_all_reduce,
        ):
            (mock_linear_in.weight.sum() + mock_linear_out.weight.sum()).backward()

        assert mock_all_reduce.call_count == 2
        for call in mock_all_reduce.call_args_list:
            grad = call.args[0]
            torch.testing.assert_close(grad, torch.ones_like(grad))
            assert call.kwargs["group"] is ep_group

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_grouped_expert_swiglu_sharded_state_dict_uses_expert_axis(
        self, mock_row_linear, mock_col_linear, mock_config
    ):
        """Shared grouped-expert SwiGLU adapters should split gate/up shards inside the expert axis."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        linear_in_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        linear_out_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        mock_linear_in.sharded_state_dict.return_value = {
            "adapter.linear_in.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_in.weight", linear_in_weight, replica_id=(0, 0, 0)
            ),
            "adapter.linear_in._extra_state": torch.tensor([1.0]),
        }
        mock_linear_out.sharded_state_dict.return_value = {
            "adapter.linear_out.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_out.weight", linear_out_weight, replica_id=(0, 0, 0)
            ),
            "adapter.linear_out._extra_state": torch.tensor([2.0]),
        }
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config.num_moe_experts = 4
        mock_config.gated_linear_unit = True
        mock_config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=1, etp_rank=0, edp_rank=2)

        adapter = ParallelLinearAdapter(
            in_features=2,
            out_features=4,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc1",
            is_expert=True,
            model_parallel_config=mock_config,
        )
        result = adapter.sharded_state_dict(prefix="adapter.")

        assert "adapter.linear_in._extra_state" not in result
        assert "adapter.linear_out._extra_state" not in result
        factory = result["adapter.linear_out.weight"]
        assert isinstance(factory, ShardedTensorFactory)
        assert factory.replica_id == (0, 0, 2)

        built = factory.build()
        assert len(built) == 4
        assert built[0].global_shape == (4, 4, 2)
        assert [shard.global_offset for shard in built] == [
            (2, 0, 0),
            (2, 2, 0),
            (3, 0, 0),
            (3, 2, 0),
        ]

        merged = factory.merge_fn(
            [
                torch.full((2, 2), 1.0),
                torch.full((2, 2), 2.0),
                torch.full((2, 2), 3.0),
                torch.full((2, 2), 5.0),
            ]
        )
        expected = torch.cat([torch.full((2, 2), 2.0), torch.full((2, 2), 3.5)], dim=0)
        torch.testing.assert_close(merged, expected)

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_grouped_expert_sharded_state_dict_keeps_extra_state_on_main_expert_rank(
        self, mock_row_linear, mock_col_linear, mock_config
    ):
        """Shared grouped-expert adapter extra state should be kept only on EP0/ETP0."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        linear_in_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        linear_out_weight = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
        linear_in_extra_state = torch.tensor([1.0])
        linear_out_extra_state = torch.tensor([2.0])
        mock_linear_in.sharded_state_dict.return_value = {
            "adapter.linear_in.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_in.weight", linear_in_weight, replica_id=(0, 0, 0)
            ),
            "adapter.linear_in._extra_state": linear_in_extra_state,
        }
        mock_linear_out.sharded_state_dict.return_value = {
            "adapter.linear_out.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_out.weight", linear_out_weight, replica_id=(0, 0, 0)
            ),
            "adapter.linear_out._extra_state": linear_out_extra_state,
        }
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config.num_moe_experts = 4
        mock_config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=0, etp_rank=0, edp_rank=0)

        adapter = ParallelLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            is_expert=True,
            model_parallel_config=mock_config,
        )
        result = adapter.sharded_state_dict(prefix="adapter.")

        assert result["adapter.linear_in._extra_state"] is linear_in_extra_state
        assert result["adapter.linear_out._extra_state"] is linear_out_extra_state
        built = result["adapter.linear_in.weight"].build()
        assert [shard.global_offset for shard in built] == [(0, 0, 0), (1, 0, 0)]

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_legacy_shared_expert_state_dict_uses_2d_shape(
        self, mock_row_linear, mock_col_linear, mock_config
    ):
        """Legacy shared grouped-expert adapter checkpoints should load as 2D tensors."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        linear_in_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        linear_out_weight = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
        mock_linear_in.sharded_state_dict.return_value = {
            "adapter.linear_in.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_in.weight", linear_in_weight, replica_id=(0, 0, 0)
            ),
        }
        mock_linear_out.sharded_state_dict.return_value = {
            "adapter.linear_out.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_out.weight", linear_out_weight, replica_id=(0, 0, 0)
            ),
        }
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config.num_moe_experts = 4
        mock_config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=1, etp_rank=0, edp_rank=3)

        adapter = ParallelLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            is_expert=True,
            model_parallel_config=mock_config,
        )
        adapter.use_legacy_shared_expert_adapter_checkpoint = True

        result = adapter.sharded_state_dict(prefix="adapter.")

        sharded_weight = result["adapter.linear_in.weight"]
        assert isinstance(sharded_weight, ShardedTensor)
        assert sharded_weight.global_shape == (2, 2)

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_enable_legacy_shared_expert_adapter_loading_detects_2d_checkpoint_metadata(
        self, mock_row_linear, mock_col_linear, mock_config, monkeypatch
    ):
        """A 2D checkpoint tensor should opt only its shared expert adapter into legacy loading."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        linear_in_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        linear_out_weight = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
        mock_linear_in.sharded_state_dict.return_value = {
            "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight": ShardedTensor.from_rank_offsets(
                "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
                linear_in_weight,
                replica_id=(0, 0, 0),
            ),
        }
        mock_linear_out.sharded_state_dict.return_value = {
            "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight": ShardedTensor.from_rank_offsets(
                "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
                linear_out_weight,
                replica_id=(0, 0, 0),
            ),
        }
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config.num_moe_experts = 4
        mock_config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=0, etp_rank=0, edp_rank=0)

        adapter = ParallelLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            is_expert=True,
            model_parallel_config=mock_config,
        )
        sharded_state_dict = {
            "model": adapter.sharded_state_dict(prefix="decoder.layers.0.mlp.experts.linear_fc2.adapter.")
        }
        metadata = {
            "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight": ShardedTensor.from_rank_offsets(
                "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
                torch.empty(2, 2),
            ).without_data()
        }
        monkeypatch.setattr(peft_utils.dist_checkpointing, "load_tensors_metadata", lambda _: metadata)

        enabled = enable_legacy_shared_expert_adapter_loading(
            [SimpleNamespace(named_modules=lambda: [("decoder.layers.0.mlp.experts.linear_fc2.adapter", adapter)])],
            sharded_state_dict,
            "/checkpoint",
        )

        assert enabled is True
        assert adapter.use_legacy_shared_expert_adapter_checkpoint is True

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_enable_legacy_shared_expert_adapter_loading_tolerates_key_name_mismatch(
        self, mock_row_linear, mock_col_linear, mock_config, monkeypatch
    ):
        """Legacy loading should still work when checkpoint keys and module names differ."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        linear_in_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        linear_out_weight = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
        global_key = "decoder.layers.8.mlp.experts.linear_fc2.adapter.linear_in.weight"
        mock_linear_in.sharded_state_dict.return_value = {
            global_key: ShardedTensor.from_rank_offsets(global_key, linear_in_weight, replica_id=(0, 0, 0)),
        }
        mock_linear_out.sharded_state_dict.return_value = {
            "decoder.layers.8.mlp.experts.linear_fc2.adapter.linear_out.weight": ShardedTensor.from_rank_offsets(
                "decoder.layers.8.mlp.experts.linear_fc2.adapter.linear_out.weight",
                linear_out_weight,
                replica_id=(0, 0, 0),
            ),
        }
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config.num_moe_experts = 4
        mock_config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=0, etp_rank=0, edp_rank=0)

        adapter = ParallelLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            is_expert=True,
            model_parallel_config=mock_config,
        )
        sharded_state_dict = {
            "model": adapter.sharded_state_dict(prefix="decoder.layers.8.mlp.experts.linear_fc2.adapter.")
        }
        metadata = {global_key: ShardedTensor.from_rank_offsets(global_key, torch.empty(2, 2)).without_data()}
        monkeypatch.setattr(peft_utils.dist_checkpointing, "load_tensors_metadata", lambda _: metadata)

        enabled = enable_legacy_shared_expert_adapter_loading(
            [SimpleNamespace(named_modules=lambda: [("decoder.layers.0.mlp.experts.linear_fc2.adapter", adapter)])],
            sharded_state_dict,
            "/checkpoint",
        )

        assert enabled is True
        assert adapter.use_legacy_shared_expert_adapter_checkpoint is True

    @patch("megatron.bridge.peft.utils.ColumnParallelLinear")
    @patch("megatron.bridge.peft.utils.RowParallelLinear")
    def test_parallel_linear_adapter_non_grouped_expert_sharded_state_dict_uses_expert_dp_replica_id(
        self, mock_row_linear, mock_col_linear, mock_config
    ):
        """Sequential local expert adapters should keep base sharding and use expert-DP replica ids."""
        mock_linear_in = Mock()
        mock_linear_out = Mock()
        linear_in_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        linear_out_weight = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
        mock_linear_in.sharded_state_dict.return_value = {
            "adapter.linear_in.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_in.weight", linear_in_weight, replica_id=(0, 1, 99)
            ),
            "adapter.linear_in._extra_state": torch.tensor([1.0]),
        }
        mock_linear_out.sharded_state_dict.return_value = {
            "adapter.linear_out.weight": ShardedTensor.from_rank_offsets(
                "adapter.linear_out.weight", linear_out_weight, replica_id=(0, 1, 99)
            ),
            "adapter.linear_out._extra_state": torch.tensor([2.0]),
        }
        mock_col_linear.side_effect = [mock_linear_in, mock_linear_out]
        mock_config._pg_collection = make_mock_pg_collection(etp_rank=0, edp_rank=3)

        adapter = ParallelLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            base_linear_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc2",
            is_expert=True,
            model_parallel_config=mock_config,
        )
        result = adapter.sharded_state_dict(prefix="adapter.")

        sharded_weight = result["adapter.linear_in.weight"]
        assert isinstance(sharded_weight, ShardedTensor)
        assert sharded_weight.global_shape == (2, 2)
        assert sharded_weight.replica_id == (0, 1, 3)
        assert "adapter.linear_in._extra_state" in result


class TestGroupedExpertLinearAdapter:
    """Tests for grouped-expert per-expert LoRA adapters."""

    @pytest.mark.parametrize("split_kwarg", ["m_splits", "tokens_per_expert"])
    def test_grouped_expert_linear_adapter_accepts_tensor_split_kwargs(self, split_kwarg):
        """Tensor-valued split kwargs should not trigger ambiguous truth-value errors."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )

        with torch.no_grad():
            adapter.linear_in.weight[0].copy_(torch.eye(2))
            adapter.linear_out.weight[0].copy_(torch.eye(2))
            adapter.linear_in.weight[1].copy_(2 * torch.eye(2))
            adapter.linear_out.weight[1].copy_(torch.eye(2))

        x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        output = adapter(x, **{split_kwarg: torch.tensor([1, 2])})

        expected = torch.tensor(
            [
                [1.0, 2.0],
                [6.0, 8.0],
                [10.0, 12.0],
            ]
        )
        torch.testing.assert_close(output, expected)

    def test_grouped_expert_linear_adapter_forward_uses_per_expert_weights(self):
        """Each local expert should use its own LoRA weights."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )

        with torch.no_grad():
            adapter.linear_in.weight[0].copy_(torch.eye(2))
            adapter.linear_out.weight[0].copy_(torch.eye(2))
            adapter.linear_in.weight[1].copy_(2 * torch.eye(2))
            adapter.linear_out.weight[1].copy_(torch.eye(2))

        x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        output = adapter(x, [1, 2])

        expected = torch.tensor(
            [
                [1.0, 2.0],
                [6.0, 8.0],
                [10.0, 12.0],
            ]
        )
        torch.testing.assert_close(output, expected)

    def test_grouped_expert_linear_adapter_keeps_checkpoint_keys_after_weight_module_wrap(self):
        """Calling weight containers should not change existing adapter checkpoint keys."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )

        state_dict = adapter.state_dict()

        assert sorted(state_dict) == ["linear_in.weight", "linear_out.weight"]
        old_style_state_dict = {
            "linear_in.weight": torch.ones_like(adapter.linear_in.weight),
            "linear_out.weight": torch.full_like(adapter.linear_out.weight, 2.0),
        }
        missing, unexpected = adapter.load_state_dict(old_style_state_dict, strict=True)
        assert missing == []
        assert unexpected == []
        torch.testing.assert_close(adapter.linear_in.weight, old_style_state_dict["linear_in.weight"])
        torch.testing.assert_close(adapter.linear_out.weight, old_style_state_dict["linear_out.weight"])

    def test_grouped_expert_linear_adapter_forward_calls_weight_modules_for_param_sync_hooks(self):
        """Grouped per-expert adapters must participate in training-time param gather.

        With EP plus expert-DP, distributed optimizer param gather is driven by
        DDP forward pre-hooks. The weight containers need to be called so live
        training weights refresh during normal forwards, not only at forced
        eval/checkpoint sync boundaries.
        """
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=3,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )

        calls = []
        adapter.linear_in.register_forward_pre_hook(lambda module, inputs: calls.append("linear_in"))
        adapter.linear_out.register_forward_pre_hook(lambda module, inputs: calls.append("linear_out"))

        def fake_grouped_mm(inputs, weights, *, offs):
            chunks = []
            start = 0
            for weight_idx, end in enumerate(offs.tolist()):
                chunks.append(inputs[start:end] @ weights[weight_idx])
                start = end
            return torch.cat(chunks, dim=0)

        x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        with (
            patch.object(GroupedExpertLinearAdapter, "_can_use_grouped_mm", return_value=True),
            patch(
                "megatron.bridge.peft.utils.nn.functional.grouped_mm",
                side_effect=fake_grouped_mm,
                create=True,
            ),
        ):
            adapter(x, [1, 0, 2])

        assert calls == ["linear_in", "linear_out"]

    def test_grouped_expert_linear_adapter_zero_token_batch_keeps_weight_grad_dependency(self):
        """Empty local expert batches should still produce zero grads for DDP hooks.

        EP routing can leave a local grouped adapter with no tokens on a step.
        The zero-sized output still needs a zero-valued dependency on the LoRA
        weights so DDP sees ready gradients instead of leaving replicas stale.
        """
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )

        output = adapter(torch.empty(0, 2), [0, 0])

        assert output.shape == (0, 2)
        output.sum().backward()
        assert adapter.linear_in.weight.grad is not None
        assert adapter.linear_out.weight.grad is not None
        torch.testing.assert_close(adapter.linear_in.weight.grad, torch.zeros_like(adapter.linear_in.weight))
        torch.testing.assert_close(adapter.linear_out.weight.grad, torch.zeros_like(adapter.linear_out.weight))

    def test_grouped_expert_linear_adapter_grouped_mm_falls_back_on_cpu(self):
        """CPU inputs should not enter the grouped_mm fast path."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )

        with torch.no_grad():
            adapter.linear_in.weight[0].copy_(torch.eye(2))
            adapter.linear_out.weight[0].copy_(torch.eye(2))
            adapter.linear_in.weight[1].copy_(2 * torch.eye(2))
            adapter.linear_out.weight[1].copy_(torch.eye(2))

        x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        with patch(
            "megatron.bridge.peft.utils.nn.functional.grouped_mm",
            side_effect=AssertionError("grouped_mm should not run on CPU"),
            create=True,
        ):
            output = adapter(x, [1, 2])

        expected = torch.tensor(
            [
                [1.0, 2.0],
                [6.0, 8.0],
                [10.0, 12.0],
            ]
        )
        torch.testing.assert_close(output, expected)

    def test_grouped_expert_linear_adapter_grouped_mm_skips_zero_split_experts(self):
        """Grouped GEMM should only include experts that actually receive tokens."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=3,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )

        with torch.no_grad():
            adapter.linear_in.weight[0].copy_(torch.eye(2))
            adapter.linear_out.weight[0].copy_(torch.eye(2))
            adapter.linear_in.weight[1].copy_(7 * torch.eye(2))
            adapter.linear_out.weight[1].copy_(torch.eye(2))
            adapter.linear_in.weight[2].copy_(3 * torch.eye(2))
            adapter.linear_out.weight[2].copy_(torch.eye(2))

        x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )

        def fake_grouped_mm(inputs, weights, *, offs):
            assert offs.dtype == torch.int32
            chunks = []
            start = 0
            for weight_idx, end in enumerate(offs.tolist()):
                chunks.append(inputs[start:end] @ weights[weight_idx])
                start = end
            return torch.cat(chunks, dim=0)

        with (
            patch.object(GroupedExpertLinearAdapter, "_can_use_grouped_mm", return_value=True),
            patch(
                "megatron.bridge.peft.utils.nn.functional.grouped_mm",
                side_effect=fake_grouped_mm,
                create=True,
            ) as mock_grouped_mm,
        ):
            output = adapter(x, [1, 0, 2])

        expected = torch.tensor(
            [
                [1.0, 2.0],
                [9.0, 12.0],
                [15.0, 18.0],
            ]
        )
        torch.testing.assert_close(output, expected)
        assert mock_grouped_mm.call_count == 2
        assert mock_grouped_mm.call_args_list[0].args[1].shape[0] == 2
        assert mock_grouped_mm.call_args_list[0].kwargs["offs"].tolist() == [1, 3]

    @pytest.mark.parametrize(
        ("in_features", "out_features", "dim"),
        [
            (16, 32, 12),
            (12, 32, 8),
            (16, 30, 8),
        ],
    )
    def test_grouped_expert_linear_adapter_grouped_mm_requires_dimension_alignment(
        self, in_features, out_features, dim
    ):
        """Grouped GEMM should be disabled when any matrix stride is not 16-byte aligned."""
        adapter = GroupedExpertLinearAdapter(
            in_features=in_features,
            out_features=out_features,
            dim=dim,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )
        fake_x = Mock(is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda"))

        with patch("megatron.bridge.peft.utils.torch.cuda.get_device_capability", return_value=(8, 0)):
            assert not adapter._can_use_grouped_mm(fake_x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BF16 grouped expert weights")
    def test_grouped_expert_linear_adapter_noncontiguous_input_uses_fallback(self):
        """Unsupported input strides should use the correct per-expert linear fallback."""
        adapter = GroupedExpertLinearAdapter(
            in_features=16,
            out_features=16,
            dim=8,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
            params_device=torch.device("cuda"),
            params_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            adapter.linear_in.weight.normal_()
            adapter.linear_out.weight.normal_()
        x = torch.randn(16, 5, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
        assert not x.is_contiguous()

        with patch(
            "megatron.bridge.peft.utils.nn.functional.grouped_mm",
            side_effect=AssertionError("non-contiguous input should use the fallback"),
            create=True,
        ):
            output = adapter(x, [2, 3])

        expected_chunks = []
        for expert_idx, expert_input in enumerate(x.split([2, 3])):
            hidden = nn.functional.linear(expert_input, adapter.linear_in.weight[expert_idx])
            expected_chunks.append(nn.functional.linear(hidden, adapter.linear_out.weight[expert_idx]))
        torch.testing.assert_close(output, torch.cat(expected_chunks), rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("te_version", ["2.14", "2.16", "2.17"])
    @pytest.mark.parametrize("grad_enabled", [True, False])
    @pytest.mark.parametrize("active_expert_indices", [(0, 1), (1, 2)])
    def test_grouped_expert_linear_adapter_fp8_te_contract(self, te_version, grad_enabled, active_expert_indices):
        """FP8 dispatch should pack the supported TE 2.14, 2.16, and 2.17 call layouts."""
        calls = []
        expected = torch.randn(3, 2)

        class TE214GroupedLinear:
            @staticmethod
            def apply(inp, non_tensor_args, *weights_and_biases):
                calls.append((inp, None, non_tensor_args, weights_and_biases))
                return expected

            @staticmethod
            def forward(ctx, inp, non_tensor_args, *weights_and_biases):
                assert ctx is None
                calls.append((inp, None, non_tensor_args, weights_and_biases))
                return expected

        class TE217GroupedLinear:
            @staticmethod
            def apply(inp, m_splits, non_tensor_args, *weights_and_biases):
                calls.append((inp, m_splits, non_tensor_args, weights_and_biases))
                return expected, []

            @staticmethod
            def forward(ctx, inp, m_splits, non_tensor_args, *weights_and_biases):
                assert ctx is None
                calls.append((inp, m_splits, non_tensor_args, weights_and_biases))
                return expected, []

        autograd_function = TE214GroupedLinear if te_version in {"2.14", "2.16"} else TE217GroupedLinear
        helper = Mock()
        helper.prepare_forward.side_effect = lambda inp, *, num_gemms: inp
        helper._get_quantizers.return_value = tuple(
            [f"quantizer-{group_idx}-{expert_idx}" for expert_idx in range(3)] for group_idx in range(6)
        )
        helper.apply_bias = False
        helper.fp8 = True
        helper.fp8_calibration = False
        helper.wgrad_store = Mock()
        helper.fuse_wgrad_accumulation = False
        helper.sequence_parallel = False
        helper.activation_dtype = torch.float32
        helper.save_original_input = False

        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=3,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )
        x = torch.randn(3, 2)
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with (
            patch.object(adapter, "_get_te_grouped_linear_helper", return_value=helper),
            patch.object(peft_utils, "TEPytorchGroupedLinearAutograd", autograd_function),
            patch.object(peft_utils, "TEPytorchIsCPUOffloadEnabled", return_value=True),
            context,
        ):
            output = adapter._forward_te_grouped_linear_fp8(
                x,
                weight=adapter.linear_in(list(active_expert_indices)),
                m_splits=[1, 2],
                projection="linear_in",
                active_expert_indices=active_expert_indices,
            )

        torch.testing.assert_close(output, expected)
        assert len(calls) == 1
        received_input, explicit_splits, non_tensor_args, weights_and_biases = calls[0]
        assert received_input is x
        if te_version in {"2.14", "2.16"}:
            assert explicit_splits is None
            assert non_tensor_args[0] == [1, 2]
            common_non_tensor_args = non_tensor_args[1:17]
            assert non_tensor_args[17] is helper
            assert non_tensor_args[18] is None
            assert non_tensor_args[19] is False
            assert non_tensor_args[20] is False
        else:
            torch.testing.assert_close(explicit_splits, torch.tensor([1, 2], dtype=torch.int64))
            assert explicit_splits.device.type == "cpu"
            common_non_tensor_args = non_tensor_args[:16]
            assert non_tensor_args[16] == [None, None]
            assert non_tensor_args[17] is False
            assert non_tensor_args[18] is None
            assert non_tensor_args[19] is False
            assert non_tensor_args[20] is False
        assert common_non_tensor_args[0] is False
        assert common_non_tensor_args[2] is True
        assert common_non_tensor_args[12] is True
        for group_idx in range(6):
            assert common_non_tensor_args[5 + group_idx] == [
                f"quantizer-{group_idx}-{expert_idx}" for expert_idx in active_expert_indices
            ]
        assert len(weights_and_biases) == 4
        helper.prepare_forward.assert_called_once_with(x, num_gemms=3)
        helper.end_forward.assert_called_once_with()

    def test_grouped_expert_linear_adapter_fp8_prefers_te_backend(self):
        """An active FP8 context should select TE instead of the public BF16 backend."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        hidden = torch.zeros(32, 2)
        padded_output = torch.zeros(32, 2)
        padded_output[0] = expected[0]
        padded_output[16:18] = expected[1:]

        with (
            patch.object(adapter, "_is_te_fp8_enabled", return_value=True),
            patch.object(adapter, "_can_use_te_grouped_linear_fp8", return_value=True),
            patch.object(
                adapter,
                "_can_use_grouped_mm",
                side_effect=AssertionError("FP8 should not probe the BF16 grouped-MM backend"),
            ),
            patch.object(adapter, "_forward_te_grouped_linear_fp8", side_effect=[hidden, padded_output]) as mock_te,
        ):
            output = adapter(x, [1, 2])

        torch.testing.assert_close(output, expected)
        assert mock_te.call_count == 2
        assert mock_te.call_args_list[0].kwargs["m_splits"] == [16, 16]
        assert mock_te.call_args_list[1].kwargs["m_splits"] == [16, 16]

    def test_grouped_expert_linear_adapter_fp8_without_te_uses_fallback(self):
        """An unsupported FP8 layout should not silently run the public BF16 grouped backend."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )
        x = torch.randn(3, 2)
        expected = torch.randn(3, 2)

        with (
            patch.object(adapter, "_is_te_fp8_enabled", return_value=True),
            patch.object(adapter, "_can_use_te_grouped_linear_fp8", return_value=False),
            patch.object(
                adapter,
                "_can_use_grouped_mm",
                side_effect=AssertionError("FP8 must not use the BF16 grouped-MM backend"),
            ),
            patch.object(adapter, "_forward_per_expert", return_value=expected) as mock_fallback,
        ):
            output = adapter(x, [1, 2])

        torch.testing.assert_close(output, expected)
        mock_fallback.assert_called_once_with(x, expert_splits=[1, 2], expert_tp_size=1)

    def test_grouped_expert_linear_adapter_fp8_requires_dimension_alignment(self):
        """A rank-eight adapter should avoid TE's 16-element-aligned FP8 grouped GEMM."""
        adapter = GroupedExpertLinearAdapter(
            in_features=16,
            out_features=16,
            dim=8,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
            params_dtype=torch.bfloat16,
        )
        fake_x = Mock(is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda:0"))

        with (
            patch.object(peft_utils, "HAVE_TE_PYTORCH_GROUPED_LINEAR", True),
            patch.object(peft_utils, "HAVE_TE_PYTORCH_GROUPED_LINEAR_AUTOGRAD", True),
            patch.object(peft_utils, "HAVE_TE_PYTORCH_CPU_OFFLOAD_STATUS", True),
            patch("megatron.bridge.peft.utils.torch.cuda.current_device", return_value=0),
        ):
            assert not adapter._can_use_te_grouped_linear_fp8(fake_x)

    def test_grouped_expert_linear_adapter_fp8_requires_current_cuda_device(self):
        """TE FP8 should fall back when autocast state belongs to another CUDA device."""
        adapter = GroupedExpertLinearAdapter(
            in_features=16,
            out_features=16,
            dim=16,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
            params_dtype=torch.bfloat16,
        )
        fake_x = Mock(is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda:1"))

        with (
            patch.object(peft_utils, "HAVE_TE_PYTORCH_GROUPED_LINEAR", True),
            patch.object(peft_utils, "HAVE_TE_PYTORCH_GROUPED_LINEAR_AUTOGRAD", True),
            patch.object(peft_utils, "HAVE_TE_PYTORCH_CPU_OFFLOAD_STATUS", True),
            patch("megatron.bridge.peft.utils.torch.cuda.current_device", return_value=0),
        ):
            assert not adapter._can_use_te_grouped_linear_fp8(fake_x)

    def test_grouped_expert_linear_adapter_te_helpers_have_stable_weight_identity(self):
        """FP8 runtime state should use bounded device/projection helpers with stable expert slots."""
        adapter = GroupedExpertLinearAdapter(
            in_features=16,
            out_features=16,
            dim=16,
            num_local_experts=3,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )
        cuda0_linear_in_helper = Mock()
        cuda0_linear_out_helper = Mock()
        cuda1_linear_in_helper = Mock()
        helper_kwargs = {
            "projection": "linear_in",
            "num_gemms": 3,
            "in_features": 16,
            "out_features": 16,
            "params_dtype": torch.bfloat16,
        }

        with patch.object(
            peft_utils,
            "TEPytorchGroupedLinear",
            side_effect=[
                cuda0_linear_in_helper,
                cuda1_linear_in_helper,
                cuda0_linear_out_helper,
            ],
        ) as grouped_linear:
            first_cuda0_helper = adapter._get_te_grouped_linear_helper(**helper_kwargs, device=torch.device("cuda:0"))
            second_cuda0_helper = adapter._get_te_grouped_linear_helper(**helper_kwargs, device=torch.device("cuda:0"))
            first_cuda1_helper = adapter._get_te_grouped_linear_helper(**helper_kwargs, device=torch.device("cuda:1"))
            cuda0_linear_out = adapter._get_te_grouped_linear_helper(
                **{**helper_kwargs, "projection": "linear_out"}, device=torch.device("cuda:0")
            )

        assert first_cuda0_helper is cuda0_linear_in_helper
        assert second_cuda0_helper is cuda0_linear_in_helper
        assert first_cuda1_helper is cuda1_linear_in_helper
        assert cuda0_linear_out is cuda0_linear_out_helper
        assert grouped_linear.call_count == 3

    def test_grouped_expert_linear_adapter_te_grouped_mlp_split_call_uses_public_grouped_mm(self):
        """TEGroupedMLP-style positional splits should work through the public backend."""
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
        )
        x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        with torch.no_grad():
            adapter.linear_in.weight[0].copy_(torch.eye(2))
            adapter.linear_out.weight[0].copy_(torch.eye(2))
            adapter.linear_in.weight[1].copy_(2 * torch.eye(2))
            adapter.linear_out.weight[1].copy_(torch.eye(2))

        def fake_grouped_mm(inputs, weights, *, offs):
            chunks = []
            start = 0
            for weight_idx, end in enumerate(offs.tolist()):
                chunks.append(inputs[start:end] @ weights[weight_idx])
                start = end
            return torch.cat(chunks, dim=0)

        with (
            patch.object(GroupedExpertLinearAdapter, "_can_use_grouped_mm", return_value=True),
            patch(
                "megatron.bridge.peft.utils.nn.functional.grouped_mm",
                side_effect=fake_grouped_mm,
                create=True,
            ) as mock_grouped_mm,
        ):
            output = adapter(x, [1, 2])

        expected = torch.tensor(
            [
                [1.0, 2.0],
                [6.0, 8.0],
                [10.0, 12.0],
            ]
        )
        torch.testing.assert_close(output, expected)
        assert mock_grouped_mm.call_count == 2
        assert mock_grouped_mm.call_args_list[0].kwargs["offs"].tolist() == [1, 3]

    @pytest.mark.skipif(
        not torch.cuda.is_available() or getattr(nn.functional, "grouped_mm", None) is None,
        reason="public grouped_mm requires a supported CUDA PyTorch build",
    )
    def test_grouped_expert_linear_adapter_public_grouped_mm_forward_backward(self):
        """The public grouped GEMM path should match per-expert linear algebra and backpropagate."""
        adapter = GroupedExpertLinearAdapter(
            in_features=16,
            out_features=16,
            dim=8,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
            params_device=torch.device("cuda"),
            params_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            adapter.linear_in.weight.normal_()
            adapter.linear_out.weight.normal_()
        x = torch.randn(5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        reference_x = x.detach().clone().requires_grad_()
        reference_linear_in = adapter.linear_in.weight.detach().clone().requires_grad_()
        reference_linear_out = adapter.linear_out.weight.detach().clone().requires_grad_()
        expected_chunks = []
        for expert_idx, expert_input in enumerate(reference_x.split([2, 3])):
            hidden = nn.functional.linear(expert_input, reference_linear_in[expert_idx])
            expected_chunks.append(nn.functional.linear(hidden, reference_linear_out[expert_idx]))
        expected = torch.cat(expected_chunks)
        expected.float().sum().backward()

        output = adapter(x, [2, 3])
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
        output.float().sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(adapter.linear_in.weight.grad, reference_linear_in.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(adapter.linear_out.weight.grad, reference_linear_out.grad, rtol=2e-2, atol=2e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="TE FP8 grouped linear requires CUDA")
    def test_grouped_expert_linear_adapter_te_fp8_forward_backward_and_inference(self):
        """The real TE FP8 path should produce finite outputs and gradients without public grouped-MM."""
        if not (
            peft_utils.HAVE_TE_FP8_GLOBAL_STATE_MANAGER
            and peft_utils.HAVE_TE_PYTORCH_GROUPED_LINEAR
            and peft_utils.HAVE_TE_PYTORCH_GROUPED_LINEAR_AUTOGRAD
            and peft_utils.HAVE_TE_PYTORCH_CPU_OFFLOAD_STATUS
        ):
            pytest.skip("Transformer Engine grouped-linear FP8 support is unavailable")
        te = pytest.importorskip("transformer_engine.pytorch")
        adapter = GroupedExpertLinearAdapter(
            in_features=16,
            out_features=16,
            dim=16,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
            params_device=torch.device("cuda"),
            params_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            adapter.linear_in.weight.normal_(std=0.1)
            adapter.linear_out.weight.normal_(std=0.1)
        x = torch.randn(5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        reference_linear_in = adapter.linear_in.weight.detach().clone().requires_grad_()
        reference_linear_out = adapter.linear_out.weight.detach().clone().requires_grad_()
        expected_chunks = []
        for expert_idx, expert_input in enumerate(reference_x.split([2, 3])):
            hidden = nn.functional.linear(expert_input, reference_linear_in[expert_idx])
            expected_chunks.append(nn.functional.linear(hidden, reference_linear_out[expert_idx]))
        expected = torch.cat(expected_chunks)
        expected.float().sum().backward()

        with (
            patch(
                "megatron.bridge.peft.utils.nn.functional.grouped_mm",
                side_effect=AssertionError("FP8 should use TE instead of public grouped-MM"),
            ),
            te.autocast(enabled=True),
        ):
            output = adapter(x, [2, 3])

        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=2e-1, atol=1e-1)
        output.float().sum().backward()
        for grad in (x.grad, adapter.linear_in.weight.grad, adapter.linear_out.weight.grad):
            assert grad is not None
            assert torch.isfinite(grad).all()
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=3e-1, atol=2e-1)
        torch.testing.assert_close(adapter.linear_in.weight.grad, reference_linear_in.grad, rtol=3e-1, atol=2e-1)
        torch.testing.assert_close(adapter.linear_out.weight.grad, reference_linear_out.grad, rtol=3e-1, atol=2e-1)

        with (
            torch.no_grad(),
            patch(
                "megatron.bridge.peft.utils.nn.functional.grouped_mm",
                side_effect=AssertionError("FP8 inference should use TE instead of public grouped-MM"),
            ),
            te.autocast(enabled=True),
        ):
            inference_output = adapter(x.detach(), [2, 3])
        assert torch.isfinite(inference_output).all()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="TE FP8 device migration requires two CUDA devices")
    def test_grouped_expert_linear_adapter_te_fp8_device_mismatch_uses_fallback(self):
        """An adapter moved away from the current CUDA device should use the safe fallback."""
        if not (
            peft_utils.HAVE_TE_FP8_GLOBAL_STATE_MANAGER
            and peft_utils.HAVE_TE_PYTORCH_GROUPED_LINEAR
            and peft_utils.HAVE_TE_PYTORCH_GROUPED_LINEAR_AUTOGRAD
            and peft_utils.HAVE_TE_PYTORCH_CPU_OFFLOAD_STATUS
        ):
            pytest.skip("Transformer Engine grouped-linear FP8 support is unavailable")
        te = pytest.importorskip("transformer_engine.pytorch")
        torch.cuda.set_device(0)
        adapter = GroupedExpertLinearAdapter(
            in_features=16,
            out_features=16,
            dim=16,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=MockModelParallelConfig(),
            params_device=torch.device("cuda:0"),
            params_dtype=torch.bfloat16,
        )

        device = torch.device("cuda:1")
        adapter.to(device)
        x = torch.randn(5, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        with (
            patch.object(
                adapter,
                "_forward_te_grouped_linear_fp8",
                side_effect=AssertionError("mismatched CUDA devices must not use TE grouped FP8"),
            ),
            te.autocast(enabled=True),
        ):
            output = adapter(x, [2, 3])
        output.float().sum().backward()
        torch.cuda.synchronize(device)
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        assert adapter._te_grouped_linear_helpers == {}

    def test_grouped_expert_linear_adapter_requires_expert_tp_group_for_gather(self):
        """Per-expert LoRA should fail clearly when expert TP is configured without initialized groups."""
        config = MockModelParallelConfig()
        config.expert_tensor_parallel_size = 2
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=1,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=config,
        )

        with patch("megatron.bridge.peft.utils.torch.distributed.all_gather") as mock_all_gather:
            with pytest.raises(
                ValueError,
                match="requires initialized expert tensor parallel state when expert_tensor_parallel_size=2",
            ):
                adapter(torch.tensor([[1.0, 2.0]]), [1])

        mock_all_gather.assert_not_called()

    def test_grouped_expert_linear_fc1_sharded_state_dict_preserves_expert_axis(self):
        """Grouped expert fc1 checkpoints should split SwiGLU on the hidden axis, not the expert axis."""
        config = MockModelParallelConfig()
        config.gated_linear_unit = True
        config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=1, edp_rank=0, etp_size=1)
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=4,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc1",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=config,
        )

        result = adapter.sharded_state_dict("adapter.")

        factory = result["adapter.linear_out.weight"]
        assert isinstance(factory, ShardedTensorFactory)

        built = factory.build()
        assert len(built) == 2
        assert built[0].local_shape == (2, 2, 2)
        assert built[1].local_shape == (2, 2, 2)
        assert built[0].global_shape == (4, 4, 2)
        assert built[1].global_shape == (4, 4, 2)
        assert built[0].global_offset == (2, 0, 0)
        assert built[1].global_offset == (2, 2, 0)

    def test_grouped_expert_linear_fc1_factory_merge_restores_gate_up_order(self):
        """Grouped expert fc1 checkpoint reload should de-interleave gate/up expert-TP shards."""
        config = MockModelParallelConfig()
        config.gated_linear_unit = True
        config._pg_collection = make_mock_pg_collection(ep_size=1, ep_rank=0, edp_rank=0, etp_size=2, etp_rank=0)
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=8,
            dim=2,
            num_local_experts=1,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc1",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=config,
        )

        factory = adapter.sharded_state_dict("adapter.")["adapter.linear_out.weight"]

        fused_tp0 = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]])
        fused_tp1 = torch.tensor([[[3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0]]])

        merged = factory.merge_fn([fused_tp0, fused_tp1])
        expected = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0], [2.0, 2.0], [2.0, 2.0], [4.0, 4.0], [4.0, 4.0]]]
        )
        torch.testing.assert_close(merged, expected)

    @pytest.mark.parametrize(("ep_size", "expected_allreduce"), [(1, True), (2, False)])
    def test_grouped_expert_linear_adapter_allreduce_flag_tracks_expert_parallelism(self, ep_size, expected_allreduce):
        """Per-expert grouped adapters should use expert-DP grad sync only when EP is enabled."""
        config = MockModelParallelConfig()
        config._pg_collection = make_mock_pg_collection(ep_size=ep_size, etp_size=1)
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=config,
        )

        assert adapter.linear_in.weight.allreduce is expected_allreduce
        assert adapter.linear_out.weight.allreduce is expected_allreduce
        assert adapter.linear_in.weight.tensor_model_parallel is True
        assert adapter.linear_out.weight.tensor_model_parallel is True
        assert adapter.linear_in.weight.partition_dim == 1
        assert adapter.linear_out.weight.partition_dim == 1

    def test_grouped_expert_linear_adapter_groups_as_expert_ddp_buffer_when_ep_enabled(self):
        """Per-expert adapter params must sync on expert-DP, not dense DP.

        EP plus DP replicates each local expert across expert-DP ranks. Marking
        these params as expert-parallel keeps replicas for the same expert in
        sync without mixing different EP-owned experts.
        """
        from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers

        config = MockModelParallelConfig()
        config._pg_collection = make_mock_pg_collection(ep_size=8, etp_size=1)
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=config,
        )

        buffer_groups = group_params_for_buffers(
            [adapter.linear_in.weight, adapter.linear_out.weight],
            grad_reduce_in_fp32=False,
        )

        assert len(buffer_groups) == 1
        buffer_key, (params, _param_indices) = next(iter(buffer_groups.items()))
        assert buffer_key.is_expert_parallel
        assert [id(param) for param in params] == [
            id(adapter.linear_in.weight),
            id(adapter.linear_out.weight),
        ]

    def test_grouped_expert_linear_sharded_state_dict_uses_expert_parallel_offsets(self):
        """Grouped-expert weights should shard only across expert EP/ETP and use expert-DP replica ids."""
        config = MockModelParallelConfig()
        config._pg_collection = make_mock_pg_collection(ep_size=2, ep_rank=1, edp_rank=4, etp_size=1, etp_rank=0)
        adapter = GroupedExpertLinearAdapter(
            in_features=2,
            out_features=2,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=False,
            model_parallel_config=config,
        )
        result = adapter.sharded_state_dict("adapter.")

        sharded_weight = result["adapter.linear_in.weight"]
        assert sharded_weight.local_shape == (2, 2, 2)
        assert sharded_weight.global_shape == (4, 2, 2)
        assert sharded_weight.global_offset == (2, 0, 0)
        assert sharded_weight.replica_id == (0, 0, 4)


class _FakeModel:
    """Fake model chunk for PEFT checkpoint helper tests."""

    def sharded_state_dict(self, **kwargs):
        self.kwargs = kwargs
        return {
            "adapter.weight": torch.tensor([1.0]),
            "base.weight": torch.tensor([2.0]),
            "adapter._extra_state": torch.tensor([3.0]),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict


class _FakePeft:
    """Fake PEFT object for utility tests."""

    def __call__(self, model, training: bool):
        self.call = (model, training)
        return model

    def set_params_to_save(self, model) -> None:
        self.saved_model = model

    def adapter_key_filter(self, key: str) -> bool:
        return key.startswith("adapter.")


def _patch_checkpointing(monkeypatch, generate_model_state_dict, filter_state_dict) -> None:
    monkeypatch.setattr(
        "megatron.bridge.training.checkpointing._generate_model_state_dict",
        generate_model_state_dict,
    )
    monkeypatch.setattr(
        "megatron.bridge.training.checkpointing.apply_peft_adapter_filter_to_state_dict",
        filter_state_dict,
    )


def test_create_peft_hook_sets_params_to_save() -> None:
    base_model = [_FakeModel()]
    peft = _FakePeft()

    hook = peft_utils.create_peft_hook(peft)

    assert hook(base_model) is base_model
    assert peft.call == (base_model, True)
    assert peft.saved_model is base_model


def test_create_peft_returns_none_when_rank_disabled() -> None:
    assert peft_utils.create_peft({"rank": 0}) is None


def test_create_peft_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported PEFT type"):
        peft_utils.create_peft({"type": "not_lora", "rank": 4})


def test_create_peft_imports_only_selected_peft_type(monkeypatch) -> None:
    real_import_module = peft_utils.import_module

    def guarded_import_module(module_name):
        if module_name in {"megatron.bridge.peft.lora", "megatron.bridge.peft.canonical_lora"}:
            raise AssertionError(f"unexpected import of {module_name}")
        return real_import_module(module_name)

    monkeypatch.setattr(peft_utils, "import_module", guarded_import_module)

    peft = peft_utils.create_peft({"type": "dora", "rank": 4})

    assert peft.__class__.__name__ == "DoRA"
    assert peft.dim == 4


def test_create_peft_reports_lora_import_failure(monkeypatch) -> None:
    real_import_module = peft_utils.import_module

    def failing_import_module(module_name):
        if module_name == "megatron.bridge.peft.lora":
            raise ModuleNotFoundError("No module named 'transformer_engine'")
        return real_import_module(module_name)

    monkeypatch.setattr(peft_utils, "import_module", failing_import_module)

    with pytest.raises(ImportError, match=r"PEFT type 'lora'.*\(megatron\.bridge\.peft\.lora:LoRA\).*\[te\] extra"):
        peft_utils.create_peft({"type": "lora", "rank": 4})


def test_create_peft_translates_rank_and_ignores_downstream_keys() -> None:
    peft = peft_utils.create_peft(
        {
            "type": "lora",
            "rank": 4,
            "alpha": 8,
            "unknown_downstream_key": True,
        },
        dtype="bf16",
    )

    assert peft.dim == 4
    assert peft.alpha == 8
    assert peft.lora_dtype is torch.bfloat16
    assert not hasattr(peft, "unknown_downstream_key")


def test_create_peft_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        peft_utils.create_peft({"type": "lora", "rank": 4}, dtype="not_a_dtype")


def test_load_peft_adapter_checkpoint_filters_and_loads(monkeypatch) -> None:
    model = [_FakeModel()]
    peft = _FakePeft()
    calls = {}

    def fake_filter(state_dict, peft):
        return {
            "model": {
                key: value
                for key, value in state_dict["model"].items()
                if peft.adapter_key_filter(key) and "_extra_state" not in key
            }
        }

    _patch_checkpointing(
        monkeypatch,
        lambda model, model_sd_kwargs, ckpt_format, pg_collection=None: {"model": model[0].sharded_state_dict()},
        fake_filter,
    )

    def fake_load(sharded_state_dict, checkpoint_path, load_strategy):
        calls["sharded_state_dict"] = sharded_state_dict
        calls["checkpoint_path"] = checkpoint_path
        calls["load_strategy"] = load_strategy
        return {"model": {"adapter.weight": torch.tensor([4.0])}}

    monkeypatch.setattr("megatron.core.dist_checkpointing.load", fake_load)

    peft_utils.load_peft_adapter_checkpoint(
        model,
        "/adapter",
        peft=peft,
        strict=False,
        fully_parallel_load=False,
        load_strategy="strategy",
    )

    assert sorted(calls["sharded_state_dict"]["model"]) == ["adapter.weight"]
    assert calls["checkpoint_path"] == "/adapter"
    assert calls["load_strategy"] == "strategy"
    assert torch.equal(model[0].loaded_state_dict["adapter.weight"], torch.tensor([4.0]))
    assert model[0].loaded_strict is False


def test_load_peft_adapter_checkpoint_builds_default_parallel_strategy(monkeypatch) -> None:
    model = [_FakeModel()]
    peft = _FakePeft()
    pg_collection = make_mock_pg_collection()
    base_strategy = object()
    parallel_strategy = object()

    _patch_checkpointing(
        monkeypatch,
        lambda model, model_sd_kwargs, ckpt_format, pg_collection=None: {"model": model[0].sharded_state_dict()},
        lambda state_dict, peft: state_dict,
    )

    with (
        patch(
            "megatron.core.dist_checkpointing.strategies.torch.TorchDistLoadShardedStrategy",
            return_value=base_strategy,
        ) as mock_strategy_cls,
        patch(
            "megatron.core.dist_checkpointing.strategies.fully_parallel.FullyParallelLoadStrategyWrapper",
            return_value=parallel_strategy,
        ) as mock_parallel_wrapper,
        patch(
            "megatron.core.dist_checkpointing.load",
            return_value={"model": {"adapter.weight": torch.tensor([4.0])}},
        ) as mock_load,
    ):
        peft_utils.load_peft_adapter_checkpoint(
            model,
            "/adapter",
            peft=peft,
            pg_collection=pg_collection,
        )

    mock_strategy_cls.assert_called_once_with()
    mock_parallel_wrapper.assert_called_once_with(base_strategy, pg_collection.dp_cp)
    mock_load.assert_called_once()
    assert mock_load.call_args.args[1:] == ("/adapter", parallel_strategy)


def test_load_peft_adapter_checkpoint_errors_for_missing_model_key(monkeypatch) -> None:
    model = [_FakeModel()]
    peft = _FakePeft()

    _patch_checkpointing(
        monkeypatch,
        lambda model, model_sd_kwargs, ckpt_format, pg_collection=None: {"model": model[0].sharded_state_dict()},
        lambda state_dict, peft: state_dict,
    )
    monkeypatch.setattr(
        "megatron.core.dist_checkpointing.load",
        lambda sharded_state_dict, checkpoint_path, load_strategy: {"optimizer": {}},
    )

    with pytest.raises(KeyError, match="Expected adapter checkpoint"):
        peft_utils.load_peft_adapter_checkpoint(
            model,
            "/adapter",
            peft=peft,
            fully_parallel_load=False,
            load_strategy="strategy",
        )


def test_load_peft_adapter_checkpoint_errors_for_missing_virtual_model_key(monkeypatch) -> None:
    model = [_FakeModel(), _FakeModel()]
    peft = _FakePeft()

    _patch_checkpointing(
        monkeypatch,
        lambda model, model_sd_kwargs, ckpt_format, pg_collection=None: {
            f"model{index}": model_chunk.sharded_state_dict() for index, model_chunk in enumerate(model)
        },
        lambda state_dict, peft: state_dict,
    )
    monkeypatch.setattr(
        "megatron.core.dist_checkpointing.load",
        lambda sharded_state_dict, checkpoint_path, load_strategy: {"model0": {}},
    )

    with pytest.raises(KeyError, match="model1"):
        peft_utils.load_peft_adapter_checkpoint(
            model,
            "/adapter",
            peft=peft,
            fully_parallel_load=False,
            load_strategy="strategy",
        )
