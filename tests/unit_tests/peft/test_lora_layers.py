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
Unit tests for LoRA PEFT components.

Tests LoRA adapters, LinearAdapter, TELinearAdapter, and patch_linear_module
functionality for Parameter-Efficient Fine-Tuning.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.bridge.peft.lora.lora import TELinearAdapter
from megatron.bridge.peft.lora.lora_layers import LinearAdapter, LoRALinear, patch_linear_module


class MockLinearWithTupleReturn(nn.Module):
    """Mock linear module that returns tuples like Megatron layers."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        """Return tuple format like Megatron linear layers."""
        output = self.linear(x)
        return output, None  # (output, bias)


class MockParallelLinearAdapter(nn.Module):
    """Mock parallel linear adapter for testing LoRALinear."""

    def __init__(self, dim=8):
        """Initialize mock parallel linear adapter."""
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.dim = dim

    def forward(self, x):
        """Forward pass returning tuple format."""
        return self.linear(x) * 0.1  # Scale down to simulate adapter


class TestLoRALinear:
    """Test the LoRALinear adapter wrapper."""

    @pytest.fixture
    def mock_linear(self):
        """Create a mock linear module."""
        return MockLinearWithTupleReturn()

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter."""
        return MockParallelLinearAdapter()

    def test_lora_linear_init(self, mock_linear, mock_adapter):
        """Test LoRALinear initialization."""
        lora_linear = LoRALinear(mock_linear, mock_adapter)

        assert lora_linear.to_wrap is mock_linear
        assert lora_linear.adapter is mock_adapter
        assert isinstance(lora_linear, LoRALinear)

    def test_lora_linear_forward(self, mock_linear, mock_adapter):
        """Test LoRALinear forward pass."""
        lora_linear = LoRALinear(mock_linear, mock_adapter)
        x = torch.randn(5, 10)

        output, bias = lora_linear(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 10)
        assert bias is None  # Mock returns None for bias

    def test_lora_linear_adds_adapter_output(self, mock_linear, mock_adapter):
        """Test that LoRALinear adds adapter output to base output."""
        lora_linear = LoRALinear(mock_linear, mock_adapter)
        x = torch.randn(5, 10)

        # Get base output
        base_output, _ = mock_linear(x)
        # Get adapter output (should be applied to layernorm_output, which equals x in this case)
        adapter_output = mock_adapter(x.contiguous())

        # Get LoRA output
        lora_output, _ = lora_linear(x)

        # Verify addition
        expected = base_output + adapter_output
        assert torch.allclose(lora_output, expected, atol=1e-6)


class TestLinearAdapter:
    """Test the LinearAdapter class."""

    @pytest.fixture
    def original_linear(self):
        """Create an original linear layer."""
        linear = nn.Linear(10, 5, bias=True)
        # Initialize with known values for testing
        nn.init.constant_(linear.weight, 1.0)
        nn.init.constant_(linear.bias, 0.1)
        return linear

    @pytest.fixture
    def original_linear_no_bias(self):
        """Create an original linear layer without bias."""
        linear = nn.Linear(10, 5, bias=False)
        nn.init.constant_(linear.weight, 1.0)
        return linear

    def test_linear_adapter_init_with_bias(self, original_linear):
        """Test LinearAdapter initialization with bias."""
        adapter = LinearAdapter(original_linear, dim=8, alpha=16)

        # Check that original weights are copied
        assert torch.equal(adapter.weight, original_linear.weight)
        assert torch.equal(adapter.bias, original_linear.bias)

        # Check LoRA components exist
        assert hasattr(adapter, "lora_a")
        assert hasattr(adapter, "lora_b")
        assert hasattr(adapter, "dropout")
        assert hasattr(adapter, "scale")

        # Check dimensions
        assert adapter.lora_a.in_features == 10
        assert adapter.lora_a.out_features == 8
        assert adapter.lora_b.in_features == 8
        assert adapter.lora_b.out_features == 5

        # Check scale
        assert adapter.scale == 16 / 8  # alpha / dim

    def test_linear_adapter_init_no_bias(self, original_linear_no_bias):
        """Test LinearAdapter initialization without bias."""
        adapter = LinearAdapter(original_linear_no_bias, dim=4, alpha=8)

        assert torch.equal(adapter.weight, original_linear_no_bias.weight)
        assert adapter.bias is None

    def test_linear_adapter_lora_b_initialized_to_zero(self, original_linear):
        """Test that LoRA B matrix is initialized to zero."""
        adapter = LinearAdapter(original_linear)

        assert torch.allclose(adapter.lora_b.weight, torch.zeros_like(adapter.lora_b.weight))

    @pytest.mark.parametrize("lora_A_init_method", ["xavier", "uniform"])
    def test_linear_adapter_lora_a_initialization(self, original_linear, lora_A_init_method):
        """Test LoRA A matrix initialization methods."""
        adapter = LinearAdapter(original_linear, lora_A_init_method=lora_A_init_method)

        # Should not be all zeros
        assert not torch.allclose(adapter.lora_a.weight, torch.zeros_like(adapter.lora_a.weight))

    def test_linear_adapter_freezes_original_weights(self, original_linear):
        """Test that original weights are frozen."""
        adapter = LinearAdapter(original_linear)

        assert not adapter.weight.requires_grad
        if adapter.bias is not None:
            assert not adapter.bias.requires_grad

    def test_linear_adapter_lora_weights_trainable(self, original_linear):
        """Test that LoRA weights are trainable."""
        adapter = LinearAdapter(original_linear)

        assert adapter.lora_a.weight.requires_grad
        assert adapter.lora_b.weight.requires_grad

    @pytest.mark.parametrize("dropout_position", ["pre", "post"])
    def test_linear_adapter_dropout_position(self, original_linear, dropout_position):
        """Test dropout position parameter."""
        adapter = LinearAdapter(original_linear, dropout=0.5, dropout_position=dropout_position)

        assert adapter.dropout_position == dropout_position
        assert isinstance(adapter.dropout, nn.Dropout)

    def test_linear_adapter_forward_basic(self, original_linear):
        """Test LinearAdapter forward pass."""
        adapter = LinearAdapter(original_linear, dim=4)
        x = torch.randn(3, 10)

        output = adapter(x)

        assert output.shape == (3, 5)
        assert isinstance(output, torch.Tensor)

    def test_linear_adapter_forward_with_dropout(self, original_linear):
        """Test LinearAdapter forward with dropout."""
        adapter = LinearAdapter(original_linear, dropout=0.5)
        x = torch.randn(3, 10)

        # Test in training mode
        adapter.train()
        output_train = adapter(x)

        # Test in eval mode
        adapter.eval()
        output_eval = adapter(x)

        assert output_train.shape == output_eval.shape == (3, 5)

    def test_linear_adapter_state_dict_preservation(self, original_linear):
        """Test that state dict keys are preserved as in NeMo tests."""
        state_init = deepcopy(original_linear.state_dict())
        adapter = LinearAdapter(original_linear)

        # Check if the original state-dict keys are preserved
        for key, val in state_init.items():
            assert key in adapter.state_dict(), f"Key {key} not found in LinearAdapter"
            assert torch.equal(val, adapter.state_dict()[key]), f"Key {key} diff. val in LinearAdapter"

        # Make sure the additional keys are in the allow list
        for key, val in adapter.state_dict().items():
            if key in state_init:
                continue
            assert key in ["lora_a.weight", "lora_b.weight"]

    def test_linear_adapter_zero_output_initially(self, original_linear):
        """Test that adapter produces zero output initially (LoRA B is zero)."""
        # Create adapter with specific initialization
        adapter = LinearAdapter(original_linear, dim=4)
        x = torch.randn(3, 10)

        # Get original output
        with torch.no_grad():
            original_output = torch.nn.functional.linear(x, original_linear.weight, original_linear.bias)

        # Get adapter output
        with torch.no_grad():
            adapter_output = adapter(x)

        # Initially, LoRA should add approximately zero
        # (not exactly zero due to random initialization of lora_a, but very small)
        lora_contribution = adapter_output - original_output
        assert torch.allclose(lora_contribution, torch.zeros_like(lora_contribution), atol=1e-2)


class TestPatchLinearModule:
    """Test the patch_linear_module function."""

    def test_patch_linear_module_basic(self):
        """Test basic patching of nn.Linear module."""
        linear = nn.Linear(10, 5)
        state_init = deepcopy(linear.state_dict())

        patched_linear = patch_linear_module(linear, dim=4, alpha=8)

        # Should return the same object (in-place modification)
        assert patched_linear is linear

        # Check if the state-dict keys are preserved
        for key, val in state_init.items():
            assert key in patched_linear.state_dict(), f"Key {key} not found in patched module"
            assert torch.equal(val, patched_linear.state_dict()[key]), f"Key {key} different in patched module"

        # Make sure the additional keys are in the allow list
        for key, val in patched_linear.state_dict().items():
            if key in state_init:
                continue
            assert key in ["lora_a.weight", "lora_b.weight"]

    def test_patch_linear_module_attributes(self):
        """Test that patched module has required LoRA attributes."""
        linear = nn.Linear(10, 5)
        patched_linear = patch_linear_module(linear)

        state_dict = patched_linear.state_dict()
        for key in ["lora_a", "lora_b"]:
            assert hasattr(patched_linear, key), f"Expected {key} to be in module"
            assert f"{key}.weight" in state_dict, f"Expected {key} to be in state dict"
            assert getattr(patched_linear, key).weight.requires_grad == True, f"Expected {key} to require_grad"

    def test_patch_linear_module_already_patched_error(self):
        """Test error when trying to patch already patched module."""
        linear = nn.Linear(10, 5)
        linear.super_fwd = lambda x: x  # Simulate already patched

        with pytest.raises(AssertionError):
            patch_linear_module(linear)

    def test_patch_te_linear_module(self):
        """Test patching TELinear module."""
        te_linear = te.Linear(10, 5, device="cuda")

        patched_linear = patch_linear_module(te_linear, dim=4)

        # Should return the same object
        assert patched_linear is te_linear

        # Check LoRA attributes exist
        assert hasattr(patched_linear, "lora_a")
        assert hasattr(patched_linear, "lora_b")

    def test_patch_linear_module_unsupported_type(self):
        """Test error with unsupported module type."""
        conv = nn.Conv2d(3, 3, 3)

        with pytest.raises(AssertionError):
            patch_linear_module(conv)

    @pytest.mark.parametrize("dim,alpha", [(4, 8), (8, 16), (16, 32)])
    def test_patch_linear_module_parameters(self, dim, alpha):
        """Test patching with different parameters."""
        linear = nn.Linear(10, 5)
        patched_linear = patch_linear_module(linear, dim=dim, alpha=alpha)

        assert patched_linear.dim == dim
        assert patched_linear.scale == alpha / dim
        assert patched_linear.lora_a.out_features == dim
        assert patched_linear.lora_b.in_features == dim


class TestTELinearAdapter:
    """Test the TELinearAdapter class."""

    @pytest.fixture
    def te_linear(self):
        """Create a TE linear layer."""
        return te.Linear(10, 5)

    def test_te_linear_adapter_init(self, te_linear):
        """Test TELinearAdapter initialization."""
        adapter = TELinearAdapter(te_linear, dim=8, alpha=16)

        # Check that it's properly initialized
        assert hasattr(adapter, "lora_a")
        assert hasattr(adapter, "lora_b")
        assert adapter.scale == 16 / 8

        # Check dimensions
        assert adapter.lora_a.in_features == 10
        assert adapter.lora_a.out_features == 8
        assert adapter.lora_b.in_features == 8
        assert adapter.lora_b.out_features == 5

    def test_te_linear_adapter_forward(self, te_linear):
        """Test TELinearAdapter forward pass."""
        adapter = TELinearAdapter(te_linear, dim=4)
        x = torch.randn(3, 10, device="cuda")

        output = adapter(x)

        assert output.shape == (3, 5)
        assert isinstance(output, torch.Tensor)

    def test_te_linear_adapter_weights_frozen(self, te_linear):
        """Test that original TE weights are frozen."""
        adapter = TELinearAdapter(te_linear)

        assert not adapter.weight.requires_grad
        if adapter.bias is not None and adapter.bias.shape[0] != 0:
            assert not adapter.bias.requires_grad


class TestLoRAUtilities:
    """Test utility functions and edge cases."""

    def test_linear_adapter_custom_dtype(self):
        """Test LinearAdapter with custom dtype."""
        linear = nn.Linear(10, 5)
        adapter = LinearAdapter(linear, lora_dtype=torch.float16)

        assert adapter.lora_a.weight.dtype == torch.float16
        assert adapter.lora_b.weight.dtype == torch.float16

    def test_linear_adapter_different_dropout_values(self):
        """Test LinearAdapter with different dropout values."""
        linear = nn.Linear(10, 5)

        # Test zero dropout
        adapter_no_dropout = LinearAdapter(linear, dropout=0.0)

        # Test with dropout
        adapter_with_dropout = LinearAdapter(linear, dropout=0.3)

        x = torch.randn(3, 10)

        # Both should work
        output1 = adapter_no_dropout(x)
        output2 = adapter_with_dropout(x)

        assert output1.shape == output2.shape == (3, 5)

    def test_linear_adapter_math_correctness(self):
        """Test that LinearAdapter math is correct."""
        linear = nn.Linear(10, 5, bias=False)
        nn.init.constant_(linear.weight, 1.0)

        adapter = LinearAdapter(linear, dim=2, alpha=4)

        # Manually set LoRA weights for predictable output
        with torch.no_grad():
            nn.init.constant_(adapter.lora_a.weight, 0.1)
            nn.init.constant_(adapter.lora_b.weight, 0.1)

        x = torch.ones(1, 10)

        # Expected: original + lora_scale * lora_b(lora_a(x))
        # original = x @ linear.weight.T = 1*10 @ 1_{5,10}.T = 10 * ones(1,5)
        # lora_a(x) = x @ lora_a.weight.T = 1*10 @ 0.1_{2,10}.T = 1.0 * ones(1,2)
        # lora_b(lora_a(x)) = 1.0 @ 0.1_{5,2}.T = 0.2 * ones(1,5)
        # lora_scale = alpha/dim = 4/2 = 2
        # final = 10 + 2 * 0.2 = 10.4

        output = adapter(x)
        expected = torch.full((1, 5), 10.4)

        assert torch.allclose(output, expected, atol=1e-6)
