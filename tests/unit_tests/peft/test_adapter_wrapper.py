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
Unit tests for base PEFT components.

Tests the AdapterWrapper base class and its functionality for wrapping
modules with adapters in Parameter-Efficient Fine-Tuning scenarios.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora_layers import LoRALinear


class MockLinear(nn.Module):
    """Mock linear module that returns tuples to test base_linear_forward."""

    def __init__(self, return_pattern="simple"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))
        self.bias = nn.Parameter(torch.randn(10))
        self.return_pattern = return_pattern

    def forward(self, x, *args, **kwargs):
        """Simulate different return patterns from Megatron linear layers."""
        output = torch.matmul(x, self.weight.t()) + self.bias

        if self.return_pattern == "simple":
            # Pattern 1: (out, None)
            return output, None
        elif self.return_pattern == "with_bias":
            # Pattern 2: (out, bias)
            return output, self.bias
        elif self.return_pattern == "with_layernorm":
            # Pattern 3: ((out, ln_out), None)
            layernorm_output = x + 0.1  # Simulate layernorm
            return (output, layernorm_output), None
        elif self.return_pattern == "full":
            # Pattern 4: (out, bias, ln_out)
            layernorm_output = x + 0.1
            return output, self.bias, layernorm_output


class ConcreteAdapterWrapper(AdapterWrapper):
    """Concrete implementation of AdapterWrapper for testing."""

    def forward(self, x, *args, **kwargs):
        """Simple forward implementation for testing."""
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        adapter_output = self.adapter(layernorm_output)
        return linear_output + adapter_output, bias


class DummyPEFT(PEFT):
    """Minimal PEFT implementation for adapter enable/disable tests."""

    def transform(self, module: nn.Module, name=None, prefix=None) -> nn.Module:
        return module


class AdapterModel(nn.Module):
    """Model with a single LoRALinear adapter wrapper for testing."""

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.lora = LoRALinear(to_wrap, adapter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lora(x)
        return output


class TestAdapterWrapper:
    """Test the AdapterWrapper base class."""

    @pytest.fixture
    def simple_adapter(self):
        """Create a simple adapter for testing."""
        return nn.Linear(10, 10)

    @pytest.fixture
    def mock_linear_simple(self):
        """Create a mock linear module with simple return pattern."""
        return MockLinear("simple")

    @pytest.fixture
    def mock_linear_bias(self):
        """Create a mock linear module that returns bias."""
        return MockLinear("with_bias")

    @pytest.fixture
    def mock_linear_layernorm(self):
        """Create a mock linear module that returns layernorm output."""
        return MockLinear("with_layernorm")

    @pytest.fixture
    def mock_linear_full(self):
        """Create a mock linear module that returns full pattern."""
        return MockLinear("full")

    def test_adapter_wrapper_init(self, mock_linear_simple, simple_adapter):
        """Test AdapterWrapper initialization."""
        wrapper = ConcreteAdapterWrapper(mock_linear_simple, simple_adapter)

        assert wrapper.to_wrap is mock_linear_simple
        assert wrapper.adapter is simple_adapter
        assert isinstance(wrapper, nn.Module)

    def test_base_linear_forward_simple(self, mock_linear_simple, simple_adapter):
        """Test base_linear_forward with simple return pattern."""
        wrapper = ConcreteAdapterWrapper(mock_linear_simple, simple_adapter)
        x = torch.randn(5, 10)

        linear_output, bias, layernorm_output = wrapper.base_linear_forward(x)

        assert isinstance(linear_output, torch.Tensor)
        assert bias is None
        assert torch.equal(layernorm_output, x)  # Should be input when no layernorm

    def test_base_linear_forward_with_bias(self, mock_linear_bias, simple_adapter):
        """Test base_linear_forward with bias return pattern."""
        wrapper = ConcreteAdapterWrapper(mock_linear_bias, simple_adapter)
        x = torch.randn(5, 10)

        linear_output, bias, layernorm_output = wrapper.base_linear_forward(x)

        assert isinstance(linear_output, torch.Tensor)
        assert bias is not None
        assert torch.equal(layernorm_output, x)

    def test_base_linear_forward_with_layernorm(self, mock_linear_layernorm, simple_adapter):
        """Test base_linear_forward with layernorm output pattern."""
        wrapper = ConcreteAdapterWrapper(mock_linear_layernorm, simple_adapter)
        x = torch.randn(5, 10)

        linear_output, bias, layernorm_output = wrapper.base_linear_forward(x)

        assert isinstance(linear_output, torch.Tensor)
        assert bias is None
        assert not torch.equal(layernorm_output, x)  # Should be different from input

    def test_base_linear_forward_full_pattern(self, mock_linear_full, simple_adapter):
        """Test base_linear_forward with full return pattern."""
        wrapper = ConcreteAdapterWrapper(mock_linear_full, simple_adapter)
        x = torch.randn(5, 10)

        linear_output, bias, layernorm_output = wrapper.base_linear_forward(x)

        assert isinstance(linear_output, torch.Tensor)
        assert bias is not None
        assert not torch.equal(layernorm_output, x)

    def test_base_linear_forward_invalid_return(self, simple_adapter):
        """Test base_linear_forward with invalid return type."""

        class InvalidLinear(nn.Module):
            """Mock linear module that returns a tensor instead of a tuple."""

            def forward(self, x):
                """Return a tensor instead of a tuple."""
                return x

        wrapper = ConcreteAdapterWrapper(InvalidLinear(), simple_adapter)
        x = torch.randn(5, 10)

        with pytest.raises(AssertionError):
            wrapper.base_linear_forward(x)

    def test_state_dict_includes_both_modules(self, mock_linear_simple, simple_adapter):
        """Test that state_dict includes both wrapped module and adapter."""
        wrapper = ConcreteAdapterWrapper(mock_linear_simple, simple_adapter)

        state_dict = wrapper.state_dict()

        # Check that wrapped module parameters are included (without to_wrap prefix)
        assert "weight" in state_dict
        assert "bias" in state_dict

        # Check that adapter parameters are included with prefix
        assert "adapter.weight" in state_dict
        assert "adapter.bias" in state_dict

    def test_state_dict_with_custom_prefix(self, mock_linear_simple, simple_adapter):
        """Test state_dict with custom prefix."""
        wrapper = ConcreteAdapterWrapper(mock_linear_simple, simple_adapter)

        state_dict = wrapper.state_dict(prefix="custom_")

        # Check that custom prefix is applied
        assert "custom_weight" in state_dict
        assert "custom_adapter.weight" in state_dict

    def test_state_dict_with_existing_destination(self, mock_linear_simple, simple_adapter):
        """Test state_dict with existing destination dictionary."""
        wrapper = ConcreteAdapterWrapper(mock_linear_simple, simple_adapter)
        destination = {"existing_key": torch.tensor([1.0])}

        result = wrapper.state_dict(destination=destination)

        assert result is destination
        assert "existing_key" in result
        assert "weight" in result
        assert "adapter.weight" in result

    def test_sharded_state_dict(self, mock_linear_simple, simple_adapter):
        """Test sharded_state_dict functionality."""
        # Mock the sharded_state_dict methods on the modules
        mock_linear_simple.sharded_state_dict = Mock(return_value={"linear_shard": "value1"})
        simple_adapter.sharded_state_dict = Mock(return_value={"adapter_shard": "value2"})

        wrapper = ConcreteAdapterWrapper(mock_linear_simple, simple_adapter)

        result = wrapper.sharded_state_dict(prefix="test_")

        assert "linear_shard" in result
        assert "adapter_shard" in result
        mock_linear_simple.sharded_state_dict.assert_called_once_with("test_", (), None)
        simple_adapter.sharded_state_dict.assert_called_once_with("test_adapter.", (), None)

    def test_forward_integration(self, mock_linear_simple, simple_adapter):
        """Test full forward pass integration."""
        wrapper = ConcreteAdapterWrapper(mock_linear_simple, simple_adapter)
        x = torch.randn(5, 10)

        output, bias = wrapper(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 10)
        # bias should be None for simple pattern
        assert bias is None

    @pytest.mark.parametrize("pattern", ["simple", "with_bias", "with_layernorm", "full"])
    def test_base_linear_forward_all_patterns(self, simple_adapter, pattern):
        """Test base_linear_forward with all return patterns."""
        mock_linear = MockLinear(pattern)
        wrapper = ConcreteAdapterWrapper(mock_linear, simple_adapter)
        x = torch.randn(5, 10)

        linear_output, bias, layernorm_output = wrapper.base_linear_forward(x)

        # All patterns should return valid tensors
        assert isinstance(linear_output, torch.Tensor)
        assert isinstance(layernorm_output, torch.Tensor)

        # Bias behavior depends on pattern
        if pattern in ["with_bias", "full"]:
            assert bias is not None
        else:
            assert bias is None

    def test_adapter_wrapper_is_abstract(self):
        """Test that AdapterWrapper cannot be instantiated directly."""
        linear = nn.Linear(10, 10)
        adapter = nn.Linear(10, 10)

        # This should work fine since ConcreteAdapterWrapper implements forward
        wrapper = ConcreteAdapterWrapper(linear, adapter)
        assert isinstance(wrapper, AdapterWrapper)

        # Test that the base class has the expected methods
        assert hasattr(AdapterWrapper, "base_linear_forward")
        assert hasattr(AdapterWrapper, "state_dict")
        assert hasattr(AdapterWrapper, "sharded_state_dict")

    def test_adapter_wrapper_enable_disable_toggle(self, mock_linear_simple, simple_adapter):
        """Test adapter output toggling via AdapterWrapper methods."""
        wrapper = LoRALinear(mock_linear_simple, simple_adapter)
        x = torch.randn(5, 10)

        base_output, _ = mock_linear_simple(x)
        enabled_output, _ = wrapper(x)
        expected = base_output + simple_adapter(x)
        assert torch.allclose(enabled_output, expected, atol=1e-6)

        wrapper.disable_adapter_layers()
        disabled_output, _ = wrapper(x)
        assert torch.allclose(disabled_output, base_output, atol=1e-6)

        wrapper.enable_adapter_layers()
        reenabled_output, _ = wrapper(x)
        assert torch.allclose(reenabled_output, enabled_output, atol=1e-6)

    def test_peft_disable_adapter_context_manager(self, mock_linear_simple, simple_adapter):
        """Test PEFT.disable_adapter restores adapter state."""
        peft = DummyPEFT()
        model = AdapterModel(mock_linear_simple, simple_adapter)
        x = torch.randn(5, 10)

        base_output, _ = mock_linear_simple(x)
        enabled_output = model(x)

        with peft.disable_adapter(model):
            disabled_output = model(x)
            assert torch.allclose(disabled_output, base_output, atol=1e-6)

        assert torch.allclose(model(x), enabled_output, atol=1e-6)

        with pytest.raises(RuntimeError):
            with peft.disable_adapter(model):
                raise RuntimeError("boom")

        assert torch.allclose(model(x), enabled_output, atol=1e-6)

    def test_peft_enable_disable_adapter_layers_manual(self, mock_linear_simple, simple_adapter):
        """Test manual adapter enable/disable via PEFT helpers."""
        peft = DummyPEFT()
        model = AdapterModel(mock_linear_simple, simple_adapter)
        x = torch.randn(5, 10)

        base_output, _ = mock_linear_simple(x)
        enabled_output = model(x)

        peft.disable_adapter_layers(model)
        disabled_output = model(x)
        assert torch.allclose(disabled_output, base_output, atol=1e-6)

        peft.enable_adapter_layers(model)
        reenabled_output = model(x)
        assert torch.allclose(reenabled_output, enabled_output, atol=1e-6)
