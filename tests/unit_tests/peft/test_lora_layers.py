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

import os
from copy import deepcopy
from types import SimpleNamespace

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.bridge.peft.lora import LoRA, TELinearAdapter
from megatron.bridge.peft.lora_layers import (
    LinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TEFusedLoRALinear,
    patch_linear_module,
)


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
        assert hasattr(adapter, "linear_in")
        assert hasattr(adapter, "linear_out")
        assert hasattr(adapter, "dropout")
        assert hasattr(adapter, "scale")

        # Check dimensions
        assert adapter.linear_in.in_features == 10
        assert adapter.linear_in.out_features == 8
        assert adapter.linear_out.in_features == 8
        assert adapter.linear_out.out_features == 5

        # Check scale
        assert adapter.scale == 16 / 8  # alpha / dim

    def test_linear_adapter_init_no_bias(self, original_linear_no_bias):
        """Test LinearAdapter initialization without bias."""
        adapter = LinearAdapter(original_linear_no_bias, dim=4, alpha=8)

        assert torch.equal(adapter.weight, original_linear_no_bias.weight)
        assert adapter.bias is None

    def test_linear_adapter_linear_out_initialized_to_zero(self, original_linear):
        """Test that LoRA B matrix is initialized to zero."""
        adapter = LinearAdapter(original_linear)

        assert torch.allclose(adapter.linear_out.weight, torch.zeros_like(adapter.linear_out.weight))

    @pytest.mark.parametrize("lora_A_init_method", ["xavier", "uniform"])
    def test_linear_adapter_linear_in_initialization(self, original_linear, lora_A_init_method):
        """Test LoRA A matrix initialization methods."""
        adapter = LinearAdapter(original_linear, lora_A_init_method=lora_A_init_method)

        # Should not be all zeros
        assert not torch.allclose(adapter.linear_in.weight, torch.zeros_like(adapter.linear_in.weight))

    def test_linear_adapter_freezes_original_weights(self, original_linear):
        """Test that original weights are frozen."""
        adapter = LinearAdapter(original_linear)

        assert not adapter.weight.requires_grad
        if adapter.bias is not None:
            assert not adapter.bias.requires_grad

    def test_linear_adapter_lora_weights_trainable(self, original_linear):
        """Test that LoRA weights are trainable."""
        adapter = LinearAdapter(original_linear)

        assert adapter.linear_in.weight.requires_grad
        assert adapter.linear_out.weight.requires_grad

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
            assert key in ["linear_in.weight", "linear_out.weight"]

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
        # (not exactly zero due to random initialization of linear_in, but very small)
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
            assert key in ["linear_in.weight", "linear_out.weight"]

    def test_patch_linear_module_attributes(self):
        """Test that patched module has required LoRA attributes."""
        linear = nn.Linear(10, 5)
        patched_linear = patch_linear_module(linear)

        state_dict = patched_linear.state_dict()
        for key in ["linear_in", "linear_out"]:
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
        assert hasattr(patched_linear, "linear_in")
        assert hasattr(patched_linear, "linear_out")

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
        assert patched_linear.linear_in.out_features == dim
        assert patched_linear.linear_out.in_features == dim


class TestTEFusedLoRALinear:
    """Test the TEFusedLoRALinear adapter wrapper with fused operations."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        """Setup and teardown parallel state for Megatron tests."""

        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            init_process_group_kwargs = {
                "backend": "nccl" if device_count > 0 else "gloo",
                "world_size": 1,
                "rank": 0,
            }

            dist.init_process_group(**init_process_group_kwargs)

        assert dist.is_initialized(), "Distributed backend not initialized"
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"

        from megatron.bridge.training.initialize import _set_random_seed

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
        )

        yield

        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                # Clean up environment variables
                for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    @pytest.fixture
    def te_layer_norm_linear_layernorm(self):
        """Create a TELayerNormLinear with LayerNorm."""
        return te.LayerNormLinear(in_features=10, out_features=5, bias=True, normalization="LayerNorm", device="cuda")

    @pytest.fixture
    def te_layer_norm_linear_rmsnorm(self):
        """Create a TELayerNormLinear with RMSNorm."""
        return te.LayerNormLinear(in_features=10, out_features=5, bias=True, normalization="RMSNorm", device="cuda")

    @pytest.fixture
    def te_linear(self):
        """Create a basic TE linear layer."""
        return te.Linear(10, 5, device="cuda")

    @pytest.fixture
    def linear_adapter(self):
        """Create a LinearAdapter for LoRA."""
        linear = nn.Linear(10, 5, device="cuda")
        return LinearAdapter(linear, dim=4, alpha=8)

    @pytest.fixture
    def parallel_linear_adapter(self):
        """Create a ParallelLinearAdapter for LoRA."""
        from megatron.bridge.peft.utils import ParallelLinearAdapter

        return ParallelLinearAdapter(
            in_features=10,
            out_features=5,
            dim=4,
            base_linear_name="test_linear",
            alpha=8,
            dropout=0.0,
        ).cuda()

    def test_fused_lora_linear_with_layernorm(self, te_layer_norm_linear_layernorm, linear_adapter):
        """Test TEFusedLoRALinear with LayerNormLinear (LayerNorm variant)."""
        fused_lora = TEFusedLoRALinear(te_layer_norm_linear_layernorm, linear_adapter)
        x = torch.randn(3, 10, device="cuda")

        output, bias = fused_lora(x)

        assert output.shape == (3, 5)
        assert bias is None

    def test_fused_lora_linear_with_rmsnorm(self, te_layer_norm_linear_rmsnorm, linear_adapter):
        """Test TEFusedLoRALinear with LayerNormLinear (RMSNorm variant)."""
        fused_lora = TEFusedLoRALinear(te_layer_norm_linear_rmsnorm, linear_adapter)
        x = torch.randn(3, 10, device="cuda")

        output, bias = fused_lora(x)

        assert output.shape == (3, 5)
        assert bias is None

    def test_fused_lora_linear_with_te_linear(self, te_linear, linear_adapter):
        """Test TEFusedLoRALinear with basic TELinear."""
        fused_lora = TEFusedLoRALinear(te_linear, linear_adapter)
        x = torch.randn(3, 10, device="cuda")

        output, bias = fused_lora(x)

        assert output.shape == (3, 5)
        assert bias is None

    def test_fused_lora_linear_with_parallel_adapter(self, te_linear, parallel_linear_adapter):
        """Test TEFusedLoRALinear with ParallelLinearAdapter."""
        fused_lora = TEFusedLoRALinear(te_linear, parallel_linear_adapter)
        x = torch.randn(3, 10, device="cuda")

        output, bias = fused_lora(x)

        assert output.shape == (3, 5)
        assert bias is None

    def test_fused_lora_linear_with_te_linear_adapter(self, te_linear):
        """Test TEFusedLoRALinear with TELinearAdapter."""
        te_adapter = TELinearAdapter(te_linear, dim=4, alpha=8)
        fused_lora = TEFusedLoRALinear(te_linear, te_adapter)
        x = torch.randn(3, 10, device="cuda")

        output, bias = fused_lora(x)

        assert output.shape == (3, 5)
        assert bias is None

    def test_fused_lora_linear_unsupported_normalization(self, te_linear, linear_adapter):
        """Test TEFusedLoRALinear with unsupported normalization type."""
        # Manually create a LayerNormLinear with an unsupported normalization
        te_layer_norm = te.LayerNormLinear(10, 5, device="cuda", normalization="LayerNorm")
        # Hack the normalization type to trigger the error
        te_layer_norm.normalization = "UnsupportedNorm"

        fused_lora = TEFusedLoRALinear(te_layer_norm, linear_adapter)
        x = torch.randn(3, 10, device="cuda")

        with pytest.raises(ValueError, match="Unsupported normalization"):
            fused_lora(x)

    def test_fused_lora_linear_unsupported_adapter(self, te_linear):
        """Test TEFusedLoRALinear with unsupported adapter type."""
        # Create an unsupported adapter type
        unsupported_adapter = nn.Linear(10, 5, device="cuda")

        fused_lora = TEFusedLoRALinear(te_linear, unsupported_adapter)
        x = torch.randn(3, 10, device="cuda")

        with pytest.raises(ValueError, match="Unsupported class for LoRA adapter"):
            fused_lora(x)

    def test_fused_lora_linear_unsupported_wrapped_module(self, linear_adapter):
        """Test TEFusedLoRALinear with unsupported wrapped module type."""
        # Create an unsupported wrapped module
        conv = nn.Conv2d(3, 3, 3).cuda()

        fused_lora = TEFusedLoRALinear(conv, linear_adapter)
        x = torch.randn(1, 3, 5, 5, device="cuda")

        with pytest.raises(ValueError, match="Unsupported class for wrapped linear"):
            fused_lora(x)

    def test_fused_lora_linear_multiple_forward_passes(self, te_linear, linear_adapter):
        """Test that fused branches are reused across forward passes."""
        fused_lora = TEFusedLoRALinear(te_linear, linear_adapter)
        x = torch.randn(3, 10, device="cuda")

        # First forward pass initializes fused branches
        output1, _ = fused_lora(x)
        assert fused_lora._fused_branches is not None

        # Store reference to fused branches
        fused_branches_ref = fused_lora._fused_branches

        # Second forward pass should reuse the same fused branches
        output2, _ = fused_lora(x)
        assert fused_lora._fused_branches is fused_branches_ref

    def test_fused_lora_linear_with_dropout(self, te_linear):
        """Test TEFusedLoRALinear with dropout in adapter."""
        adapter = LinearAdapter(nn.Linear(10, 5, device="cuda"), dim=4, dropout=0.5)
        fused_lora = TEFusedLoRALinear(te_linear, adapter)
        x = torch.randn(3, 10, device="cuda")

        # Train mode
        fused_lora.train()
        output_train, _ = fused_lora(x)

        # Eval mode
        fused_lora.eval()
        output_eval, _ = fused_lora(x)

        assert output_train.shape == output_eval.shape == (3, 5)

    def test_fused_lora_linear_with_dropout_pre(self, te_linear):
        """Test TEFusedLoRALinear with pre-dropout position."""
        adapter = LinearAdapter(nn.Linear(10, 5, device="cuda"), dim=4, dropout=0.3, dropout_position="pre")
        fused_lora = TEFusedLoRALinear(te_linear, adapter)
        x = torch.randn(3, 10, device="cuda")

        output, _ = fused_lora(x)
        assert output.shape == (3, 5)

    def test_fused_lora_linear_with_scale(self, te_linear):
        """Test TEFusedLoRALinear with different scale values."""
        adapter = LinearAdapter(nn.Linear(10, 5, device="cuda"), dim=4, alpha=16)
        fused_lora = TEFusedLoRALinear(te_linear, adapter)
        x = torch.randn(3, 10, device="cuda")

        output, _ = fused_lora(x)
        assert output.shape == (3, 5)
        # Verify scale is correctly set (alpha/dim = 16/4 = 4)
        assert adapter.scale == 4.0


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
        assert hasattr(adapter, "linear_in")
        assert hasattr(adapter, "linear_out")
        assert adapter.scale == 16 / 8

        # Check dimensions
        assert adapter.linear_in.in_features == 10
        assert adapter.linear_in.out_features == 8
        assert adapter.linear_out.in_features == 8
        assert adapter.linear_out.out_features == 5

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

        assert adapter.linear_in.weight.dtype == torch.float16
        assert adapter.linear_out.weight.dtype == torch.float16

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
            nn.init.constant_(adapter.linear_in.weight, 0.1)
            nn.init.constant_(adapter.linear_out.weight, 0.1)

        x = torch.ones(1, 10)

        # Expected: original + lora_scale * linear_out(linear_in(x))
        # original = x @ linear.weight.T = 1*10 @ 1_{5,10}.T = 10 * ones(1,5)
        # linear_in(x) = x @ linear_in.weight.T = 1*10 @ 0.1_{2,10}.T = 1.0 * ones(1,2)
        # linear_out(linear_in(x)) = 1.0 @ 0.1_{5,2}.T = 0.2 * ones(1,5)
        # lora_scale = alpha/dim = 4/2 = 2
        # final = 10 + 2 * 0.2 = 10.4

        output = adapter(x)
        expected = torch.full((1, 5), 10.4)

        assert torch.allclose(output, expected, atol=1e-6)


class DummyRouter(nn.Module):
    def __init__(self, hidden_size: int = 4, num_experts: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))
        self.expert_bias = torch.zeros(num_experts)
        self.config = SimpleNamespace(
            moe_router_force_load_balancing=False,
            sequence_parallel=False,
        )

    def _maintain_float32_expert_bias(self) -> None:
        if isinstance(self.expert_bias, torch.Tensor):
            self.expert_bias = self.expert_bias.float()

    def apply_input_jitter(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def gating(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()

    def routing(self, logits: torch.Tensor):
        return logits, logits > 0


class RouterModel(nn.Module):
    def __init__(self, router_cls: type[DummyRouter]) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.router = router_cls()


class TestLoRATopKRouter:
    """Test LoRA router wrapper behavior."""

    def test_forward_adds_adapter_delta(self) -> None:
        hidden_size = 5
        num_experts = 4
        router = DummyRouter(hidden_size=hidden_size, num_experts=num_experts)
        adapter = nn.Linear(hidden_size, num_experts, bias=False)
        wrapper = LoRATopKRouter(router, adapter)

        x = torch.randn(2, hidden_size)
        expected_logits = router.gating(x) + adapter(x)

        logits, routing_map = wrapper(x)

        assert torch.allclose(logits, expected_logits)
        assert routing_map.shape == expected_logits.shape

    def test_forward_skips_adapter_when_disabled(self) -> None:
        hidden_size = 6
        num_experts = 2
        router = DummyRouter(hidden_size=hidden_size, num_experts=num_experts)
        adapter = nn.Linear(hidden_size, num_experts, bias=False)
        wrapper = LoRATopKRouter(router, adapter)
        wrapper.disable_adapter_layers()

        x = torch.randn(2, hidden_size)
        expected_logits = router.gating(x)

        logits, routing_map = wrapper(x)

        assert torch.allclose(logits, expected_logits)
        assert routing_map.shape == expected_logits.shape

    def test_forward_applies_force_load_balancing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from megatron.bridge.peft import lora_layers as lora_layers_module

        hidden_size = 4
        num_experts = 3
        router = DummyRouter(hidden_size=hidden_size, num_experts=num_experts)
        router.config.moe_router_force_load_balancing = True
        adapter = nn.Linear(hidden_size, num_experts, bias=False)
        wrapper = LoRATopKRouter(router, adapter)

        x = torch.randn(2, hidden_size)
        expected_logits = router.gating(x) + adapter(x)

        def fake_random_logits(logits: torch.Tensor) -> torch.Tensor:
            return logits + 1.0

        monkeypatch.setattr(lora_layers_module, "apply_random_logits", fake_random_logits, raising=True)

        logits, _ = wrapper(x)

        assert torch.allclose(logits, expected_logits + 1.0)

    def test_lora_wraps_router_with_lora_topk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from megatron.bridge.peft import lora as lora_module

        class DummyTopKRouter(DummyRouter):
            pass

        def fake_adapter(in_features, out_features, *args, **kwargs):
            return nn.Linear(in_features, out_features, bias=False)

        def fake_attrs(*args, **kwargs):
            return False, 4, 3, False, True, False

        monkeypatch.setattr(lora_module, "TopKRouter", DummyTopKRouter, raising=True)
        monkeypatch.setattr(lora_module, "ParallelLinearAdapter", fake_adapter, raising=True)
        monkeypatch.setattr(lora_module, "get_adapter_attributes_from_linear", fake_attrs, raising=True)

        model = RouterModel(DummyTopKRouter)
        lora = LoRA(target_modules=["router"])
        transformed = lora(model, training=True)

        assert isinstance(transformed.mlp.router, LoRATopKRouter)


class TestLoRATopKRouterAdapters:
    def test_get_adapter_attributes_topkrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from megatron.bridge.peft import utils as peft_utils

        class DummyTopKRouter(DummyRouter):
            pass

        router = DummyTopKRouter(hidden_size=7, num_experts=5)
        router.config.sequence_parallel = True
        router.parallel_mode = "test"

        monkeypatch.setattr(peft_utils, "TopKRouter", DummyTopKRouter, raising=True)
        monkeypatch.setattr(
            peft_utils.parallel_state,
            "get_tensor_model_parallel_world_size",
            lambda: 1,
            raising=True,
        )

        attrs = peft_utils.get_adapter_attributes_from_linear(router)
        input_is_parallel, in_features, out_features, disable_tp_comm, disable_sp_comm, base_linear_is_parallel = attrs

        assert input_is_parallel is False
        assert in_features == router.weight.shape[1]
        assert out_features == router.weight.shape[0]
        assert disable_tp_comm is False
        assert disable_sp_comm is True
        assert base_linear_is_parallel is False


class TestCanonicalLoRATopKRouter:
    def test_canonical_lora_wraps_router_with_lora_topk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from megatron.bridge.peft import canonical_lora as canonical_module

        class DummyTopKRouter(DummyRouter):
            pass

        def fake_adapter(in_features, out_features, *args, **kwargs):
            return nn.Linear(in_features, out_features, bias=False)

        def fake_attrs(*args, **kwargs):
            return False, 4, 3, False, True, False

        monkeypatch.setattr(canonical_module, "TopKRouter", DummyTopKRouter, raising=True)
        monkeypatch.setattr(canonical_module, "ParallelLinearAdapter", fake_adapter, raising=True)
        monkeypatch.setattr(canonical_module, "get_adapter_attributes_from_linear", fake_attrs, raising=True)

        model = RouterModel(DummyTopKRouter)
        lora = canonical_module.CanonicalLoRA(target_modules=["router"])
        transformed = lora(model, training=True)

        assert isinstance(transformed.mlp.router, LoRATopKRouter)
