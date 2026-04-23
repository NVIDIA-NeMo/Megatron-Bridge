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

"""Tests for LoRA TopKRouter merge/unmerge and LoRAMerge router support.

Validates that:
- LoRATopKRouter.merge_adapter folds the LoRA delta into the router weight
- LoRATopKRouter.unmerge_adapter restores the original weight
- LoRAMerge.transform handles LoRATopKRouter modules correctly
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft.lora_layers import LoRATopKRouter


class MockAdapter(nn.Module):
    """Minimal adapter mimicking ParallelLinearAdapter's interface."""

    def __init__(self, in_features, out_features, rank, alpha=32):
        super().__init__()
        self.dim = rank
        self.alpha = alpha
        self.linear_in = nn.Linear(in_features, rank, bias=False)
        self.linear_out = nn.Linear(rank, out_features, bias=False)
        nn.init.ones_(self.linear_in.weight)
        nn.init.ones_(self.linear_out.weight)

    def forward(self, x):
        return self.linear_out(self.linear_in(x)) * (self.alpha / self.dim)


class MockRouter(nn.Module):
    """Minimal TopKRouter stand-in with a weight attribute."""

    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.config = SimpleNamespace(moe_router_force_load_balancing=False)

    def _maintain_float32_expert_bias(self):
        pass

    def apply_input_jitter(self, x):
        return x

    def gating(self, x):
        return torch.matmul(x, self.weight.t())

    def routing(self, logits, *args, **kwargs):
        return logits


class TestLoRATopKRouterMergeUnmerge:
    @pytest.fixture
    def router_with_lora(self):
        hidden_size = 16
        num_experts = 4
        rank = 4

        router = MockRouter(hidden_size, num_experts)
        adapter = MockAdapter(hidden_size, num_experts, rank, alpha=rank)
        return LoRATopKRouter(router, adapter)

    def test_merge_modifies_weight(self, router_with_lora):
        original_weight = router_with_lora.to_wrap.weight.data.clone()
        router_with_lora.merge_adapter()

        assert not torch.equal(router_with_lora.to_wrap.weight.data, original_weight)
        assert router_with_lora._merged is True

    def test_merge_is_idempotent(self, router_with_lora):
        router_with_lora.merge_adapter()
        weight_after_first = router_with_lora.to_wrap.weight.data.clone()
        router_with_lora.merge_adapter()  # second call should be no-op
        assert torch.equal(router_with_lora.to_wrap.weight.data, weight_after_first)

    def test_unmerge_restores_weight(self, router_with_lora):
        original_weight = router_with_lora.to_wrap.weight.data.clone()
        router_with_lora.merge_adapter()
        router_with_lora.unmerge_adapter()

        assert torch.allclose(router_with_lora.to_wrap.weight.data, original_weight, atol=1e-6)
        assert router_with_lora._merged is False

    def test_unmerge_without_merge_is_noop(self, router_with_lora):
        original_weight = router_with_lora.to_wrap.weight.data.clone()
        router_with_lora.unmerge_adapter()  # no-op
        assert torch.equal(router_with_lora.to_wrap.weight.data, original_weight)

    def test_forward_skips_adapter_when_merged(self, router_with_lora):
        x = torch.randn(2, 16)

        # Forward with adapter (not merged)
        out_with_adapter = router_with_lora(x)

        # Forward with merged weights
        router_with_lora.merge_adapter()
        out_merged = router_with_lora(x)

        assert torch.allclose(out_with_adapter, out_merged, atol=1e-5)

    def test_merge_delta_correctness(self, router_with_lora):
        """Verify the merged weight equals W + (alpha/dim) * B @ A."""
        original_weight = router_with_lora.to_wrap.weight.data.clone()
        adapter = router_with_lora.adapter
        expected_delta = (adapter.alpha / adapter.dim) * (
            adapter.linear_out.weight.data @ adapter.linear_in.weight.data
        )
        router_with_lora.merge_adapter()
        actual = router_with_lora.to_wrap.weight.data
        expected = original_weight + expected_delta
        assert torch.allclose(actual, expected, atol=1e-6)


class TestLoRAMergeTransformRouter:
    @pytest.fixture
    def router_module(self):
        hidden_size = 16
        num_experts = 4
        rank = 4

        router = MockRouter(hidden_size, num_experts)
        adapter = MockAdapter(hidden_size, num_experts, rank, alpha=rank)
        return LoRATopKRouter(router, adapter)

    @patch(
        "megatron.bridge.peft.lora.parallel_state.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    def test_merge_transform_handles_router(self, mock_tp, router_module):
        from megatron.bridge.peft.lora import LoRAMerge

        original_weight = router_module.to_wrap.weight.data.clone()
        merger = LoRAMerge()
        result = merger.transform(router_module, name="router", prefix="mlp")

        assert result is router_module
        assert not torch.equal(router_module.to_wrap.weight.data, original_weight)
