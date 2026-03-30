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

import pytest
import torch

from megatron.bridge.diffusion.models.common.normalization import RMSNorm


pytestmark = [pytest.mark.unit]


class TestRMSNormInitialization:
    """Tests for RMSNorm constructor."""

    def test_default_initialization(self):
        norm = RMSNorm(hidden_size=64)
        assert norm.eps == 1e-6
        assert norm.weight.shape == (64,)
        assert torch.allclose(norm.weight.data, torch.ones(64))

    def test_custom_eps(self):
        norm = RMSNorm(hidden_size=32, eps=1e-5)
        assert norm.eps == 1e-5

    def test_config_parameter_accepted(self):
        norm = RMSNorm(hidden_size=16, config="ignored_value")
        assert norm.weight.shape == (16,)

    def test_different_hidden_sizes(self):
        for size in [1, 8, 128, 1024]:
            norm = RMSNorm(hidden_size=size)
            assert norm.weight.shape == (size,)


class TestRMSNormForward:
    """Tests for RMSNorm forward pass."""

    def test_forward_2d_input(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(4, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_forward_3d_input(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_numerical_correctness(self):
        norm = RMSNorm(hidden_size=4, eps=1e-6)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = norm(x)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = (x / rms) * norm.weight
        assert torch.allclose(out, expected, atol=1e-5)

    def test_preserves_dtype_bfloat16(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16

    def test_preserves_dtype_float16(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(2, 32, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_preserves_dtype_float32(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(2, 32, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32

    def test_output_is_finite(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(4, 10, 64)
        out = norm(x)
        assert torch.isfinite(out).all()

    def test_zero_input(self):
        norm = RMSNorm(hidden_size=16)
        x = torch.zeros(2, 16)
        out = norm(x)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_weight_scaling(self):
        norm = RMSNorm(hidden_size=8)
        norm.weight.data.fill_(2.0)
        x = torch.ones(1, 8)
        out = norm(x)

        norm_ref = RMSNorm(hidden_size=8)
        out_ref = norm_ref(x)
        assert torch.allclose(out, 2.0 * out_ref, atol=1e-5)

    def test_gradient_flow(self):
        norm = RMSNorm(hidden_size=16)
        x = torch.randn(3, 16, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_batch_independence(self):
        norm = RMSNorm(hidden_size=8)
        x = torch.randn(4, 8)
        out_full = norm(x)
        for i in range(4):
            out_single = norm(x[i : i + 1])
            assert torch.allclose(out_full[i : i + 1], out_single, atol=1e-6)

    def test_large_values(self):
        norm = RMSNorm(hidden_size=16)
        x = torch.randn(2, 16) * 1000.0
        out = norm(x)
        assert torch.isfinite(out).all()
