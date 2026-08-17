# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the CPU-evaluable Kimi K3 numerical operators."""

import pytest
import torch
from torch import nn

from megatron.bridge.models.kimi.kimi_k3_ops import (
    KimiRMSNorm,
    SiTUAndMul,
    attn_res_aggregate,
    situ_and_mul,
)


pytestmark = pytest.mark.unit


class TestSituAndMul:
    def test_halves_last_dimension(self):
        out = situ_and_mul(torch.randn(2, 3, 8))

        assert out.shape == (2, 3, 4)

    def test_matches_reference_formula(self):
        torch.manual_seed(0)
        inputs = torch.randn(4, 6, dtype=torch.float32)
        beta, linear_beta = 4.0, 25.0

        out = situ_and_mul(inputs, beta=beta, linear_beta=linear_beta)

        gate, linear = torch.chunk(inputs, 2, dim=-1)
        expected = (beta * torch.tanh(gate / beta) * torch.sigmoid(gate)) * (
            linear_beta * torch.tanh(linear / linear_beta)
        )
        assert torch.allclose(out, expected, atol=1e-6)

    def test_preserves_input_dtype(self):
        out = situ_and_mul(torch.randn(2, 8, dtype=torch.bfloat16))

        assert out.dtype == torch.bfloat16

    def test_output_is_bounded_by_betas(self):
        # tanh saturation bounds each factor by its beta.
        inputs = torch.full((1, 2), 1e4)

        out = situ_and_mul(inputs, beta=4.0, linear_beta=25.0)

        assert out.abs().max() <= 4.0 * 25.0

    def test_module_wrapper_matches_function(self):
        torch.manual_seed(1)
        inputs = torch.randn(3, 10)
        module = SiTUAndMul(config=None)

        assert torch.equal(module(inputs), situ_and_mul(inputs))


class TestKimiRMSNorm:
    def test_unit_weight_matches_reference_rmsnorm(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, 16)
        norm = KimiRMSNorm(hidden_size=16, eps=1e-6)

        out = norm(hidden)

        expected = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1e-6)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_weight_scales_output(self):
        hidden = torch.randn(4, 8)
        norm = KimiRMSNorm(hidden_size=8, eps=1e-6)
        with torch.no_grad():
            norm.weight.mul_(2.0)

        doubled = norm(hidden)
        norm_unit = KimiRMSNorm(hidden_size=8, eps=1e-6)

        assert torch.allclose(doubled, 2.0 * norm_unit(hidden), atol=1e-6)

    def test_normalizes_row_scale_away(self):
        hidden = torch.randn(1, 8)
        norm = KimiRMSNorm(hidden_size=8, eps=1e-12)

        assert torch.allclose(norm(hidden), norm(hidden * 1000.0), atol=1e-4)

    def test_weight_is_learnable_parameter_of_ones(self):
        norm = KimiRMSNorm(hidden_size=8, eps=1e-6)

        assert isinstance(norm.weight, nn.Parameter)
        assert torch.equal(norm.weight.detach(), torch.ones(8))


class TestAttnResAggregate:
    def _components(self, hidden_size, seed=0):
        torch.manual_seed(seed)
        score_proj = nn.Linear(hidden_size, 1, bias=False)
        score_norm = KimiRMSNorm(hidden_size=hidden_size, eps=1e-6)
        return score_proj, score_norm

    def test_output_shape_drops_snapshot_dimension(self):
        hidden_size, num_snapshots = 8, 3
        prefix_sum = torch.randn(2, 4, hidden_size)
        block_residual = torch.randn(2, 4, num_snapshots, hidden_size)
        score_proj, score_norm = self._components(hidden_size)

        out = attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, nn.Identity())

        assert out.shape == (2, 4, hidden_size)

    def test_zero_score_weights_give_uniform_mixture(self):
        hidden_size = 8
        prefix_sum = torch.randn(1, hidden_size)
        block_residual = torch.randn(1, 2, hidden_size)
        score_proj, score_norm = self._components(hidden_size)
        with torch.no_grad():
            score_proj.weight.zero_()

        out = attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, nn.Identity())

        rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
        assert torch.allclose(out, rows.mean(dim=-2), atol=1e-6)

    def test_matches_reference_implementation(self):
        hidden_size = 8
        torch.manual_seed(3)
        prefix_sum = torch.randn(2, hidden_size)
        block_residual = torch.randn(2, 3, hidden_size)
        score_proj, score_norm = self._components(hidden_size, seed=4)

        out = attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, nn.Identity())

        rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2).float()
        normalized = rows * torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + score_norm.eps)
        score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
        probabilities = torch.softmax((normalized * score_weight).sum(dim=-1), dim=-1)
        expected = (probabilities.unsqueeze(-1) * rows).sum(dim=-2)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_output_norm_is_applied(self):
        hidden_size = 8
        prefix_sum = torch.randn(1, hidden_size)
        block_residual = torch.randn(1, 2, hidden_size)
        score_proj, score_norm = self._components(hidden_size)
        output_norm = KimiRMSNorm(hidden_size=hidden_size, eps=1e-6)

        with_norm = attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, output_norm)
        without_norm = attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, nn.Identity())

        assert torch.allclose(with_norm, output_norm(without_norm), atol=1e-6)

    def test_mixture_dtype_follows_input_rows(self):
        hidden_size = 8
        prefix_sum = torch.randn(1, hidden_size, dtype=torch.bfloat16)
        block_residual = torch.randn(1, 2, hidden_size, dtype=torch.bfloat16)
        score_proj = nn.Linear(hidden_size, 1, bias=False).to(torch.bfloat16)
        score_norm = KimiRMSNorm(hidden_size=hidden_size, eps=1e-6, dtype=torch.bfloat16)

        out = attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, nn.Identity())

        assert out.dtype == torch.bfloat16
