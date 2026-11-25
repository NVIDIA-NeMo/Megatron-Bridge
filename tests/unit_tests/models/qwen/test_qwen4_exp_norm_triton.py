# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Fused norm dispatch, forward/backward arithmetic and bounded workspace."""

import pytest
import torch

from megatron.bridge.models.qwen.modeling_qwen4_exp import norms


@pytest.mark.parametrize("budget", [1, 40959, 40960, 64 << 20])
def test_backward_partials_respect_workspace(budget):
    partitions = norms._backward_partitions(2048, 10240, 4, budget, 148)
    assert partitions * 10240 * 4 <= budget
    assert 0 <= partitions <= 74
    assert norms._backward_partitions(3, 10240, 4, budget, 148) <= 3


@pytest.mark.skipif(not torch.cuda.is_available() or norms.triton is None, reason="Requires CUDA and Triton")
@pytest.mark.parametrize("channels", [129, 2560])
@pytest.mark.parametrize(
    "dtype,weight_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_fused_norm_forward_backward_and_accumulation(monkeypatch, channels, dtype, weight_dtype):
    torch.manual_seed(73)
    x = torch.randn(17, 4 * channels, device="cuda", dtype=dtype)
    x[0], x[1] = 0, 1e-7
    x.requires_grad_()
    weight = (torch.randn(4 * channels, device="cuda", dtype=weight_dtype) * 0.1).requires_grad_()
    with torch.no_grad():
        weight[::19] = -1
    dy = torch.randn_like(x)
    launches = []

    class Witness:
        def __init__(self, original, name):
            self.original, self.name = original, name

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launches.append(self.name)
                assert kwargs["enable_fp_fusion"] is False
                return self.original[grid](*args, **kwargs)

            return run

    for name in ("_grouped_rmsnorm_fwd_kernel", "_grouped_rmsnorm_bwd_kernel", "_grouped_rmsnorm_dw_kernel"):
        monkeypatch.setattr(norms, name, Witness(getattr(norms, name), name))
    y = norms.grouped_rmsnorm(x, weight, 4, 1e-6)
    rstd = y.grad_fn.saved_tensors[2]
    # Independent eager oracle checks both forward statistics and full gradients;
    # the saved-statistic oracle below separately isolates backward arithmetic.
    ref_x, ref_w = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    values = ref_x.float().reshape(17, 4, channels)
    expected_rstd = torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    expected_y = ((values * expected_rstd).flatten(-2) * (1 + ref_w.float())).to(dtype)
    torch.testing.assert_close(rstd, expected_rstd, rtol=3e-6, atol=3e-6)
    eager_gradients = torch.autograd.grad(expected_y, (ref_x, ref_w), dy)
    z = x.detach().float().reshape(17, 4, channels) * rstd
    gradient = dy.float().reshape_as(z)
    scaled = gradient * (1 + weight.detach().float()).reshape(4, channels)
    expected_dx = (rstd * (scaled - z * (scaled * z).mean(-1, keepdim=True))).reshape_as(x).to(dtype)
    expected_dw = (gradient * z).flatten(-2).sum(0).to(weight_dtype)
    gradients = torch.autograd.grad(y, (x, weight), dy, retain_graph=True)
    repeated = torch.autograd.grad(y, (x, weight), dy, retain_graph=True)
    assert (
        launches == ["_grouped_rmsnorm_fwd_kernel"] + ["_grouped_rmsnorm_bwd_kernel", "_grouped_rmsnorm_dw_kernel"] * 2
    )
    for actual, again in zip(gradients, repeated):
        assert torch.equal(actual.contiguous().view(torch.uint8), again.contiguous().view(torch.uint8))
    tolerance = 3e-5 if dtype == torch.float32 else (3e-3 if dtype == torch.float16 else 2e-2)
    for actual, expected in zip((y, *gradients), (expected_y, *eager_gradients)):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(gradients, (expected_dx, expected_dw)):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert x.grad is None and weight.grad is None
    y.backward(dy, retain_graph=True)
    first = weight.grad.clone()
    y.backward(dy)
    torch.testing.assert_close(weight.grad, first + first, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or norms.triton is None, reason="Requires CUDA and Triton")
@pytest.mark.parametrize("layout", ["expanded", "strided", "tiny_workspace"])
def test_backward_unsupported_gradient_or_workspace_falls_back(monkeypatch, layout):
    x = torch.randn(9, 516, device="cuda", requires_grad=True)
    weight = torch.randn(516, device="cuda", requires_grad=True)
    budget = 1 if layout == "tiny_workspace" else 64 << 20
    y = norms.grouped_rmsnorm(x, weight, 4, 1e-6, workspace_bytes=budget)
    if layout == "expanded":
        dy = torch.randn(1, 516, device="cuda").expand_as(x)
    elif layout == "strided":
        dy = torch.randn(9, 1032, device="cuda")[:, ::2]
    else:
        dy = torch.randn_like(x)

    class MustNotLaunch:
        def __getitem__(self, _):
            raise AssertionError("unsupported backward must retain eager path")

    monkeypatch.setattr(norms, "_grouped_rmsnorm_bwd_kernel", MustNotLaunch())
    actual = torch.autograd.grad(y, (x, weight), dy)
    assert all(torch.isfinite(g).all() for g in actual)
