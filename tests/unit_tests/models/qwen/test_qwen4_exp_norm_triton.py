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

"""Optional fused-forward dispatch contracts; numerical GPU qualification is separate."""

import builtins
import importlib.util
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.qwen.modeling_qwen4_exp import norms


def test_cpu_and_strided_inputs_keep_eager_path():
    x = torch.randn(3, 32)[..., ::2].requires_grad_()
    weight = torch.randn(16, requires_grad=True)
    assert not norms._can_use_triton(x, weight)
    actual = norms.grouped_rmsnorm(x, weight, 4, 1e-6)
    values = x.float().reshape(3, 4, 4)
    expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)).flatten(-2)
    expected = expected * (1 + weight.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(actual.sum(), (x, weight)))


def test_import_without_triton(monkeypatch):
    original = builtins.__import__

    def optional_import(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("optional Triton unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", optional_import)
    spec = importlib.util.spec_from_file_location("norms_without_triton", norms.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.triton is None
    x, weight = torch.randn(2, 16), torch.randn(16)
    torch.testing.assert_close(module.grouped_rmsnorm(x, weight, 4, 1e-6), norms.grouped_rmsnorm(x, weight, 4, 1e-6))


def test_launch_uses_input_device_and_keeps_saved_state(monkeypatch):
    events = []

    @contextmanager
    def device_context(device):
        events.append(("enter", device))
        yield
        events.append(("exit", device))

    class Launch:
        def __getitem__(self, grid):
            assert grid == (2, 4)

            def kernel(x, weight, rstd, output, rows, eps, **kwargs):
                assert events == [("enter", x.device)]
                assert kwargs == dict(GROUPS=4, CHANNELS=16, BM=8, BC=128, num_warps=8, enable_fp_fusion=False)
                values = x.float().reshape(rows, 4, 16)
                rstd.copy_(torch.rsqrt(values.square().mean(-1, keepdim=True) + eps))
                output.copy_((values * rstd * (1 + weight.float()).reshape(4, 16)).flatten(-2))

            return kernel

    monkeypatch.setattr(norms, "_can_use_triton", lambda *_: True)
    monkeypatch.setattr(norms, "triton", SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d))
    monkeypatch.setattr(norms, "_grouped_rmsnorm_fwd_kernel", Launch(), raising=False)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    x, weight = torch.randn(9, 64, requires_grad=True), torch.randn(64, requires_grad=True)
    result = norms.grouped_rmsnorm(x, weight, 4, 1e-6)
    assert events == [("enter", x.device), ("exit", x.device)]
    assert result.grad_fn.saved_tensors[2].dtype == torch.float32
    assert result.grad_fn.chunk == norms.norm_chunk_rows(64, 64 << 20)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(result.sum(), (x, weight)))


@pytest.mark.skipif(not torch.cuda.is_available() or norms.triton is None, reason="Requires CUDA and Triton")
@pytest.mark.parametrize("channels", [129, 2560])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_forward_statistics_and_gradients(monkeypatch, channels, dtype):
    """Guard real fused dispatch and gradients at channel/row tails and model width."""
    torch.manual_seed(29)
    x = torch.randn(9, 1, 4 * channels, device="cuda", dtype=dtype)
    x[0], x[1] = 0, 1e-7
    x.requires_grad_()
    weight = torch.randn(4 * channels, device="cuda", dtype=dtype) * 0.1
    weight[0] = -1
    weight.requires_grad_()
    ref_x, ref_w = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    original, launches = norms._grouped_rmsnorm_fwd_kernel, []

    class Witness:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                compiled = original[grid](*args, **kwargs)
                launches.append(grid)
                return compiled

            return run

    monkeypatch.setattr(norms, "_grouped_rmsnorm_fwd_kernel", Witness())
    assert norms._can_use_triton(x, weight)
    actual = norms.grouped_rmsnorm(x, weight, 4, 1e-6)
    assert launches == [(2, 4)]
    actual_rstd = actual.grad_fn.saved_tensors[2]
    values = ref_x.float().unflatten(-1, (4, channels))
    expected_rstd = torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    expected = ((values * expected_rstd).flatten(-2) * (1 + ref_w.float())).to(dtype)
    dy = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, weight), dy)
    expected_grads = torch.autograd.grad(expected, (ref_x, ref_w), dy)
    torch.testing.assert_close(actual_rstd, expected_rstd.reshape(9, 4, 1), rtol=3e-6, atol=3e-6)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-2
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)


def test_large_index_layout_falls_back(monkeypatch):
    monkeypatch.setattr(norms, "triton", object())
    metadata = dict(is_cuda=True, is_contiguous=lambda: True, device=torch.device("cuda"), dtype=torch.bfloat16)
    x = SimpleNamespace(**metadata, numel=lambda: 2**31)
    weight = SimpleNamespace(**metadata)
    assert not norms._can_use_triton(x, weight)
