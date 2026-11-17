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

"""GPU tests of the ERNIE 5.5 fused fp32 mHC kernels (``ernie_fused_mhc``) against the eager fp32 operations.

The kernels sum in their own order, so they are compared with tolerances, not bitwise:

* mappings ``h_pre``, ``h_post``, ``H_res``: ``rtol=1e-5, atol=1e-6`` (bounded values; the norm and the
  20 Sinkhorn rounds add a few fp32 ulps);
* aggregate and residual update (fp32 streams): ``rtol=1e-5, atol=1e-5 * max|eager|``; bf16 streams (two single
  roundings of fp32 sums): ``|a - b| <= ulp_bf16(max(|a|, |b|)) + 1e-5 * max|eager|`` and fewer than 0.1% of the
  elements differ;
* gradients: relative norm ``||fused - eager|| / ||eager||`` at most 1e-5, or 5e-5 for the mapping weight and the
  alphas (sums over all tokens in a different GEMM layout), plus an elementwise check.

They must be bitwise reproducible: two calls, and a ``no_grad`` call against a grad call, give equal outputs, and
two backward passes give equal gradients. The launch count per hyper-connection unit is checked with the
profiler. TF32 is pinned off (the projection follows PyTorch's TF32 setting in both paths).

Needs a CUDA device and Triton; without them the tests skip. One GPU is enough::

    python -m pytest -rA tests/unit_tests/models/ernie/test_ernie55_fused_mhc_gpu.py
"""

import dataclasses
import math
import os

import pytest
import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.modeling_ernie55 import fused_mhc
from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import triton_available
from megatron.bridge.models.ernie.modeling_ernie55.hyper_connection import (
    ernie55_h_aggregate,
    ernie55_h_post_bda,
    use_ernie55_hyper_connections,
)
from tests.unit_tests.models.ernie.ernie55_test_utils import isolate_fused_dispatch


pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or not triton_available(), reason="the fused mHC kernels need CUDA and Triton"
    ),
]

STREAMS = 4
# (s, b, C); the second is the full-size shape per context-parallel rank (n * C = 16384).
SHAPES = [(64, 2, 512), (4096, 1, 4096)]
SHAPE_IDS = ["small", "full"]


@pytest.fixture(autouse=True)
def _device_and_tf32():
    """One device per rank under ``torch.distributed.run``; TF32 off for the fp32 GEMMs, restored afterwards."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


@dataclasses.dataclass
class _MhcConfig(TransformerConfig):
    """Megatron-Core's config plus the one ERNIE field the hyper-connection module reads."""

    ernie_fused_mhc: bool = False


def _module(*, channels: int, iterations: int, fused: bool, seed: int = 0) -> HyperConnectionModule:
    """An ERNIE hyper-connection module on the GPU, every parameter drawn at O(1) logit scale."""
    config = _MhcConfig(
        num_layers=1,
        hidden_size=channels,
        num_attention_heads=4,
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_sinkhorn_iterations=iterations,
        use_cpu_initialization=True,
        ernie_fused_mhc=fused,
    )
    module = HyperConnectionModule(config, layer_number=1)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        weight = module.mapping_proj.weight
        weight.copy_(torch.randn(weight.shape, generator=generator) / math.sqrt(weight.shape[1]))
        module.bias.copy_(torch.randn(module.bias.shape, generator=generator))
        for alpha in (module.alpha_pre, module.alpha_post, module.alpha_res):
            alpha.copy_(0.6 + 0.3 * torch.rand(1, generator=generator))
    use_ernie55_hyper_connections(module)
    return module.cuda()


def _pair(channels: int, iterations: int):
    """The same parameters with the eager and the fused operations."""
    eager = _module(channels=channels, iterations=iterations, fused=False)
    fused = _module(channels=channels, iterations=iterations, fused=True)
    fused.load_state_dict(eager.state_dict())
    assert fused._ernie_fused_mhc and not eager._ernie_fused_mhc
    return eager, fused


def _inputs(shape, stream_dtype, *, seed: int = 1):
    s, b, channels = shape
    generator = torch.Generator().manual_seed(seed)
    x = (3.0 * torch.randn(s, b, STREAMS * channels, generator=generator)).to(stream_dtype)
    y = torch.randn(s, b, channels, generator=generator).to(torch.bfloat16)
    return x.cuda(), y.cuda()


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.double()
    return ((actual.double() - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def _assert_grad_close(actual: torch.Tensor, expected: torch.Tensor, name: str, bound: float = 1e-5) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
    if expected.dtype == torch.bfloat16:
        # Both paths round fp32 values that agree to fp32 rounding; only a few elements may land one ulp apart.
        assert _relative(actual, expected) <= 1e-3, (name, _relative(actual, expected))
        return
    assert _relative(actual, expected) <= bound, (name, _relative(actual, expected))
    scale = expected.abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5 * scale, msg=name)


def _assert_bf16_single_rounding_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float) -> None:
    """Two single roundings to bf16 of fp32 values that agree to ``atol``; fewer than 0.1% of the elements differ.

    Both paths sum in fp32 (in different orders) and round once. Rounding two fp32 values ``a`` and ``b`` to bf16 can
    land them one bf16 ulp apart, where the ulp is that of the larger magnitude (the smaller one may sit in the
    binade below); on top of that, the fp32 values themselves differ by the summation-order error, which is
    absolute (a few fp32 ulps of the summed terms). Where the update nearly cancels, that error exceeds one bf16 ulp
    of the small result, so a pure ulp bound fails for two correct implementations. Elementwise:
    ``|a - b| <= ulp_bf16(max(|a|, |b|)) + atol``.
    """
    assert actual.dtype == expected.dtype == torch.bfloat16
    a, b = actual.float(), expected.float()
    _, exponent = torch.frexp(torch.maximum(a.abs(), b.abs()))
    ulp = torch.ldexp(torch.ones_like(b), exponent - 8)  # bf16: 8 significant bits
    difference = (a - b).abs()
    excess = difference - (ulp + atol)
    assert bool((excess <= 0).all()), (difference.max().item(), excess.max().item())
    assert (actual != expected).float().mean().item() < 1e-3


def _unit(module, x, y):
    """One hyper-connection unit: mappings, aggregate and residual update, as the hybrid layer runs them."""
    aggregated, h_res, h_post, residual = module(x, return_residual=True)
    out = module.fused_h_res_h_post_bda(
        h_res, residual, h_post, (y, None), dropout_prob=0.0, training=True, fused=False
    )
    return aggregated, out


def _run_unit(module, x, y, *, seed: int = 2):
    """Forward and backward of one unit with fixed cotangents; returns outputs and every gradient."""
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    aggregated, out = _unit(module, x, y)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    g_agg = torch.randn(aggregated.shape, generator=generator, device="cuda", dtype=aggregated.dtype)
    g_out = torch.randn(out.shape, generator=generator, device="cuda").to(out.dtype)
    torch.autograd.backward((aggregated, out), (g_agg, g_out))
    grads = {"x": x.grad, "y": y.grad}
    grads.update({name: p.grad for name, p in module.named_parameters()})
    return {"aggregated": aggregated.detach(), "out": out.detach()}, grads


# ---------------------------------------------------------------------------------------------
# Against the eager operations
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
@pytest.mark.parametrize("iterations", [1, 20])
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mappings_match_eager(monkeypatch, shape, iterations, stream_dtype):
    dispatch = isolate_fused_dispatch(monkeypatch)
    eager, fused = _pair(shape[2], iterations)
    x, _ = _inputs(shape, stream_dtype)
    expected = eager.compute_mappings(x)
    actual = fused.compute_mappings(x)
    assert dispatch.FUSED_OP_CALLS["mhc_mappings@hyper_connection"] == 1
    for name, a, e in zip(("h_pre", "h_post", "h_res"), actual, expected, strict=True):
        assert a.dtype == torch.float32 and a.shape == e.shape and a.is_contiguous(), name
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_aggregate_and_residual_update_match_eager(shape, stream_dtype):
    s, b, channels = shape
    generator = torch.Generator().manual_seed(3)
    streams = (3.0 * torch.randn(s, b, STREAMS, channels, generator=generator)).to(stream_dtype).cuda()
    h_pre = (torch.rand(s, b, STREAMS, generator=generator) + 1e-6).cuda()
    h_post = (2 * torch.rand(s, b, STREAMS, generator=generator)).cuda()
    h_res = torch.rand(s, b, STREAMS, STREAMS, generator=generator).cuda()
    h_res = h_res / h_res.sum(dim=-2, keepdim=True)
    y = torch.randn(s, b, channels, generator=generator).to(torch.bfloat16).cuda()

    aggregated = fused_mhc.fused_mhc_aggregate(streams, h_pre)
    expected = ernie55_h_aggregate(streams, h_pre)
    assert aggregated.dtype == torch.float32 and aggregated.shape == expected.shape
    torch.testing.assert_close(aggregated, expected, rtol=1e-5, atol=1e-5 * expected.abs().max().item())

    updated = fused_mhc.fused_mhc_post_bda(h_res, streams, h_post, y)
    expected = ernie55_h_post_bda(h_res, streams, h_post, y, None)
    assert updated.dtype == stream_dtype and updated.shape == expected.shape and updated.is_contiguous()
    if stream_dtype == torch.bfloat16:
        _assert_bf16_single_rounding_close(updated, expected, atol=1e-5 * expected.float().abs().max().item())
    else:
        torch.testing.assert_close(updated, expected, rtol=1e-5, atol=1e-5 * expected.abs().max().item())


@pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
@pytest.mark.parametrize("iterations", [1, 20])
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_unit_forward_and_gradients_match_eager(monkeypatch, shape, iterations, stream_dtype):
    dispatch = isolate_fused_dispatch(monkeypatch)
    eager, fused = _pair(shape[2], iterations)
    x, y = _inputs(shape, stream_dtype)
    expected_out, expected_grads = _run_unit(eager, x, y)
    assert not any(key.startswith("mhc_") for key in dispatch.FUSED_OP_CALLS)
    actual_out, actual_grads = _run_unit(fused, x, y)
    for op in ("mhc_mappings", "mhc_aggregate", "mhc_post_bda"):
        assert dispatch.FUSED_OP_CALLS[f"{op}@hyper_connection"] == 1, dict(dispatch.FUSED_OP_CALLS)

    reference = expected_out["aggregated"]
    torch.testing.assert_close(
        actual_out["aggregated"], reference, rtol=1e-5, atol=1e-5 * reference.abs().max().item()
    )
    if stream_dtype == torch.bfloat16:
        reference = expected_out["out"]
        _assert_bf16_single_rounding_close(
            actual_out["out"], reference, atol=1e-5 * reference.float().abs().max().item()
        )
    else:
        reference = expected_out["out"]
        torch.testing.assert_close(actual_out["out"], reference, rtol=1e-5, atol=1e-5 * reference.abs().max().item())

    assert actual_grads.keys() == expected_grads.keys()
    loose = {"mapping_proj.weight", "alpha_pre", "alpha_post", "alpha_res"}
    for name, expected in expected_grads.items():
        _assert_grad_close(actual_grads[name], expected, name, bound=5e-5 if name in loose else 1e-5)


# ---------------------------------------------------------------------------------------------
# Determinism and launch count
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
def test_forward_and_backward_are_bitwise_reproducible(monkeypatch, stream_dtype):
    """Run to run and no_grad against grad (the full-recompute replay), every output and gradient is equal."""
    dispatch = isolate_fused_dispatch(monkeypatch)
    shape = SHAPES[0]
    module = _module(channels=shape[2], iterations=20, fused=True)
    x, y = _inputs(shape, stream_dtype)
    with torch.no_grad():
        first = [t.clone() for t in _unit(module, x, y)]
        mappings_no_grad = [t.clone() for t in module.compute_mappings(x)]
    outputs_a, grads_a = _run_unit(module, x, y)
    outputs_b, grads_b = _run_unit(module, x, y)
    for no_grad_tensor, name in zip(first, ("aggregated", "out"), strict=True):
        assert torch.equal(no_grad_tensor, outputs_a[name]), name
        assert torch.equal(outputs_a[name], outputs_b[name]), name
    with torch.enable_grad():
        mappings_grad = module.compute_mappings(x.detach().clone().requires_grad_(True))
    for a, b in zip(mappings_no_grad, mappings_grad, strict=True):
        assert torch.equal(a, b.detach())
    # The fused kernels ran (an all-eager fallback would replay bitwise too and prove nothing).
    for op in ("mhc_mappings", "mhc_aggregate", "mhc_post_bda"):
        assert dispatch.FUSED_OP_CALLS[f"{op}@hyper_connection"] > 0, dict(dispatch.FUSED_OP_CALLS)
    for name, grad in grads_a.items():
        assert torch.equal(grad, grads_b[name]), name


def _kernel_launches(fn) -> int:
    """CUDA kernels (not copies or memsets) that ``fn`` launches."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA and not event.name.startswith(("Memcpy", "Memset"))
    )


def _unit_launches(module, x, y) -> tuple[int, int]:
    """``(forward, backward)`` kernel launches of one unit, after a warm-up step (Triton and torch.compile)."""
    state = {}

    def forward():
        state["outputs"] = _unit(module, *state["inputs"])

    def backward():
        torch.autograd.backward(state["outputs"], state["cotangents"])

    for count in (False, True):
        # Leaf inputs without a copy; gradients reset to None so that no accumulation kernel runs.
        state["inputs"] = (x.detach().requires_grad_(True), y.detach().requires_grad_(True))
        module.zero_grad(set_to_none=True)
        if not count:
            forward()
            state["cotangents"] = tuple(torch.ones_like(t) for t in state["outputs"])
            backward()
            continue
        forward_launches = _kernel_launches(forward)
        backward_launches = _kernel_launches(backward)
    return forward_launches, backward_launches


@pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
def test_launches_per_unit(stream_dtype):
    """Fused: at most 5 forward kernels (6 for bf16 streams) and 9 backward kernels per unit; eager: over 100.

    Forward: the projection GEMM (cuBLAS may split its K = n*C reduction into a second kernel), the mapping epilogue,
    the aggregate and the residual update, plus the fp32 cast of bf16 streams for the GEMM. The fused path multiplies
    by the transposed view of ``W``; the eager path's contiguous ``W^T`` copy was one more kernel.
    """
    shape = SHAPES[0]
    x, y = _inputs(shape, stream_dtype)
    eager_forward, eager_backward = _unit_launches(_module(channels=shape[2], iterations=20, fused=False), x, y)
    fused_forward, fused_backward = _unit_launches(_module(channels=shape[2], iterations=20, fused=True), x, y)
    assert eager_forward >= 100, eager_forward
    assert fused_forward <= (5 if stream_dtype == torch.float32 else 6), fused_forward
    # 8 (post-update, aggregate, mapping epilogue, partial sum, two GEMMs, norm term, Megatron-Core's stream-gradient
    # add), plus one if autograd materializes a zero gradient.
    assert fused_backward <= 9, (fused_backward, eager_backward)
