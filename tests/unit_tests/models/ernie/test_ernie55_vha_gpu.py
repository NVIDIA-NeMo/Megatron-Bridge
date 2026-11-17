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

"""GPU tests of the batched VHA head mix (``ernie_vha_batched_mix``) against ``torch.matmul`` at ERNIE 5.5 shapes.

:func:`~megatron.bridge.models.ernie.modeling_ernie55.ops.vha_postmix` with ``batched_mix=True`` applies the bf16 mix
``M = I + V U^T`` as one strided-batched GEMM over the tokens (stride-0 ``M``) with an explicit backward, where
``torch.matmul`` folds the tokens into one GEMM over the head-last transpose and copies the activation in and out.
The shapes are one context-parallel rank's micro-batch of the 64-GPU SFT recipe (4096 rows,
:mod:`ernie55_vha_utils`): the H core output ``[4096, 1, 64, 512]`` (contiguous) and the D layer's V-up einsum
result ``[4096, 1, 64, 256]`` (head-major, made by the D core's own einsum), plus odd shapes.

Pass criteria:

* V1 forward: the batched output equals ``torch.matmul``'s bitwise (``torch.equal``) at every shape.
* V2 input gradient: bitwise equal as well (the same 64 products per entry, accumulated in fp32 by cuBLAS).
* V3 mix gradient: ``dM`` within one bf16 ulp at its own scale of ``torch.matmul``'s (largest difference at most 2^-8
  of the largest ``|dM|``: both round one fp32 sum of the same ``tokens * D`` products, in different orders, so entries
  near zero may differ by more ulps of their own); the ``U`` and ``V`` gradients within a relative max error of 1e-2
  and a cosine of at least 0.99999 of ``torch.matmul``'s. The ``dM`` bars are tighter because the ``U`` and ``V``
  bars alone also pass a ``dM`` that rounds every token's product to bf16 (CPU: 3e-3 of the largest
  entry): the token-product sum that ``dM`` rounds (``_sum_of_token_products`` on CUDA, ``bmm`` with
  ``out_dtype=float32``) is fp32 and within 2^-14 of the largest entry of the fp64 sum of the same products (room for
  tensor-core accumulation; the CPU's fp32 path is at 2e-7), and the bf16 ``dM`` equals that fp64 sum rounded once
  in at least 99% of the entries (per-token bf16 rounding: 58% on the CPU). ``torch.matmul``'s share is logged.
* V4 copies: a forward and backward of the batched mix (activation a ``[t, 1, H * D]`` view, as the H core hands it
  over) records no ``aten::copy_`` of a tensor of the activation's size in the torch profiler; ``torch.matmul``'s
  records at least 3 (sanity check of the probe). The CUDA kernels of both are logged, except under nsys
  (``ERNIE55_UNDER_NSYS=1`` under Nsight Systems), which owns CUPTI: then the profiler records CPU ops only,
  which the check reads.
* V5 CUDA graph: forward and backward captured in one CUDA graph and replayed on new input values give bitwise the
  eager results on those values.
* V6 time: an eager forward plus backward of the batched mix at the H shape takes at most 0.75 of ``torch.matmul``'s
  (median of 20 after warmup); the graph replay times are logged.
* V7 batch count: 65539 tokens (more batches than a CUDA grid dimension holds; 64 heads of 8): the output and the
  input gradient within a relative max error of 2^-7 of ``torch.matmul``'s, the ``U`` and ``V`` gradients within the
  V3 bars.

The tests skip without a CUDA device (``ERNIE55_GPU_STRICT=1`` turns the skip into a failure). One GPU::

    ERNIE55_GPU_STRICT=1 python -m pytest -rA -o log_cli=true -o log_cli_level=INFO \\
        tests/unit_tests/models/ernie/test_ernie55_vha_gpu.py

Under ``nsys profile --capture-range=cudaProfilerApi`` the capture is V6's timed loops (the file's only
``cudaProfilerStart``/``Stop`` range, NVTX ``vha/<case>/<matmul|batched>/<eager|graph>``).
"""

import logging
import math
import os
import statistics

import pytest
import torch

from tests.unit_tests.models.ernie import ernie55_vha_utils as bench


logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on("GPU")]

_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"
_UNDER_NSYS = os.environ.get("ERNIE55_UNDER_NSYS") == "1"

# (case, rows, batch): the SFT recipe's cases, one token, an odd token count and a batch of 3 sequences.
_SHAPES = [("h", 4096, 1), ("d", 4096, 1), ("h", 1, 1), ("h", 37, 1), ("d", 5, 3)]
_MAX_FACTOR_REL_ERR = 1e-2
_MIN_FACTOR_COSINE = 0.99999
_MAX_TIME_RATIO = 0.75
_MAX_SUM_REL_ERR = 2**-14
_MIN_CORRECTLY_ROUNDED = 0.99


def _skip(reason: str) -> None:
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def device():
    if not torch.cuda.is_available():
        _skip("needs a CUDA device")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    yield torch.device("cuda", torch.cuda.current_device())


def _inputs(case: str, rows: int, batch: int, device: torch.device, seed: int):
    """Benchmark inputs, with ``batch`` sequences of ``rows`` rows (``[rows, batch, H, D]``) when ``batch > 1``."""
    out, u, v, grad = bench.make_inputs(case, device=device, rows=rows * batch, seed=seed)
    if batch == 1:
        return out, u, v, grad
    heads, dim = bench.CASES[case]
    # The same layout family as the single-sequence case: [rows, batch] over the one token axis.
    out = out.detach().reshape(rows, batch, heads, dim) if out.is_contiguous() else _head_major(out, rows, batch)
    return out.requires_grad_(True), u, v, grad.reshape(rows, batch, heads, dim)


def _head_major(out: torch.Tensor, rows: int, batch: int) -> torch.Tensor:
    """``[rows * batch, 1, H, D]`` head-major -> ``[rows, batch, H, D]`` with the heads still outermost in memory."""
    heads, dim = out.shape[-2:]
    storage = out.detach().permute(2, 0, 1, 3).contiguous()  # [H, rows * batch, 1, D]
    return storage.view(heads, rows, batch, dim).permute(1, 2, 0, 3)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-300)).item()


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected.double()).abs().max() / expected.double().abs().max().clamp_min(1e-300)).item()


@pytest.mark.parametrize(("case", "rows", "batch"), _SHAPES, ids=lambda value: str(value))
def test_forward_and_input_gradient_are_bitwise(device, case, rows, batch):
    """V1, V2."""
    inputs = _inputs(case, rows, batch, device, seed=11)
    if case == "d":
        assert not inputs[0].is_contiguous(), "the D case must keep the V-up einsum's head-major layout"
    matmul = bench.vha_step(*inputs, batched_mix=False)
    batched = bench.vha_step(*inputs, batched_mix=True)
    for name, expected, actual in zip(("output", "d_out"), matmul[:2], batched[:2]):
        differing = int((expected != actual).sum().item())
        logger.info(
            "V1/V2 %s %s: %d of %d entries differ, %.3g bf16 ulps at most",
            (case, rows, batch),
            name,
            differing,
            expected.numel(),
            bench.bf16_ulp_distance(actual, expected),
        )
        assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape, name
        assert differing == 0, f"{name}: {differing} of {expected.numel()} entries differ from torch.matmul"


@pytest.mark.parametrize(("case", "rows", "batch"), _SHAPES, ids=lambda value: str(value))
def test_mix_gradient_within_summation_order(device, case, rows, batch):
    """V3."""
    from megatron.bridge.models.ernie.modeling_ernie55 import ops

    out, u, v, grad = _inputs(case, rows, batch, device, seed=12)
    mix = bench.mix_gradients(out, u=u, v=v, grad=grad)
    scale_rel_err = _rel_err(mix["batched"], mix["matmul"])
    differing = int((mix["batched"] != mix["matmul"]).sum().item())
    matmul = bench.vha_step(out, u, v, grad, batched_mix=False)
    batched = bench.vha_step(out, u, v, grad, batched_mix=True)
    factors = {
        name: (_rel_err(actual, expected), _cosine(actual, expected))
        for name, expected, actual in zip(("d_u", "d_v"), matmul[2:], batched[2:])
    }
    # The dM contract itself: one fp32 sum of the token products, rounded to bf16 once.
    heads, dim = out.shape[-2:]
    rows_3d, grad_3d = out.detach().reshape(-1, heads, dim), grad.reshape(-1, heads, dim)
    exact = torch.einsum("nkd,nhd->kh", grad_3d.double(), rows_3d.double())
    with torch.no_grad():  # as in the backward: bmm with out_dtype=float32 on CUDA
        summed = ops._sum_of_token_products(grad_3d, rows_3d)
    sum_rel_err = _rel_err(summed, exact)
    rounded_once = exact.to(torch.bfloat16)
    correctly_rounded = {name: (value == rounded_once).double().mean().item() for name, value in mix.items()}
    logger.info(
        "V3 %s: dM %d of %d differ, largest difference %.3g of max |dM| (%.3g ulps of its own entry); factors %s; "
        "fp32 token sum %.3g of max |fp64 sum|; dM = fp64 sum rounded once in %s of the entries",
        (case, rows, batch),
        differing,
        mix["matmul"].numel(),
        scale_rel_err,
        bench.bf16_ulp_distance(mix["batched"], mix["matmul"]),
        factors,
        sum_rel_err,
        {name: f"{100 * share:.2f}%" for name, share in correctly_rounded.items()},
    )
    assert mix["batched"].dtype == mix["matmul"].dtype == torch.bfloat16
    assert scale_rel_err <= 2**-8, f"dM differs from torch.matmul's by {scale_rel_err:.3g} of its scale"
    for name, (rel_err, cosine) in factors.items():
        assert rel_err <= _MAX_FACTOR_REL_ERR and cosine >= _MIN_FACTOR_COSINE, (name, rel_err, cosine)
    assert summed.dtype == torch.float32, f"the token products are summed in {summed.dtype}"
    assert sum_rel_err <= _MAX_SUM_REL_ERR, f"the fp32 token sum is {sum_rel_err:.3g} of its scale off the fp64 sum"
    assert correctly_rounded["batched"] >= _MIN_CORRECTLY_ROUNDED, (
        f"dM is the fp64 sum rounded once in only {100 * correctly_rounded['batched']:.2f}% of the entries"
    )


def test_no_activation_copies(device):
    """V4, at the H shape: the activation reaches the core as a view of ``[t, 1, H * D]``."""
    heads, dim = bench.CASES["h"]
    out, u, v, grad = bench.make_inputs("h", device=device, seed=13)
    flat = out.detach().reshape(bench.ROWS, 1, heads * dim).requires_grad_(True)
    numel = flat.numel()

    def step(batched_mix: bool):
        from megatron.bridge.models.ernie.modeling_ernie55 import ops

        mixed = ops.vha_postmix(flat.view(bench.ROWS, 1, heads, dim), u=u, v=v, bf16_mix=True, batched_mix=batched_mix)
        mixed = mixed.reshape(flat.shape)
        return torch.autograd.grad(mixed, (flat, u, v), grad.reshape(flat.shape))

    copies, kernels = {}, {}
    # Under nsys, which owns CUPTI, record CPU ops only: the copies are aten::copy_ CPU events either way.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if not _UNDER_NSYS:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    for name, batched_mix in (("matmul", False), ("batched", True)):
        step(batched_mix)  # warm up cuBLAS
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            step(batched_mix)
            torch.cuda.synchronize()
        copies[name] = [
            event.input_shapes
            for event in prof.events()
            if event.name == "aten::copy_" and any(math.prod(shape) >= numel for shape in event.input_shapes if shape)
        ]
        kernels[name] = sorted(
            {event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA}
        )
        logger.info(
            "V4 %s: %d activation-sized copies %s; CUDA kernels %s",
            name,
            len(copies[name]),
            copies[name],
            kernels[name],
        )
    assert len(copies["matmul"]) >= 3, f"the probe found too few torch.matmul copies: {copies['matmul']}"
    assert copies["batched"] == [], f"the batched mix copied the activation: {copies['batched']}"


@pytest.mark.parametrize("case", ["h", "d"])
def test_graph_replay_is_bitwise_eager(device, case):
    """V5."""
    inputs = bench.make_inputs(case, device=device, seed=14)
    graph, static = bench.capture_graph(lambda: bench.vha_step(*inputs, batched_mix=True))
    fresh = bench.make_inputs(case, device=device, seed=15)
    with torch.no_grad():
        for target, source in zip(inputs, fresh):
            target.copy_(source)
    graph.replay()
    replayed = [tensor.detach().clone() for tensor in static]
    eager = bench.vha_step(*inputs, batched_mix=True)
    for name, expected, actual in zip(("output", "d_out", "d_u", "d_v"), eager, replayed):
        assert torch.equal(actual, expected), (
            f"{name}: graph replay differs from eager by {_rel_err(actual, expected):.3g}"
        )


def test_batched_mix_is_faster(device):
    """V6, at the H shape (CUDA events; median of 20 after 5 warmup runs, in matmul-batched-batched-matmul order)."""
    results = bench.benchmark(["h", "d"], device=device, iters=20, warmup=5)
    for case, result in results.items():
        logger.info(
            "V6 %s: eager %.3f -> %.3f ms (x%.2f), graph %.3f -> %.3f ms (x%.2f), eager peak extra %.0f -> %.0f MiB",
            case,
            result["matmul"]["eager_ms_median"],
            result["batched"]["eager_ms_median"],
            result["eager_speedup"],
            result["matmul"]["graph_ms_median"],
            result["batched"]["graph_ms_median"],
            result["graph_speedup"],
            result["matmul"]["eager_peak_extra_mib"],
            result["batched"]["eager_peak_extra_mib"],
        )
    ratio = results["h"]["batched"]["eager_ms_median"] / results["h"]["matmul"]["eager_ms_median"]
    assert ratio <= _MAX_TIME_RATIO, f"batched / matmul eager time at the H shape is {ratio:.2f} > {_MAX_TIME_RATIO}"
    assert statistics.median([results["h"]["graph_speedup"], results["d"]["graph_speedup"]]) > 1.0


def test_more_tokens_than_a_grid_dimension(device):
    """V7."""
    generator = torch.Generator(device=device).manual_seed(16)
    tokens, heads, dim = 2**16 + 3, 64, 8
    out = torch.randn(tokens, 1, heads, dim, device=device, generator=generator).bfloat16().requires_grad_(True)
    u = (0.3 * torch.randn(heads, bench.VHA_RANK, device=device, generator=generator)).requires_grad_(True)
    v = (0.3 * torch.randn(heads, bench.VHA_RANK, device=device, generator=generator)).requires_grad_(True)
    grad = torch.randn(tokens, 1, heads, dim, device=device, generator=generator).bfloat16()
    matmul = bench.vha_step(out, u, v, grad, batched_mix=False)
    batched = bench.vha_step(out, u, v, grad, batched_mix=True)
    for name, expected, actual in zip(("output", "d_out"), matmul[:2], batched[:2]):
        rel_err = _rel_err(actual, expected)
        logger.info(
            "V7 %s: %d entries differ, relative max error %.3g", name, int((actual != expected).sum()), rel_err
        )
        assert rel_err <= 2**-7, (name, rel_err)
    for name, expected, actual in zip(("d_u", "d_v"), matmul[2:], batched[2:]):
        rel_err, cosine = _rel_err(actual, expected), _cosine(actual, expected)
        assert rel_err <= _MAX_FACTOR_REL_ERR and cosine >= _MIN_FACTOR_COSINE, (name, rel_err, cosine)
