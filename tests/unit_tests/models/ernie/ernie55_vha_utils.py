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

"""Shapes, inputs and timing helpers for the ERNIE 5.5 VHA head-mix GPU tests (``test_ernie55_vha_gpu.py``).

``ernie_vha_batched_mix`` applies the D and H layers' head mix ``M = I + V U^T`` (64 x 64, bf16) as one
strided-batched GEMM over the tokens instead of ``torch.matmul(M, out)``
(:func:`~megatron.bridge.models.ernie.modeling_ernie55.ops.vha_postmix`). These helpers build one forward and
backward of both at the shapes of one context-parallel rank of the 64-GPU SFT recipe (4096 rows), eagerly and
replayed from a CUDA graph, and time them with CUDA events.

Cases: ``h`` is the H core output ``[4096, 1, 64, 512]`` (contiguous); ``d`` is the D layer's V-up result
``[4096, 1, 64, 256]``, made by the same ``torch.einsum`` the D core runs, so it has that einsum's head-major layout.
"""

from __future__ import annotations

import statistics

import torch


ROWS = 4096
"""Query rows of one context-parallel rank in one micro-batch (262144 / 64)."""

CASES = {"h": (64, 512), "d": (64, 256)}
"""Case name -> (heads, head dim): the H core output and the D layer's V-up result at the ERNIE 5.5 geometry."""

IMPLEMENTATIONS = {"matmul": False, "batched": True}
"""Implementation name -> ``batched_mix`` of :func:`~megatron.bridge.models.ernie.modeling_ernie55.ops.vha_postmix`."""

VHA_RANK = 16
D_KV_LORA_RANK = 512


def make_inputs(
    case: str, *, device: torch.device, rows: int = ROWS, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``out`` (bf16, a leaf with the case's layout), ``U``, ``V`` (fp32 leaves) and a bf16 output gradient."""
    heads, dim = CASES[case]
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*size: int, scale: float = 1.0) -> torch.Tensor:
        return scale * torch.randn(*size, device=device, generator=generator)

    if case == "d":
        latent = randn(rows, 1, heads, D_KV_LORA_RANK).bfloat16()
        up_v = randn(heads, dim, D_KV_LORA_RANK, scale=D_KV_LORA_RANK**-0.5).bfloat16()
        out = torch.einsum("sbhc,hdc->sbhd", latent, up_v)
    else:
        out = randn(rows, 1, heads, dim).bfloat16()
    u = randn(heads, VHA_RANK, scale=0.3)
    v = randn(heads, VHA_RANK, scale=0.3)
    grad = randn(rows, 1, heads, dim).bfloat16()
    return out.detach().requires_grad_(True), u.requires_grad_(True), v.requires_grad_(True), grad


def vha_step(
    out: torch.Tensor, u: torch.Tensor, v: torch.Tensor, grad: torch.Tensor, *, batched_mix: bool, nvtx: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward and backward of the bf16 VHA mix: the output and the gradients of ``out``, ``U`` and ``V``.

    ``torch.autograd.grad`` instead of ``backward``, so no gradient-accumulation copy runs on top of the op.
    """
    from megatron.bridge.models.ernie.modeling_ernie55 import ops

    if nvtx:
        torch.cuda.nvtx.range_push("fwd")
    mixed = ops.vha_postmix(out, u=u, v=v, bf16_mix=True, batched_mix=batched_mix)
    if nvtx:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("bwd")
    d_out, d_u, d_v = torch.autograd.grad(mixed, (out, u, v), grad)
    if nvtx:
        torch.cuda.nvtx.range_pop()
    return mixed, d_out, d_u, d_v


def capture_graph(fn, *, warmup: int = 3) -> tuple[torch.cuda.CUDAGraph, object]:
    """Capture ``fn()`` (forward and backward) in a CUDA graph after ``warmup`` runs on a side stream."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_result = fn()
    torch.cuda.synchronize()
    return graph, static_result


def bf16_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Largest ``|actual - expected|`` in units of the bf16 spacing at ``max(|actual|, |expected|)``."""
    actual, expected = actual.float(), expected.float()
    magnitude = torch.maximum(actual.abs(), expected.abs()).clamp_min(torch.finfo(torch.bfloat16).tiny)
    spacing = torch.exp2(torch.floor(torch.log2(magnitude)) - 7)
    return ((actual - expected).abs() / spacing).max().item()


def mix_gradients(
    out: torch.Tensor, *, u: torch.Tensor, v: torch.Tensor, grad: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The gradient of the bf16 mix ``M`` itself through ``torch.matmul`` and through the batched GEMM."""
    from megatron.bridge.models.ernie.modeling_ernie55.ops import _TokenBatchedHeadMix

    heads = out.size(-2)
    with torch.no_grad():
        identity = torch.eye(heads, dtype=out.dtype, device=out.device)
        mix = torch.addmm(identity, v.to(out.dtype), u.to(out.dtype).t())
    result = {}
    for name in ("matmul", "batched"):
        leaf = mix.detach().requires_grad_(True)
        source = out.detach()
        if name == "matmul":
            mixed = torch.matmul(leaf, source)
        else:
            mixed = _TokenBatchedHeadMix.apply(leaf, source.reshape(-1, heads, source.size(-1))).view(source.shape)
        (result[name],) = torch.autograd.grad(mixed, (leaf,), grad)
    return result


def compare_numerics(case: str, *, device: torch.device, seed: int = 0, rows: int = ROWS) -> dict[str, float]:
    """Batched against ``torch.matmul`` at one case: differing elements, mix-gradient ulps, factor-gradient errors."""
    inputs = make_inputs(case, device=device, rows=rows, seed=seed)
    matmul = vha_step(*inputs, batched_mix=False)
    batched = vha_step(*inputs, batched_mix=True)
    report = {}
    for name, expected, actual in zip(("output", "d_out", "d_u", "d_v"), matmul, batched):
        report[f"{name}_differing"] = int((expected != actual).sum().item())
        report[f"{name}_max_rel"] = (
            (actual.float() - expected.float()).abs().max() / expected.float().abs().max()
        ).item()
    out, u, v, grad = inputs
    mix = mix_gradients(out, u=u, v=v, grad=grad)
    report["d_mix_differing"] = int((mix["matmul"] != mix["batched"]).sum().item())
    difference = (mix["batched"].float() - mix["matmul"].float()).abs().max()
    report["d_mix_max_rel"] = (difference / mix["matmul"].float().abs().max()).item()
    report["d_mix_max_ulps_own_entry"] = bf16_ulp_distance(mix["batched"], mix["matmul"])
    return report


def _time(run, *, iters: int, label: str) -> list[float]:
    """Milliseconds of ``run()`` per iteration (CUDA events), each iteration in an NVTX range ``label``."""
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(label)
        start.record()
        run()
        end.record()
        torch.cuda.nvtx.range_pop()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _prepare(case: str, impl: str, *, device: torch.device, warmup: int) -> dict[str, object]:
    """Inputs, a warmed-up eager step, its captured graph, and the peak extra memory of one eager step."""
    inputs = make_inputs(case, device=device, seed=1)
    batched_mix = IMPLEMENTATIONS[impl]

    def eager():
        return vha_step(*inputs, batched_mix=batched_mix, nvtx=True)

    for _ in range(warmup):
        eager()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    eager()
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated(device) - resident
    graph, _ = capture_graph(lambda: vha_step(*inputs, batched_mix=batched_mix))
    for _ in range(warmup):
        graph.replay()
    return {"inputs": inputs, "eager": eager, "graph": graph, "eager_peak_extra_mib": peak_extra / 2**20}


def benchmark(cases: list[str], *, device: torch.device, iters: int, warmup: int) -> dict[str, dict]:
    """Median forward-plus-backward time of both implementations, eager and replayed from a graph, per case.

    Everything is warmed up and captured first; the timed loops then run in one ``cudaProfilerStart``/``Stop``
    window (the nsys capture range), in the order matmul, batched, batched, matmul with half the iterations each.
    """
    prepared = {
        (case, impl): _prepare(case, impl, device=device, warmup=warmup) for case in cases for impl in IMPLEMENTATIONS
    }
    torch.cuda.synchronize()
    times = {key: {"eager": [], "graph": []} for key in prepared}
    half = max(1, iters // 2)
    torch.cuda.profiler.start()
    for case in cases:
        for impl in ("matmul", "batched", "batched", "matmul"):
            state = prepared[case, impl]
            times[case, impl]["eager"] += _time(state["eager"], iters=half, label=f"vha/{case}/{impl}/eager")
            times[case, impl]["graph"] += _time(state["graph"].replay, iters=half, label=f"vha/{case}/{impl}/graph")
    torch.cuda.profiler.stop()
    results = {}
    for case in cases:
        result = {}
        for impl in IMPLEMENTATIONS:
            measured = times[case, impl]
            result[impl] = {
                "eager_ms_median": statistics.median(measured["eager"]),
                "eager_ms_min": min(measured["eager"]),
                "graph_ms_median": statistics.median(measured["graph"]),
                "graph_ms_min": min(measured["graph"]),
                "iterations": len(measured["eager"]),
                "eager_peak_extra_mib": prepared[case, impl]["eager_peak_extra_mib"],
            }
        for mode in ("eager", "graph"):
            result[f"{mode}_speedup"] = result["matmul"][f"{mode}_ms_median"] / result["batched"][f"{mode}_ms_median"]
        results[case] = result
    return results
