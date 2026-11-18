#!/usr/bin/env python3
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

"""
Time the ERNIE 5.5 routed activation unfused and fused (``ernie_fused_situ_prob``) on one GPU.

The unfused path is what the routed experts run without the flag: Bridge's SiTU-GLU Triton kernel, then
Megatron-Core's product with the fp32 routing probabilities and the cast back to bf16 (``(act * probs).to(bf16)``),
three kernels forward and six backward. The fused path is ``fused_situ_prob.fused_situ_glu_prob``, one kernel per
direction. For every row count the script times the forward alone and forward plus backward of both paths (CUDA
events, after warm-up), checks that the outputs and the input gradients are bitwise equal, and reports the largest
probability-gradient difference relative to its row's fp32 summation-order bound.

Shapes: ``--width`` is the expert FFN width (``moe_intermediate_size`` in the model's config.json; the fc1 output
has twice as many columns). ``--rows`` are the routed rows a rank's experts receive per micro-batch: the rank's tokens
times the router's top-k on average, and more on its busiest experts. Set both from the model's config and the
parallel layout; the defaults are only example shapes.

The script first prepares every row count: inputs, the bitwise checks (each path's first launch, Triton compilation
included) and the warm-up calls. Only then does it run the timed loops, back to back, each in an NVTX range
(``situ_prob/<path>/<direction>/rows=<n>``). With ``--profile`` just those loops sit between ``cudaProfilerStart``
and ``cudaProfilerStop``, so ``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`` records the
timed calls and nothing else: ``--iters`` calls per path, direction and row count. Per row count the fused path's
forward kernel thus runs ``2 * iters`` times (``iters`` in each direction) and its backward kernel ``iters`` times;
the reported means are wall times per call, launch gaps included. Under ``torch.distributed.run`` every rank uses its
own GPU (``LOCAL_RANK``) and runs the same sweep.

Usage:

    uv run python examples/models/ernie55/benchmark_situ_prob.py --rows 40960 81920 122880

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx --sample=none \
        -o situ_prob python examples/models/ernie55/benchmark_situ_prob.py --profile --output situ_prob.json

Requires CUDA and Triton.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable

import torch

from megatron.bridge.models.ernie.modeling_ernie55.fused_ops import fused_situ_glu
from megatron.bridge.models.ernie.modeling_ernie55.fused_situ_prob import fused_situ_glu_prob


logger = logging.getLogger(__name__)
_U = 2.0**-24


def _unfused(x: torch.Tensor, probs: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    act = fused_situ_glu(x, beta, linear_beta)
    return (act * probs).to(act.dtype)


def _fused(x: torch.Tensor, probs: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    return fused_situ_glu_prob(x, probs, beta, linear_beta)


_PATHS = {"unfused": _unfused, "fused": _fused}


def _time_ms(step, iters: int, label: str) -> float:
    """Mean milliseconds of ``step()`` over ``iters`` calls, inside one NVTX range."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(label)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _prepare(rows: int, args: argparse.Namespace) -> tuple[dict[str, object], dict[tuple[str, str], Callable]]:
    """Inputs of one row count, the bitwise checks of both paths and the warmed-up steps to time.

    Runs before the profiler's capture range: the checks, the first launches and the warm-up calls stay out of it.
    """
    generator = torch.Generator(device="cuda").manual_seed(rows)
    x = (args.sigma * torch.randn(rows, 2 * args.width, device="cuda", generator=generator)).to(torch.bfloat16)
    probs = 3 * torch.rand(rows, 1, device="cuda", generator=generator)
    grad = torch.randn(rows, args.width, device="cuda", generator=generator).to(torch.bfloat16)
    leaf_x, leaf_probs = x.clone().requires_grad_(True), probs.clone().requires_grad_(True)
    result: dict[str, object] = {"rows": rows, "width": args.width}

    outputs = {}
    for name, path in _PATHS.items():
        leaf_x.grad = leaf_probs.grad = None
        out = path(leaf_x, leaf_probs, args.beta, args.linear_beta)
        out.backward(grad)
        outputs[name] = (out.detach(), leaf_x.grad.clone(), leaf_probs.grad.clone())
    (out_u, dx_u, dp_u), (out_f, dx_f, dp_f) = outputs["unfused"], outputs["fused"]
    with torch.no_grad():
        act = fused_situ_glu(x, args.beta, args.linear_beta)
        bound = 2 * args.width * _U * (grad.float() * act.float()).abs().sum(-1, keepdim=True) + 1e-30
    result["forward_bitwise"] = torch.equal(out_u.view(torch.int16), out_f.view(torch.int16))
    result["input_grad_bitwise"] = torch.equal(dx_u.view(torch.int16), dx_f.view(torch.int16))
    result["prob_grad_max_diff_over_bound"] = ((dp_u - dp_f).abs() / bound).max().item()

    def forward_step(path):
        def step():
            with torch.no_grad():
                path(x, probs, args.beta, args.linear_beta)

        return step

    def train_step(path):
        def step():
            leaf_x.grad = leaf_probs.grad = None
            path(leaf_x, leaf_probs, args.beta, args.linear_beta).backward(grad)

        return step

    steps = {}
    for name, path in _PATHS.items():
        for direction, make in (("fwd", forward_step), ("fwd+bwd", train_step)):
            step = steps[name, direction] = make(path)
            for _ in range(args.warmup):
                step()
    torch.cuda.synchronize()
    return result, steps


def _time(result: dict[str, object], steps: dict[tuple[str, str], Callable], iters: int) -> None:
    """Time the prepared steps of one row count into ``result``: only timed calls, each loop in its NVTX range."""
    for (name, direction), step in steps.items():
        label = f"situ_prob/{name}/{direction}/rows={result['rows']}"
        result[f"{name}_{direction}_ms"] = _time_ms(step, iters, label)
    for direction in ("fwd", "fwd+bwd"):
        result[f"speedup_{direction}"] = result[f"unfused_{direction}_ms"] / result[f"fused_{direction}_ms"]


def main() -> None:
    """Run the sweep and log one line per row count; optionally write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[40960, 81920, 122880])
    parser.add_argument("--width", type=int, default=2048, help="moe_intermediate_size (half the fc1 width)")
    parser.add_argument("--beta", type=float, default=4.0, help="activation_func_tanh_clamp_scale")
    parser.add_argument("--linear-beta", type=float, default=25.0, help="activation_func_tanh_clamp_scale_linear")
    parser.add_argument("--no-linear-beta", action="store_true", help="leave the up half linear")
    parser.add_argument("--sigma", type=float, default=4.0, help="standard deviation of the fc1 output")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--profile", action="store_true", help="bracket only the timed loops with cudaProfilerStart/Stop"
    )
    parser.add_argument("--output", default=None, help="JSON results; '.rank<LOCAL_RANK>' is added before .json")
    args = parser.parse_args()
    if args.no_linear_beta:
        args.linear_beta = None
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.get_device_name()
    logger.info("ERNIE 5.5 routed SiTU-GLU x probability on %s (rank %d): %s", device, local_rank, vars(args))

    # Every row count's checks and warm-up first, then the timed loops alone: with --profile, only they are captured.
    prepared = [_prepare(rows, args) for rows in args.rows]
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for result, steps in prepared:
            _time(result, steps, args.iters)
    finally:
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()
    results = [result for result, _ in prepared]
    for result in results:
        logger.info(
            "rows %7d: fwd %.3f -> %.3f ms (x%.2f), fwd+bwd %.3f -> %.3f ms (x%.2f); forward bitwise %s, input "
            "gradient bitwise %s, probability gradient max |diff| / bound %.3g",
            result["rows"],
            result["unfused_fwd_ms"],
            result["fused_fwd_ms"],
            result["speedup_fwd"],
            result["unfused_fwd+bwd_ms"],
            result["fused_fwd+bwd_ms"],
            result["speedup_fwd+bwd"],
            result["forward_bitwise"],
            result["input_grad_bitwise"],
            result["prob_grad_max_diff_over_bound"],
        )
    if args.output:
        root, ext = os.path.splitext(args.output)
        path = f"{root}.rank{local_rank}{ext or '.json'}"
        with open(path, "w") as handle:
            json.dump({"device": device, "args": vars(args), "results": results}, handle, indent=1)
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
