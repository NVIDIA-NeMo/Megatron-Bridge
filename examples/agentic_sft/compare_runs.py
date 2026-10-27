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

"""Compare the steady-state step time of two training logs, for example static versus dynamic context parallelism.

Both runs must have consumed the same data in the same order. The script reads Bridge's per-iteration log lines, drops
the warmup iterations (which include one-time compilation), and reports seconds per step and, when the mock dataset
manifest is given, runtime tokens per second. Rows are consumed in file order, so iteration ``i`` processes rows
``(i - 1) * global_batch_size`` to ``i * global_batch_size - 1``.

Example::

    python examples/agentic_sft/compare_runs.py --baseline static.log --candidate dynamic.log --warmup 10 \\
        --manifest agentic_sft_workspace/data/mock/manifest.json --global-batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from pathlib import Path


logger = logging.getLogger(__name__)

_ITERATION = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+\s*\|\s*consumed samples:\s*(?P<samples>\d+)\s*\|\s*"
    r"elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+)"
)
_LM_LOSS = re.compile(r"lm loss:\s*(?P<loss>[0-9.]+E[+-]\d+)")


def parse_log(text: str) -> dict[int, dict[str, float]]:
    """Return ``{iteration: {"seconds": ..., "loss": ...}}`` from Bridge training log text.

    Raises:
        ValueError: If the log has no iteration lines.
    """
    steps: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        match = _ITERATION.search(line)
        if match is None:
            continue
        step = {"seconds": float(match["ms"]) / 1000.0}
        loss = _LM_LOSS.search(line)
        if loss is not None:
            step["loss"] = float(loss["loss"])
        steps[int(match["iteration"])] = step
    if not steps:
        raise ValueError("No iteration lines found; is this a Bridge training log?")
    return steps


def tokens_per_step(lengths: list[int], global_batch_size: int, iterations: list[int]) -> dict[int, int]:
    """Runtime tokens consumed by each iteration, with rows taken in file order."""
    return {
        iteration: sum(lengths[(iteration - 1) * global_batch_size : iteration * global_batch_size])
        for iteration in iterations
    }


def summarize(
    steps: dict[int, dict[str, float]], warmup: int, tokens: dict[int, int] | None = None
) -> dict[str, float | int]:
    """Steady-state statistics over iterations after ``warmup``.

    Raises:
        ValueError: If no iteration remains after the warmup.
    """
    measured = sorted(iteration for iteration in steps if iteration > warmup)
    if not measured:
        raise ValueError(f"No iterations after warmup {warmup}; the log has {len(steps)}")
    seconds = [steps[iteration]["seconds"] for iteration in measured]
    summary: dict[str, float | int] = {
        "steps": len(measured),
        "first_step": measured[0],
        "last_step": measured[-1],
        "mean_s": statistics.mean(seconds),
        "median_s": statistics.median(seconds),
        "stdev_s": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
        "total_s": sum(seconds),
    }
    if tokens is not None:
        summary["tokens"] = sum(tokens[iteration] for iteration in measured)
        summary["tokens_per_s"] = summary["tokens"] / summary["total_s"]
    return summary


def relative_gain(baseline: dict[str, float | int], candidate: dict[str, float | int]) -> float:
    """Throughput change of ``candidate`` over ``baseline`` on identical work: ``baseline_time / candidate_time - 1``."""
    return float(baseline["total_s"]) / float(candidate["total_s"]) - 1.0


def main(argv: list[str] | None = None) -> None:
    """Print the comparison table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="Log of the reference run, e.g. static CP.")
    parser.add_argument("--candidate", type=Path, required=True, help="Log of the run being evaluated.")
    parser.add_argument("--warmup", type=int, default=10, help="Leading iterations to exclude.")
    parser.add_argument("--manifest", type=Path, help="Mock dataset manifest, for tokens per second.")
    parser.add_argument("--global-batch-size", type=int, help="Required with --manifest.")
    args = parser.parse_args(argv)
    if (args.manifest is None) != (args.global_batch_size is None):
        parser.error("--manifest and --global-batch-size go together")

    runs = {"baseline": parse_log(args.baseline.read_text()), "candidate": parse_log(args.candidate.read_text())}
    common = sorted(set(runs["baseline"]) & set(runs["candidate"]))
    tokens = None
    if args.manifest is not None:
        lengths = json.loads(args.manifest.read_text())["splits"]["training"]["runtime_lengths"]
        tokens = tokens_per_step(lengths, args.global_batch_size, common)
    summaries = {name: summarize({i: steps[i] for i in common}, args.warmup, tokens) for name, steps in runs.items()}
    for name, summary in summaries.items():
        line = (
            f"{name:9s} steps {summary['first_step']}-{summary['last_step']} ({summary['steps']}): "
            f"{summary['mean_s']:.2f} s/step mean, {summary['median_s']:.2f} median, {summary['stdev_s']:.2f} stdev"
        )
        if "tokens_per_s" in summary:
            line += f", {summary['tokens_per_s']:,.0f} runtime tokens/s"
        logger.info(line)
    gain = relative_gain(summaries["baseline"], summaries["candidate"])
    logger.info("candidate vs baseline on identical inputs: %+.2f%% throughput", 100 * gain)


if __name__ == "__main__":
    main()
