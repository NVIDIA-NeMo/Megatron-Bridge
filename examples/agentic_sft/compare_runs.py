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

"""Compare the steady-state throughput of two training logs, for example static versus dynamic context parallelism.

The script reads Bridge's per-iteration log lines, drops the warmup iterations (which include one-time compilation),
and reports seconds per step. For runs of the ``qwen35_text_35b_a3b_4gpu_gb200_fp8mx_*dynamic_cp_config`` recipes it
also reports runtime tokens per second: the ``MockGPTSFTDatasetConfig`` lengths are deterministic, and samples (or,
with offline packing, packed rows) are consumed in order, so iteration ``i`` processes items
``(i - 1) * global_batch_size`` to ``i * global_batch_size - 1``. The gain is then the ratio of tokens per second, so
runs that process different work per step, such as online and offline packing, can be compared. Without token counts,
both runs must have consumed the same data in the same order and the gain is the ratio of step times.

Example::

    python examples/agentic_sft/compare_runs.py --baseline online.log --candidate offline.log --warmup 10 \\
        --mock-seq-length 65536 --train-iters 200 --baseline-global-batch-size 32 \\
        --candidate-global-batch-size 8 --candidate-packed-sequence-size 65536
"""

from __future__ import annotations

import argparse
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
    """Throughput change of ``candidate`` over ``baseline``.

    The ratio of runtime tokens per second when both summaries have token counts; otherwise, for identical work,
    ``baseline_time / candidate_time - 1``.
    """
    if "tokens_per_s" in baseline and "tokens_per_s" in candidate:
        return float(candidate["tokens_per_s"]) / float(baseline["tokens_per_s"]) - 1.0
    return float(baseline["total_s"]) / float(candidate["total_s"]) - 1.0


def mock_item_lengths(
    seq_length: int, seed: int, num_items: int, packed_sequence_size: int | None, pad_seq_to_mult: int
) -> list[int]:
    """Runtime tokens of each item a mock run consumes: samples, or packed rows with offline packing."""
    from megatron.bridge.data.builders import MockGPTSFTDatasetConfig
    from megatron.bridge.data.builders.mock_gpt_sft import mock_sft_packed_rows, mock_sft_sequence_lengths
    from megatron.bridge.data.packing import PackedSequenceSpecs

    if packed_sequence_size is None:
        mock = MockGPTSFTDatasetConfig(seq_length=seq_length, vocab_size=2, random_seed=seed)
        return mock_sft_sequence_lengths(mock, num_items).tolist()
    mock = MockGPTSFTDatasetConfig(
        seq_length=seq_length,
        vocab_size=2,
        random_seed=seed,
        enable_offline_packing=True,
        offline_packing_specs=PackedSequenceSpecs(
            packed_sequence_size=packed_sequence_size, pad_seq_to_mult=pad_seq_to_mult
        ),
    )
    lengths, rows = mock_sft_packed_rows(mock, num_items)
    return [int(sum(lengths[sample] for sample in row)) for row in rows]


def main(argv: list[str] | None = None) -> None:
    """Print the comparison table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="Log of the reference run, e.g. static CP.")
    parser.add_argument("--candidate", type=Path, required=True, help="Log of the run being evaluated.")
    parser.add_argument("--warmup", type=int, default=10, help="Leading iterations to exclude.")
    parser.add_argument(
        "--mock-seq-length", type=int, help="seq_length of the runs' MockGPTSFTDatasetConfig, for tokens per second."
    )
    parser.add_argument(
        "--mock-seed", type=int, default=1234, help="random_seed of the runs' MockGPTSFTDatasetConfig."
    )
    parser.add_argument(
        "--train-iters", type=int, help="train.train_iters of the runs; required with --mock-seq-length."
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        help="Global batch size of both runs; with --mock-seq-length, unless set per run.",
    )
    parser.add_argument("--packed-pad-to-multiple-of", type=int, default=8, help="Offline packing alignment.")
    for run in ("baseline", "candidate"):
        parser.add_argument(f"--{run}-global-batch-size", type=int, help=f"Global batch size of the {run} run.")
        parser.add_argument(
            f"--{run}-packed-sequence-size", type=int, help=f"Offline packed row size of the {run} run, if packed."
        )
    args = parser.parse_args(argv)
    batch_sizes = {
        run: getattr(args, f"{run}_global_batch_size") or args.global_batch_size for run in ("baseline", "candidate")
    }
    if args.mock_seq_length is not None and (args.train_iters is None or None in batch_sizes.values()):
        parser.error("--mock-seq-length needs --train-iters and a global batch size for each run")

    runs = {"baseline": parse_log(args.baseline.read_text()), "candidate": parse_log(args.candidate.read_text())}
    common = sorted(set(runs["baseline"]) & set(runs["candidate"]))
    tokens: dict[str, dict[int, int] | None] = {"baseline": None, "candidate": None}
    if args.mock_seq_length is not None:
        for run, batch_size in batch_sizes.items():
            lengths = mock_item_lengths(
                args.mock_seq_length,
                args.mock_seed,
                args.train_iters * batch_size,
                getattr(args, f"{run}_packed_sequence_size"),
                args.packed_pad_to_multiple_of,
            )
            tokens[run] = tokens_per_step(lengths, batch_size, common)
    summaries = {
        name: summarize({i: steps[i] for i in common}, args.warmup, tokens[name]) for name, steps in runs.items()
    }
    for name, summary in summaries.items():
        line = (
            f"{name:9s} steps {summary['first_step']}-{summary['last_step']} ({summary['steps']}): "
            f"{summary['mean_s']:.2f} s/step mean, {summary['median_s']:.2f} median, {summary['stdev_s']:.2f} stdev"
        )
        if "tokens_per_s" in summary:
            line += f", {summary['tokens_per_s']:,.0f} runtime tokens/s"
        logger.info(line)
    gain = relative_gain(summaries["baseline"], summaries["candidate"])
    basis = "runtime tokens/s" if "tokens_per_s" in summaries["candidate"] else "step time on identical inputs"
    logger.info("candidate vs baseline: %+.2f%% throughput (%s)", 100 * gain, basis)


if __name__ == "__main__":
    main()
