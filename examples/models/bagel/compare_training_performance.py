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

"""Compare steady-state official BAGEL and Bridge iteration latency."""

import argparse
import json
import logging
import re
import statistics
from pathlib import Path


logger = logging.getLogger(__name__)
BRIDGE_ITERATION_PATTERN = re.compile(r"iteration\s+\d+/\s*\d+.*?elapsed time per iteration \(ms\): ([0-9.]+)")


def parse_args() -> argparse.Namespace:
    """Parse performance-comparison arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True, help="Official BAGEL JSONL trace")
    parser.add_argument("--bridge-log", type=Path, required=True, help="Bridge rank-zero training log")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    """Calculate an interpolated percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize iteration latency in milliseconds."""
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.9),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def load_official_latencies(path: Path) -> list[float]:
    """Derive official iteration latency from its token-throughput trace."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [1000.0 * float(row["tokens_per_step"]) / float(row["tokens_per_sec"]) for row in rows]


def load_bridge_latencies(path: Path) -> list[float]:
    """Extract Bridge iteration latency from the rank-zero training log."""
    return [float(value) for value in BRIDGE_ITERATION_PATTERN.findall(path.read_text(encoding="utf-8"))]


def main() -> None:
    """Write an official-versus-Bridge steady-state latency report."""
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be nonnegative")
    official = load_official_latencies(args.official)
    bridge = load_bridge_latencies(args.bridge_log)
    if len(official) != len(bridge):
        raise ValueError(f"Training trace lengths differ: official={len(official)}, bridge={len(bridge)}")
    if len(official) <= args.warmup_steps:
        raise ValueError("Warmup consumes the complete training trace")
    official = official[args.warmup_steps :]
    bridge = bridge[args.warmup_steps :]
    official_summary = summarize(official)
    bridge_summary = summarize(bridge)
    report = {
        "warmup_steps": args.warmup_steps,
        "measured_steps": len(official),
        "official": official_summary,
        "bridge": bridge_summary,
        "bridge_mean_speedup": official_summary["mean_ms"] / bridge_summary["mean_ms"],
        "bridge_mean_latency_reduction_percent": 100.0
        * (1.0 - bridge_summary["mean_ms"] / official_summary["mean_ms"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d-step BAGEL performance comparison to %s", len(official), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
