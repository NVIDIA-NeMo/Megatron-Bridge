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

"""Replay device_traces for a specific training step and show source-grouped allocations."""

# Annotations are lazy so the PEP 604 `X | None` syntax below does not need to
# evaluate at import time; these scripts stay runnable on the system python3
# (3.9 on macOS) even though the repo itself targets 3.10+.
from __future__ import annotations

import argparse
import json
import logging
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    TRUST_EPILOG,
    compute_baseline,
    compute_step_start_deltas,
    find_active_device,
    format_size,
    get_profiler_steps,
    get_step_annotations,
    get_step_events,
    group_by_source,
    live_set_before,
    load_snapshot,
    replay_events,
)


logger = logging.getLogger(__name__)


def replay_one_step(
    traces: list,
    annotations: list,
    step_num: int,
    step_info: dict,
    top_n: int = 15,
    frame_depth: int = 1,
    as_json: bool = False,
    baseline_at_start: int = 0,
    step_start_delta: int = 0,
) -> dict:
    """Replay a single step and return results."""
    start = step_info["start"]
    end = step_info["end"]
    events = get_step_events(traces, start, end)
    # Seed with what was already live when the step opened, so the source table
    # accounts for memory carried into the step, not only what it allocated.
    result = replay_events(events, initial_live=live_set_before(traces, start))
    sources = group_by_source(result.peak_live_set, depth=frame_depth)
    ann_counts = get_step_annotations(annotations, start, end)

    complete = end is not None
    duration_ms = (end - start) / 1000 if complete else None

    absolute_peak = baseline_at_start + step_start_delta + result.peak_delta

    step_result = {
        "step": step_num,
        "complete": complete,
        "duration_ms": duration_ms,
        "alloc_count": result.alloc_count,
        "free_count": result.free_count,
        "total_throughput": result.total_alloc_bytes,
        "peak_delta": result.peak_delta,
        "absolute_peak": absolute_peak,
        "end_delta": result.end_delta,
        "unmatched_frees": result.unmatched_free_count,
        "unmatched_free_bytes": result.unmatched_free_bytes,
        "top_sources_at_peak": sources[:top_n],
        "annotations": ann_counts,
    }

    if not as_json:
        print(f"\n{'=' * 70}")
        print(f"  Step {step_num}" + (" (incomplete)" if not complete else ""))
        print(f"{'=' * 70}")
        if duration_ms:
            print(f"  Duration:       {duration_ms:.1f} ms")
        print(f"  Allocs:         {result.alloc_count:,}")
        print(f"  Frees:          {result.free_count:,}")
        print(f"  Throughput:     {format_size(result.total_alloc_bytes)}")
        print(f"  Peak delta:     {format_size(result.peak_delta)}")
        if baseline_at_start > 0:
            print(f"  Absolute peak:  {format_size(absolute_peak)}")
        print(f"  End delta:      {format_size(result.end_delta)}")
        if result.unmatched_free_count > 0:
            print(f"  Pre-existing frees: {result.unmatched_free_count} ({format_size(result.unmatched_free_bytes)})")

        if ann_counts:
            print("\n  --- Active Annotations ---")
            for name, count in sorted(ann_counts.items(), key=lambda x: -x[1]):
                print(f"    {count:>5}x  {name}")

        print(f"\n  --- Top {min(top_n, len(sources))} Sources at Peak (by size) ---")
        if sources:
            print(f"  {'#':>3}  {'Size':>12}  {'Count':>6}  Source")
            print(f"  {'─' * 3}  {'─' * 12}  {'─' * 6}  {'─' * 40}")
            for i, (key, total, count) in enumerate(sources[:top_n], 1):
                print(f"  {i:>3}  {format_size(total):>12}  {count:>6}  {key}")
        else:
            print("  (no live allocations at peak)")

    return step_result


def main() -> None:
    """Parse arguments and replay the requested training step(s)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Replay device_traces for a training step.",
        epilog=TRUST_EPILOG,
    )
    parser.add_argument("pickle_path", help="Path to the snapshot .pickle file")
    parser.add_argument("--step", type=int, help="Step number to replay")
    parser.add_argument("--all-steps", action="store_true", help="Replay all complete steps")
    parser.add_argument("--top", type=int, default=15, help="Number of top sources to show (default: 15)")
    parser.add_argument(
        "--frame-depth", type=int, default=1, help="Stack frame depth for source grouping (default: 1)"
    )
    parser.add_argument("--device", type=int, help="Device index (default: auto-detect)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.step is None and not args.all_steps:
        parser.error("Specify --step N or --all-steps")

    snapshot = load_snapshot(args.pickle_path)
    annotations = snapshot.get("external_annotations", [])
    steps = get_profiler_steps(annotations)

    if not steps:
        logger.error("no ProfilerStep annotations found.")
        sys.exit(1)

    device_idx = args.device if args.device is not None else find_active_device(snapshot)
    if device_idx is None:
        logger.error("no device traces found.")
        sys.exit(1)
    traces = snapshot["device_traces"][device_idx]

    # Compute baseline and per-step starting deltas
    baseline = compute_baseline(snapshot, device_idx)
    step_deltas = compute_step_start_deltas(traces, steps)

    if not args.json:
        print(f"\n  Baseline at profiling start: {format_size(baseline.baseline_at_start)}")

    if args.all_steps:
        step_nums = sorted(steps.keys())
    else:
        if args.step not in steps:
            available = sorted(steps.keys())
            logger.error(f"step {args.step} not found. Available: {available}")
            sys.exit(1)
        step_nums = [args.step]

    all_results = []
    for num in step_nums:
        info = steps[num]
        if not info["start"]:
            continue
        r = replay_one_step(
            traces,
            annotations,
            num,
            info,
            top_n=args.top,
            frame_depth=args.frame_depth,
            as_json=args.json,
            baseline_at_start=baseline.baseline_at_start,
            step_start_delta=step_deltas.get(num, 0),
        )
        all_results.append(r)

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))
    else:
        print()


if __name__ == "__main__":
    main()
