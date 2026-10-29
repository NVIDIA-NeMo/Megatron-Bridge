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

"""Drill down into CUDA memory state at a specific point in time."""

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
    find_active_device,
    format_size,
    group_by_source,
    load_snapshot,
    replay_to_time,
)


logger = logging.getLogger(__name__)


def _trace_duration_s(traces):
    """Return elapsed seconds between first and last event."""
    first = last = None
    for e in traces:
        t = e.get("time_us", 0)
        if first is None:
            first = t
        last = t
    if first is None or last is None:
        return 0.0
    return (last - first) / 1e6


def drill_single(path: str, time_s: float, top_n: int, frame_depth: int, as_json: bool) -> None:
    """Report what is live in a single snapshot at a given elapsed time.

    Args:
        path: Path to the snapshot pickle file.
        time_s: Elapsed seconds from the first trace event to replay up to.
        top_n: Number of top allocation sources to show.
        frame_depth: Stack frames to include in each source key.
        as_json: Emit machine-readable JSON instead of a text report.
    """
    snap = load_snapshot(path)
    dev = find_active_device(snap)
    if dev is None:
        logger.error("no device traces found.")
        sys.exit(1)

    traces = snap["device_traces"][dev]
    duration = _trace_duration_s(traces)
    baseline = compute_baseline(snap, dev)
    base = baseline.baseline_at_start

    target_us = int(time_s * 1e6)
    result = replay_to_time(traces, target_us)
    groups = group_by_source(result.live_set, depth=frame_depth)

    abs_mem = base + result.current_delta

    if as_json:
        out = {
            "file": path,
            "time_s": time_s,
            "trace_duration_s": round(duration, 2),
            "baseline": base,
            "delta": result.current_delta,
            "absolute": abs_mem,
            "alloc_count": result.alloc_count,
            "free_count": result.free_count,
            "live_count": len(result.live_set),
            "sources": [{"source": src, "bytes": total, "count": cnt} for src, total, cnt in groups[:top_n]],
        }
        print(json.dumps(out, indent=2, default=str))
        return

    name = os.path.basename(path)
    print("=" * 80)
    print(f"  Memory State at t = {time_s:.2f}s")
    print("=" * 80)
    print(f"\n  File:     {name}")
    print(f"  Duration: {duration:.2f}s total trace")
    print(f"\n  Baseline:          {format_size(base):>12}")
    print(f"  Delta at t={time_s:.2f}s:  {format_size(result.current_delta):>12}")
    print(f"  Absolute:          {format_size(abs_mem):>12}")
    print(f"  Live allocations:  {len(result.live_set):>12,}")
    print(f"  Allocs so far:     {result.alloc_count:>12,}")
    print(f"  Frees so far:      {result.free_count:>12,}")

    print(f"\n--- Live Allocations by Source (top {top_n}) ---\n")
    if groups:
        print(f"  {'Source':<60s}  {'Size':>12s}  {'Count':>6s}")
        print(f"  {'─' * 60}  {'─' * 12}  {'─' * 6}")
        for src, total, cnt in groups[:top_n]:
            display = src if len(src) <= 60 else src[:57] + "..."
            print(f"  {display:<60s}  {format_size(total):>12s}  {cnt:>6,}")
        print(f"\n  ({len(groups)} total sources, showing top {top_n})")
    else:
        print("  (no live allocations)")
    print()


def drill_compare(path_a: str, path_b: str, time_s: float, top_n: int, frame_depth: int, as_json: bool) -> None:
    """Compare what is live in two snapshots at the same elapsed time.

    Sources are sorted by absolute delta so the allocations responsible for a
    divergence surface first, with only-in-A and only-in-B totals reported
    separately.

    Args:
        path_a: Path to the first snapshot pickle file.
        path_b: Path to the second snapshot pickle file.
        time_s: Elapsed seconds from each trace's first event to replay up to.
        top_n: Number of top allocation sources to show.
        frame_depth: Stack frames to include in each source key.
        as_json: Emit machine-readable JSON instead of a text report.
    """
    snap_a = load_snapshot(path_a)
    snap_b = load_snapshot(path_b)

    dev_a = find_active_device(snap_a)
    dev_b = find_active_device(snap_b)
    if dev_a is None or dev_b is None:
        logger.error("one or both snapshots have no device traces.")
        sys.exit(1)

    traces_a = snap_a["device_traces"][dev_a]
    traces_b = snap_b["device_traces"][dev_b]
    baseline_a = compute_baseline(snap_a, dev_a)
    baseline_b = compute_baseline(snap_b, dev_b)
    base_a = baseline_a.baseline_at_start
    base_b = baseline_b.baseline_at_start

    target_us = int(time_s * 1e6)
    res_a = replay_to_time(traces_a, target_us)
    res_b = replay_to_time(traces_b, target_us)

    groups_a = group_by_source(res_a.live_set, depth=frame_depth)
    groups_b = group_by_source(res_b.live_set, depth=frame_depth)

    abs_a = base_a + res_a.current_delta
    abs_b = base_b + res_b.current_delta

    # Merge sources
    map_a = {src: (total, cnt) for src, total, cnt in groups_a}
    map_b = {src: (total, cnt) for src, total, cnt in groups_b}
    all_sources = set(map_a.keys()) | set(map_b.keys())

    rows = []
    for src in all_sources:
        ta, ca = map_a.get(src, (0, 0))
        tb, cb = map_b.get(src, (0, 0))
        rows.append((abs(ta - tb), ta - tb, ta, ca, tb, cb, src))
    rows.sort(reverse=True)

    if as_json:
        out = {
            "file_a": path_a,
            "file_b": path_b,
            "time_s": time_s,
            "baseline_a": base_a,
            "baseline_b": base_b,
            "delta_a": res_a.current_delta,
            "delta_b": res_b.current_delta,
            "absolute_a": abs_a,
            "absolute_b": abs_b,
            "live_count_a": len(res_a.live_set),
            "live_count_b": len(res_b.live_set),
            "sources": [
                {
                    "source": src,
                    "bytes_a": ta,
                    "count_a": ca,
                    "bytes_b": tb,
                    "count_b": cb,
                    "delta": ta - tb,
                }
                for _, _, ta, ca, tb, cb, src in rows[:top_n]
            ],
        }
        print(json.dumps(out, indent=2, default=str))
        return

    name_a = os.path.basename(path_a)
    name_b = os.path.basename(path_b)

    print("=" * 100)
    print(f"  Memory Comparison at t = {time_s:.2f}s")
    print("=" * 100)
    print(f"\n  A: {name_a}")
    print(f"  B: {name_b}")

    print(f"\n  {'':45s}  {'A':>14s}  {'B':>14s}  {'Delta(A-B)':>14s}")
    print(f"  {'─' * 45}  {'─' * 14}  {'─' * 14}  {'─' * 14}")
    print(
        f"  {'Baseline':45s}  {format_size(base_a):>14s}  {format_size(base_b):>14s}  {format_size(base_a - base_b):>14s}"
    )
    print(
        f"  {'Delta at t=' + f'{time_s:.2f}s':45s}  {format_size(res_a.current_delta):>14s}  {format_size(res_b.current_delta):>14s}  {format_size(res_a.current_delta - res_b.current_delta):>14s}"
    )
    print(
        f"  {'Absolute':45s}  {format_size(abs_a):>14s}  {format_size(abs_b):>14s}  {format_size(abs_a - abs_b):>14s}"
    )
    print(
        f"  {'Live allocations':45s}  {len(res_a.live_set):>14,}  {len(res_b.live_set):>14,}  {len(res_a.live_set) - len(res_b.live_set):>+14,}"
    )

    print(f"\n--- Live Allocations at t={time_s:.2f}s (top {top_n} by |delta|) ---\n")
    print(f"  {'Source':<60s}  {'A':>12s}  {'#A':>5s}  {'B':>12s}  {'#B':>5s}  {'Delta(A-B)':>12s}")
    print(f"  {'─' * 60}  {'─' * 12}  {'─' * 5}  {'─' * 12}  {'─' * 5}  {'─' * 12}")

    for _, diff, ta, ca, tb, cb, src in rows[:top_n]:
        sign = "+" if diff > 0 else ""
        display = src if len(src) <= 60 else src[:57] + "..."
        print(
            f"  {display:<60s}  "
            f"{format_size(ta):>12s}  {ca:>5,}  "
            f"{format_size(tb):>12s}  {cb:>5,}  "
            f"{sign}{format_size(diff):>11s}"
        )

    print(f"\n  ({len(rows)} total sources, showing top {top_n})")

    # Only-in-A / only-in-B summaries
    only_a = [(src, ta, ca) for _, _, ta, ca, tb, cb, src in rows if tb == 0 and ta > 1024 * 1024]
    only_b = [(src, tb, cb) for _, _, ta, ca, tb, cb, src in rows if ta == 0 and tb > 1024 * 1024]
    if only_a:
        total = sum(t for _, t, _ in only_a)
        print(f"\n  Only in A (> 1 MiB): {len(only_a)} sources, {format_size(total)} total")
    if only_b:
        total = sum(t for _, t, _ in only_b)
        print(f"  Only in B (> 1 MiB): {len(only_b)} sources, {format_size(total)} total")
    print()


def main() -> None:
    """Parse arguments and run the point-in-time drill-down."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Drill down into CUDA memory state at a specific elapsed time.",
        epilog=TRUST_EPILOG,
    )
    parser.add_argument("pickles", nargs="+", help="One or two snapshot pickle files")
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        required=True,
        help="Elapsed seconds from profiling start to inspect",
    )
    parser.add_argument("--top", type=int, default=25, help="Number of top sources (default: 25)")
    parser.add_argument("--frame-depth", type=int, default=1, help="Stack frame depth (default: 1)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if len(args.pickles) > 2:
        parser.error("At most two pickle files supported")

    if len(args.pickles) == 1:
        drill_single(args.pickles[0], args.time, args.top, args.frame_depth, args.json)
    else:
        drill_compare(args.pickles[0], args.pickles[1], args.time, args.top, args.frame_depth, args.json)


if __name__ == "__main__":
    main()
