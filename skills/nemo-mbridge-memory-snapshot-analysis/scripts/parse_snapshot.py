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

"""Single-file overview of a PyTorch CUDA memory snapshot."""

# Annotations are lazy so the PEP 604 `X | None` syntax below does not need to
# evaluate at import time; these scripts stay runnable on the system python3
# (3.9 on macOS) even though the repo itself targets 3.10+.
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


# Allow running as script from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    TRUST_EPILOG,
    compute_baseline,
    find_active_device,
    format_size,
    get_profiler_steps,
    get_source_key,
    load_snapshot,
)


def print_overview(snapshot: dict, path: str, as_json: bool = False) -> None:
    """Print a one-file overview: settings, segments, baseline, steps, top sources.

    Args:
        snapshot: Deserialized snapshot dictionary.
        path: Path the snapshot was loaded from, used for the file-size line.
        as_json: Emit machine-readable JSON instead of a text report.
    """
    file_size = os.path.getsize(path)

    # --- Segments ---
    # allocated_size counts blocks currently handed out; active_size also counts
    # active_awaiting_free (freed by the user, still held pending stream sync).
    # Reusable memory is total - active, so computing inactive from allocated
    # would report awaiting-free blocks as fragmentation.
    segments = snapshot.get("segments", [])
    total_size = sum(s.get("total_size", 0) for s in segments)
    allocated_size = sum(s.get("allocated_size", 0) for s in segments)
    active_size = sum(s.get("active_size", s.get("allocated_size", 0)) for s in segments)
    awaiting_free_size = active_size - allocated_size
    inactive_size = total_size - active_size

    # --- Profiler steps ---
    annotations = snapshot.get("external_annotations", [])
    steps = get_profiler_steps(annotations)

    # --- Device traces ---
    device_idx = find_active_device(snapshot)
    device_traces = snapshot.get("device_traces", [])
    traces = device_traces[device_idx] if device_idx is not None and device_idx < len(device_traces) else []
    action_counts = defaultdict(int)
    for e in traces:
        action_counts[e["action"]] += 1

    # --- Top allocation sources by throughput ---
    source_throughput = defaultdict(lambda: [0, 0])  # key -> [bytes, count]
    for e in traces:
        if e["action"] == "alloc":
            key = get_source_key(e.get("frames", []))
            source_throughput[key][0] += e["size"]
            source_throughput[key][1] += 1
    top_sources = sorted(source_throughput.items(), key=lambda x: -x[1][0])[:15]

    # --- Baseline memory ---
    baseline = compute_baseline(snapshot, device_idx) if device_idx is not None else None

    if as_json:
        result = {
            "file": path,
            "file_size_bytes": file_size,
            "segments": {
                "count": len(segments),
                "total_size": total_size,
                "allocated_size": allocated_size,
                "active_size": active_size,
                "awaiting_free_size": awaiting_free_size,
                "inactive_size": inactive_size,
            },
            "baseline": {
                "baseline_at_start": baseline.baseline_at_start,
                "segments_active": baseline.segments_active,
                "pre_existing_freed": baseline.pre_existing_freed,
                "profiling_still_live": baseline.profiling_still_live,
            }
            if baseline
            else None,
            "profiler_steps": {
                num: {
                    "start_us": info["start"],
                    "end_us": info["end"],
                    "duration_ms": round((info["end"] - info["start"]) / 1000, 1)
                    if info["start"] and info["end"]
                    else None,
                    "complete": info["end"] is not None,
                }
                for num, info in sorted(steps.items())
            },
            "device_traces": {
                "device": device_idx,
                "event_count": len(traces),
                "actions": dict(action_counts),
            },
            "top_sources": [{"source": k, "throughput_bytes": v[0], "alloc_count": v[1]} for k, v in top_sources],
        }
        print(json.dumps(result, indent=2))
        return

    # --- Print report ---
    print("=" * 70)
    print("  CUDA Memory Snapshot Overview")
    print("=" * 70)

    print(f"\n  File: {path}")
    print(f"  Size: {format_size(file_size)}")

    # Allocator settings
    settings = snapshot.get("allocator_settings", {})
    if settings:
        print("\n--- Allocator Settings ---")
        for k, v in sorted(settings.items()):
            # roundup_power2_divisions is usually an empty dict, but when set it
            # changes allocation rounding and can explain a reserved-memory or
            # fragmentation difference, so report it rather than hiding it.
            if isinstance(v, dict):
                v = dict(sorted(v.items(), key=lambda kv: str(kv[0]))) or "(unset)"
            print(f"  {k}: {v}")

    # Segments
    print("\n--- Segments ---")
    print(f"  Count:               {len(segments)}")
    print(f"  Reserved (total):    {format_size(total_size)}")
    print(f"  Allocated:           {format_size(allocated_size)}")
    print(f"  Active:              {format_size(active_size)}")
    print(f"  Awaiting free:       {format_size(awaiting_free_size)}")
    print(f"  Inactive (reusable): {format_size(inactive_size)}")
    if total_size > 0:
        frag_pct = inactive_size / total_size * 100
        print(f"  Fragmentation:       {frag_pct:.1f}%  (inactive / reserved)")

    # Baseline memory
    if baseline:
        print("\n--- Baseline Memory (at profiling start) ---")
        print(f"  Baseline:           {format_size(baseline.baseline_at_start)}")
        print(f"  Pre-existing freed: {format_size(baseline.pre_existing_freed)}")
        print(f"  Profiling live:     {format_size(baseline.profiling_still_live)}")

    # Profiler steps
    print("\n--- Profiler Steps ---")
    if not steps:
        print("  (none detected)")
    else:
        print(f"  {'Step':>6}  {'Duration':>10}  {'Status'}")
        print(f"  {'─' * 6}  {'─' * 10}  {'─' * 12}")
        for num, info in sorted(steps.items()):
            if info["start"] and info["end"]:
                dur_ms = (info["end"] - info["start"]) / 1000
                status = "complete"
                dur_str = f"{dur_ms:.1f} ms"
            else:
                status = "incomplete"
                dur_str = "—"
            print(f"  {num:>6}  {dur_str:>10}  {status}")

    # Device traces
    print("\n--- Device Traces ---")
    if device_idx is None:
        print("  (no traces found)")
    else:
        print(f"  Device:       {device_idx}")
        print(f"  Total events: {len(traces):,}")
        for action, count in sorted(action_counts.items()):
            print(f"    {action:20s}: {count:>8,}")

    # Top sources
    print("\n--- Top Allocation Sources (by throughput) ---")
    if not top_sources:
        print("  (no allocations)")
    else:
        print(f"  {'#':>3}  {'Throughput':>12}  {'Allocs':>8}  Source")
        print(f"  {'─' * 3}  {'─' * 12}  {'─' * 8}  {'─' * 40}")
        for i, (key, (total, count)) in enumerate(top_sources, 1):
            print(f"  {i:>3}  {format_size(total):>12}  {count:>8,}  {key}")

    print()


def main() -> None:
    """Parse arguments and print the snapshot overview."""
    parser = argparse.ArgumentParser(
        description="Overview of a CUDA memory snapshot pickle file.",
        epilog=TRUST_EPILOG,
    )
    parser.add_argument("pickle_path", help="Path to the snapshot .pickle file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    snapshot = load_snapshot(args.pickle_path)
    print_overview(snapshot, args.pickle_path, as_json=args.json)


if __name__ == "__main__":
    main()
