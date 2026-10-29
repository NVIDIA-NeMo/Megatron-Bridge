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

"""Compare two PyTorch CUDA memory snapshots side-by-side."""

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
    get_step_events,
    group_by_source,
    live_set_before,
    load_snapshot,
    replay_events,
)


logger = logging.getLogger(__name__)


def _normalize_setting(value: object) -> object:
    """Render a setting in a stable, comparable form.

    ``roundup_power2_divisions`` arrives as a dict whose key order is not
    meaningful, which makes a naive ``!=`` report spurious differences and reads
    badly in the report. Sorting it gives a stable representation so the setting
    can be compared and shown rather than suppressed — it changes allocator
    rounding and is a legitimate explanation for two runs differing in reserved
    memory or fragmentation.
    """
    if isinstance(value, dict):
        return dict(sorted(value.items(), key=lambda kv: str(kv[0])))
    return value


def diff_settings(settings_a: dict, settings_b: dict) -> list:
    """Return list of (key, val_a, val_b) for differing settings."""
    all_keys = sorted(set(list(settings_a.keys()) + list(settings_b.keys())))
    diffs = []
    for k in all_keys:
        va = _normalize_setting(settings_a.get(k))
        vb = _normalize_setting(settings_b.get(k))
        if va != vb:
            diffs.append((k, va, vb))
    return diffs


def compare(
    path_a: str,
    path_b: str,
    step: int | None = None,
    top_n: int = 15,
    frame_depth: int = 1,
    as_json: bool = False,
) -> None:
    """Compare two snapshots and drill into the step with the largest divergence.

    Falls back to a full-trace replay when neither snapshot carries matching
    ``ProfilerStep`` annotations, which is the common case for Megatron Bridge
    runs that enable ``record_memory_history`` without the PyTorch profiler.

    Args:
        path_a: Path to the first snapshot pickle file.
        path_b: Path to the second snapshot pickle file.
        step: Step to drill into. Defaults to the largest peak difference.
        top_n: Number of top allocation sources to show.
        frame_depth: Stack frames to include in each source key.
        as_json: Emit machine-readable JSON instead of a text report.
    """
    snap_a = load_snapshot(path_a)
    snap_b = load_snapshot(path_b)

    # --- Config diff ---
    settings_a = snap_a.get("allocator_settings", {})
    settings_b = snap_b.get("allocator_settings", {})
    config_diffs = diff_settings(settings_a, settings_b)

    # --- Segments ---
    def seg_summary(snap):
        # allocated_size counts blocks currently handed out; active_size also
        # counts active_awaiting_free (freed by the user, still held pending
        # stream sync). Reusable memory is total - active, so computing inactive
        # from allocated would report awaiting-free blocks as fragmentation.
        segs = snap.get("segments", [])
        total = sum(s.get("total_size", 0) for s in segs)
        allocated = sum(s.get("allocated_size", 0) for s in segs)
        active = sum(s.get("active_size", s.get("allocated_size", 0)) for s in segs)
        return {
            "count": len(segs),
            "total": total,
            "allocated": allocated,
            "active": active,
            "awaiting_free": active - allocated,
            "inactive": total - active,
        }

    seg_a = seg_summary(snap_a)
    seg_b = seg_summary(snap_b)

    # --- Steps ---
    ann_a = snap_a.get("external_annotations", [])
    ann_b = snap_b.get("external_annotations", [])
    steps_a = get_profiler_steps(ann_a)
    steps_b = get_profiler_steps(ann_b)

    # Find matching complete steps
    complete_a = {n for n, s in steps_a.items() if s["start"] and s["end"]}
    complete_b = {n for n, s in steps_b.items() if s["start"] and s["end"]}
    matching = sorted(complete_a & complete_b)

    # --- Device traces ---
    dev_a = find_active_device(snap_a)
    dev_b = find_active_device(snap_b)
    traces_a = snap_a["device_traces"][dev_a] if dev_a is not None else []
    traces_b = snap_b["device_traces"][dev_b] if dev_b is not None else []

    # --- Baseline memory ---
    baseline_a = compute_baseline(snap_a, dev_a) if dev_a is not None else None
    baseline_b = compute_baseline(snap_b, dev_b) if dev_b is not None else None
    base_a = baseline_a.baseline_at_start if baseline_a else 0
    base_b = baseline_b.baseline_at_start if baseline_b else 0

    # --- Per-step starting deltas ---
    deltas_a = compute_step_start_deltas(traces_a, steps_a)
    deltas_b = compute_step_start_deltas(traces_b, steps_b)

    # --- Replay each matching step ---
    step_results = []
    for num in matching:
        events_a = get_step_events(traces_a, steps_a[num]["start"], steps_a[num]["end"])
        events_b = get_step_events(traces_b, steps_b[num]["start"], steps_b[num]["end"])
        # Seed each replay with what was already live when the step opened, so
        # the source table accounts for memory carried into the step rather than
        # only what the step itself allocated.
        replay_a = replay_events(events_a, initial_live=live_set_before(traces_a, steps_a[num]["start"]))
        replay_b = replay_events(events_b, initial_live=live_set_before(traces_b, steps_b[num]["start"]))
        abs_peak_a = base_a + deltas_a.get(num, 0) + replay_a.peak_delta
        abs_peak_b = base_b + deltas_b.get(num, 0) + replay_b.peak_delta
        step_results.append(
            {
                "step": num,
                "peak_a": replay_a.peak_delta,
                "peak_b": replay_b.peak_delta,
                "peak_diff": replay_b.peak_delta - replay_a.peak_delta,
                "abs_peak_a": abs_peak_a,
                "abs_peak_b": abs_peak_b,
                "abs_peak_diff": abs_peak_b - abs_peak_a,
                "allocs_a": replay_a.alloc_count,
                "allocs_b": replay_b.alloc_count,
                "throughput_a": replay_a.total_alloc_bytes,
                "throughput_b": replay_b.total_alloc_bytes,
                "replay_a": replay_a,
                "replay_b": replay_b,
            }
        )

    # --- Full-trace fallback when no step annotations ---
    full_trace_mode = not matching and traces_a and traces_b
    full_trace_result = None
    if full_trace_mode:
        logger.info("  No matching ProfilerStep annotations; falling back to full-trace replay.")
        replay_a_full = replay_events(traces_a)
        replay_b_full = replay_events(traces_b)
        abs_peak_a_full = base_a + replay_a_full.peak_delta
        abs_peak_b_full = base_b + replay_b_full.peak_delta
        full_trace_result = {
            "peak_a": replay_a_full.peak_delta,
            "peak_b": replay_b_full.peak_delta,
            "peak_diff": replay_b_full.peak_delta - replay_a_full.peak_delta,
            "abs_peak_a": abs_peak_a_full,
            "abs_peak_b": abs_peak_b_full,
            "allocs_a": replay_a_full.alloc_count,
            "allocs_b": replay_b_full.alloc_count,
            "throughput_a": replay_a_full.total_alloc_bytes,
            "throughput_b": replay_b_full.total_alloc_bytes,
            "unmatched_a": replay_a_full.unmatched_free_count,
            "unmatched_b": replay_b_full.unmatched_free_count,
            "replay_a": replay_a_full,
            "replay_b": replay_b_full,
        }

    # --- Pick drill-down step ---
    if step is not None:
        drill_step = step
        if step not in [r["step"] for r in step_results]:
            logger.warning(f"step {step} not in matching complete steps {matching}")
            drill_step = None
    elif step_results:
        # Rank by absolute peak, not the within-step delta: a step can allocate
        # modestly yet still sit at the run's memory high-water mark because of
        # what it inherited, and that is the step worth drilling into.
        drill_step = max(step_results, key=lambda r: abs(r["abs_peak_diff"]))["step"]
    else:
        drill_step = None

    # --- Drill-down source comparison ---
    drill_data = None

    # Helper to build drill_data from two replay results
    def _build_drill_data(label, replay_a_res, replay_b_res, peak_diff):
        sources_a = group_by_source(replay_a_res.peak_live_set, depth=frame_depth)
        sources_b = group_by_source(replay_b_res.peak_live_set, depth=frame_depth)
        src_map_a = {k: (total, count) for k, total, count in sources_a}
        src_map_b = {k: (total, count) for k, total, count in sources_b}
        all_keys = sorted(
            set(list(src_map_a.keys()) + list(src_map_b.keys())),
            key=lambda k: -abs(src_map_b.get(k, (0, 0))[0] - src_map_a.get(k, (0, 0))[0]),
        )
        return {
            "step": label,
            "peak_diff": peak_diff,
            "sources": [
                {
                    "source": k,
                    "peak_a": src_map_a.get(k, (0, 0))[0],
                    "peak_b": src_map_b.get(k, (0, 0))[0],
                    "delta": src_map_b.get(k, (0, 0))[0] - src_map_a.get(k, (0, 0))[0],
                }
                for k in all_keys
            ],
        }

    if drill_step is not None:
        dr = next((r for r in step_results if r["step"] == drill_step), None)
        if dr:
            drill_data = _build_drill_data(
                drill_step,
                dr["replay_a"],
                dr["replay_b"],
                dr["peak_diff"],
            )
    elif full_trace_result:
        drill_data = _build_drill_data(
            "full-trace",
            full_trace_result["replay_a"],
            full_trace_result["replay_b"],
            full_trace_result["peak_diff"],
        )

    # --- Observations ---
    observations = []

    # Alloc ratio observations (per-step or full-trace)
    obs_items = step_results or ([full_trace_result] if full_trace_result else [])
    for r in obs_items:
        if r["allocs_a"] > 0 and r["allocs_b"] > 0:
            ratio = max(r["allocs_a"], r["allocs_b"]) / min(r["allocs_a"], r["allocs_b"])
            if ratio > 10:
                fewer = "A" if r["allocs_a"] < r["allocs_b"] else "B"
                scope = f"step {r['step']}" if "step" in r and isinstance(r.get("step"), int) else "the full trace"
                observations.append(
                    f"Snapshot {fewer} has ~{ratio:.0f}x fewer allocs in {scope} "
                    f"({min(r['allocs_a'], r['allocs_b']):,} vs {max(r['allocs_a'], r['allocs_b']):,}), "
                    f"suggesting CUDA graph replay is active."
                )
    if drill_data and drill_data["sources"]:
        top5_delta = sum(abs(s["delta"]) for s in drill_data["sources"][:5])
        scope = f"step {drill_data['step']}" if isinstance(drill_data["step"], int) else "the full trace"
        observations.append(f"Top 5 sources account for {format_size(top5_delta)} of the peak difference in {scope}.")

    # --- Output ---
    if as_json:
        result = {
            "file_a": path_a,
            "file_b": path_b,
            "config_diffs": config_diffs,
            "segments_a": seg_a,
            "segments_b": seg_b,
            "baseline_a": baseline_a.baseline_at_start if baseline_a else None,
            "baseline_b": baseline_b.baseline_at_start if baseline_b else None,
            "matching_steps": matching,
            "step_results": [{k: v for k, v in r.items() if k not in ("replay_a", "replay_b")} for r in step_results],
            "full_trace": (
                {k: v for k, v in full_trace_result.items() if k not in ("replay_a", "replay_b")}
                if full_trace_result
                else None
            ),
            "drill_down": drill_data,
            "observations": observations,
        }
        print(json.dumps(result, indent=2, default=str))
        return

    # --- Text output ---
    name_a = os.path.basename(path_a)
    name_b = os.path.basename(path_b)

    print("=" * 70)
    print("  CUDA Memory Snapshot Comparison")
    print("=" * 70)
    print(f"\n  A: {name_a}")
    print(f"  B: {name_b}")

    # Config diff
    print("\n--- Allocator Settings Diff ---")
    if not config_diffs:
        print("  (identical)")
    else:
        for k, va, vb in config_diffs:
            print(f"  {k}:")
            print(f"    A: {va}")
            print(f"    B: {vb}")

    # Segments
    print("\n--- Segments Baseline ---")
    print(f"  {'':20s}  {'Snapshot A':>14}  {'Snapshot B':>14}  {'Delta':>14}")
    print(f"  {'─' * 20}  {'─' * 14}  {'─' * 14}  {'─' * 14}")
    for label, ka, kb in [
        ("Segments", seg_a["count"], seg_b["count"]),
    ]:
        delta = kb - ka
        sign = "+" if delta > 0 else ""
        print(f"  {label:20s}  {ka:>14}  {kb:>14}  {sign}{delta:>13}")
    for label, ka, kb in [
        ("Reserved (total)", seg_a["total"], seg_b["total"]),
        ("Allocated", seg_a["allocated"], seg_b["allocated"]),
        ("Active", seg_a["active"], seg_b["active"]),
        ("Awaiting free", seg_a["awaiting_free"], seg_b["awaiting_free"]),
        ("Inactive (reusable)", seg_a["inactive"], seg_b["inactive"]),
        ("Baseline at start", base_a, base_b),
    ]:
        delta = kb - ka
        sign = "+" if delta > 0 else ""
        print(f"  {label:20s}  {format_size(ka):>14}  {format_size(kb):>14}  {sign}{format_size(delta):>13}")

    # Full-trace summary (when no step annotations)
    if full_trace_mode and full_trace_result:
        ftr = full_trace_result
        print("\n--- Full-Trace Summary (no ProfilerStep annotations) ---")
        print(f"  {'':35s}  {'Snapshot A':>14}  {'Snapshot B':>14}  {'Delta(B-A)':>14}")
        print(f"  {'─' * 35}  {'─' * 14}  {'─' * 14}  {'─' * 14}")
        for label, va, vb, is_size in [
            ("Peak delta (from trace start)", ftr["peak_a"], ftr["peak_b"], True),
            ("Absolute peak (baseline+delta)", ftr["abs_peak_a"], ftr["abs_peak_b"], True),
            ("Total alloc throughput", ftr["throughput_a"], ftr["throughput_b"], True),
        ]:
            diff = vb - va
            sign = "+" if diff > 0 else ""
            if is_size:
                print(f"  {label:35s}  {format_size(va):>14}  {format_size(vb):>14}  {sign}{format_size(diff):>13}")
            else:
                print(f"  {label:35s}  {va:>14,}  {vb:>14,}  {sign}{diff:>13,}")
        for label, va, vb in [
            ("Alloc count", ftr["allocs_a"], ftr["allocs_b"]),
            ("Unmatched frees (pre-existing)", ftr["unmatched_a"], ftr["unmatched_b"]),
        ]:
            diff = vb - va
            sign = "+" if diff > 0 else ""
            print(f"  {label:35s}  {va:>14,}  {vb:>14,}  {sign}{diff:>13,}")

    # Per-step summary
    print("\n--- Per-Step Summary ---")
    if not step_results and not full_trace_mode:
        print("  (no matching complete steps)")
    elif not step_results and full_trace_mode:
        print("  (skipped — see full-trace summary above)")
    else:
        has_baseline = base_a > 0 or base_b > 0
        if has_baseline:
            print(
                f"  {'Step':>5}  {'Peak Δ A':>12}  {'Peak Δ B':>12}  "
                f"{'Diff(B-A)':>12}  {'Abs Peak A':>12}  {'Abs Peak B':>12}  "
                f"{'Allocs A':>10}  {'Allocs B':>10}"
            )
            print(f"  {'─' * 5}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
        else:
            print(
                f"  {'Step':>5}  {'Peak A':>12}  {'Peak B':>12}  "
                f"{'Diff(B-A)':>12}  {'Allocs A':>10}  {'Allocs B':>10}  "
                f"{'Thruput A':>12}  {'Thruput B':>12}"
            )
            print(f"  {'─' * 5}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 12}  {'─' * 12}")
        for r in step_results:
            diff = r["peak_diff"]
            sign = "+" if diff > 0 else ""
            if has_baseline:
                print(
                    f"  {r['step']:>5}  "
                    f"{format_size(r['peak_a']):>12}  "
                    f"{format_size(r['peak_b']):>12}  "
                    f"{sign}{format_size(diff):>11}  "
                    f"{format_size(r['abs_peak_a']):>12}  "
                    f"{format_size(r['abs_peak_b']):>12}  "
                    f"{r['allocs_a']:>10,}  "
                    f"{r['allocs_b']:>10,}"
                )
            else:
                print(
                    f"  {r['step']:>5}  "
                    f"{format_size(r['peak_a']):>12}  "
                    f"{format_size(r['peak_b']):>12}  "
                    f"{sign}{format_size(diff):>11}  "
                    f"{r['allocs_a']:>10,}  "
                    f"{r['allocs_b']:>10,}  "
                    f"{format_size(r['throughput_a']):>12}  "
                    f"{format_size(r['throughput_b']):>12}"
                )

    # Excluded steps
    excluded_a = sorted(set(steps_a.keys()) - complete_a)
    excluded_b = sorted(set(steps_b.keys()) - complete_b)
    only_a = sorted(complete_a - complete_b)
    only_b = sorted(complete_b - complete_a)
    notes = []
    if excluded_a or excluded_b:
        all_exc = sorted(set(excluded_a) | set(excluded_b))
        notes.append(f"Step(s) {all_exc} excluded (incomplete, no END annotation)")
    if only_a:
        notes.append(f"Step(s) {only_a} only in A")
    if only_b:
        notes.append(f"Step(s) {only_b} only in B")
    for n in notes:
        print(f"  * {n}")

    # Drill-down
    if drill_data:
        step_label = f"Step {drill_data['step']}" if isinstance(drill_data["step"], int) else "Full Trace"
        print(
            f"\n--- Drill-Down: {step_label} "
            f"(peak diff: {'+' if drill_data['peak_diff'] > 0 else ''}"
            f"{format_size(drill_data['peak_diff'])}) ---"
        )
        sources = drill_data["sources"][:top_n]
        if sources:
            print(f"  {'Source':50s}  {'Peak A':>12}  {'Peak B':>12}  {'Delta(B-A)':>12}")
            print(f"  {'─' * 50}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
            for s in sources:
                delta = s["delta"]
                sign = "+" if delta > 0 else ""
                src_display = s["source"][:50]
                print(
                    f"  {src_display:50s}  "
                    f"{format_size(s['peak_a']):>12}  "
                    f"{format_size(s['peak_b']):>12}  "
                    f"{sign}{format_size(delta):>11}"
                )
        print(f"  ({len(drill_data['sources'])} total sources, showing top {top_n})")

    # Observations
    if observations:
        print("\n--- Observations ---")
        for i, obs in enumerate(observations, 1):
            print(f"  {i}. {obs}")

    print()


def main() -> None:
    """Parse arguments and run the two-snapshot comparison."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Compare two CUDA memory snapshots.",
        epilog=TRUST_EPILOG,
    )
    parser.add_argument("pickle_a", help="Path to snapshot A")
    parser.add_argument("pickle_b", help="Path to snapshot B")
    parser.add_argument("--step", type=int, help="Specific step to drill into (default: largest divergence)")
    parser.add_argument("--top", type=int, default=15, help="Number of top sources in drill-down (default: 15)")
    parser.add_argument(
        "--frame-depth", type=int, default=1, help="Stack frame depth for source grouping (default: 1)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    compare(
        args.pickle_a, args.pickle_b, step=args.step, top_n=args.top, frame_depth=args.frame_depth, as_json=args.json
    )


if __name__ == "__main__":
    main()
