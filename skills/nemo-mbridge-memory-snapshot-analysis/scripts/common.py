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

"""Shared utilities for CUDA memory snapshot analysis."""

# Annotations are lazy so the PEP 604 `X | None` syntax below does not need to
# evaluate at import time; these scripts stay runnable on the system python3
# (3.9 on macOS) even though the repo itself targets 3.10+.
from __future__ import annotations

import logging
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Frame filtering – paths considered "internal" (not user code)
# ---------------------------------------------------------------------------
_INTERNAL_PATH_FRAGMENTS = (
    "/torch/",
    "/torch_",
    "torch/nn/modules/module.py",
    "torch/autograd/",
    "torch/amp/",
    "torch/cuda/",
    "torch/distributed/",
    "/cuda/",
    "/hydra/",
    "/python3.",
    "/importlib/",
    "/runpy.py",
)

_INTERNAL_FUNC_NAMES = {"<module>", "<lambda>", "_call_impl", "_wrapped_call_impl", "inner"}


logger = logging.getLogger(__name__)

TRUST_EPILOG = (
    "Security: snapshots are unpickled, and pickle.load executes arbitrary code. "
    "Only analyze snapshot files from training runs you trust."
)


_KIB = 1024
_MIB = 1024**2
_GIB = 1024**3


def format_size(num_bytes: int) -> str:
    """Human-readable byte size in binary units, e.g. '37.91 GiB', '58.72 MiB'.

    Binary rather than decimal so the numbers line up with ``nvidia-smi`` and
    ``torch.cuda.memory_allocated`` reporting. A decimal "GB" reads ~7.4% larger
    than the same quantity in GiB, which is enough to make a cross-check against
    ``nvidia-smi`` look like a real discrepancy when nothing is wrong.
    """
    abs_b = abs(num_bytes)
    sign = "-" if num_bytes < 0 else ""
    if abs_b >= _GIB:
        return f"{sign}{abs_b / _GIB:.2f} GiB"
    if abs_b >= _MIB:
        return f"{sign}{abs_b / _MIB:.2f} MiB"
    if abs_b >= _KIB:
        return f"{sign}{abs_b / _KIB:.2f} KiB"
    return f"{sign}{abs_b} B"


def load_snapshot(path: str) -> dict:
    """Load a CUDA memory snapshot pickle file.

    Snapshots are unpickled, and ``pickle.load`` executes arbitrary code, so
    only load files produced by a training run you trust.
    """
    if not os.path.isfile(path):
        logger.error(f"file not found: {path}")
        sys.exit(1)
    size_mib = os.path.getsize(path) / _MIB
    logger.info(f"Loading {path} ({size_mib:.1f} MiB) — unpickling executes code; load trusted files only.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        logger.error(f"expected dict, got {type(data).__name__}")
        sys.exit(1)
    return data


def find_active_device(snapshot: dict) -> int | None:
    """Return the index of the first device_traces entry with events."""
    traces = snapshot.get("device_traces", [])
    for i, trace in enumerate(traces):
        if trace and len(trace) > 0:
            return i
    return None


def get_profiler_steps(annotations: list) -> dict[int, dict[str, int | None]]:
    """Parse ProfilerStep#N annotations into {step_num: {'start': us, 'end': us}}.

    Incomplete steps (missing END) are included with end=None.
    """
    steps: dict[int, dict[str, int | None]] = {}
    for ann in annotations:
        name = ann.get("name", "")
        if "ProfilerStep#" not in name:
            continue
        try:
            step_num = int(name.split("#")[1])
        except (IndexError, ValueError):
            continue
        if step_num not in steps:
            steps[step_num] = {"start": None, "end": None}
        stage = ann.get("stage", "")
        time_us = ann.get("time_us", 0)
        if stage == "START":
            steps[step_num]["start"] = time_us
        elif stage == "END" and time_us > 0:
            steps[step_num]["end"] = time_us
    return steps


def get_step_events(traces: list, start_us: int, end_us: int | None) -> list:
    """Filter trace events to a time range [start_us, end_us]."""
    if end_us is None:
        return [e for e in traces if e["time_us"] >= start_us]
    return [e for e in traces if start_us <= e["time_us"] <= end_us]


def get_source_key(frames: list, depth: int = 1) -> str:
    """Extract a meaningful source key from stack frames.

    Walks the frames top-to-bottom, skipping PyTorch/CUDA internals,
    and returns `function@file.py:line` for the first `depth` user-level frames.
    """
    relevant = []
    for f in frames:
        fn = f.get("filename", "")
        name = f.get("name", "")
        # Skip internal frames
        if any(frag in fn for frag in _INTERNAL_PATH_FRAGMENTS):
            continue
        if name in _INTERNAL_FUNC_NAMES:
            continue
        short_file = fn.rsplit("/", 1)[-1] if "/" in fn else fn
        relevant.append(f"{name}@{short_file}:{f.get('line', '?')}")
        if len(relevant) >= depth:
            break
    if not relevant:
        # Fallback: use top frame even if internal
        for f in frames[:depth]:
            fn = f.get("filename", "")
            short_file = fn.rsplit("/", 1)[-1] if "/" in fn else fn
            relevant.append(f"{f.get('name', '?')}@{short_file}:{f.get('line', '?')}")
    return " -> ".join(relevant) if relevant else "<no frames>"


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------


@dataclass
class BaselineInfo:
    """Baseline memory at the start of profiling.

    Derived by cross-referencing segments (point-in-time allocator state)
    with device_traces (alloc/free event log during profiling).
    """

    segments_active: int = 0  # Active bytes in segments at snapshot time
    pre_existing_freed: int = 0  # Pre-existing allocs freed during profiling
    profiling_still_live: int = 0  # Profiling-period allocs still live at snapshot
    baseline_at_start: int = 0  # Estimated memory at profiling start


def compute_baseline(snapshot: dict, device_idx: int) -> BaselineInfo:
    """Estimate memory allocated before profiling started.

    At snapshot time, segments_active = pre_existing_still_live + profiling_still_live.
    At profiling start, memory = pre_existing_still_live + pre_existing_freed
    (the freed ones hadn't been freed yet).
    """
    segments = snapshot.get("segments", [])
    # active_size, not allocated_size: the trace replay below only releases a
    # block on free_completed, so a block in active_awaiting_free is still live
    # from the replay's point of view. Using allocated_size here would subtract
    # those blocks on one side of the equation but not the other.
    segments_active = sum(s.get("active_size", s.get("allocated_size", 0)) for s in segments)

    traces = snapshot.get("device_traces", [])
    device_trace = traces[device_idx] if device_idx < len(traces) else []

    allocated: dict[int, int] = {}  # addr -> size for profiling-period allocs
    pre_existing_freed = 0

    for ev in device_trace:
        action = ev.get("action")
        if action == "alloc":
            allocated[ev["addr"]] = ev["size"]
        elif action == "free_completed":
            if ev["addr"] in allocated:
                del allocated[ev["addr"]]
            else:
                pre_existing_freed += ev["size"]

    profiling_still_live = sum(allocated.values())
    baseline_at_start = segments_active - profiling_still_live + pre_existing_freed

    return BaselineInfo(
        segments_active=segments_active,
        pre_existing_freed=pre_existing_freed,
        profiling_still_live=profiling_still_live,
        baseline_at_start=baseline_at_start,
    )


def compute_step_start_deltas(traces: list, steps: dict) -> dict[int, int]:
    """Compute the cumulative memory delta at the start of each profiler step.

    Replays the full trace chronologically, recording the running net delta
    (allocs minus frees) at the moment each step begins.  Combined with
    baseline_at_start, this gives the absolute memory at each step boundary.
    """
    step_starts = sorted(
        [(info["start"], num) for num, info in steps.items() if info["start"]],
        key=lambda x: x[0],
    )
    if not step_starts:
        return {}

    current_delta = 0
    result: dict[int, int] = {}
    step_idx = 0

    for ev in traces:
        t = ev.get("time_us", 0)
        while step_idx < len(step_starts) and t >= step_starts[step_idx][0]:
            result[step_starts[step_idx][1]] = current_delta
            step_idx += 1
        action = ev.get("action")
        if action == "alloc":
            current_delta += ev["size"]
        elif action == "free_completed":
            current_delta -= ev["size"]

    # Handle steps that start after the last event
    while step_idx < len(step_starts):
        result[step_starts[step_idx][1]] = current_delta
        step_idx += 1

    return result


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Result of replaying device_traces events."""

    peak_delta: int = 0
    peak_live_set: dict[int, tuple[int, list]] = field(default_factory=dict)
    end_delta: int = 0
    total_alloc_bytes: int = 0
    alloc_count: int = 0
    free_count: int = 0
    unmatched_free_count: int = 0
    unmatched_free_bytes: int = 0


def live_set_before(traces: list, cutoff_us: int) -> dict[int, tuple[int, list]]:
    """Return allocations still live immediately before *cutoff_us*.

    Used to seed a windowed replay so that memory allocated earlier in the run
    and still held (weights, optimizer state, graph pools, activations carried
    across a step boundary) is attributable, not just the allocations that
    happen to occur inside the window.
    """
    live: dict[int, tuple[int, list]] = {}
    for e in traces:
        if e.get("time_us", 0) >= cutoff_us:
            break
        action = e.get("action")
        if action == "alloc":
            live[e["addr"]] = (e["size"], e.get("frames", []))
        elif action == "free_completed":
            live.pop(e["addr"], None)
    return live


def replay_events(events: list, initial_live: dict[int, tuple[int, list]] | None = None) -> ReplayResult:
    """Replay alloc/free_completed events to track memory over time.

    Returns peak delta from start, the set of live allocations at peak,
    and summary statistics.

    Args:
        events: Trace events to replay, in chronological order.
        initial_live: Allocations already live when the window opens, as
            returned by :func:`live_set_before`. These are carried into
            ``peak_live_set`` so source attribution accounts for them, but they
            do not contribute to ``peak_delta``, which stays a delta measured
            from the start of the window.
    """
    live: dict[int, tuple[int, list]] = dict(initial_live) if initial_live else {}  # addr -> (size, frames)
    current_delta = 0
    peak_delta = 0
    # Seeded state is the peak until an alloc beats it, so a window that only
    # frees still reports the allocations that were live when it opened.
    peak_snapshot: dict[int, tuple[int, list]] = dict(live)
    total_alloc = 0
    alloc_count = 0
    free_count = 0
    unmatched_frees = 0
    unmatched_free_bytes = 0

    for e in events:
        action = e["action"]
        if action == "alloc":
            addr = e["addr"]
            size = e["size"]
            frames = e.get("frames", [])
            live[addr] = (size, frames)
            current_delta += size
            total_alloc += size
            alloc_count += 1
            if current_delta > peak_delta:
                peak_delta = current_delta
                # Snapshot the live set at peak — copy the dict
                peak_snapshot = dict(live)
        elif action == "free_completed":
            addr = e["addr"]
            size = e["size"]
            free_count += 1
            if addr in live:
                current_delta -= live[addr][0]
                del live[addr]
            else:
                # Pre-existing allocation freed during trace
                current_delta -= size
                unmatched_frees += 1
                unmatched_free_bytes += size

    return ReplayResult(
        peak_delta=peak_delta,
        peak_live_set=peak_snapshot,
        end_delta=current_delta,
        total_alloc_bytes=total_alloc,
        alloc_count=alloc_count,
        free_count=free_count,
        unmatched_free_count=unmatched_frees,
        unmatched_free_bytes=unmatched_free_bytes,
    )


def group_by_source(live_set: dict[int, tuple[int, list]], depth: int = 1) -> list[tuple[str, int, int]]:
    """Group a live allocation set by source key.

    Returns sorted list of (source_key, total_bytes, count), descending by bytes.
    """
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for addr, (size, frames) in live_set.items():
        key = get_source_key(frames, depth)
        groups[key][0] += size
        groups[key][1] += 1
    result = [(key, total, count) for key, (total, count) in groups.items()]
    result.sort(key=lambda x: -x[1])
    return result


@dataclass
class TimeReplayResult:
    """Result of replaying device_traces up to a specific elapsed time."""

    live_set: dict[int, tuple[int, list]] = field(default_factory=dict)
    current_delta: int = 0
    alloc_count: int = 0
    free_count: int = 0
    unmatched_free_count: int = 0


def replay_to_time(events: list, target_elapsed_us: int) -> TimeReplayResult:
    """Replay alloc/free events up to *target_elapsed_us* from profiling start.

    *Profiling start* is defined as the timestamp of the very first event in
    the trace (which is typically a ``segment_map``), so ``--time T`` means
    T seconds from when profiling began — matching external logs and the
    timeline plot's x-axis.

    Returns the live allocation set (addr -> (size, frames)) and cumulative
    delta at that point, suitable for source grouping with ``group_by_source``.
    """
    if not events:
        return TimeReplayResult()
    t0 = events[0].get("time_us", 0)

    cutoff = t0 + target_elapsed_us
    live: dict[int, tuple[int, list]] = {}
    current_delta = 0
    alloc_count = 0
    free_count = 0
    unmatched = 0

    for e in events:
        if e.get("time_us", 0) > cutoff:
            break
        action = e["action"]
        if action == "alloc":
            addr = e["addr"]
            size = e["size"]
            live[addr] = (size, e.get("frames", []))
            current_delta += size
            alloc_count += 1
        elif action == "free_completed":
            addr = e["addr"]
            free_count += 1
            if addr in live:
                current_delta -= live[addr][0]
                del live[addr]
            else:
                current_delta -= e["size"]
                unmatched += 1

    return TimeReplayResult(
        live_set=live,
        current_delta=current_delta,
        alloc_count=alloc_count,
        free_count=free_count,
        unmatched_free_count=unmatched,
    )


def get_step_annotations(annotations: list, start_us: int, end_us: int | None) -> dict[str, int]:
    """Count annotation names active during a step's time range."""
    counts: dict[str, int] = defaultdict(int)
    for ann in annotations:
        if ann.get("name", "").startswith("ProfilerStep"):
            continue
        t = ann.get("time_us", 0)
        if t < start_us:
            continue
        if end_us is not None and t > end_us:
            continue
        if ann.get("stage") == "START":
            counts[ann["name"]] += 1
    return dict(counts)
