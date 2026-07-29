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

"""Generate an interactive HTML timeline of CUDA memory usage from snapshots."""

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
from common import TRUST_EPILOG, compute_baseline, find_active_device, load_snapshot


COLORS = [
    "#00d2ff",
    "#ff6b6b",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#e74c3c",
    "#1abc9c",
    "#e67e22",
]


# ---------------------------------------------------------------------------
# Replay helpers (local – only needed for timeline generation)
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)


def _replay_timeseries(traces):
    """Replay trace events into parallel lists for both x-axis modes.

    Uses ``free_completed`` to match PyTorch memory_viz behaviour.

    Returns (times_us, event_indices, delta_bytes) where:
    - *times_us*: wall-clock timestamp of each state change
    - *event_indices*: synthetic counter (1 per alloc/free) for event-index mode
    - *delta_bytes*: cumulative allocated bytes at each point
    """
    times = []
    indices = []
    deltas = []
    live = {}
    current = 0
    idx = 0

    # Seed the state at trace start. Without this the series begins at the first
    # alloc/free rather than at delta 0, so a trace whose earliest visible event
    # is an unmatched free (older captures, or a ring buffer that dropped the
    # matching allocs) yields only negative deltas and reports a peak below the
    # reconstructed baseline — when the true peak was the baseline itself.
    if traces:
        times.append(traces[0].get("time_us", 0))
        indices.append(idx)
        deltas.append(0)
        idx += 1

    for e in traces:
        action = e["action"]
        if action == "alloc":
            addr = e["addr"]
            live[addr] = e["size"]
            current += e["size"]
            times.append(e["time_us"])
            indices.append(idx)
            deltas.append(current)
            idx += 1
        elif action == "free_completed":
            addr = e["addr"]
            if addr in live:
                current -= live.pop(addr)
            else:
                current -= e["size"]
            times.append(e["time_us"])
            indices.append(idx)
            deltas.append(current)
            idx += 1

    return times, indices, deltas


def _downsample_by_time(times_us, values, precision_us):
    """Downsample by fixed-width time buckets, preserving min/max per bucket."""
    n = len(times_us)
    if n == 0:
        return [], []

    out_t, out_v = [], []
    bucket_end = times_us[0] + precision_us
    min_idx = max_idx = 0

    for i in range(1, n):
        if times_us[i] >= bucket_end:
            # Emit current bucket
            first, second = (min_idx, max_idx) if min_idx <= max_idx else (max_idx, min_idx)
            out_t.append(times_us[first])
            out_v.append(values[first])
            if first != second:
                out_t.append(times_us[second])
                out_v.append(values[second])
            # Start new bucket at this event
            bucket_end = times_us[i] + precision_us
            min_idx = max_idx = i
        else:
            if values[i] < values[min_idx]:
                min_idx = i
            if values[i] > values[max_idx]:
                max_idx = i

    # Emit last bucket
    first, second = (min_idx, max_idx) if min_idx <= max_idx else (max_idx, min_idx)
    out_t.append(times_us[first])
    out_v.append(values[first])
    if first != second:
        out_t.append(times_us[second])
        out_v.append(values[second])

    return out_t, out_v


def _downsample_by_count(xs, values, target):
    """Downsample to a target point count via min/max bucketing."""
    target = max(target, 4)
    n = len(xs)
    if n <= target:
        return xs, values

    bucket_size = max(1, n // (target // 2))
    out_x, out_v = [], []

    for i in range(0, n, bucket_size):
        end = min(i + bucket_size, n)
        min_idx = max_idx = i
        for j in range(i + 1, end):
            if values[j] < values[min_idx]:
                min_idx = j
            if values[j] > values[max_idx]:
                max_idx = j

        first, second = (min_idx, max_idx) if min_idx <= max_idx else (max_idx, min_idx)
        out_x.append(xs[first])
        out_v.append(values[first])
        if first != second:
            out_x.append(xs[second])
            out_v.append(values[second])

    return out_x, out_v


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CUDA Memory Timeline</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 20px;
    background: #1a1a2e; color: #e0e0e0;
  }
  h1 { text-align: center; margin-bottom: 5px; }
  .subtitle { text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }
  .container { max-width: 1400px; margin: 0 auto; }
  .controls {
    display: flex; gap: 20px; justify-content: center;
    margin-bottom: 15px; flex-wrap: wrap;
  }
  .control-group {
    background: #16213e; border-radius: 8px; padding: 10px 18px;
    display: flex; align-items: center; gap: 8px;
  }
  .control-group label { font-size: 14px; cursor: pointer; }
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 15px; margin-bottom: 20px;
  }
  .stat-card {
    background: #16213e; border-radius: 8px; padding: 15px;
    border-left: 4px solid #666;
  }
  .stat-card h3 { margin: 0 0 8px; font-size: 14px; color: #aaa; }
  .stat-card .value { font-size: 24px; font-weight: bold; }
  .stat-card .detail { font-size: 12px; color: #888; margin-top: 4px; }
  #plot { width: 100%; height: 600px; }
  select, input[type=number] {
    background: #0f3460; color: #e0e0e0; border: 1px solid #444;
    border-radius: 4px; padding: 4px 8px; font-size: 13px;
  }
  input[type=number] { width: 70px; }
  .point-count { font-size: 12px; color: #888; margin-left: 4px; }
</style>
</head>
<body>
<div class="container">
  <h1>CUDA Memory Timeline</h1>
  <p class="subtitle" id="subtitle"></p>
  <div class="stats" id="stat-cards"></div>
  <div class="controls">
    <div class="control-group">
      <label>X-axis:</label>
      <select id="xMode" onchange="updatePlot()">
        <option value="time">Wall Clock (seconds)</option>
        <option value="event">Event Index</option>
      </select>
    </div>
    <div class="control-group">
      <label>Y-axis:</label>
      <select id="yMode" onchange="updatePlot()">
        <option value="absolute">Absolute (baseline + delta)</option>
        <option value="delta">Delta from trace start</option>
      </select>
    </div>
    <div class="control-group" id="precisionGroup">
      <label>Precision:</label>
      <input type="number" id="precisionMs" value="__PRECISION_MS__"
             min="__PRECISION_MS__" step="__PRECISION_MS__"
             onchange="updatePlot()">
      <label>ms</label>
      <span class="point-count" id="pointCount"></span>
    </div>
  </div>
  <div id="plot"></div>
</div>
<script>
const DATA = __DATA_PLACEHOLDER__;
const BASE_PRECISION_MS = __PRECISION_MS__;

document.getElementById('subtitle').textContent = DATA.map(d => d.label).join(' vs ');

// Build stat cards (static — peak values don't change with precision)
const cardsDiv = document.getElementById('stat-cards');
DATA.forEach(d => {
  const card = document.createElement('div');
  card.className = 'stat-card';
  card.style.borderLeftColor = d.color;
  card.innerHTML = `
    <h3>${d.label} — Peak</h3>
    <div class="value" style="color:${d.color}">${d.peak_abs_gib.toFixed(2)} GiB</div>
    <div class="detail">
      Baseline: ${d.baseline_gib.toFixed(2)} GiB
      &nbsp;|&nbsp; Peak delta: ${d.peak_delta_gib.toFixed(2)} GiB
      &nbsp;|&nbsp; ${d.alloc_count.toLocaleString()} alloc events
    </div>`;
  cardsDiv.appendChild(card);
});

if (DATA.length === 2) {
  const diff = DATA[0].peak_abs_gib - DATA[1].peak_abs_gib;
  const card = document.createElement('div');
  card.className = 'stat-card';
  card.style.borderLeftColor = '#ffd93d';
  const sign = diff >= 0 ? '+' : '';
  card.innerHTML = `
    <h3>Delta (${DATA[0].label} - ${DATA[1].label})</h3>
    <div class="value" style="color:#ffd93d">${sign}${diff.toFixed(2)} GiB</div>
    <div class="detail">Difference in absolute peak memory</div>`;
  cardsDiv.appendChild(card);
}

// Client-side time-based downsampling (coarsen from embedded resolution)
function downsampleByTime(times, values, precisionS) {
  const n = times.length;
  if (n === 0) return { x: [], y: [] };
  const ox = [], oy = [];
  let bEnd = times[0] + precisionS;
  let minI = 0, maxI = 0;
  for (let i = 1; i < n; i++) {
    if (times[i] >= bEnd) {
      const f = minI <= maxI ? minI : maxI;
      const s = minI <= maxI ? maxI : minI;
      ox.push(times[f]); oy.push(values[f]);
      if (f !== s) { ox.push(times[s]); oy.push(values[s]); }
      bEnd = times[i] + precisionS;
      minI = maxI = i;
    } else {
      if (values[i] < values[minI]) minI = i;
      if (values[i] > values[maxI]) maxI = i;
    }
  }
  const f = minI <= maxI ? minI : maxI;
  const s = minI <= maxI ? maxI : minI;
  ox.push(times[f]); oy.push(values[f]);
  if (f !== s) { ox.push(times[s]); oy.push(values[s]); }
  return { x: ox, y: oy };
}

function updatePlot() {
  const useEvent = document.getElementById('xMode').value === 'event';
  const useAbs = document.getElementById('yMode').value === 'absolute';
  const precMs = Math.max(BASE_PRECISION_MS, parseFloat(document.getElementById('precisionMs').value) || BASE_PRECISION_MS);

  // Show/hide precision control based on x-axis mode
  document.getElementById('precisionGroup').style.opacity = useEvent ? '0.4' : '1';

  let totalPoints = 0;
  const traces = DATA.map(d => {
    let xData, deltaData;
    if (useEvent) {
      xData = d.event_idx;
      deltaData = d.delta_gib_event;
    } else if (precMs <= BASE_PRECISION_MS) {
      // At base precision — use embedded data directly
      xData = d.times_s;
      deltaData = d.delta_gib;
    } else {
      // Coarser than base — downsample client-side
      const ds = downsampleByTime(d.times_s, d.delta_gib, precMs / 1000);
      xData = ds.x;
      deltaData = ds.y;
    }
    const yData = useAbs ? deltaData.map(v => v + d.baseline_gib) : deltaData;
    totalPoints += xData.length;
    return {
      x: xData,
      y: yData,
      type: 'scattergl',
      mode: 'lines',
      name: d.label,
      line: { color: d.color, width: 1.5 },
      hovertemplate: useEvent
        ? `<b>${d.label}</b><br>Event: %{x:,}<br>Memory: %{y:.2f} GiB<extra></extra>`
        : `<b>${d.label}</b><br>Time: %{x:.2f}s<br>Memory: %{y:.2f} GiB<extra></extra>`
    };
  });
  document.getElementById('pointCount').textContent =
    useEvent ? '' : `(${totalPoints.toLocaleString()} pts)`;
  const layout = {
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    font: { color: '#e0e0e0', family: '-apple-system, BlinkMacSystemFont, sans-serif' },
    xaxis: {
      title: useEvent ? 'Event Index (alloc/free sequence)' : 'Time (seconds from profiling start)',
      gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a',
      tickformat: useEvent ? ',d' : '.1f'
    },
    yaxis: {
      title: useAbs ? 'GPU Memory (GiB)' : 'Memory Delta from Start (GiB)',
      gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a', ticksuffix: ' GiB'
    },
    legend: {
      orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'center', x: 0.5,
      bgcolor: 'rgba(22,33,62,0.8)', bordercolor: '#2a2a4a', borderwidth: 1,
      font: { size: 13 }
    },
    hovermode: 'x unified',
    margin: { l: 70, r: 30, t: 40, b: 60 }
  };
  Plotly.react('plot', traces, layout, {
    responsive: true, displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'], displaylogo: false
  });
}
updatePlot();
</script>
</body>
</html>"""


def generate(pickle_paths: list[str], labels: list[str], output: str, precision_ms: float) -> None:
    """Write a standalone interactive HTML timeline for one or more snapshots.

    Args:
        pickle_paths: Snapshot pickle files to plot, one trace per file.
        labels: Display label for each trace, parallel to ``pickle_paths``.
        output: Path the HTML file is written to.
        precision_ms: Time bucket width in milliseconds for the wall-clock view.
    """
    GIB = 1024**3
    precision_us = int(precision_ms * 1000)
    traces_data = []

    for i, path in enumerate(pickle_paths):
        snap = load_snapshot(path)
        dev = find_active_device(snap)
        if dev is None:
            logger.warning(f"no device traces in {path}, skipping.")
            continue
        trace = snap["device_traces"][dev]
        baseline = compute_baseline(snap, dev)
        base_gib = baseline.baseline_at_start / GIB

        # Anchor to profiling start (first event of any kind)
        t0 = trace[0].get("time_us", 0) if trace else 0

        logger.info(f"  Replaying {len(trace):,} events ...")
        raw_times, raw_indices, raw_deltas = _replay_timeseries(trace)
        if not raw_times:
            logger.warning(f"empty trace in {path}, skipping.")
            continue

        # Wall-clock view: downsample by time precision
        ds_times, ds_deltas_t = _downsample_by_time(raw_times, raw_deltas, precision_us)

        # Event-index view: match the wall-clock point count for similar file size
        time_points = len(ds_times)
        ds_indices, ds_deltas_e = _downsample_by_count(
            raw_indices,
            raw_deltas,
            time_points,
        )

        times_s = [(t - t0) / 1e6 for t in ds_times]
        delta_gib_time = [d / GIB for d in ds_deltas_t]
        event_idx = list(ds_indices)
        delta_gib_event = [d / GIB for d in ds_deltas_e]

        peak_delta_gib = max(d / GIB for d in raw_deltas) if raw_deltas else 0
        peak_abs_gib = peak_delta_gib + base_gib

        label = labels[i] if i < len(labels) else os.path.basename(path)
        color = COLORS[i % len(COLORS)]
        alloc_count = sum(1 for e in trace if e["action"] == "alloc")

        logger.info(f"  {len(times_s):,} time points ({precision_ms}ms), {len(event_idx):,} event-index points")

        traces_data.append(
            {
                "label": label,
                "color": color,
                "baseline_gib": round(base_gib, 4),
                "alloc_count": alloc_count,
                "peak_delta_gib": round(peak_delta_gib, 4),
                "peak_abs_gib": round(peak_abs_gib, 4),
                "points": len(times_s),
                # Time-based x-axis data
                "times_s": times_s,
                "delta_gib": delta_gib_time,
                # Event-index x-axis data
                "event_idx": event_idx,
                "delta_gib_event": delta_gib_event,
            }
        )

    if not traces_data:
        logger.error("no valid traces to plot.")
        sys.exit(1)

    html = _HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(traces_data)).replace(
        "__PRECISION_MS__", str(precision_ms)
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        f.write(html)

    size_kib = os.path.getsize(output) / 1024
    logger.info(f"Written to {output} ({size_kib:.0f} KiB)")


def main() -> None:
    """Parse arguments and write the interactive timeline HTML."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML timeline of CUDA memory usage.",
        epilog=TRUST_EPILOG,
    )
    parser.add_argument("pickles", nargs="+", help="Snapshot pickle file(s)")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[],
        help="Display labels for each snapshot (default: filenames)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: cuda_memory_timeline.html in working dir)",
    )

    def _positive_float(value):
        fval = float(value)
        if fval <= 0:
            raise argparse.ArgumentTypeError(f"--precision must be strictly positive, got {value}")
        return fval

    parser.add_argument(
        "--precision",
        type=_positive_float,
        default=1.0,
        help="Time bucket width in milliseconds for wall-clock mode (default: 1.0). "
        "Smaller values produce more plot points and finer detail.",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.pickles):
        parser.error("Number of --labels must match number of pickle files")

    output = args.output or "cuda_memory_timeline.html"
    generate(args.pickles, args.labels, output, args.precision)


if __name__ == "__main__":
    main()
