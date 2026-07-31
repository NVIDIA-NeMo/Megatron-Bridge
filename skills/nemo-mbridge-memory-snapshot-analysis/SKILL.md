---
name: nemo-mbridge-memory-snapshot-analysis
description: Analyze and compare PyTorch CUDA memory snapshots produced by Megatron Bridge's ProfilingConfig(record_memory_history=True). Replays the recorded allocation timeline to plot memory over time, compare two runs, and attribute peak memory to source code locations.
license: Apache-2.0
when_to_use: Debugging an OOM or peak-memory regression from a captured snapshot pickle, plotting GPU memory over time, comparing memory between two configurations, or attributing memory to specific allocations; 'snapshot.pickle', 'snapshot_0.pickle', 'memory snapshot', 'record_memory_history', 'memory_snapshot_path', 'device_traces', 'plot memory usage', 'graph memory over time', 'memory timeline', 'visualize GPU memory', 'memory usage chart', 'why does this config use more memory', 'peak memory analysis', 'OOM pickle'.
---

# Memory Snapshot Analysis

Stable documentation: @docs/training/profiling.md
Related skills: @skills/nemo-mbridge-perf-memory-tuning/SKILL.md (how to *fix*
the memory problem this skill helps you *find*)

## What It Is

Bridge writes PyTorch CUDA memory snapshots when `record_memory_history=True`
is set on `ProfilingConfig`. Each snapshot pickle contains:

| Key | Contents |
|---|---|
| `device_traces` | Chronological alloc/free events with Python stack context — the bulk of the file |
| `segments` | Point-in-time allocator state at the moment of the dump |
| `external_annotations` | `ProfilerStep#N` START/END markers — **only present when the PyTorch profiler also ran** |
| `allocator_settings` | `PYTORCH_CUDA_ALLOC_CONF` and related allocator config |

The bundled scripts replay `device_traces` to reconstruct memory over time,
group allocations by source location, and diff two snapshots to explain *which
tensors* account for a memory difference.

PyTorch's own `memory_viz` also renders a timeline from these files and is worth
using for a single snapshot. What it cannot do is put **two runs on one axis** —
which is the whole question when you are asking why config B peaks higher than
config A. `plot_timeline.py` overlays them, and `compare_snapshots.py` then
attributes the gap to specific allocation sites. Add scriptable JSON output on
top and that is the reason this skill exists alongside `memory_viz`.

**No dependencies** — Python stdlib only, and they run on any `python3` back to
3.9, so a login node or laptop system interpreter is fine (macOS ships 3.9;
`from __future__ import annotations` keeps the newer type syntax from being
evaluated at import). No virtualenv, no `uv sync`, no GPU. The HTML plot loads
Plotly.js from a CDN in the browser, so it adds nothing to the Python
environment.

## Pick a Script

| If the question is… | Run | Output |
|---|---|---|
| "what's even in this file?" | `parse_snapshot.py` | text / JSON |
| "**plot / graph / visualize** memory over time" | `plot_timeline.py` | **standalone HTML** |
| "why does B peak higher than A?" | `compare_snapshots.py` | text / JSON |
| "what's live at *this* moment?" | `replay_to_time.py` | text / JSON |
| "what happened during step N?" | `replay_step.py` (needs step markers) | text / JSON |

`plot_timeline.py` is the only script that writes a file; the rest print to
stdout. It also takes **two** snapshots and overlays them, which is usually the
fastest way to see *where* two runs diverge before asking `compare_snapshots.py`
*why*.

A good default loop for a memory regression: overview both files, overlay them
on a timeline, read off the time where they split, then drill in at that moment.

```bash
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/plot_timeline.py A_0.pickle B_0.pickle --labels A B -o compare.html
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/replay_to_time.py A_0.pickle B_0.pickle --time 83.0
```

## How Bridge Produces These Files

Read this before interpreting any numbers. Bridge's capture path differs from a
hand-rolled `_record_memory_history()` call in ways that change what the
numbers mean.

```python
from megatron.bridge.training.config import ProfilingConfig

cfg.profiling = ProfilingConfig(
    record_memory_history=True,
    memory_snapshot_path="/nemo_run/snapshot.pickle",
    profile_step_end=15,
    profile_ranks=[0],
)
```

Five consequences worth internalizing:

**1. `external_annotations` is populated, but never with `ProfilerStep`.**
This is the single most misleading thing about these files. The list is *not*
empty — a real Bridge pretrain snapshot carries thousands of entries, but all of
them are allocator-level `record_function` markers:

```text
  1448  nccl:all_gather_into_tensor_coalesced
   588  nccl:all_reduce
   504  nccl:reduce_scatter_tensor_coalesced
   462  nccl:_all_gather_base
    96  enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__
    44  Optimizer.step#FusedAdam.step
    24  CustomFSDP.forward
```

`ProfilerStep#N` markers come from `torch.profiler`'s step boundaries, and
`record_memory_history` and `use_pytorch_profiler` are independent flags. So a
snapshot can be rich in annotations and still have zero step boundaries.
Consequences:

- `replay_step.py` cannot run — it needs step boundaries, and exits 1.
- `compare_snapshots.py` automatically falls back to full-trace replay.
- `plot_timeline.py` and `replay_to_time.py` are unaffected — they never needed steps.
- The NCCL/optimizer/dataloader markers *are* still useful for orienting
  yourself in the timeline, and `replay_step.py` surfaces them per step when
  step boundaries do exist.

To get per-step drill-down, set `use_pytorch_profiler=True` alongside
`record_memory_history=True`. Note `finalize()` forbids combining it with
`use_nsys_profiler`.

**2. Where the trace starts depends on the Bridge version.** Since
`fix: record CUDA memory history before snapshot` (`9e9e58b9f`, 2026-04-23),
`start_memory_history_recording()` runs at `setup.py:330`, *before*
`_build_distributed_model()`. Weights and optimizer state are then allocated
**inside** the trace, so they appear as ordinary `alloc` events attributable to
their construction site and `baseline_at_start` is small.

Snapshots captured before that commit — or by an external callback that enables
recording at train start — begin after model construction instead, and their
`baseline_at_start` is the whole model. A verified pre-fix capture shows a
26.29 GiB baseline for exactly this reason. Check the earliest frames in the
trace to tell which regime you are in.

**3. The trace is capped at 100k entries.** `trace_alloc_max_entries=100_000`
is hardcoded in `start_memory_history_recording()`. PyTorch keeps the most
recent entries once the cap is hit, so a long enough run silently loses its
beginning. Symptom: a large `unmatched frees` count and a `baseline_at_start`
that no longer reconciles with `nvidia-smi`. If you need the start of the run,
dump earlier by lowering `profile_step_end`. Snapshots captured outside this
code path are not subject to the cap and can be much larger — a real 113-second
capture contained 184,679 events.

**4. One dump per rank, at `profile_step_end`.** The snapshot is written when
`iteration == profile_step_end`, with the rank appended before the extension.
`memory_snapshot_path="/nemo_run/snapshot.pickle"` with `profile_ranks=[0]`
yields `/nemo_run/snapshot_0.pickle`. Compare like for like — rank 0 against
rank 0.

**5. OOM produces its own snapshot automatically.** An out-of-memory observer is
attached at capture time and dumps `snapshot_oom_rank-{N}.pickle` at the moment
of failure. This file is usually the most valuable one you have after a crash:
it captures exactly what was live when the allocator gave up. Analyze it with
`parse_snapshot.py` and `replay_to_time.py`.

## Workflows

Scripts live in `skills/nemo-mbridge-memory-snapshot-analysis/scripts/`.

> **Load trusted snapshots only.** Every script below starts by unpickling the
> file, and `pickle.load` executes arbitrary code embedded in it. Analyze
> snapshots produced by your own training runs or by someone you trust — never a
> pickle from an untrusted issue attachment, bucket, or download. There is no
> safe-mode parse: the format requires full deserialization.
>
> Snapshots also carry absolute paths and stack frames from the machine that
> produced them, so check before sharing one outside your organization.

### 1. Single snapshot overview

Start here to see what a file actually contains — especially whether it has step
annotations and how many events survived the ring buffer.

```bash
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/parse_snapshot.py snapshot_0.pickle
```

Shows allocator settings, segments totals and fragmentation, baseline memory,
detected steps, event counts by action, and the top 15 allocation sources by
throughput. Add `--json` for machine-readable output.

### 2. Compare two runs

The primary use case: explaining why config B uses more memory than config A.

```bash
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/compare_snapshots.py A_0.pickle B_0.pickle
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/compare_snapshots.py A_0.pickle B_0.pickle --top 20 --frame-depth 2
```

Produces an allocator-settings diff, a side-by-side segments baseline, a
per-step table (or a full-trace summary when annotations are absent), a
drill-down at the largest divergence grouped by source and sorted by delta, and
automatic observations.

### 3. Visualize the timeline

```bash
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/plot_timeline.py snapshot_0.pickle -o timeline.html
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/plot_timeline.py A_0.pickle B_0.pickle --labels "TP2" "TP4" -o compare.html
```

Standalone HTML with a WebGL Plotly timeline. Two toggles matter:

- **X-axis**: "Wall Clock" vs "Event Index". Event-index mode matches
  `memory_viz`, stretching busy iterations and compressing idle gaps so
  sawtooth allocation patterns become visible.
- **Y-axis**: "Absolute" (baseline + delta) vs "Delta from trace start".

Wall-clock mode buckets by time (default 1 ms, `--precision`) preserving min/max
per bucket, so busy traces produce more points.

### 4. Drill into a specific moment

The natural follow-up to the plot: spot a divergence visually, then find out
what is responsible.

```bash
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/replay_to_time.py snapshot_0.pickle --time 83.0
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/replay_to_time.py A_0.pickle B_0.pickle --time 83.0 --frame-depth 2
```

Replays to the given elapsed second and reports baseline/delta/absolute memory,
live allocation counts, and live allocations grouped by source. In comparison
mode it sorts by absolute delta and separately totals only-in-A and only-in-B
sources.

### 5. Per-step drill-down (requires annotations)

Only works when the snapshot was captured with `use_pytorch_profiler=True`.

```bash
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/replay_step.py snapshot_0.pickle --step 7
python3 skills/nemo-mbridge-memory-snapshot-analysis/scripts/replay_step.py snapshot_0.pickle --all-steps
```

Per step: alloc/free counts, throughput, peak delta and absolute peak, live
allocations at peak grouped by source, and active annotation phases.

## Example Output

```text
--- Segments Baseline ---
                            Snapshot A      Snapshot B           Delta
  ────────────────────  ──────────────  ──────────────  ──────────────
  Segments                          24              24               0
  Reserved (total)           53.21 GiB       73.49 GiB  +    20.27 GiB
  Allocated                  45.10 GiB       62.28 GiB  +    17.18 GiB
  Active                     45.10 GiB       62.28 GiB  +    17.18 GiB
  Awaiting free                    0 B             0 B            0 B
  Inactive (reusable)         8.12 GiB       11.21 GiB  +     3.09 GiB
  Baseline at start          39.61 GiB       52.77 GiB  +    13.15 GiB

--- Full-Trace Summary (no ProfilerStep annotations) ---
                                           Snapshot A      Snapshot B      Delta(B-A)
  ───────────────────────────────────  ──────────────  ──────────────  ──────────────
  Peak delta (from trace start)              9.65 GiB       19.15 GiB  +     9.50 GiB
  Absolute peak (baseline+delta)            49.26 GiB       71.92 GiB  +    22.65 GiB
  Alloc count                                     760           1,840  +        1,080

--- Drill-Down: Full Trace (peak diff: +9.50 GiB) ---
  Source                                            Peak A        Peak B    Delta(B-A)
  ────────────────────────────────────────────  ──────────  ──────────  ────────────
  router_forward@router.py:145                    1.81 GiB    4.51 GiB  +   2.70 GiB
  mlp_forward@mlp.py:301                          1.59 GiB    4.28 GiB  +   2.69 GiB
  attention_forward@attention.py:88               1.77 GiB    3.99 GiB  +   2.21 GiB
```

## Interpreting Results

All sizes are **binary** — KiB/MiB/GiB, 1024-based — so they line up directly
with `nvidia-smi` and `torch.cuda.memory_allocated`. A decimal "GB" reads about
7.4% larger for the same bytes, which is enough to make a correct cross-check
look like a discrepancy.

### Segments vs device traces

`segments` is the allocator's state at dump time — use it for total memory
accounting. `device_traces` is the event log — use it to understand what changes
over time. The two are cross-referenced to derive `baseline_at_start`, which is
what turns relative deltas into real GPU memory figures. A timeline alone cannot
tell you how much memory the model was actually using.

### Reserved, allocated, active, inactive

The four segment numbers are distinct and easy to conflate:

| Term | Meaning |
|---|---|
| Reserved (total) | Memory the allocator holds from the driver |
| Allocated | Blocks currently handed out to tensors |
| Active | Allocated **plus** `active_awaiting_free` — freed by the caller but still held pending stream sync |
| Inactive (reusable) | `reserved - active`; free blocks the allocator can hand out again |

Fragmentation is reported as `inactive / reserved`. It is deliberately computed
from *active*, not *allocated*: awaiting-free blocks are not reusable yet and
are not fragmentation, so counting them as such over-reports the problem. When
`Awaiting free` is non-zero, expect the two definitions to disagree.

### Absolute peak

`absolute_peak = baseline_at_start + cumulative_delta_at_step_start + step_peak_delta`.
Compare this against `nvidia-smi` or the `memory/` metrics Bridge logs to
TensorBoard to sanity-check that the trace covers what you think it covers.

Per-step replays seed their live set with whatever was already live when the
step opened, so the source table accounts for memory carried into the step
(weights, optimizer state, graph pools, activations held across the boundary)
and not just what the step itself allocated. On a real annotated capture this
recovered about 3.4 GiB per step that the table previously omitted.

**The table will not sum to `absolute_peak`, and the gap is not a single clean
quantity.** The table shows allocations the trace can see and attribute to a
stack frame; `absolute_peak` also includes memory that predates the trace and
has no frames, and the two are related through `step_start_delta`, which nets
frees of pre-trace memory against later allocations. Treat the table as "what
is attributable and how it ranks", not as a decomposition of the peak. Use the
segments numbers for total accounting.

### Unmatched frees

`free_completed` events with no matching `alloc` are allocations that predate
the visible trace. A small count is normal and harmless. A *large* count means
the trace does not reach back to the allocations it is freeing — either the
100k cap dropped the beginning of the run, or recording started after model
construction (see point 2 above). Either way, `baseline_at_start` is doing more
guessing than measuring, so cross-check it against `nvidia-smi` before trusting
absolute numbers.

### Source grouping

Allocations are grouped by their first non-internal stack frame, rendered as
`function@file.py:line`. PyTorch, CUDA, Hydra, and stdlib frames are skipped.
When results look over-aggregated (everything attributed to one wrapper), raise
`--frame-depth 2` to split by two levels of user code.

### CUDA graphs

A snapshot with dramatically fewer allocs indicates CUDA graph replay. Graphs
pre-allocate at capture time and reuse on replay, so those allocations stop
appearing in `device_traces`. The memory did not disappear — it moved into the
graph's pinned pool, which is why enabling graphs can *raise* peak memory even
as the trace gets quieter.

A matched pair of Bridge captures — same model and config, graphs off vs on —
shows both halves of this:

| | Graphs off | Graphs on |
|---|---|---|
| Alloc count | 281,949 | 62,271 |
| Total alloc throughput | 23,683 GiB | 7,682 GiB |
| Absolute peak | 85.89 GiB | 119.80 GiB |

The trace got 4.5x quieter and peak memory went *up* 34 GiB. Running
`compare_snapshots.py` on the pair attributes most of that increase to the graph
memory pool rather than to any model tensor — which is the signal you want, and
the reason to diff rather than eyeball a single file. See
@skills/nemo-mbridge-perf-cuda-graphs/SKILL.md.

## Options

| Flag | Scripts | Description |
|---|---|---|
| `--frame-depth N` | compare, replay_step, replay_to_time | Stack frames per source key (default 1) |
| `--top N` | compare, replay_step, replay_to_time | Top sources to show |
| `--step N` | compare, replay_step | Specific step; compare defaults to largest divergence |
| `--all-steps` | replay_step | Replay every complete step |
| `--time T` | replay_to_time | Elapsed seconds from trace start |
| `--labels L1 L2` | plot_timeline | Trace display labels (default: filenames) |
| `--output PATH` | plot_timeline | Output HTML path |
| `--precision MS` | plot_timeline | Wall-clock bucket width in ms (default 1.0) |
| `--device D` | replay_step | Force device index (default: auto-detect) |
| `--json` | all except plot_timeline | Machine-readable output |

Diagnostics go to stderr via `logging`; report bodies go to stdout, so `--json`
output pipes cleanly.

## Troubleshooting

**"no ProfilerStep annotations found"** — Expected for a Bridge run with
`record_memory_history=True` and no PyTorch profiler, and not a sign of a broken
snapshot: the file still has NCCL and optimizer annotations, just no step
boundaries. Use `compare_snapshots.py` (auto-falls back), `plot_timeline.py`, or
`replay_to_time.py`. Add `use_pytorch_profiler=True` on the next run if you need
per-step analysis.

**"no device traces found"** — The pickle has only `segments`. Either
`record_memory_history` was False, or the snapshot came from somewhere other
than Bridge's capture path. `parse_snapshot.py` still summarizes segments.

**Numbers don't match `nvidia-smi`** — Check the unmatched-free count first; a
wrapped ring buffer makes `baseline_at_start` meaningless. Also confirm you are
looking at the rank that actually peaked — with pipeline parallelism the first
PP stage usually holds the most memory, and `profile_ranks` defaults to `[0]`.

**Steps don't line up between two snapshots** — Matching is by step number, so
different warmup counts pair the wrong steps. Force alignment with `--step N`.

**Large files** — The whole pickle is deserialized into memory; budget roughly
10-20x the file size in RAM. Loading progress is logged to stderr.
