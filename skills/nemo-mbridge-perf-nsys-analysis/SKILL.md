---
name: nemo-mbridge-perf-nsys-analysis
description: Analyze NVIDIA Nsight Systems `.nsys-rep` and exported `.sqlite` traces for Megatron Bridge training. Use for single-trace diagnosis, before/after comparisons, multi-rank surveys, slow-rank and pipeline-stage analysis, MFU or step-time investigations, GPU busy/compute-absent accounting, communication overlap and rank-jitter analysis, CUDA launch starvation, CPU offload or memcpy investigations, source-level attribution, and evidence-backed gain estimates. Do not use as a substitute for Nsight Compute kernel roofline analysis.
---

# Nsight Systems Analysis for Megatron Bridge

Turn an Nsight Systems trace into a critical-path diagnosis and a ranked,
evidence-backed optimization plan. Treat the profile as timing evidence, not as
a list of long kernels to optimize.

Use these principles throughout:

- Use the performance model as the ceiling and the profiler as the agenda.
- Compute wall time with interval unions, never summed stream or kernel time.
- Treat visible communication and copies as coincident work until a dependency
  proves that they blocked compute.
- Separate a measured phenomenon from its source-level root cause.
- Report a gain ceiling from recoverable critical-path time; do not invent a
  likely gain without comparable measured evidence.

## Establish the analysis contract

Record before calculating:

1. Trace path, rank id, hostname, GPU, Nsight Systems version, Bridge and MCore
   revisions, container, and whether CUDA graphs were enabled.
2. Model, precision, sequence length, micro/global batch sizes, task, and the
   measured step definition.
3. TP, PP, VPP, CP, EP, ETP, and DP sizes plus the rank-to-stage mapping.
4. The steady-state iteration range and the number of analyzed iterations.
5. Whether source for the exact profiled revision is available.

Never infer a rank id or pipeline stage from a filename. Never compare ranks
that hold different model parts as if they performed identical work.

Preserve the input report. Export a separate SQLite file when necessary:

```bash
nsys export --type sqlite -o profile.sqlite profile.nsys-rep
```

Check the available tables before using a query because schemas vary by Nsight
version. Read [references/sql-recipes.md](references/sql-recipes.md) for
portable inspection queries and [references/pitfalls.md](references/pitfalls.md)
before making a bottleneck claim.

## Choose the workflow

- **One implementation, one rank:** produce an absolute time budget and ranked
  optimization opportunities.
- **One implementation, several ranks:** survey every supplied rank before
  selecting representatives; diagnose stage imbalance, jitter, and the
  slowest-rank step.
- **Two implementations:** verify identical workload and topology, then
  reconcile their median per-iteration delta.

For distributed runs, declare representatives rather than selecting them
silently. Cover at least the slowest trainer-step rank and one representative
for every distinct PP stage or rank fingerprint relevant to the question. Rank
0 is not automatically representative. If the mapping is unavailable, report
the rank survey first and label any provisional selection.

## Run the analysis

### 1. Establish iteration windows

Prefer an NVTX range that exactly matches the user's step definition. Otherwise
select a recurring anchor whose count is stable per iteration:

1. a repeated optimizer or train-step range;
2. a repeated NCCL collective sequence;
3. optimizer kernels;
4. a stable recurring kernel sequence.

Drop capture warmup, graph capture, and cooldown iterations. Report median,
minimum, maximum, and `n`. Cross-check with a second anchor when possible. Stop
and state the ambiguity when candidate anchors imply materially different step
times.

For a comparison, require matching iteration semantics, model inputs, batch
shape, precision, topology, and capture mode. Describe mismatches before
showing a delta.

### 2. Survey ranks before drilling down

For every supplied rank, report:

- median/min/max iteration time and `n`;
- non-communication busy time;
- compute, collective, memcpy, and idle fingerprints;
- collective types and counts;
- pipeline stage or model part when known.

Group ranks only when their fingerprints and model parts match. Within each
declared part, report the fastest and slowest rank, the spread, and whether the
same rank is repeatedly slow or the straggler rotates.

Relate timestamps from different exports only after rebasing with
`TARGET_INFO_SESSION_START_TIME`. Same-host clocks can be compared after this
rebase. Across hosts, refine only with matched collective end timestamps and
state the remaining alignment error. Duration-based spread remains usable when
cross-host arrival time cannot be established.

### 3. Build the per-iteration device budget

Clip all intervals to each iteration window and compute interval unions across
all streams:

| Metric | Definition |
|---|---|
| Device busy | Union of kernels, memcpy, and memset |
| Device idle | Iteration minus device busy |
| Non-transfer busy | Union after removing NCCL and HtoD/DtoH copies |
| Compute busy | Union of compute kernels only |
| Compute-absent | Iteration minus compute busy |
| Occupied-not-computing | Device busy minus compute busy |

Require `compute busy <= non-transfer busy <= device busy <= iteration`.
Reconcile `device busy + device idle` to iteration time. Show kernel-duration
sums only as work volume and label them explicitly; never present them as wall
time.

### 4. Attribute compute-absent time

Split each interval with no compute on any stream into:

- **Launch-starved:** the next compute kernel had not been issued by the host.
- **Blocking:** the kernel was issued and a resolved producer finished after
  the gap began; require a device dependency and negative slack.
- **Dependency-stalled:** the kernel was issued, but no resolved producer
  explains the remaining delay.

Follow relayed event dependencies to the operation that produced the awaited
event. Classify that operation, not its stream. Join CUDA event records using
`eventSyncId`, not the reused `eventId` handle. Report unresolved wait share and
whether the blocking measurement is an upper or lower bound.

CUDA graph replay may give many kernels one launch and omit per-kernel launch
rows. In that case, describe the missing dependency resolution and avoid a
false host-launch or call-site conclusion.

Verify:

```text
launch-starved + blocking + dependency-stalled ~= compute-absent
```

Investigate a residual above 0.5 ms per iteration instead of absorbing it into
the largest bucket.

### 5. Analyze communication and overlap

Report three different quantities:

1. **Communication volume:** collective count and total device work.
2. **Exposed communication:** NCCL interval union not overlapped by any
   non-NCCL device operation. Treat it as an upper bound.
3. **Blocking communication:** compute-absent time whose resolved dependency
   producer is a collective. Use this as the recoverable critical-path bound.

Never call a collective a bottleneck from NCCL duration or exposed time alone.
Report exposed and blocking values together.

When all collective participants are available, match instances and split a
rank's collective residence into:

- **Transfer proxy:** the shortest participant duration for the instance.
- **Jitter wait:** this rank's duration minus that proxy.

Report both together with matched/unmatched instance counts. If jitter wait
dominates, pursue the late participant or load imbalance rather than network
bandwidth. If only one participant is present, do not make a bandwidth claim.

Calculate overlap as an observed timeline property, not as proof of causality:

```text
overlapped_comm = total_comm_union - exposed_comm
overlap_pct = overlapped_comm / total_comm_union
```

Use blocking communication, available independent compute, and matched A/B
measurements to judge whether an overlap knob can help.

### 6. Classify compute and transfers

Break compute work into at least GEMM, attention, normalization, elementwise or
fused, optimizer, and other. Keep NCCL, memcpy, and memset separate. Inspect
every regex-matched kernel above 1% of iteration time and list material
unclassified kernels. Treat observed exact names as stronger evidence than
regexes.

For copies, report direction, bytes, stream, occupancy, achieved bandwidth,
and pinned versus pageable memory when present. Require a saturated engine or a
blocking dependency before claiming a copy-bandwidth bottleneck.

Use Nsight Compute for kernel-level SOL, instruction, occupancy, or memory
roofline conclusions. Nsight Systems shows placement and duration, not why an
individual kernel underuses the GPU.

### 7. Use metrics without overclaiming

Prefer a separate, short metric-sampling capture over representative ranks at
100 kHz. Keep timing and metrics passes separate because sampling greatly
increases trace size. Join them by operator name, not individual launch.

For every utilization number, report sample count and exclusivity—the share of
samples in which only that operator was resident. Mark low-exclusivity rows as
contaminated instead of correcting them.

- Use `SM Issue` and `Tensor Active` as compute-throughput evidence.
- Use DRAM read/write throughput for memory pressure.
- Use NVLink **response** user-data throughput for interconnect saturation.
- Use `SMs Active` only as residency context, not compute throughput.
- Treat an unsampled operator as unknown, never zero.

If GPU metrics are absent or permission is denied, continue with timing
analysis and state the limitation.

### 8. Link kernels to code carefully

Prefer source evidence from the exact revision. Nsight Runtime API correlation
can connect a kernel to a CUDA API launch but often not to a Python or framework
call site.

If a separate CUDA call-stack capture is available, correlate launches by
`(rank, thread, launch ordinal)` against the same workload, configuration, ROI,
and declared ranks. Do not capture call stacks under Nsight Systems. Report the
link rate beside every call-site-derived figure. A missing capture means
regex-only taxonomy and no source-scope claim; zero links does not mean zero
launches, especially with CUDA graphs.

For each root cause, provide:

1. the measured trace phenomenon;
2. the source/config mechanism that creates it;
3. the exact MBridge knob or code anchor to change;
4. status: `trace-verified`, `source-verified`, `inferred`, or `unverified`.

Without source, stop at trace phenomena and proposed verification steps.

### 9. Estimate gain without double counting

Rank opportunities by recoverable critical-path milliseconds per iteration.
Use one of these evidence bases:

- matched before/after median delta on the same workload;
- launch-starved time affected by a verified launch-reduction mechanism;
- blocking time attributed through a dependency edge;
- a non-overlapping, source-verified critical-path interval.

For current step time `T` and recoverable bound `R`:

```text
new_step_ceiling = T - R
throughput_gain_ceiling_pct = (T / (T - R) - 1) * 100
new_mfu_ceiling = old_mfu * T / (T - R)
```

Use the MFU formula only when the algorithmic numerator and precision are
unchanged. Call these ceilings, not forecasts. Give a likely gain only when a
comparable measured implementation supports it, and cite that measurement.
Do not add opportunity bounds unless their intervals are disjoint and their
fixes are independent.

## Map findings to MBridge actions

| Proven dominant cost | Next skill or action |
|---|---|
| Launch-starved | `@skills/nemo-mbridge-perf-cuda-graphs/SKILL.md`; inspect CPU affinity, Python work, GC, and microbatch size |
| Blocking TP/DP/PP communication | `@skills/nemo-mbridge-perf-tp-dp-comm-overlap/SKILL.md` |
| Blocking MoE dispatch/combine | `@skills/nemo-mbridge-perf-moe-comm-overlap/SKILL.md` and dispatcher selection |
| Long-context communication/layout | `@skills/nemo-mbridge-perf-hierarchical-context-parallel/SKILL.md` and parallelism strategies |
| Blocking HtoD/DtoH | `@skills/nemo-mbridge-perf-cpu-offloading/SKILL.md`; inspect prefetch distance and pinned memory |
| Memory-constrained layout | `@skills/nemo-mbridge-perf-memory-tuning/SKILL.md` |
| Low kernel SOL | Capture Nsight Compute and inspect precision, fusion, shape, and kernel choice |
| Rank jitter | Inspect slow rank, topology, data imbalance, CPU affinity, and straggler telemetry |

Enable knobs only after the corresponding critical-path cost is established.

## Output contract

Return sections in this order:

1. **Verdict:** one paragraph naming the dominant cost and trace limitations.
2. **Capture quality:** inputs, ranks/stages, iteration anchor, `n`, CUDA graph
   state, source availability, metric availability, and call-stack link rate.
3. **Per-step budget:** iteration, device busy/idle, compute busy/absent, and
   launch-starved/blocking/dependency-stalled reconciliation.
4. **Rank and communication findings:** transfer proxy, jitter wait, exposed
   comm, blocking comm, overlap, and straggler verdict where supported.
5. **Ranked opportunities:** evidence, recoverable bound, throughput/MFU
   ceiling, confidence, action, and verification step.
6. **Claim status:** distinguish source-verified facts from trace inference.

State all times per iteration and include `n`. Include an uncertainty range or
the observed min/max when estimating a gain.

## Design note

Keep this workflow standalone. It does not require an external orchestrator or
proprietary trace-processing service; use standard Nsight Systems exports and
the evidence available for the profiled Megatron Bridge revision.
