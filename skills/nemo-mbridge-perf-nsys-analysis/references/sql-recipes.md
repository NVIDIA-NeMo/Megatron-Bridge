# Nsight Systems SQLite recipes

Use these as inspection starting points. Inspect `.schema <table>` first and
adapt column names to the installed Nsight Systems version. Durations are
nanoseconds unless the schema says otherwise.

## Contents

- [Export a report](#export-a-report)
- [Discover tables and metadata](#discover-tables-and-metadata)
- [Capture diagnostics and recorded profiler activity](#capture-diagnostics-and-recorded-profiler-activity)
- [Top kernels by work volume](#top-kernels-by-work-volume)
- [Distributed backend kernel inventory](#distributed-backend-kernel-inventory)
- [Per-stream work volume and span](#per-stream-work-volume-and-span)
- [Runtime API and launch correlation](#runtime-api-summary)
- [Synchronization and dependencies](#synchronization-and-device-dependency-edges)
- [Memcpy volume and bandwidth](#memcpy-volume-and-bandwidth)
- [Recurring collective anchors](#recurring-collective-anchors)
- [Cross-rank timestamp alignment](#rebase-two-rank-exports)
- [Metric inventory](#metric-inventory)

## Export a report

```bash
nsys export --type sqlite -o profile.sqlite profile.nsys-rep
```

## Discover tables and metadata

```sql
.tables
SELECT sqlite_version();
SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name = 'Hostname';
SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME;
```

## Capture diagnostics and recorded profiler activity

Inspect warnings before trusting absence, scheduling, or source attribution.
These tables are version-dependent; query `sqlite_master` before using them.

```sql
SELECT
  severity.label AS severity,
  source.label AS source,
  d.text
FROM DIAGNOSTIC_EVENT AS d
LEFT JOIN ENUM_DIAGNOSTIC_SEVERITY_LEVEL AS severity
  ON severity.id = d.severity
LEFT JOIN ENUM_DIAGNOSTIC_SOURCE_TYPE AS source
  ON source.id = d.source
WHERE d.severity > 1
ORDER BY d.timestamp;
```

Summarize activity that Nsight recorded as profiler overhead:

```sql
SELECT
  COALESCE(s.value, '<unknown>') AS activity,
  e.label AS overhead_type,
  COUNT(*) AS events,
  SUM(p.end - p.start) / 1e6 AS work_ms
FROM PROFILER_OVERHEAD AS p
LEFT JOIN StringIds AS s ON s.id = p.nameId
LEFT JOIN ENUM_CUPTI_OVERHEAD_TYPE AS e ON e.id = p.overheadType
GROUP BY activity, overhead_type
ORDER BY work_ms DESC;
```

This is recorded profiler activity, not total training slowdown. Calculate
end-to-end perturbation from same-run profiled and unprofiled trainer steps.

## Top kernels by work volume

This is summed device work, not wall time.

```sql
SELECT
  COALESCE(s.value, '<unknown>') AS kernel,
  COUNT(*) AS launches,
  SUM(k.end - k.start) / 1e6 AS work_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN StringIds AS s ON s.id = k.demangledName
GROUP BY kernel
ORDER BY work_ms DESC
LIMIT 40;
```

## Distributed backend kernel inventory

Do not stop at NCCL. Inventory exact material names from the configured
dispatcher or transport before defining a communication union.

```sql
SELECT
  COALESCE(s.value, '<unknown>') AS kernel,
  COUNT(*) AS launches,
  SUM(k.end - k.start) / 1e6 AS work_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN StringIds AS s ON s.id = k.demangledName
WHERE LOWER(s.value) LIKE '%nccl%'
   OR LOWER(s.value) LIKE '%hybrid_ep%'
   OR LOWER(s.value) LIKE '%hybridep%'
   OR LOWER(s.value) LIKE '%deepep%'
   OR LOWER(s.value) LIKE '%nvshmem%'
   OR LOWER(s.value) LIKE '%dispatch%'
   OR LOWER(s.value) LIKE '%combine%'
   OR LOWER(s.value) LIKE '%rdma%'
   OR LOWER(s.value) LIKE '%device_sync%'
GROUP BY kernel
ORDER BY work_ms DESC;
```

Inspect every material match and material unclassified kernel. Split NCCL,
backend transport/synchronization, and packing/routing/control where evidence
allows. A dispatch/combine union is dispatcher occupancy, not automatically
network transfer.

## Per-stream work volume and span

Do not call `work_ms` or `span_ms` wall time. Compute interval unions outside
SQL before comparing them with an iteration.

```sql
SELECT
  streamId,
  COUNT(*) AS launches,
  SUM(end - start) / 1e6 AS work_ms,
  (MAX(end) - MIN(start)) / 1e6 AS span_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY streamId
ORDER BY work_ms DESC;
```

## Runtime API summary

```sql
SELECT
  COALESCE(s.value, '<unknown>') AS api,
  COUNT(*) AS calls,
  SUM(r.end - r.start) / 1e6 AS host_api_ms,
  AVG(r.end - r.start) / 1e3 AS avg_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
LEFT JOIN StringIds AS s ON s.id = r.nameId
GROUP BY api
ORDER BY host_api_ms DESC;
```

Intersect API intervals with device-idle unions before calling host API time a
stall. A synchronization call may correctly wait while the GPU is busy.

## Launch-to-start queueing

```sql
SELECT
  k.streamId,
  (k.start - r.end) / 1e3 AS launch_to_start_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
  ON r.correlationId = k.correlationId
ORDER BY launch_to_start_us;
```

Validate that the join is one-to-one. CUDA graph nodes commonly share a launch
and cannot be interpreted as independent host dispatches.

Measure kernel-to-CUDA-runtime correlation separately:

```sql
WITH runtime_correlations AS (
  SELECT DISTINCT correlationId
  FROM CUPTI_ACTIVITY_KIND_RUNTIME
)
SELECT
  COUNT(*) AS kernel_launches,
  COUNT(r.correlationId) AS runtime_correlated_launches
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN runtime_correlations AS r USING (correlationId);
```

This join does not measure source call-stack coverage. Report source links only
from an independently captured and validated call-stack dataset.

## Synchronization and device dependency edges

Join event records by `eventSyncId`, not `eventId`.

```sql
SELECT
  y.start AS wait_timestamp,
  y.streamId AS waiting_stream,
  e.streamId AS recording_stream,
  waiting_name.value AS waiting_api,
  recording_name.value AS recording_api
FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION AS y
JOIN CUPTI_ACTIVITY_KIND_CUDA_EVENT AS e
  ON e.eventSyncId = y.eventSyncId
LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS waiting_runtime
  ON waiting_runtime.correlationId = y.correlationId
LEFT JOIN StringIds AS waiting_name
  ON waiting_name.id = waiting_runtime.nameId
LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS recording_runtime
  ON recording_runtime.correlationId = e.correlationId
LEFT JOIN StringIds AS recording_name
  ON recording_name.id = recording_runtime.nameId
WHERE y.syncType = 2
ORDER BY y.start;
```

Reconstruct the producer completion from device stream order. Do not use
`CUPTI_ACTIVITY_KIND_CUDA_EVENT.timestamp`; it may be zero.

## Memcpy volume and bandwidth

```sql
SELECT
  copyKind,
  streamId,
  COUNT(*) AS copies,
  SUM(bytes) AS bytes,
  SUM(end - start) / 1e6 AS work_ms,
  SUM(bytes) / 1e9 / (SUM(end - start) / 1e9) AS achieved_GBps
FROM CUPTI_ACTIVITY_KIND_MEMCPY
GROUP BY copyKind, streamId
ORDER BY work_ms DESC;
```

Check pinned versus pageable memory:

```sql
SELECT
  src.label AS src_kind,
  dst.label AS dst_kind,
  COUNT(*) AS copies,
  SUM(m.end - m.start) / 1e6 AS work_ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
LEFT JOIN ENUM_CUDA_MEM_KIND AS src ON src.id = m.srcKind
LEFT JOIN ENUM_CUDA_MEM_KIND AS dst ON dst.id = m.dstKind
GROUP BY src_kind, dst_kind
ORDER BY work_ms DESC;
```

## Recurring collective anchors

```sql
SELECT
  k.start,
  k.end,
  COALESCE(s.value, '<unknown>') AS kernel
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN StringIds AS s ON s.id = k.demangledName
WHERE s.value LIKE '%AllGather%'
   OR s.value LIKE '%ReduceScatter%'
   OR s.value LIKE '%AllReduce%'
   OR s.value LIKE '%AllToAll%'
ORDER BY k.start;
```

Use only a recurring sequence with stable count and position per iteration.

## Rebase two rank exports

Add each report's `TARGET_INFO_SESSION_START_TIME.utcEpochNs` to raw activity
timestamps before comparing ranks. Match collective counts and order first.

```sql
ATTACH 'rank1.sqlite' AS rank1;

WITH base0 AS (
  SELECT utcEpochNs AS epoch FROM main.TARGET_INFO_SESSION_START_TIME
), base1 AS (
  SELECT utcEpochNs AS epoch FROM rank1.TARGET_INFO_SESSION_START_TIME
), rank0_ops AS (
  SELECT ROW_NUMBER() OVER (ORDER BY k.start) AS seq,
         k.start + (SELECT epoch FROM base0) AS start_ns,
         k.end + (SELECT epoch FROM base0) AS end_ns
  FROM main.CUPTI_ACTIVITY_KIND_KERNEL AS k
  JOIN main.StringIds AS s ON s.id = k.demangledName
  WHERE s.value LIKE '%AllReduce%'
), rank1_ops AS (
  SELECT ROW_NUMBER() OVER (ORDER BY k.start) AS seq,
         k.start + (SELECT epoch FROM base1) AS start_ns,
         k.end + (SELECT epoch FROM base1) AS end_ns
  FROM rank1.CUPTI_ACTIVITY_KIND_KERNEL AS k
  JOIN rank1.StringIds AS s ON s.id = k.demangledName
  WHERE s.value LIKE '%AllReduce%'
)
SELECT
  rank0_ops.seq,
  (rank0_ops.start_ns - rank1_ops.start_ns) / 1e3 AS start_delta_us,
  (rank0_ops.end_ns - rank1_ops.end_ns) / 1e3 AS end_delta_us
FROM rank0_ops
JOIN rank1_ops USING (seq)
ORDER BY rank0_ops.seq;
```

Across hosts, match collective name, order, and participant group—not just a
single global ordinal. Use many matched collective end deltas to estimate a
constant clock offset with a robust center such as the median, then report
residual p05/p95 or another error interval after correction. Compare aligned
starts and ends separately. Do not call a longer window a straggler when it
started earlier and ended with its peer, and do not claim exact arrival lateness
when alignment is weak.

## Metric inventory

Resolve metrics by name because ids change between reports.

```sql
SELECT metricId, typeId, metricName
FROM TARGET_INFO_GPU_METRICS
ORDER BY metricId, typeId;
```

Consume only established throughput metrics. Join metric samples to operation
windows by timestamp and device. Report sample count and exclusivity because
samples are device-wide and carry no stream or correlation id.
