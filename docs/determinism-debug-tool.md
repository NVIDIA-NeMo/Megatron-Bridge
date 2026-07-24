# Determinism Debug Tool

A low-overhead tracer for locating the **first** point where two training runs
that *should* be identical diverge — e.g. the same deterministic recipe that
reproduces at 24 nodes but breaks at 48. It answers one question: *which
operation, in execution order, is the first whose output differs between two
runs?* Everything before that point matched, so that op (or the collective it
feeds) is the root cause.

> Status: debugging aid, env-gated and inert unless explicitly enabled. Not a
> training feature — it adds overhead and is meant for isolated repro runs.

## When to use it

- A run is internally reproducible but two launches disagree (topology- or
  reduction-order-dependent non-determinism — the classic multi-node break).
- A determinism flag "should" make a run bit-exact but doesn't, and you need to
  see *where* the first byte diverges rather than guessing.
- You need per-layer / per-collective attribution of a divergence, not just a
  final-loss mismatch.

## High-level design

Two divergence classes, one tool:

- **Class A (same process):** re-run inside one job; non-determinism from
  kernels/algorithms. An independent oracle (e.g. `torch.use_deterministic_algorithms`)
  is the check.
- **Class B (cross process):** two separate launches disagree — the topology-
  dependent floating-point reduction-order bug. This is the tool's main target.

The tracer records an ordered **fingerprint stream per logical rank**, then an
offline diff aligns the two runs and reports the first divergence. Three capture
layers stack from cheap/coarse to detailed:

| Layer | File | Captures | Cost |
|-------|------|----------|------|
| **Collectives** | `collective_trace.py` | Every `torch.distributed` reduce/gather/all-to-all (input + output signature), incl. Megatron's import-time-captured SP refs | low — default |
| **Ops** | `op_trace.py` | Every ATen op output (via `TorchDispatchMode`) — names the exact compute op that first diverges | higher — opt-in |
| **Layers** | `module_scope.py` | Pushes the enclosing module name onto each record so a divergence reads `decoder.layers.7.mlp` not `aten.view` | negligible |

Supporting pieces:

- **`signature.py` — the fingerprint.** A 64-bit digest from the PyTorch-native
  `torch.hash_tensor`, computed **on the tensor's own device** and returned as a
  `uint64` *tensor* — so the tracer **stages** it (keeps the GPU scalar) and defers
  the single `.item()` to the step boundary; nothing but an 8-byte scalar crosses
  the PCIe bus, and never mid-iteration. `hash_tensor` upcasts each element to its
  64-bit equivalent and **xor-reduces**, so the digest is **order-independent →
  identical across processes, GPUs, and topologies** — the property a cross-job key
  needs — and sensitive to single-element (1-ULP) changes. (Trade-off: xor reduction
  is a strong *screen* for value divergence, not permutation-sensitive or
  collision-proof; `shape`/`dtype` are compared alongside the digest.)
- **Logical-coordinate alignment.** Streams are keyed by the *minimal unique
  GPU coordinate* `(pp, tp, cp, dp)`, not physical rank — so the same logical
  GPU lines up between two jobs even when it lands on different nodes. Records
  align on `(window, group, op, align_idx)`.
- **`diff_streams.py` — offline diff.** Walks job A in execution order, finds the
  first record whose *output* signature differs from job B's. If that record's
  *inputs* matched, the op itself is the reduction-order root cause; otherwise the
  cause is upstream. Refuses to compare two runs with different parallel config.

## How to use

The tracer is wired into the training loop and gated entirely on env vars —
unset, it does nothing.

| Env var | Meaning | Default |
|---------|---------|---------|
| `DET_TRACE_OUT_DIR` | Directory for per-rank `.fp` streams (use a **shared FS** for multi-node). Setting it enables the tracer. | unset (off) |
| `DET_TRACE_ITERS` | Iterations to capture: `all`, a range `a-b`, and/or a list `1,5,40-44`. | `1` |
| `DET_TRACE_OPS` | Also fingerprint every ATen op output (heavier; pinpoints the exact compute op). | off |
| `DET_TRACE_BWD_SCOPE` | Add `[bwd]` layer labels. **PP-incompatible** (backward hooks alias outputs → break pipeline deallocation) — TP-only debugging. | off |

Workflow:

1. **Run job A** with `DET_TRACE_OUT_DIR=/lustre/.../streams/A` (+ the iters you
   want). Then **run job B** identically with `.../streams/B`.
2. **Diff them:**
   ```bash
   python src/megatron/bridge/training/utils/determinism/diff_streams.py \
       /lustre/.../streams/A  /lustre/.../streams/B
   ```
3. **Read the verdict** — either every logical rank reports `OK — all outputs
   match`, or:
   ```
   [pp0_tp0_cp0_dp3] FIRST DIVERGENCE: reduce_scatter_tensor on group 'dp'
       (window=7, align_idx=12, seq_a=418)
       layer=decoder.layers.7.mlp  caller_a=...  caller_b=...
       — INPUTS MATCHED → reduction-order/topology root cause
   ```

Two-stage strategy for cost: trace **collectives only** first (cheap, catches
cross-process reduction-order breaks — the usual multi-node bug). Only add
`DET_TRACE_OPS=1` on the narrowed iteration window if you need to see *inside*
the compute of the divergent layer.

### Multi-node notes

- Streams must land on a shared filesystem so `diff_streams` sees both jobs.
- The tracer force-syncs each collective so the captured output is complete;
  this removes comm/compute overlap, so **expect the traced run to be slower** —
  trace a bounded iteration window rather than a full job, and raise
  `distributed_timeout_minutes` so the slower iterations don't trip the NCCL
  watchdog.

## Limitations

- **Hybrid-EP / DeepEP dispatch is a blind spot.** The `flex` MoE dispatcher's
  token dispatch/combine runs in a custom CUDA/RDMA extension (`fused_a2a` /
  `hybrid_ep_cpp`), which is neither a `torch.distributed` call nor an ATen op,
  so it is not captured directly (a divergence there surfaces one hop later, at
  the first ATen op that consumes the dispatched tokens). The plain `alltoall`
  dispatcher is fully covered (its exchange is `all_to_all_single`).
- `DET_TRACE_BWD_SCOPE` cannot be used with pipeline parallelism.
- `force_sync` trades throughput for complete value capture — this is a debug
  tool, not a training-time feature.
