# Hybrid / Hierarchical Context Parallel

This page covers the stable Bridge-facing meaning of hierarchical context
parallelism, especially the `a2a+p2p` transport path and
`hierarchical_context_parallel_sizes`.

For operational setup, code anchors, and verification commands, see
`skills/perf-techniques/hybrid-context-parallel.md`.

## What It Is

Context parallelism (CP) splits the input sequence across GPUs so each rank
processes a chunk. The GPUs must communicate KV data during attention. There are
several CP communication backends:

| `cp_comm_type` | Mechanism | Async / Overlap | Constraint |
|---|---|---|---|
| `"p2p"` | Ring-exchange of KV chunks | Yes | None |
| `"all_gather"` | All-gather full KV before attention | No | None |
| `"a2a"` | All-to-all: scatter heads, gather full sequence (Ulysses-style) | N/A | **CP <= num_kv_heads** |
| `"a2a+p2p"` | Hierarchical: a2a within inner group, p2p across outer group | Partial (p2p part) | Requires `hierarchical_context_parallel_sizes` |

**HCP (`a2a+p2p`)** exists to scale CP beyond the KV head count by combining
a2a (fast, head-parallel) on intra-node links with p2p (async,
sequence-parallel) on inter-node links.

It is important to separate this from the upstream boolean
`hybrid_context_parallel`, which is a different feature for balancing packed or
variable-length workloads. The two concepts should not be treated as
interchangeable.

### Why a2a is limited by KV heads

a2a transposes the parallelism dimension: each rank trades its sequence chunk
for a subset of attention heads. After the all-to-all, every rank has the
**full sequence** but only `heads / CP` heads. This means:

- `heads / CP` must be a positive integer.
- The bottleneck is KV heads (not Q heads), because in GQA the KV heads are the
  indivisible unit.
- If the model has 8 KV heads, pure a2a supports at most CP=8.

HCP breaks this limit by applying a2a only within a sub-group small enough to
fit within the KV head count.

## When to Use It

**Use HCP when ALL of these are true:**

1. You need CP larger than `num_kv_heads / TP` (pure a2a won't fit).
2. You cannot (or don't want to) increase TP to shrink CP.
3. Your cluster has a clear bandwidth hierarchy (e.g., NVLink intra-node >> IB
   inter-node).

**Prefer pure `a2a` when:**

- You can adjust TP so that `CP <= num_kv_heads / TP`. This is simpler, avoids
  the p2p overhead, and often yields the same throughput with better memory
  headroom.

**Prefer pure `p2p` when:**

- You have very few KV heads or want maximum CP flexibility.
- Your workload can hide the p2p latency behind compute (long sequences help).

### Decision example

Model: 8 KV heads. Cluster: 4 nodes x 8 GPUs. Goal: train 128K sequences.

| Option | TP | CP | `cp_comm_type` | Notes |
|---|---|---|---|---|
| A | 1 | 16 | `a2a+p2p` with `[8,2]` | a2a intra-node (8 GPUs), p2p across 2 node-groups |
| B | 2 | 4 | `a2a` | CP=4 <= 8 KV heads. Simpler. Often same throughput. |
| C | 1 | 16 | `p2p` | Works but no a2a bandwidth benefit intra-node |

In practice, **option B is usually preferred** -- benchmarks showed identical
throughput to option A with more memory headroom.

It should be treated as an advanced feature rather than a default recommendation.

## Stable Bridge Limitation

The most important Bridge-specific limitation is that hierarchical context
parallelism is currently supported only on the MPU initialization path.

In practice, that means:

- `dist.use_decentralized_pg=False` is the supported Bridge path
- the decentralized process-group path should not be assumed to materialize HCP
  groups

## Stable Constraints

The durable constraints are:

- `hierarchical_context_parallel_sizes` must match
  `context_parallel_size` multiplicatively
- the usual CP sequence-length divisibility rules still apply
- Transformer Engine version support matters for `a2a+p2p`

## Recommendation Level

Use hierarchical context parallelism in Bridge only when you intentionally want
that transport path and are prepared to validate execution-path details. It is
not yet the kind of feature that should be presented as universally safe across
all Bridge initialization modes.

## Related Docs

- `docs/performance-guide.md`
- `docs/training/communication-overlap.md`
- `skills/perf-techniques/hybrid-context-parallel/SKILL.md`
- `skills/perf-techniques/hybrid-context-parallel/card.yaml`
