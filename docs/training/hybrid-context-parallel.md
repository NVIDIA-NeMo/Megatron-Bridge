# Hybrid / Hierarchical Context Parallel

This page covers the stable Bridge-facing meaning of hierarchical context
parallelism, especially the `a2a+p2p` transport path and
`hierarchical_context_parallel_sizes`.

For operational setup, code anchors, and verification commands, see
`skills/perf-techniques/hybrid-context-parallel.md`.

## What It Is

In upstream Megatron-Core, `cp_comm_type="a2a+p2p"` plus
`hierarchical_context_parallel_sizes` enables a hierarchical context-parallel
transport path. This is the Bridge-relevant form of hierarchical context
parallelism.

It is important to separate that from the upstream boolean
`hybrid_context_parallel`, which is a different feature for balancing packed or
variable-length workloads. The two concepts should not be treated as
interchangeable.

## When to Use It

Hierarchical context parallelism is relevant when:

- plain context parallelism is already required
- larger CP sizes make flat `p2p` less attractive
- you specifically want the hierarchical `a2a+p2p` transport path

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
