# Communication Overlap

Communication overlap reduces exposed communication cost in distributed training
by overlapping collectives or point-to-point transfers with useful compute.
Megatron Bridge supports overlap across several parallelism dimensions, but the
available behavior is not identical for every mode.

This page is the stable overview for what communication overlap is, when to use
it, and which constraints are durable. For operational setup, code anchors, and
verification commands, see:

- [skills/perf-techniques/tp-dp-comm-overlap/SKILL.md](../../skills/perf-techniques/tp-dp-comm-overlap/SKILL.md)
- [skills/perf-techniques/expert-parallel-overlap/SKILL.md](../../skills/perf-techniques/expert-parallel-overlap/SKILL.md)

## What It Is

In Bridge, communication overlap spans several related subfeatures:

- data-parallel overlap for gradient reduce-scatter and parameter all-gather
- tensor-parallel overlap for TP communication under GEMM work
- pipeline-parallel overlap for PP send and receive behavior
- context-parallel overlap built into context-parallel execution paths
- MoE expert-parallel overlap for expert token dispatch communication

These are related performance techniques, but they do not share the same gates,
defaults, or operational risks.

## When to Use It

Communication overlap is a good fit when:

- the model already needs TP, DP, PP, CP, or EP for scale
- communication is a meaningful part of step time
- correctness is already established and you are tuning for throughput

It is less appropriate when:

- you are still bringing up a new training path and want minimal moving parts
- the feature combination is branch-sensitive or weakly validated
- launch-time environment tuning is likely to conflict with another technique

## Stable Per-Mode Guidance

### Data Parallel

DP overlap is tied to the distributed-optimizer path. It is the natural overlap
mechanism for sharded optimizer-state training and should be reasoned about
together with distributed optimizer behavior rather than as an isolated knob.

### Tensor Parallel

TP overlap is conceptually tied to sequence parallelism. If sequence
parallelism is not available or not enabled, TP overlap should not be assumed to
remain active.

### Pipeline Parallel

PP overlap is not a blanket property of all pipeline-parallel training. In
practice, interleaved pipeline schedules are the most important positive case.

### Context Parallel

CP overlap is part of Bridge's context-parallel execution model rather than a
separate standalone technique page. For hierarchical or `a2a+p2p` CP guidance,
see `docs/training/hybrid-context-parallel.md`.

### MoE Expert Parallel

MoE expert-parallel overlap hides the cost of token dispatch/combine all-to-all
communication by overlapping it with expert FFN compute. Optionally, delayed
expert weight-gradient computation (`moe_delay_wgrad_compute`) provides
additional overlap.

MoE overlap should be treated separately from generic TP, DP, and PP overlap.
Its constraints depend on dispatcher choice (`alltoall` or `flex`), expert
parallelism degree, precision (BF16/FP16), and runtime support. When pipeline
parallelism is used, virtual pipeline parallelism is required for the overlap
scheduling to interleave correctly.

## Stable Constraints and Caveats

The most durable caveats are:

1. Not all overlap modes are auto-enabled in the same situations.
2. Some overlap-related precision settings are owned by mixed-precision config,
   not by standalone overlap tuning alone.
3. Launch-time environment settings are part of the technique in practice,
   especially for TP, CP, and MoE overlap paths.
4. Recipe defaults are often conservative; feature existence does not imply that
   every public recipe enables the corresponding overlap path.

## Recommendation Level

Treat communication overlap as a tuning layer on top of a working distributed
configuration, not as the first knob to reach for when basic correctness is
still uncertain.

For most teams, the right order is:

1. establish a correct distributed configuration
2. choose the necessary parallelism strategy
3. enable or tune overlap for the specific communication bottleneck

## Related Docs

- [docs/performance-guide.md](../performance-guide.md)
- [docs/training/hybrid-context-parallel.md](hybrid-context-parallel.md)
- [skills/perf-techniques/tp-dp-comm-overlap/SKILL.md](../../skills/perf-techniques/tp-dp-comm-overlap/SKILL.md)
- [skills/perf-techniques/expert-parallel-overlap/SKILL.md](../../skills/perf-techniques/expert-parallel-overlap/SKILL.md)
- [skills/perf-techniques/moe-comm-overlap/SKILL.md](../../skills/perf-techniques/moe-comm-overlap/SKILL.md)
- [skills/perf-techniques/moe-comm-overlap/card.yaml](../../skills/perf-techniques/moe-comm-overlap/card.yaml)
