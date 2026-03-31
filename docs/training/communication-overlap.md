# Communication Overlap

Communication overlap reduces exposed communication cost in distributed training
by hiding collectives or point-to-point transfers under useful compute.

This page is the stable guide for what communication overlap is, when it tends
to help, and which boundaries are durable across Megatron Bridge. For exact
knobs, code anchors, and verification commands, see:

- `skills/perf-techniques/tp-dp-comm-overlap/SKILL.md`
- `skills/perf-techniques/expert-parallel-overlap/SKILL.md`

## What It Is

In Bridge, communication overlap is a family of related techniques rather than a
single switch:

| Mode | What gets hidden | Main gate |
|---|---|---|
| DP | gradient reduce-scatter and parameter all-gather | distributed-optimizer overlap path |
| TP | tensor-parallel collectives under layer compute | `CommOverlapConfig.tp_comm_overlap` plus sequence parallelism |
| PP | pipeline send/recv work under schedule execution | pipeline schedule and virtual pipeline layout |
| CP | context-parallel communication inside CP execution paths | CP implementation choice |
| EP | MoE token dispatch/combine communication under expert compute | `overlap_moe_expert_parallel_comm` |

These paths share the same goal, but they do not share the same enablement
rules, evidence level, or failure modes.

## What Problem It Solves

Distributed training often becomes communication-bound before it becomes
compute-bound. Once TP, DP, PP, CP, or EP traffic is visible on the critical
path, adding more GPUs may raise communication time faster than it raises useful
compute.

Communication overlap addresses that by moving communication earlier or later in
the step so the same transfer can happen while some other part of the model is
already doing useful work. It does not change the training objective. It tries
to reduce idle time.

## Impacted Training Dimensions

| Dimension | Effect | Confidence | Why |
|---|---|---|---|
| `speed` | ~0-15% faster step time, mode-dependent | medium | The whole point is to hide communication time, but gain depends strongly on which overlap mode is active and whether communication is actually exposed. EP overlap measured flat to ~13% slower on small-EP Qwen3-30B-A3B, so gains are not guaranteed. |
| `memory` | neutral (some modes add ~1-2 GB for buffers) | low | Overlap itself is usually not a primary memory technique, although some implementations (e.g., TP userbuffers) add buffer or scheduling constraints. |
| `scale` | positive at higher parallelism degrees | medium | Overlap becomes more valuable as communication dominates larger distributed runs. |
| `convergence` | no change expected | medium | The intent is to preserve the same training math, though schedule changes can alter floating-point accumulation order. |
| `stability` | adds operational constraints | medium | More overlap usually means tighter requirements around schedule shape, precision, runtime versions, and feature combinations. |

## When to Use It

Enable communication overlap when all of the following are mostly true:

- the distributed configuration already works correctly without overlap
- communication is a meaningful part of step time
- you are tuning throughput or utilization, not doing first bring-up
- you can benchmark the specific overlap mode you plan to use

As a rule of thumb:

| Mode | Good first use case | Recommendation |
|---|---|---|
| DP | distributed optimizer on multi-GPU or multi-node training | Usually worth considering early once optimizer sharding is already chosen. |
| TP | `TP >= 2` with sequence parallelism and TE-enabled path | Benchmark when TP collectives are visible in the profile. |
| PP | interleaved pipeline schedules where p2p overhead is visible | Treat as schedule tuning, not a blanket PP default. |
| CP | large-context runs already using CP | Follow the CP-specific guidance rather than treating it as a separate generic knob. |
| EP | large-scale MoE with many micro-batches and inter-node A2A cost | Most promising at larger EP and with higher-latency dispatcher backends. |

Measured repo evidence today is strongest for MoE EP overlap. On
Qwen3-30B-A3B with EP=4 and `alltoall` on 2 H100 nodes, EP overlap is
numerically safe at GBS=8 but provides no speedup, and it is about 13% slower
at GBS=64. On Qwen3-Next-80B-A3B with EP=8 and `alltoall` on 8 nodes, the
overlap variants are stable while the non-overlap baseline NaNs, but
`delay_wgrad_compute` is still about 4.8% slower than overlap-only. That makes
EP overlap correctness-backed in this repo, but not yet broadly speedup-backed.

## When Not to Use It

Avoid communication overlap when any of these are true:

- you are still debugging a new distributed setup
- the profile is compute-bound rather than communication-bound
- the required companion feature is missing, such as sequence parallelism for TP
- another feature already imposes conflicting runtime constraints
- you have not benchmarked the exact model and parallelism shape

For MoE EP overlap specifically, avoid treating it as a default when:

- `EP <= 4` with `alltoall` on `<= 2` nodes
- the run has very few pipeline micro-batches
- `moe_shared_expert_overlap` must stay enabled
- full recompute or recompute scheduling incompatible with EP overlap is required

## Feature Interactions

The most important interactions are:

- DP overlap is tied to distributed-optimizer behavior rather than a fully independent tuning path.
- TP overlap depends on sequence parallelism and the supported TE overlap path.
- PP and EP overlap interact with virtual pipeline layout when `PP > 1`.
- CP overlap should be reasoned about together with the chosen CP communication type.
- EP overlap with DeepEP or HybridEP requires explicitly switching the dispatcher to `flex`.
- EP overlap and `moe_shared_expert_overlap` are mutually exclusive.
- CUDA graphs plus `delay_wgrad_compute` adds extra TE-version and graph-scope restrictions.
- Launch-time environment tuning can conflict across overlap paths, especially TP or CP overlap versus DeepEP or HybridEP tuning.

## Bridge Configuration

Communication overlap is configured through `CommOverlapConfig` plus
mode-specific model settings. There is no single universal toggle — DP, TP,
PP, CP, and EP each have different prerequisites and should be enabled based
on the actual bottleneck.

For config examples and minimal runnable commands, see:

- [skills/perf-techniques/tp-dp-comm-overlap/SKILL.md](../skills/perf-techniques/tp-dp-comm-overlap/SKILL.md)
- [skills/perf-techniques/expert-parallel-overlap/SKILL.md](../skills/perf-techniques/expert-parallel-overlap/SKILL.md)

## Expected Metric Changes

| Metric | Expected Change | Conditions | Evidence |
|---|---|---|---|
| `step_time` | down | DP overlap with distributed optimizer on communication-heavy runs | expected |
| `step_time` | down | TP overlap with `TP >= 2`, sequence parallelism, and supported TE path | expected |
| `pipeline_idle_time` | down | interleaved PP where p2p cost is visible | expected |
| `step_time` | flat | Qwen3-30B-A3B, EP=4, `alltoall`, 2 nodes, GBS=8 | measured: 822ms baseline vs 827ms overlap |
| `step_time` | up | same model/config, GBS=64 | measured: 4889ms baseline vs 5538ms overlap |
| `step_time` | up | Qwen3-Next-80B-A3B, EP=8, `alltoall`, 8 nodes, `delay_wgrad_compute=True` vs overlap-only | measured: 4912ms vs 4686ms |

Do not assume one overlap win transfers automatically to another mode. The
correct question is always "which communication path is exposed in this run?"

## Common Failure Modes

- TP overlap silently disables itself when sequence parallelism is off or `TP < 2`.
- PP overlap expectations are wrong when the schedule is non-interleaved or VPP is missing.
- EP overlap asserts when `PP > 1` but `virtual_pipeline_model_parallel_size` is unset.
- EP overlap asserts when full recompute, recompute method, or shared-expert overlap stays enabled.
- Setting `moe_flex_dispatcher_backend` alone does not activate DeepEP or HybridEP; the dispatcher must actually switch to `flex`.
- Small-EP `alltoall` MoE runs can get slower because scheduling overhead is larger than the communication being hidden.

## Related Docs

- [docs/performance-guide.md](../performance-guide.md)
- [docs/training/cuda-graphs.md](cuda-graphs.md)
- [docs/training/hybrid-context-parallel.md](hybrid-context-parallel.md)
- [skills/perf-techniques/tp-dp-comm-overlap/SKILL.md](../skills/perf-techniques/tp-dp-comm-overlap/SKILL.md)
- [skills/perf-techniques/expert-parallel-overlap/SKILL.md](../skills/perf-techniques/expert-parallel-overlap/SKILL.md)
- [skills/perf-techniques/moe-comm-overlap/SKILL.md](../skills/perf-techniques/moe-comm-overlap/SKILL.md)
- [skills/perf-techniques/moe-comm-overlap/card.yaml](../skills/perf-techniques/moe-comm-overlap/card.yaml)
