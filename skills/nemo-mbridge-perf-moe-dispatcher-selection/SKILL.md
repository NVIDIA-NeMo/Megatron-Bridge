---
name: nemo-mbridge-perf-moe-dispatcher-selection
description: Choose the right MoE token dispatcher (`alltoall`, DeepEP, or HybridEP) for the hardware, EP degree, and optimization stage. Summarizes patterns from DSV3, Qwen3, Qwen3-Next, and VLM bring-up work.
license: Apache-2.0
when_to_use: Choosing a MoE token dispatcher, or tracing a MoE regression or crash to a dispatcher config change; 'which dispatcher', 'alltoall vs DeepEP', 'HybridEP', 'MoE dispatcher', 'flex backend', 'EP dispatcher selection'.
---

# MoE Dispatcher Selection Guide

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-dispatcher-selection/card.yaml

## Quick Decision

### By hardware

| Hardware | First choice | Why |
|---|---|---|
| H100 | A/B DeepEP and HybridEP after proving both runtime paths | DeepEP is a useful Hopper baseline, but Qwen3.5 on EOS favored HybridEP and the available DeepEP path timed out |
| B200 | DeepEP, if the runtime package is installed | Good first choice unless a platform-specific HybridEP path is available |
| GB200 / GB300 NVL72 | HybridEP, if the runtime package is installed | Best fit for NVLink-domain-aware dispatch and lower memory pressure |
| Unknown or first bring-up | `alltoall` | Easiest path for correctness and debugging |

### By EP degree

| EP size | Guidance |
|---|---|
| Small EP | Dispatcher choice is usually second-order; start with `alltoall` or DeepEP |
| Medium EP | DeepEP often becomes worthwhile |
| Large EP | HybridEP is usually the best target on NVL72 systems |

## Model-Family Patterns

| Workload | Common best path | Notes |
|---|---|---|
| DSV3 at large scale | HybridEP on GB200 or GB300, DeepEP on H100 | Dispatcher choice matters more as EP and PP both grow |
| Qwen3 235B | DeepEP on H100, HybridEP on GB200 | HybridEP usually wins on GB200 and often uses less memory |
| Qwen3 30B | DeepEP | Smaller models still benefit, but the absolute gap is smaller |
| Qwen3.5 35B-A3B text | HybridEP on the measured EOS H100 stack | About 20% faster than the matching native all-to-all baseline; DeepEP timed out at the first dispatch |
| Qwen3-Next | Close race in BF16, HybridEP stronger in FP8 or memory-tight runs | Good reminder to test, not assume |
| MoE VLMs | Start simple, then test HybridEP on GB200-class systems | Vision workloads are sensitive to both memory and host overhead |

## Rounded Evidence Summary

### Backend availability gate

Do not interpret a dispatcher timing until the container has proven that the
selected backend package is available. `--moe_flex_dispatcher_backend None`
selects the standard `alltoall` dispatcher, while `deepep` and `hybridep`
select `moe_token_dispatcher_type="flex"` and then require their corresponding
runtime packages at model construction time. If DeepEP or HybridEP is missing,
record the import failure as an environment limitation and treat `alltoall` as
the only measured correctness fallback for that run.

### Qwen3 30B A3B on H100

A short 2026-05-17 H100 smoke run used Qwen3 30B A3B BF16, 16 GPUs, EP=16,
the recipe's Transformer Engine CUDA graph scopes (`moe_router`,
`moe_preprocess`), and `model.moe_permute_fusion=false` due to a Triton JIT
compatibility issue in the run container. The `alltoall` fallback completed five
steps with 45.65 s mean step time after warmup, 132.9 mean TFLOP/s/GPU after
warmup, final loss 11.44050, and 61.351 GB peak max allocated memory. DeepEP
and HybridEP selected the requested flex backend in the dumped configs but
failed before the first iteration because the packages were not installed. This
confirms the availability gate; it is not a throughput ranking for flex
dispatchers on H100.

### Qwen3.5 35B-A3B text on EOS H100

A 2026-07-25 two-node, 16×H100 bring-up held PP=1, EP=16, MBS=1,
BF16 precision-aware optimizer state, routing mode, and recompute policy
constant. The native all-to-all baseline reached about 157 model TFLOP/s/GPU.
Switching only to HybridEP raised the result to about 189 model
TFLOP/s/GPU, a 20.25% gain. Adding scoped Transformer Engine graphs later
raised the same HybridEP path further, so keep dispatcher and graph gains as
separate rows in the tuning waterfall.

The same container had a working DeepEP import and completed model setup, but
an EP=16 DeepEP run failed before its first measured iteration with
`DeepEP error: timeout (dispatch CPU)`. A successful import is therefore only
the first availability gate: require at least one complete dispatch/combine
and a steady-state iteration before treating the backend as usable.

Static HybridEP receive capacity is also stack-specific. On this H100 stack,
combining `moe_expert_rank_capacity_factor=1.5` with the TE op fuser padded
the expert input to 49,152 tokens while GroupedLinear still split by the
roughly 32,768 actual routed tokens. The first forward pass failed with a
`split_with_sizes` mismatch. Do not copy the Blackwell static-capacity/fused
MLP stack to Hopper without a direct shape-path validation.

Standard per-expert capacity is not an equivalent workaround. Setting
`moe_expert_capacity_factor=1.0` with
`moe_pad_expert_input_to_capacity=true` removed the dynamic
`tokens_per_expert.sum()` sizing path and reached 224.9 model TFLOPS/GPU,
only about 3.1% above the valid dropless 218.121 result. Because random
force-balanced routing is balanced statistically rather than exactly per
expert, factor 1.0 can drop routed tokens. Use this experiment only to bound
the host-synchronization cost. Never accept a higher benchmark number that
comes from executing fewer routed-expert tokens.

### DSV3 on GB200 or GB300

The broad trend is more important than any single row in the tracker:

- plain `alltoall` is usually the conservative baseline
- DeepEP improves that baseline once EP communication becomes visible
- HybridEP adds another step up on NVL72 systems, especially after CUDA graphs,
  routing improvements, and CPU-side cleanup are already in place

In practice, the stack often moves from roughly "low-teens MFU" territory with
an untuned baseline into "high-teens to low-20s MFU" territory after the full
dispatcher and kernel stack is tuned.

### Qwen3 235B on GB200

For Qwen3 235B, the practical ordering is usually:

1. `alltoall` for initial bring-up
2. DeepEP if you want a familiar tuned path
3. HybridEP for the strongest steady-state result on GB200

HybridEP is usually modestly faster than `alltoall` on this workload and often
has noticeably better memory headroom.

### Qwen3-Next on GB200

This family is a good reminder that dispatcher wins are workload-dependent:

- in BF16, `alltoall` and HybridEP can be close
- in FP8 or memory-constrained settings, HybridEP tends to look better
- pipeline layout and grouped-GEMM changes can matter almost as much as the
  dispatcher itself

## Tuning Parameters

### DeepEP

DeepEP is selected by setting
`moe_token_dispatcher_type="flex"` and `moe_flex_dispatcher_backend="deepep"`.

```bash
--moe-deepep-num-sms 20
```

Tune the SM count allocated to DeepEP communication kernels (default 20).
The optimal value depends on the workload and EP degree.
First confirm the DeepEP package imports in the target container; a missing
package fails during model construction, before any dispatcher timing is
available. Then require a complete first dispatch: the package can import and
initialize successfully yet still fail with a dispatch-side CPU timeout.

### HybridEP

HybridEP is selected by setting
`moe_token_dispatcher_type="flex"` and `moe_flex_dispatcher_backend="hybridep"`.

```bash
--moe-hybridep-num-sms 16
```

Tune the SM count allocated to HybridEP communication (default 16). The
performance harness uses 32 for HybridEP workloads. Sweep between 16 and 32
for the target hardware. Set
`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` to match the NVLink domain size of
the deployment. If it does not match the actual topology, performance and
sometimes correctness will suffer.
First confirm the HybridEP package imports in the target container; a missing
package fails during model construction, before any dispatcher timing is
available.

Leave `moe_expert_rank_capacity_factor=None` for the first Hopper run. Static
capacity removes a host-side dynamic-size synchronization, but it also changes
the expert input shape and requires a compatible fused GroupedLinear path.
Validate that combination independently before using it in a benchmark recipe.
Do not substitute `moe_expert_capacity_factor` plus pad-to-capacity unless
token dropping and padded work are part of the intended training semantics.
For a dropless benchmark, verify that every configured route is still executed.

Validate the EP group against the physical ranks per node before launch. On an
8-GPU H100 node, the measured HybridEP runtime requires the EP group size to be
divisible by 8. A Qwen3.5 CP4/EP4 topology reached the first forward and then
failed because the four-rank EP group was not divisible by the eight ranks per
node. Keep EP=8, 16, or another compatible multiple for that runtime, or use a
different dispatcher and remeasure its communication cost.

### Routing mode

```bash
--moe-router-force-load-balancing
```

For performance benchmarking, force-balance routing is the safer default. It
usually outperforms dropless routing in large-scale benchmarks and makes results
more comparable across dispatcher backends.

## Key Interactions

| Feature | Interaction |
|---|---|
| CUDA graphs | Best paired with `attn moe_router moe_preprocess` on dropless MoE |
| EP overlap | Helps when dispatcher time is still visible after backend tuning |
| FP8 | Often increases the relative importance of communication and host overhead |
| CPU affinity | Can matter as much as dispatcher choice on GB200 or GB300 |
| Pipeline layout | Poor PP or VPP layout can erase dispatcher gains |

## When To Use Each

### `alltoall`

- first correctness bring-up
- small EP configurations
- debugging communication regressions

### DeepEP

- Hopper or B200 deployments
- cross-node EP is clearly visible in profiles
- you want a mature intermediate step before testing HybridEP
- the target container completes a real dispatch/combine, not merely an import

### HybridEP

- GB200 or GB300 NVL72 systems
- large EP degrees
- memory headroom matters in addition to throughput
- measured Hopper workloads where a same-stack A/B beats DeepEP or `alltoall`

## Pitfalls

1. **Do not compare dispatchers on different stacks**: container, routing mode,
   PP layout, and CUDA-graph scope can move the result as much as the dispatcher.

2. **HybridEP is topology-sensitive**: it is not a universal win outside the
   hardware it was designed for.

3. **Both dispatchers need SM tuning**: default `moe_deepep_num_sms` (20) and
   `moe_hybridep_num_sms` (16) are reasonable starting points but rarely optimal.

4. **Force-balance and dropless are not interchangeable baselines**: keep the
   routing mode fixed when comparing dispatcher backends.

5. **Memory and throughput can trade off differently by model**: Qwen3-style
   runs may show a smaller speed delta than DSV3, but still justify HybridEP for
   memory headroom.

6. **Backend import failures are not performance data**: if DeepEP or HybridEP
   is missing from the container, do not compare its failed job against a
   completed `alltoall` job. Fix the environment first, then rerun the same
   stack.

7. **Backend initialization is still not performance data**: a DeepEP run can
   import, build the model, and then time out in its first dispatch. Require a
   completed steady-state iteration.

8. **Static HybridEP capacity is not portable by default**: on H100, a padded
   receive buffer can be incompatible with the installed TE GroupedLinear
   shape path even when the op fuser is enabled. Test the first expert forward
   before launching a long benchmark.

9. **Capacity factors can change benchmark work**: per-expert capacity 1.0
   made the measured Qwen3.5 path only about 3.1% faster, while random routing
   can exceed an individual expert's capacity and drop routes. Treat it as a
   synchronization-cost diagnostic, not a valid dropless performance result.

10. **HybridEP constrains the parallel mesh**: the backend can initialize with
    an invalid EP group and fail only at the first dispatch. Check that the EP
    group size is divisible by the configured ranks per physical node; on the
    measured 8-GPU H100 stack, EP4 was invalid even though CP4×EP4 divided the
    16-GPU world size.
