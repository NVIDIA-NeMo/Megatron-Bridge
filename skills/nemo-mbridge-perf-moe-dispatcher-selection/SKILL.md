---
name: nemo-mbridge-perf-moe-dispatcher-selection
description: >-
  Select and validate an MoE token dispatcher (`alltoall`, DeepEP, or
  HybridEP) for a fixed workload and runtime. Covers backend availability,
  topology, matched A/B evidence, routing semantics, and failure diagnosis.
  Use when choosing a dispatcher or tracing a regression or crash to the MoE
  dispatcher configuration.
license: Apache-2.0
---

# MoE Dispatcher Selection Guide

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-dispatcher-selection/card.yaml

## Quick Decision

### By hardware

| Hardware | Bring-up path | Tuned candidates |
|---|---|---|
| H100 | `alltoall` | A/B DeepEP and HybridEP when installed; the current 16×H100 Qwen3 30B winner is HybridEP |
| B200 | `alltoall` | A/B DeepEP and HybridEP when supported by the target runtime |
| GB200 / GB300 NVL72 | `alltoall` | HybridEP is the strongest topology-informed candidate; compare DeepEP when available |
| Unknown | `alltoall` | Add one flex backend only after the correctness baseline is stable |

Hardware narrows the candidate set; it does not select the winner. Hold the
model, routing, batch shape, parallelism, overlap, graph scope, container, and
timing window fixed during the comparison.

### By EP degree

| EP size | Guidance |
|---|---|
| Small EP | Dispatcher choice may be second-order; start with `alltoall` |
| Medium EP | Profile first, then A/B the installed flex backends |
| Large EP | Prioritize topology-aware candidates, but still require a matched A/B |

On one NVL8 domain in BF16, treat `alltoall` and HybridEP as matched candidates:
their throughput can be close once the full stack is held fixed. HybridEP is a
high-priority tuning path, not a reason to skip the correctness baseline.

## Model-Family Patterns

| Workload | Common best path | Notes |
|---|---|---|
| DSV3 at large scale | Measured snapshots use HybridEP on GB200/GB300 and DeepEP on H100 | Revalidate against the target container and topology |
| Qwen3 235B | Current H100 recipe uses `alltoall` plus overlap; measured GB200 snapshots use HybridEP | Do not replace the current recipe from a hardware rule alone |
| Qwen3 30B | Current canonical 16×H100 recipe uses HybridEP | Direct counterexample to H100 → DeepEP mapping |
| Qwen3-Next | Workload-dependent | Precision, memory, PP layout, and kernels can change the ordering |
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

The current canonical 16×H100 BF16 performance recipe uses HybridEP, 32
HybridEP SMs, 64-token combine chunks, plain expert-parallel communication
overlap, delayed weight-gradient compute disabled, and TE graphs over
`moe_router` and `moe_preprocess`. Its verified 50-step run averaged 20.14729 s
and 299.352 model TFLOPS/GPU over steps 41–50. This proves HybridEP can win on
NVL8 H100; it does not prove HybridEP is universal.

An earlier matched overlap A/B on the same broad shape isolated a rise from
244.039 to 287.305 TFLOPS/GPU. Keep that causal result separate from the later
multi-knob canonical winner.

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
available.

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

### Routing mode

```bash
--moe-router-force-load-balancing
```

Forced load balancing is a **benchmark-only** control that can reduce routing
variance across dispatcher backends. It changes routing semantics, so keep it
fixed within the dispatcher A/B and do not use it to accept a
training-equivalent or convergence-sensitive result. Validate the production
winner again with natural routing.

## Key Interactions

| Feature | Interaction |
|---|---|
| CUDA graphs | Profile-driven candidate; start narrow and re-test after dispatcher changes |
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

- any supported target runtime where DeepEP imports successfully
- cross-node EP is clearly visible in profiles
- a matched steady-state A/B beats the alternatives

### HybridEP

- NVL72 systems, where the topology makes it a high-priority candidate
- NVL8 systems when the package supports the topology and a matched A/B wins
- large EP degrees
- memory headroom matters in addition to throughput
- an NVL8 BF16 matched A/B beats or materially improves headroom over
  `alltoall`; a small or negative delta is a valid reason to keep `alltoall`

## Pitfalls

1. **Do not compare dispatchers on different stacks**: container, routing mode,
   PP layout, and CUDA-graph scope can move the result as much as the dispatcher.

2. **HybridEP is topology-sensitive**: configure the actual NVLink domain and
   do not infer support or performance from the GPU SKU alone.

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

7. **Forced routing is not training equivalence**: use it only as a disclosed
   benchmark control, then validate natural routing separately.

8. **Config selection is not backend proof**: require runtime evidence that the
   requested flex backend initialized and completed steady iterations.
