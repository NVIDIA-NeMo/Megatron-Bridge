---
name: nemo-mbridge-perf-moe-optimization-workflow
description: Systematic workflow for MoE training optimization in Megatron Bridge, based on the Megatron-Core MoE paper. Covers the Three Walls framework, parallel folding, recompute strategy, dispatcher choice, and CUDA-graph bring-up.
license: Apache-2.0
when_to_use: Full MoE throughput tuning sweep, or diagnosing a MoE throughput regression after a commit or config change; 'optimize MoE throughput', 'MoE perf tuning', 'Three Walls', 'memory wall', 'communication wall', 'compute wall'.
---

# MoE Training Optimization Workflow

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-optimization-workflow/card.yaml
Source: [Scalable Training of MoE Models with Megatron Core](https://arxiv.org/abs/2603.07685)

## Quick Reference

Think in terms of the paper's Three Walls:

- memory wall
- communication wall
- compute and host-overhead wall

MoE tuning is iterative. Fixing one wall usually exposes the next one, so the
best workflow is: fit first, scale second, profile third, then retune.

## First Answer Checklist

For MoE optimization workflow prompts, present the response in this order:

1. **Fit**: make the model memory-feasible first. Use the smallest model
   parallelism that fits, prefer selective recompute before full recompute, add
   offloading only after recompute and parallelism are insufficient, and use
   `--fake-init-process-group` to sanity-check large layouts.
2. **Scale**: maximize DP after the model fits, keep hot communication inside
   the fastest interconnect, use PP plus VPP for multi-node scaling, prefer EP
   over extra TP for expert layers, and add CP when long context makes attention
   memory dominant.
3. **Profile**: identify the dominant wall: memory, communication, host
   overhead, or compute.
4. **Retune**: change dispatcher, overlap, FP8 mode, CUDA graphs, or recompute
   based on the profiled bottleneck.
5. **Validate multiple optimizer steps**: a finite first step is not enough.
   Optimizer-state allocation, JIT compilation, graph capture, or a
   dispatcher/overlap stall may appear only on later steps. Require at least
   three completed steps with finite loss and zero skipped/NaN iterations
   before treating a candidate as viable.
6. Include the exact Parallel Folding meshes: `Attention: TP x CP x DP x PP`
   and `MoE: ETP x EP x EDP x PP`.
7. Include the default mappings: `alltoall` for safe bring-up,
   `flex` + `deepep` for H100/B200-style systems, `flex` + `hybridep` for
   GB200/GB300/NVL72 systems, Hopper to FP8 blockwise, Blackwell to MXFP8, and
   dropless MoE TE-scoped CUDA graphs over `attn`, `moe_router`, and
   `moe_preprocess`.

## Phase 1: Make The Run Memory-Feasible

Start with a configuration that fits reliably before chasing throughput.

Recommended order:

1. Use the smallest amount of model parallelism that still fits.
2. Identify whether the peak comes from activations or delayed optimizer-state
   allocation. An OOM on step 2 after a finite step 1 is often optimizer-state
   memory, not an activation peak; inspect precision-aware optimizer dtypes
   before paying for recompute or more PP.
3. Turn on selective recompute before falling back to full recompute.
4. Add offloading only when recompute and parallelism are still insufficient.
5. Use `--fake-init-process-group` to sanity-check large parallel layouts on a
   single GPU before burning cluster time.

### Recompute guidance

Prefer selective recompute for MoE runs:

- good first choices: `layernorm`, `core_attn`, `moe_act`, `mlp`, or
  model-specific modules (`shared_experts`, `mla_up_proj`)
- use full recompute only when the run still does not fit
- revisit recompute after enabling CUDA graphs, because some graph scopes and
  full recompute paths do not mix well

As a rule of thumb, fine-grained recompute often recovers most of the needed
memory while keeping throughput much closer to the non-recompute baseline than
full-layer recompute does.

## Phase 2: Choose Parallelism For Scale

Priority order:

1. Maximize DP once the model fits.
2. Keep the hot communication path inside the fast interconnect when possible.
3. Use PP, plus VPP if needed, for multi-node scaling.
4. Prefer EP over extra TP for expert layers.
5. Add CP for long context once sequence length makes attention memory dominant.

### Parallel Folding

Parallel Folding decouples attention and MoE parallelism so you do not have to
pick a single compromise layout:

```text
Attention: TP × CP × DP × PP
MoE:       ETP × EP × EDP × PP
```

Key knobs:

- `--expert-model-parallel-size`
- `--expert-tensor-parallel-size`

Use it when attention prefers some TP or CP, but expert layers benefit from a
larger EP degree than the dense layers can tolerate.

## Phase 3: Profile The Dominant Bottleneck

| Bottleneck | What it looks like | Primary fixes |
|---|---|---|
| Memory | Run fits only with aggressive full recompute or OOMs during warmup | selective recompute, FP8, offloading, better PP layout |
| Communication | Nsight shows large all-to-all or collective blocks | DeepEP or HybridEP, EP overlap, DP/TP overlap, better PP layout |
| Host overhead | GPU gaps, launch-bound traces, Python overhead | CUDA graphs, `--manual-gc`, higher MBS, CPU affinity tuning |
| Compute | Low SM utilization after comm and host issues are addressed | grouped GEMM, fusion work, FP8, dispatcher-specific kernel tuning |

### Profile overlap without misreading it

Use unprofiled steady iterations for the acceptance metric and a matched
profile for causal explanation:

1. Change one overlap or dispatcher variable at a time; keep routing, graph
   scopes, parallelism, batch shape, and runtime fixed.
2. Build interval unions for communication kernels and compute kernels, then
   measure their intersection to quantify hidden communication.
3. Do not add kernel durations and call the result wall time. Concurrent
   kernels may run longer because of SM or bandwidth contention even while the
   exposed GPU-active union and end-to-end step time fall.
4. Corroborate the trace with dispatch/combine NVTX ranges, steady step time,
   model TFLOPS/GPU, loss finiteness, skipped/NaN counts, and peak memory.
5. Separate one-time compilation from steady execution. Persist
   `TORCHINDUCTOR_CACHE_DIR` and, for TileLang kernels,
   `TILELANG_CACHE_DIR` on a mounted cache path. Container `HOME` may point to
   an ephemeral filesystem even when the host home is persistent.
6. Exclude the first iteration, JIT compilation, graph warmup, and graph
   capture from acceptance timing. Use a fixed post-warmup iteration window.

On a controlled 16×H100 Qwen3 30B-A3B HybridEP run, plain EP overlap increased
communication hidden by GEMM/attention from 0.11% to 36.55%. The unprofiled
step fell from 24.7138s to 20.9920s and throughput rose from 244.039 to 287.305
model TFLOPS/GPU. `delay_wgrad_compute` remained disabled.

Do not generalize that overlap result across model families. On a matched
16×H100 Qwen3.5-35B-A3B GDN-MoE run, changing native all-to-all to HybridEP
improved steady BF16 throughput from 157.248 to 189.096 model TFLOPS/GPU
(+20.25%), but then enabling plain EP overlap reduced it to 184.57
TFLOPS/GPU (-2.45%). Keep dispatcher and overlap as separate A/B dimensions;
an overlap stream can contend with GDN, expert, or dispatcher kernels even
when the same knob helps an attention-only MoE.

## Dispatcher And Overlap Guidance

Use dispatcher choice as a bottleneck fix, not as the first tuning knob.

- `moe_token_dispatcher_type="alltoall"`: safest bring-up path, fine for
  smaller EP sizes
- `moe_token_dispatcher_type="flex"` + `moe_flex_dispatcher_backend="deepep"`:
  strong default for H100 and B200 style deployments
- `moe_token_dispatcher_type="flex"` + `moe_flex_dispatcher_backend="hybridep"`:
  strongest starting point on GB200 or GB300 NVL72 systems

Treat these as starting points, not hard platform rules. HybridEP plus plain EP
overlap was the measured winner for the 16×H100 Qwen3 30B-A3B shape. Benchmark
backend compatibility and throughput in the target container.

If the all-to-all path is visible in profiles, combine dispatcher tuning with:

- `--overlap-moe-expert-parallel-comm`
- `--overlap-grad-reduce`
- `--tp-comm-overlap`

### Hybrid GDN MoE bring-up

Hybrid models with Gated DeltaNet or another JIT-backed linear-attention block
need a stricter bring-up sequence than attention-only MoE models:

1. Start with eager execution, native `alltoall`, no EP overlap, and a fixed
   routing mode.
2. Balance PP stages by parameter count and block cost, not layer count alone.
   Embeddings, output logits, MTP, and the mix of attention/GDN layers can make
   equal layer splits badly imbalanced.
3. Complete at least three optimizer steps before calling the layout stable.
   A first-step pass can still become a step-2 OOM when Adam states are first
   materialized.
4. Measure a matched kernel A/B with identical topology, precision, routing,
   batch shape, container, and cache state.
5. Only then test a flex dispatcher, EP overlap, scoped CUDA graphs, or delayed
   wgrad, one variable at a time.

FlashQLA-style TileLang backends can compile separate forward, recompute, MTP,
and backward variants. The cold iteration may therefore be minutes rather than
seconds. Cache every target shape and report only steady replay iterations.

## FP8 Recipe Quick Decision

| Platform | Recommended starting recipe |
|---|---|
| Hopper | FP8 blockwise |
| Blackwell | MXFP8 |
| Blackwell, speed-first exploration | NVFP4 after the BF16 or FP8 path is stable |

Keep the router in FP32. The largest wins usually come from expert GEMMs and
other heavy matrix math, not from trying to quantize every small MoE component.

## CUDA Graphs For MoE

For dropless MoE, start with partial TE-scoped graphs:

- `attn`
- `moe_router`
- `moe_preprocess`

That path usually gives a meaningful step-time win while keeping the dynamic
expert work outside the graph. Expect a moderate speedup when launch overhead is
visible, but budget several extra GB of memory and verify that shapes remain
static.

Use full-iteration graphs only for graph-friendly workloads such as drop-and-pad
or tightly controlled static-shape experiments.

Related references:

- @skills/nemo-mbridge-perf-cuda-graphs/SKILL.md
- @docs/training/cuda-graphs.md
- @docs/training/activation-recomputation.md

## Pitfalls

1. **Do not optimize in the wrong order**: fitting the model and selecting sane
   parallelism matter more than micro-optimizations.

2. **Platform changes the limiting wall**: H100-class runs often feel more
   communication-bound, while GB200 or GB300 runs often expose CPU or launch
   overhead earlier.

3. **FP8 MFU can look misleadingly low**: compare absolute throughput as well as
   MFU when switching precision modes.

4. **CUDA graphs and recompute interact**: TE-scoped graphs are usually paired
   with selective recompute, not blanket full recompute.

5. **Parallel Folding is not optional at large scale**: once attention and expert
   layers want clearly different layouts, a single shared TP or EP plan becomes
   a tax on both.

6. **Summed kernel time is not exposed time**: use interval unions and
   communication/compute intersection when validating overlap.

7. **A finite first step is not a stability result**: validate later optimizer
   steps and watch both high-power compute and low-power 100%-SM communication
   spin states.

8. **Current-scaling FP8 is not an automatic Hopper win**: small projections,
   GDN kernels, and quantization overhead can make it slower than BF16. Start
   Hopper exploration with blockwise FP8 after a stable BF16 baseline, and keep
   only a measured win.

9. **Blockwise FP8 is still shape-specific on Hopper**: on the controlled
   16×H100 Qwen3.5-35B-A3B HybridEP shape, blockwise FP8 reduced throughput
   from 189.096 to 174.38 model TFLOPS/GPU (-7.8%) and increased rank-0 peak
   allocated memory from 69.939 to 72.730 GiB. Its cold first iteration also
   took 129.1 seconds for kernel compilation. Exclude compilation from timing,
   but reject the precision mode when its steady-state result still loses.

10. **Tensorwise current scaling can lose even more on small experts**: on the
    pinned 16×H100 Qwen3.5-35B-A3B EP16 HybridEP shape, the first iteration
    took 192.33 seconds to compile and iterations 3-10 averaged only
    145.3 model TFLOPS/GPU. The run was finite with zero skipped/NaN
    iterations, so this was a performance rejection rather than a correctness
    failure. Treat FP8 mode and scaling granularity as separate A/Bs.

11. **Do not optimize the metric by dropping MoE work**: a Qwen3.5
    `moe_expert_capacity_factor=1.0` diagnostic removed HybridEP's dynamic
    token-count synchronization but improved the valid dropless result by only
    about 3.1%. Random force-balanced routing can exceed individual expert
    capacities, so the setting may drop routes. Use it to bound synchronization
    overhead, not to accept throughput.

12. **Revalidate at the acceptance batch**: the Qwen3.5 GBS128 winner reached
    218.121 model TFLOPS/GPU, while an exact GBS1024 replay stabilized at
    225.8-225.9 model TFLOPS/GPU over its last four steps. Larger gradient
    accumulation helped modestly but did not remove the model-family gap to
    Qwen3. Report the exact acceptance batch instead of extrapolating from a
    short A/B batch.

13. **Shared-expert overlap is a separate concurrency A/B**: enabling only
    `moe_shared_expert_overlap` on the same Qwen3.5 HybridEP + scoped-graph
    stack reduced throughput from 218.121 to about 207.0 model TFLOPS/GPU
    (-5.1%) and slightly increased peak memory. Do not infer a win from either
    dispatcher selection or EP overlap; the shared-expert stream can introduce
    its own GDN/expert/communication contention.
