---
name: nemo-mbridge-perf-moe-optimization-workflow
description: >-
  Evidence-gated workflow for MoE performance optimization in Megatron Bridge.
  Covers measurement contracts, the Three Walls framework, parallel folding,
  profiling, matched A/B tuning, and final validation. Use for full MoE
  throughput tuning or diagnosing a regression involving memory,
  communication, compute, or host/launch overhead.
license: Apache-2.0
---

# MoE Training Optimization Workflow

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-optimization-workflow/card.yaml
Source: [Scalable Training of MoE Models with Megatron Core](https://arxiv.org/abs/2603.07685)

## Quick Reference

Start with the paper's Three Walls:

- memory wall
- communication wall
- compute-efficiency wall

For operational diagnosis, split the compute-efficiency wall into **compute**
and **host/launch** bottlenecks. They need different evidence and different
fixes. MoE tuning is iterative, so use this order:

```text
freeze the measurement contract -> fit -> scale -> profile -> retune -> validate
```

## First Answer Checklist

For MoE optimization workflow prompts, present the response in this order:

1. **Freeze the measurement contract**: record the exact model and task,
   hardware and topology, container and commits, data and routing semantics,
   precision, sequence and batch shape, parallelism, graph scopes, and the
   steady-state metric window. Label each candidate as training-equivalent or
   benchmark-only.
2. **Fit**: make the model memory-feasible first. Use the smallest model
   parallelism that fits: keep dense TP as low as capacity allows while using
   a large legal EP for expert weights. Prefer selective recompute before full
   recompute, add offloading only after recompute and parallelism are
   insufficient, and use `--fake-init-process-group` to sanity-check large
   layouts.
3. **Scale**: maximize DP after the model fits, keep hot communication inside
   the fastest interconnect, use PP plus VPP for multi-node scaling, prefer EP
   over extra TP for expert layers, and add CP when long context makes attention
   memory dominant.
4. **Profile**: identify the dominant wall: memory, communication, host
   overhead, or compute.
5. **Retune**: change one variable at a time based on the profiled bottleneck.
   Dispatcher, overlap, lower precision, CUDA graphs, and recompute are
   candidates, not hardware defaults.
6. **Validate**: use short matched screens to reject candidates, then run the
   winner for at least 50 steps. Verify the requested backend or graph replay
   actually ran, time a declared post-warmup window, and report loss health,
   skipped/NaN iterations, memory, step time, and model TFLOPS/GPU.
7. Include the exact Parallel Folding meshes: `Attention: TP x CP x DP x PP`
   and `MoE: ETP x EP x EDP x PP`.
8. Use `alltoall` for safe bring-up, then A/B `flex` + `deepep` and `flex` +
   `hybridep` when their packages and target topology support them. Start from
   BF16 and eager execution; introduce lower precision or the narrowest useful
   CUDA-graph scope only after profiling justifies it.

## Phase 0: Freeze The Measurement Contract

A comparison is valid only when the following stay fixed unless they are the
single variable under test:

- Bridge, MCore, Transformer Engine, container, CUDA, and NCCL versions
- GPU count, SKU, node topology, and launcher/environment settings
- model, task, data path, sequence length, MBS, GBS, and optimizer settings
- TP, PP, VPP, CP, EP, ETP, and DP layout
- routing semantics, precision, recompute, dispatcher, overlap, and graph scope
- warmup and steady-state timing windows

Separate two acceptance classes:

- **Training-equivalent** changes preserve the intended routing, loss, data,
  optimizer, and checkpoint/resume behavior.
- **Benchmark-only** changes such as forced load balancing are useful for
  controlled kernel studies, but cannot establish production-training
  correctness or convergence.

## Phase 1: Make The Run Memory-Feasible

Start with a configuration that fits reliably before chasing throughput.

Recommended order:

1. Use the lowest dense TP that still fits; avoid paying extra TP communication
   merely to reduce activation memory.
2. Use the largest legal EP that fits the expert count and intended topology.
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

For Qwen-family models, choose boundaries by architecture. Standard-attention
Qwen recipes can often start with `core_attn`. Qwen3.5 includes GDN layers, so
`core_attn` alone does not cover their retained normalization output; the
current pinned Megatron Core exposes `gdn_norm_out`. Test it on the exact stack,
and fall back to full-layer recompute when selective boundaries still do not
make the complete optimizer step fit.

## Phase 2: Choose Parallelism For Scale

Priority order:

1. Maximize DP once the model fits.
2. Keep the hot communication path inside the fast interconnect when possible.
3. Use PP, plus VPP if needed, for multi-node scaling.
4. Prefer EP over extra TP for expert layers.
5. Add CP for long context once sequence length makes attention memory dominant.

For common MoE allocations, EP=8 is a strong first candidate on an 8-GPU node
and EP=32 is a strong first candidate on 32 GPUs when the expert count and mesh
are divisible. This is a capacity heuristic, not a correctness rule. EP shards
only expert parameters: EP=8 can still leave a large model dependent on full
recompute, while EP=32 often creates enough expert-weight headroom to retest
selective recompute.

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

### Fill Available Headroom

Once the run completes optimizer initialization and several steady steps,
inspect W&B memory and confirm every rank's peak allocated and reserved memory
in the runtime logs; rank 0 can miss the hot EP or PP rank. Use any safe
headroom to A/B one of these changes at a time:

- lower TP by one legal step, if the additional dense state still fits;
- increase MBS, while adjusting gradient accumulation to keep GBS and
  optimizer boundaries unchanged.

Judge the result by end-to-end tokens/s/GPU or step time, not memory utilization
alone. Preserve margin for checkpoint/eval transients, routing imbalance,
dispatcher workspaces, and CUDA-graph private pools.

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

On a controlled 16×H100 Qwen3 30B-A3B HybridEP run, plain EP overlap increased
communication hidden by GEMM/attention from 0.11% to 36.55%. The unprofiled
step fell from 24.7138s to 20.9920s and throughput rose from 244.039 to 287.305
model TFLOPS/GPU. `delay_wgrad_compute` remained disabled.

## Phase 4: Retune From Evidence

Choose the smallest candidate that targets the profiled bottleneck and change
one variable at a time.

### Dispatcher And Overlap Guidance

Use dispatcher choice as a bottleneck fix, not as a hardware lookup table.

- `moe_token_dispatcher_type="alltoall"`: safest bring-up path, fine for
  smaller EP sizes
- `moe_token_dispatcher_type="flex"` + `moe_flex_dispatcher_backend="deepep"`:
  candidate when DeepEP is installed and communication is exposed
- `moe_token_dispatcher_type="flex"` + `moe_flex_dispatcher_backend="hybridep"`:
  topology-sensitive candidate on both NVL8 and NVL72 systems when HybridEP is
  installed

HybridEP plus plain EP overlap is the current measured winner for the canonical
16×H100 Qwen3 30B-A3B shape, while the canonical 256×H100 Qwen3 235B recipe
uses standard `alltoall` plus overlap. Benchmark backend compatibility and
throughput in the target container; neither GPU name nor EP degree determines
the winner by itself.

On a single NVL8 domain in BF16, `alltoall` and HybridEP can be close enough
that the simpler baseline wins after setup overhead. Prefer HybridEP as a tuned
candidate, not an assumption, and keep a matched `alltoall` control.

If the all-to-all path is visible in profiles, combine dispatcher tuning with:

- `--overlap-moe-expert-parallel-comm`
- `--overlap-grad-reduce`
- `--tp-comm-overlap`

Test plain EP overlap, shared-expert overlap, and delayed weight-gradient
compute as separate candidates first. A combination can regress even when one
component helped on another model.

### Lower-Precision Candidate Matrix

Start with a verified BF16 baseline. Hardware capability only determines which
lower-precision candidates are legal; it does not guarantee a speedup.

| Platform | Candidate after BF16 is stable |
|---|---|
| Hopper | per-tensor, current-scaling, or blockwise FP8 supported by the target stack |
| Blackwell | MXFP8 or another supported FP8 recipe |
| Blackwell, speed-first exploration | NVFP4 after the BF16/FP8 path is stable |

Keep the router in FP32. The largest wins usually come from expert GEMMs and
other heavy matrix math, not from trying to quantize every small MoE component.
Require logs or traces showing that the intended kernels ran, and judge the
candidate by end-to-end steady step time rather than theoretical peak FLOPS.

### CUDA Graphs For MoE

Use CUDA graphs only after a profile shows meaningful host/launch gaps. For
dropless MoE, start with the narrowest partial TE-scoped graph candidate:

- `moe_router`
- `moe_preprocess`

Add `attn` only if it is supported for the model and improves the same matched
stack. A successful capture is not evidence of a speedup, and a graph win can
disappear after dispatcher, overlap, or precision changes.

This path keeps dynamic expert work outside the graph. Budget extra memory,
verify that shapes remain static, confirm replay rather than capture alone, and
time only post-capture iterations.

Use full-iteration graphs only for graph-friendly workloads such as drop-and-pad
or tightly controlled static-shape experiments.

Related references:

- @skills/nemo-mbridge-perf-cuda-graphs/SKILL.md
- @docs/training/cuda-graphs.md
- @docs/training/activation-recomputation.md

## Phase 5: Validate And Package Evidence

Use 6–12 post-warmup iterations for inexpensive screening when the workload
allows it. For the selected candidate, run at least 50 steps and report a fixed
steady window such as steps 41–50. The final evidence bundle should contain:

- exact command/config diff, commits, container, hardware, and topology
- declared routing/data semantics and training-equivalent vs benchmark-only label
- proof that the intended dispatcher, precision kernels, overlap, and graph
  replay were active
- step time and model TFLOPS/GPU from the same unprofiled steady window
- finite loss, skipped/NaN counts, peak memory, and checkpoint/optimizer-state
  validation when the production path requires it
- matched A/B profile evidence for the claimed causal mechanism

Do not attribute the total gain of a final multi-change winner to one earlier
A/B. For example, the Qwen3 overlap experiment isolated a rise from 244.039 to
287.305 TFLOPS/GPU; the later canonical recipe reached 299.352 after additional
HybridEP tuning. They answer different questions.

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

7. **Benchmark-only semantics are not production acceptance**: forced routing,
   synthetic data, or disabled optimizer/checkpoint paths must be disclosed and
   validated separately from training-equivalent results.

8. **Feature activation needs evidence**: a config dump is insufficient when a
   backend can fall back, a graph can capture without helping, or a lower-
   precision recipe can miss the intended kernels.

_Last guidance refresh: 2026-08-25._
