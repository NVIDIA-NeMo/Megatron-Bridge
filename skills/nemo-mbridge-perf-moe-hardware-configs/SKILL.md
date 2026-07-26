---
name: nemo-mbridge-perf-moe-hardware-configs
description: Representative MoE training playbooks by hardware platform and model family. Summarizes rounded throughput bands, parallelism patterns, and common tuning stacks.
license: Apache-2.0
when_to_use: Hardware-specific MoE playbooks or throughput estimates; 'MoE on H100', 'GB200 config', 'expected throughput', 'MoE hardware playbook', 'parallelism for B200'.
---

# MoE Hardware Configuration Reference

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-hardware-configs/card.yaml

## Quick Platform Playbook

| Platform | Typical MoE strategy | What usually matters most |
|---|---|---|
| H100 | DeepEP or HybridEP + explicit EP overlap | communication overlap, dispatcher/runtime compatibility, and PP efficiency |
| B200 | DeepEP + MXFP8 + careful PP layout | container quality and tuned comm settings |
| GB200 | HybridEP + partial CUDA graphs + CPU cleanup | host overhead, topology-aware dispatch, memory headroom |
| GB300 | HybridEP + newer FP8 and kernel stack | same GB200 playbook, usually with a higher ceiling |

## First Answer Checklist

For hardware playbook questions, answer from these canonical rows before adding
throughput caveats:

| Workload | Hardware | Dispatcher | Layout |
|---|---|---|---|
| DSV3 | H100 | DeepEP | TP=2, EP=64, PP=8, VPP=4 |
| DSV3 | GB200/GB300 | HybridEP | TP=1, EP=64, PP=4, VPP=4 |
| Qwen3 235B | H100 | DeepEP | TP=2, EP=32, PP=8, VPP=4 |
| Qwen3 235B | GB200 | HybridEP | TP=1 or 2, EP=32-64, PP=4, VPP=unspecified |
| Qwen3 30B | 16×H100 | HybridEP | TP=1, EP=16, PP=1, plain EP overlap |
| Qwen3.5 35B-A3B | 16×H100 | HybridEP | TP=1, EP=16, PP=1, benchmark EP overlap separately |

For Qwen3 235B on GB200, explicitly say `VPP=unspecified`; do not invent or
extrapolate `VPP=12` unless a measured row provides it. Include TE-scoped CUDA
graph scopes (`attn`, `moe_router`, `moe_preprocess`),
`CUDA_DEVICE_MAX_CONNECTIONS` selection,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NCCL_GRAPH_REGISTER=0`,
GB200/GB300 CPU-side tuning, and the warning not to cargo-cult tracker rows.

## Rounded Performance Bands

These are intentionally rounded so the document stays durable as the tracker
moves. Treat them as planning ranges, not exact promises.

| Workload family | Hardware | Typical band | Representative shape |
|---|---|---|---|
| DSV3, large-scale | H100 | low-to-mid hundreds TFLOPS/GPU, high-teens MFU | TP2, EP64, PP8, DeepEP |
| DSV3, large-scale | B200 | high-hundreds TFLOPS/GPU, mid-teens MFU | TP1, EP32, PP8, DeepEP |
| DSV3, large-scale | GB200 | around 1K TFLOPS/GPU, low-20s MFU | TP1, EP64, PP4, HybridEP |
| DSV3, large-scale | GB300 | above the GB200 band, often mid-20s MFU | TP1, EP64, PP4, HybridEP |
| Qwen3 235B | H100 | low-300s TFLOPS/GPU, around 30% MFU | TP2, EP32, PP8, DeepEP |
| Qwen3 235B | GB200 | high-hundreds TFLOPS/GPU in tuned runs | TP1 or TP2, EP32-64, PP4, HybridEP |
| Qwen3 30B | H100 | high-200s TFLOPS/GPU on the validated 16-GPU shape | TP1, EP16, PP1, HybridEP + EP overlap |
| Qwen3.5 35B-A3B | H100 | low-200s TFLOPS/GPU during 16-GPU GDN-MoE bring-up | TP1, EP16, PP1, HybridEP without EP overlap, router/preprocess TE graphs |
| Qwen3-Next 80B | GB200 | low-300s TFLOPS/GPU in BF16-class runs | TP1, EP32, PP2, HybridEP |

## Representative Config Families

### DSV3 on H100

```text
Dispatcher: DeepEP
TP=2  EP=64  PP=8  VPP=4
Routing: force balance
Recompute: light-to-moderate selective recompute
Priority: overlap communication and keep PP efficient
```

### DSV3 on B200

```text
Dispatcher: DeepEP
TP=1  EP=32  PP=8  VPP=2 or similar
Precision: MXFP8-class
Recompute: selective recompute around MLA up-projection and MLP-side modules
Priority: container quality, PP layout, and DeepEP SMS tuning
```

### DSV3 on GB200 or GB300

```text
Dispatcher: HybridEP
TP=1  EP=64  PP=4  VPP=4
Precision: MXFP8-class
CUDA Graph: attn + moe_router + moe_preprocess
Priority: HybridEP, CPU optimization, and graph-friendly static shapes
```

### Qwen3 235B on H100

```text
Dispatcher: DeepEP
TP=2  EP=32  PP=8  VPP=4
Recompute: norm and activation-side selective recompute
Priority: communication overlap and router-path cleanup
```

### Qwen3 235B on GB200

```text
Dispatcher: HybridEP
TP=1 or 2  EP=32 to 64  PP=4  VPP=unspecified unless measured
CUDA Graph: attn + moe_router + moe_preprocess
Recompute: moe_act, mlp, or norm depending on memory pressure
Priority: balance throughput against memory headroom
```

### Qwen3 30B-A3B on 16 H100

```text
Dispatcher: HybridEP
TP=1  EP=16  PP=1  CP=1
Precision: BF16
Sequence: 4096
Batch: MBS1 GBS1024
Routing: force balance
EP overlap: enabled
Delayed wgrad: disabled
CUDA Graph: moe_router + moe_preprocess
Measured: 20.992s/step, 287.305 model TFLOPS/GPU over iterations 41-50
Rank-0 peak allocated memory: 62.166 GiB
```

This shape improved 17.729% over its reproduced 244.039 TFLOPS/GPU baseline
when only plain EP overlap changed. A matched Nsight A/B increased
communication hidden by GEMM/attention from 0.11% to 36.55%.

### Qwen3.5 35B-A3B on 16 H100

```text
Dispatcher: HybridEP
TP=1  EP=16  PP=1  CP=1
Precision: BF16
Sequence: 4096
Batch: MBS1 GBS128 for controlled A/B
Routing: force balance
EP overlap: disabled
Delayed wgrad: disabled
TE graph scopes: attn, moe_router, moe_preprocess
Measured bring-up: 3.3783s/step, 218.121 model TFLOPS/GPU
Rank-0 peak allocated memory: 69.939 GiB
Exact GBS1024 replay: 26.0814s/step, 225.85 model TFLOPS/GPU
```

On the FlashQLA + pre-GDR stack, switching only the dispatcher from native
all-to-all to HybridEP improved throughput by 20.25%. Enabling plain EP
overlap afterward regressed throughput by 2.45% and raised peak allocated
memory to 71.698 GiB. Enabling shared-expert overlap separately regressed the
scoped-graph result by about 5.1% to roughly 207.0 model TFLOPS/GPU and raised
peak allocated memory slightly to 70.006 GiB. GDN-heavy models therefore need
separate A/Bs for each overlap stream instead of inheriting the Qwen3 30B
overlap setting. Blockwise FP8 also
regressed to 174.38 model TFLOPS/GPU (-7.8% versus BF16) and raised peak
allocated memory to 72.730 GiB. Tensorwise current-scaling FP8 on the pinned
H100 stack was worse still: after a 192.33-second cold compile iteration,
iterations 3-10 averaged 5.0678s/step and 145.3 model TFLOPS/GPU, with
66.053 GiB peak allocated on rank 0. Both FP8 variants completed finite steps,
but neither beat BF16 for these small routed-expert shapes. TE-scoped router
and preprocessing graphs then improved the BF16
HybridEP result by 12.245% to 212.251 model TFLOPS/GPU. Adding the attention
scope improved it by another 2.766% to 218.121 model TFLOPS/GPU.
Replaying the same stack at the Qwen3 comparison batch of GBS1024 averaged
26.0814s/step and 225.85 model TFLOPS/GPU over steps 5-8. Treat the larger
batch as a required validation point, not an assumed multiplier: it improved
throughput only modestly and did not close the model-family gap.
Reducing HybridEP preprocessing SMs from the implementation default 108 to 32
regressed throughput by 0.58% (218.121 to about 216.85 model TFLOPS/GPU), so
the recipe leaves preprocessing SMs at the default.

### Qwen3-Next 80B on GB200

```text
Dispatcher: HybridEP
TP=1  EP=32  PP=2  VPP around 4
CUDA Graph: attn + moe_router + moe_preprocess
Priority: pipeline layout and grouped GEMM quality
```

## Cross-Cutting Patterns

### PP layout

- `E` = embedding
- `t` = transformer
- `m` = MTP
- `L` = loss
- `|` = stage boundary

The biggest platform difference is usually not just the dispatcher. It is the
combination of dispatcher, PP shape, and whether VPP keeps each stage balanced.

### Recompute strategy

| Memory pressure | Starting point |
|---|---|
| low | none or a very narrow selective set |
| moderate | `moe_act`, `mlp`, `norm`, or similar selective modules |
| high | model-specific up-projection plus selective MoE and MLP modules |
| extreme or long-context | full recompute only if the selective path still does not fit |

### Environment variables

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_DEVICE_MAX_CONNECTIONS=32   # common when EP overlap and CUDA graphs are combined
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_GRAPH_REGISTER=0
```

### CPU-side tuning

On GB200 and GB300, CPU affinity and general host-overhead cleanup can move the
needle almost as much as a dispatcher swap. Treat them as first-class tuning
work, not as afterthoughts.

## Pitfalls

1. **Do not cargo-cult a tracker row**: the winning config usually depends on
   routing mode, container, and PP layout as much as on hardware name.

2. **Container quality matters**: large regressions can come from the software
   stack rather than the model recipe.

3. **VPP must be intentional**: a bad VPP split can erase the gain from a better
   dispatcher.

4. **Compare absolute throughput, not only MFU**: MFU can mislead when switching
   between BF16, FP8, and other precision modes.

5. **Force-balance routing is the safer benchmark default**: keep routing mode
   fixed when comparing hardware or dispatcher stacks.

6. **Do not treat the dispatcher table as a hard platform rule**: HybridEP was
   the validated winner for the 16×H100 Qwen3 30B shape, while DeepEP failed
   bring-up in that matched runtime. Benchmark backend compatibility and
   throughput in the production container.
