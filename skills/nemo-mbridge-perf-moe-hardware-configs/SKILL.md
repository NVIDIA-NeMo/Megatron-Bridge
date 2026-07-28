---
name: nemo-mbridge-perf-moe-hardware-configs
description: Representative MoE training playbooks by hardware platform and model family. Summarizes rounded throughput bands, parallelism patterns, and common tuning stacks. Use for hardware-specific MoE playbooks, throughput estimates, MoE on H100, GB200 configuration, expected throughput, or parallelism selection for B200.
license: Apache-2.0
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
| Qwen3.5 35B-A3B | H100 | mid-200s TFLOPS/GPU during 16-GPU GDN-MoE bring-up | TP1, EP16, PP1, HybridEP without EP overlap, eager + shared-expert overlap |
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
Batch: MBS1 GBS1024
Routing: force balance
EP overlap: disabled
Delayed wgrad: disabled
CUDA graphs: eager accepted; attn/router/preprocess scopes were neutral on the final path
Shared-expert overlap: enabled, normal-priority stream
Measured bring-up: mid-200s model TFLOPS/GPU
Exact GBS1024 development replay: 235.55 model TFLOPS/GPU, 25.007s/step
Experimental grouped-MM no-overlap replay: about 250.5 model TFLOPS/GPU, 23.516s/step
Current short-run winner: about 263.67 model TFLOPS/GPU, 22.341s/step
```

On the measured development stack, HybridEP improved the matching native
all-to-all baseline by about 20%, and scoped router/preprocess/attention graphs
added roughly 15% over the earlier eager HybridEP configuration. Fused HybridEP
permutation with runtime-compatible chunk sizes then reduced exact-batch step
time by 4.12%. After static dispatcher metadata and one-connection launch
ordering, the matched eager path was noise-level faster than those graph
scopes. Plain EP overlap, tested FP8 modes, isolated GDN/QK fusions, and
non-default FlashQLA/HybridEP chunking regressed. Normal-priority shared-expert
overlap was the exception: it improved the final eager control by 0.44% to the
current 22.341-second / 263.67-TFLOP/s/GPU short-run winner. The playbook keeps
BF16, eager execution, and shared overlap while treating every overlap or
fusion as an independent A/B. This is development evidence, not yet the
conservative public recipe, and remains below the Qwen3 287.305 TFLOPS/GPU
gate.

For this exact H100 shape, connection count was not neutral once the
fine-grained combined schedule was enabled. After a dependency-aware
owner-stream storage-release fix, changing only
`CUDA_DEVICE_MAX_CONNECTIONS=32` to `1` reduced steps 2-3 from a 66.322-second
mean (about 88.9 model TFLOP/s/GPU) to 44.465 seconds (about 132.45
TFLOP/s/GPU), with the same 72.028-GiB iteration-2 peak allocation. The result
reproduced over two finite steps but was still 1.89x slower than the 23.516-
second no-overlap path. Treat connection count as a launch-order A/B for H100
overlap, not a platform-wide constant or a reason to enable a losing schedule.
The matched profile showed why: connection count 1 reduced NCCL active time
from 14.061 to 0.665 seconds, dispatch from 30.196 to 11.162 seconds, and
`cudaEventSynchronize` calls from 4,524 to 426. It also reduced useful
HybridEP/expert intersection from 2.283 to 0.156 seconds. The lower connection
count largely restored launch ordering by serializing the schedule; remaining
time was dispatch plus idle gaps, not useful overlap.

The adjacent `CUDA_DEVICE_MAX_CONNECTIONS=2` control did not find a middle
ground: its two steady steps were 47.180 and 53.532 seconds, averaging 50.356
seconds and about 117.0 model TFLOP/s/GPU. That is 13.25% slower by step time
than one connection, with 64 allocator retries at iteration 2 despite finite
numerical checks. Stop at this nearest-neighbor regression; do not spend H100
allocations on 4/8 merely because those values lie between serialized and
default launch ordering.

The same no-overlap shape rejected a hardware-scale dispatcher SM cap.
Increasing `moe_flex_dispatcher_num_sms` from 32 to 108 regressed the steps
5--8 mean from 23.516 to 28.212 seconds (about 250.5 to 208.8 model
TFLOP/s/GPU) with finite numerical checks. Reducing the cap from 32 to 20
improved the mean only to 23.335 seconds (252.45 model TFLOP/s/GPU), a 0.78%
step-time gain. Reducing it again to 16 improved only another 0.46%, to 23.228
seconds (253.60 model TFLOP/s/GPU). Keep 16 as the measured point for this
exact stack, but treat it as a shallow, diminishing-return local optimum; 108
is the preprocessing default on this implementation, not a universal
dispatch/combine optimum. The lower boundary matters too: 12 regressed the
mean by 9.28% versus 16, to 25.385 seconds and 232.06 model TFLOP/s/GPU.

That preprocessing default is an independent resource field, not an instruction
to use 108 everywhere. DeepEP 1.2.1+34152ae instantiated the fused Qwen3.5
template with preprocess/permute/unpermute at 108 blocks and dispatch/combine
at 16. Changing only preprocessing to the 32-block value used by other Bridge
performance recipes left the other budgets unchanged and regressed exact
2-node steps 5--8 from 22.341 to 22.434 seconds (263.67 to about 262.57 model
TFLOP/s/GPU). Print the live template and benchmark each field independently;
cross-model recipe values are starting hypotheses.
Changing only fused-unpermute from 108 to 32 blocks reinforced that rule:
exact 2-node steps 5--8 regressed to 23.400 seconds / about 251.73 model
TFLOP/s/GPU, 4.742% slower than the 22.341-second winner. Even equal default
block counts do not make preprocessing, permute, and unpermute interchangeable
tuning dimensions.
The same applies to chunk fields across schedule generations. A source audit
showed that the apparent combine-only 128-to-64 comparison on the final
static-dispatch/shared-overlap path never changed the live kernel template.
The compatibility shim selected the dispatch chunk and copied it into
preprocessing and combine for fused mode. Dispatch remained at its 64-token
default, so both intended settings became effective 64/64/64 templates. The
22.341- versus 22.577-second difference cannot be attributed to combine
chunking. Print the post-normalization template; environment values alone are
not configuration evidence. The corrected full-field A/B set preprocessing,
dispatch, and combine to 128. A same-stack Configurer audit returned effective
128/128/128, stages 10/4/2, blocks 108/108, and `valid=True`; the exact-2-node
run averaged 23.5981 seconds / about 249.62 model TFLOP/s/GPU over finite
iterations 2--3. That is a 5.627% step-time regression versus the effective
64/64/64 winner, so retain 64 for this H100 schedule.

An experimental asymmetric-budget follow-up likewise did not transfer from the
profile to end-to-end throughput. Keeping dispatch at 16 SMs and raising only
combine to 20 on the same exact 2-node winner changed steps 5--8 from 22.341
to 22.500 seconds (about 263.67 to 261.81 model TFLOP/s/GPU), a 0.712%
regression. The longer measured combine active union was not evidence that
combine needed more resident SMs; treat per-phase SM splits as independent
A/Bs, not proportional allocations derived from trace duration.

With that 16-SM point fixed, changing only
`CUDA_DEVICE_MAX_CONNECTIONS=32` to `1` improved the exact 2-node steps 5-8
mean from 23.228 to 22.451 seconds and throughput from 253.60 to 262.38 model
TFLOP/s/GPU. This 3.46% step-time gain is the current short-run winner and
leaves an 8.67% throughput gap to the 287.305 acceptance target. It establishes
connection count as a workload-specific launch-order knob even when explicit
EP overlap is disabled; it does not establish communication hiding.

A matched graph A/B then disabled TE-scoped graphs while holding the 16-SM
dispatcher, connections=1, and static rank capacity 1.05 fixed. Eager averaged
22.439 seconds / about 262.52 model TFLOP/s/GPU versus 22.451 / 262.38 with
scoped graphs. The 0.053% difference is noise-level; record graphs as neutral
on this final static path and keep the simpler eager configuration. The
remaining acceptance gap is about 8.63%.

The matched rank-0 trace kept kernel count at 401,840 and showed the mechanism:
connections=1 cut HybridEP active union 8.77% and expert/linear union 1.58%
versus 32, but increased idle gaps 15.02% from 4.204 to 4.836 seconds.
HybridEP/expert intersection remained zero. Treat the result as cheaper serial
launch ordering with an idle tradeoff, and test the adjacent connection count
on the same no-overlap shape before generalizing the value.

The adjacent no-overlap connections=2 control averaged 22.502 seconds / about
261.78 model TFLOP/s/GPU, 0.28% slower than connections=1. Added launch
concurrency did not recover the idle tradeoff, so keep one connection for this
exact stack and stop the sweep.

Tensorwise global FP8 was also negative on this exact final shape even when
the grouped experts stayed in BF16. It completed eight numerically healthy
steps, but steps 5-8 averaged 24.205 seconds / about 243.36 model TFLOP/s/GPU,
7.87% slower by step time and 7.30% lower by throughput than BF16. Its cold
first iteration took 154.335 seconds and observed memory approached 78.4
GiB/GPU. For H100 GDN/MoE hybrids, do not infer a win from reduced precision
alone: benchmark the full precision boundary, compile cost, and memory
high-water mark.

Changing only that mixed path from tensorwise to blockwise scaling improved
rank-0 first-step max allocated/reserved memory to 63.369/67.228 GiB, but steps
5-8 still averaged 24.213 seconds / about 243.28 model TFLOP/s/GPU. The 0.03%
step-time difference from tensorwise was noise-level. On this shape, blockwise
was a memory-headroom result, not a throughput result, while experts remained
BF16.

Enabling only shared-expert overlap on the final BF16 configuration produced a
new short-run winner: 22.341 seconds / about 263.67 model TFLOP/s/GPU over
steps 5-8, 0.44% faster by step time than the 22.439-second eager control. It
remains about 8.22% below the 287.305 target. The matched trace moved 0.690
seconds of shared-expert work to a side stream but overlapped only 0.213
seconds with the primary stream. Connections=2 regressed to 22.937 seconds,
and a high-priority shared stream regressed to 22.381 seconds. For this exact
2-node H100 stack, retain shared overlap, connections=1, and normal stream
priority.

A later reviewable fused gated-RMSNorm implementation added a second small,
attributable win without changing the recipe topology. On the same two-node
allocation, its fused-GDN control averaged 23.4425 seconds / 251.30 model
TFLOP/s/GPU over steps 5--8, and fusing the GDN output RMSNorm with its SiLU
gate averaged 23.1225 seconds / 254.75 model TFLOP/s/GPU. The 1.365% step-time
and 1.373% throughput gains passed a real-shape BF16 output/gradient parity
probe and both distributed runs exited 0:0. Retain the fusion, but keep
22.341 seconds / 263.67 as the best absolute stack measurement until a matched
fast-node rerun supersedes it; the slower-node A/B proves relative attribution,
not the 287.305 acceptance gate. That matched normal-speed follow-up subsequently
averaged 22.152850 seconds / 265.925 model TFLOP/s/GPU for the retained fusion,
which is the new absolute short-run point. It remains 7.442% below the gate and
needs another 8.040% throughput improvement.

Transformer Engine's global `fused_residual_rmsnorm` flag did not stack with
the local GDN norm+gate fusion. On the same allocation, changing only that flag
to true averaged 22.190525 seconds / 265.450 model TFLOP/s/GPU, a 0.170%
step-time and 0.179% throughput regression. Both sides completed eight finite
steps with zero skipped or NaN iterations and successful exits. Keep the local
GDN fusion and leave the global residual fusion disabled on this recipe.

Retain `NVTE_FWD_LAYERNORM_SM_MARGIN=20` and
`NVTE_BWD_LAYERNORM_SM_MARGIN=20` for this exact stack. A bracketed A/B/A on
the same two-node allocation changed only the backward margin to 16. Steps
5--8 averaged 22.199950 seconds for the first margin-20 control, 22.180425
seconds for margin 16, and 22.178600 seconds for the repeated margin-20
control. The candidate was only 0.040% faster than the two-control mean while
the controls drifted 0.096%, so the apparent benefit was noise rather than a
recipe improvement.

Do not transpose grouped-expert weights merely to make their KxN view
contiguous. At the production expert shape, a corrected BF16 primitive probe
preserved output and gradients at cosine 1.0, but the contiguous KxN layout
regressed forward+backward by 0.181% (1.800661 versus 1.797403 milliseconds).
Likewise, FP8-storing the weighted-SwiGLU activation is a memory trade, not a
standalone H100 speed optimization: it saved 34 MiB per expert call and
preserved output/gradient cosines of at least 0.999288082, but increased
forward+backward latency 14.428% (1.419549 versus 1.240563 milliseconds).
Test it end to end only when the memory saving enables a larger microbatch.

That larger-microbatch test failed the end-to-end feasibility gate. With
microbatch size 2 and FP8 activation-function input storage enabled, all 16
H100s held about 80.7--81.0 GiB, half the ranks remained in persistent
100%-busy kernels while their peers waited, and no first iteration completed
before the 15-minute job timeout. Reject the combination: it produced neither
a finite loss nor a throughput sample and cannot replace the microbatch-1
recipe.

Adding TE-scoped attention/router/preprocess graphs to that shared-overlap
winner did not improve it. Steps 5-8 averaged 22.365 seconds / about 263.39
model TFLOP/s/GPU, 0.106% slower than eager, while iteration 1 grew to 150.351
seconds and emitted AccumulateGrad stream-mismatch warnings. Keep the eager
shared-overlap configuration.

An exact delayed grouped-wgrad experiment also remained below the winner.
The real-shape primitive split had bitwise-matching gradients and shortened
the immediate dgrad-only path by 21.1%, but enabling MCore dispatch-backward /
expert-wgrad scheduling at connections=1 averaged 22.489 seconds / about
261.93 model TFLOP/s/GPU over steps 5-8. That is 0.664% slower than the eager
shared-overlap winner. Retain immediate grouped wgrad on this topology; a
theoretical overlap window must be confirmed by the full dependency schedule.

Compiler autotune did not transfer from a uniformly segmented primitive to the
full HybridEP schedule. A balanced 32,768-row grouped expert MLP appeared
26.92% faster through backward under max autotune than an ordinary PyTorch
SwiGLU control, but the full model at its 34,416 aligned-row budget averaged
22.537 seconds / about 261.38 model TFLOP/s/GPU, 0.878% slower than the eager
winner. Uniform-offset primitives at 33,424, 33,760, 34,080, and 34,416 rows
all remained about 25.0--25.6% faster than that ordinary control.
Reserving SMs from the compiler did not close that gap on the uniform
34,416-row primitive. At carveout 0/16/24, compiled forward+backward took
75.00%/75.96%/78.81% of eager and compiled-plus-shared concurrent makespan
took 75.99%/76.07%/80.21% of its eager control. The 16-SM point was neutral
for concurrency and 24 regressed, with cosine at least 0.99999356 throughout.
Stop the upward sweep rather than spending a full-model allocation. A final
34,416-buffer/32,768-active-row control included 40-call bursts and the actual
MCore fused weighted-SwiGLU eager path. Compiled single-call forward/backward
was 7.05%/5.11% slower than that production reference. Compiled bursts were
1.57%/4.60% faster, but shared-MLP concurrent makespan was 7.44% slower.
Inactive tail and repeated-launch pressure therefore did not rescue this path.
Hopper grouped-GEMM gates must replay the production fused/autograd reference,
capacity, segmentation, and concurrency before an end-to-end schedule.

Native TE and packaged TorchAO did not provide an H100 expert-FP8 endpoint.
Variable-split TE averaged about 33.18 seconds, fixed 2,304-token slots averaged
about 36.76 seconds before a later graph-capture stall, and TorchAO's optimized
MXFP8 grouped training path required SM100. Do not conflate that packaging
boundary with PyTorch kernel availability: the same container exposes
lower-level H100 `scaled_grouped_mm` primitives. Exact real-shape probes
nevertheless rejected them for this recipe. Tensorwise had no registered GEMM
implementation despite appearing in the Python enum. Rowwise reached about
0.9993 cosine similarity to BF16, but cached-scale FC1/FC2 forward was
7.39x/3.52x slower; dynamic scale generation was slower still. The container
also lacked the setter needed to enable the newer cuBLASLt grouped-GEMM
preference. Its NVIDIA PyTorch `2.12.0a0+0291f960b6.nv26.04` build also lacked
the current-main public
`torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm` property, as an
exact-2-node fail-closed probe confirmed before timing. Treat API presence,
numerical support, and competitive latency as three separate gates.

A fixed-capacity BF16 batched-GEMM alternative was also noncompetitive. With
16 experts, 2,048 valid rows/expert, 2,304-row capacity, and all required
repack/gather/SwiGLU work included, it was 2.06x slower than grouped MM in
forward and 2.56x slower through backward, while using 0.238 GiB more peak
memory. Do not replace variable grouped MM with padded `bmm` on this shape.

Host placement was also neutral on the accepted GPU-bound path. An exact
2-node launcher bound local ranks 0--3 to NUMA node 0 and ranks 4--7 to NUMA
node 1 on each H100 host, with all 16 bindings verified in startup output.
Steps 5--8 averaged 22.341025 seconds / about 263.6688 model TFLOP/s/GPU,
only 0.000448% slower than the unbound 22.340925-second winner. Keep explicit
NUMA placement as launcher hygiene when it improves reproducibility, but do
not assign it a throughput gain without a matched end-to-end A/B.

A workload-equivalent folded-topology A/B also retained TP1. TP2/EP16/ETP1
with MBS2 and sequence parallel kept expert EDP1, 64 microsteps, and the
winner's approximate routed rows/rank and rows/local-expert. Bridge enabled
its default TP AG/RS overlap. The exact 2-node run completed eight finite steps,
but steps 5--8 averaged 26.219175 seconds / about 224.669 model TFLOP/s/GPU,
17.359% slower by step time than TP1. Rank-0 iteration-2 peak
allocated/reserved memory rose to 66.504/68.373 GiB. Preserve expert shapes
when comparing folded layouts, but still account for dense shard efficiency,
TP/SP communication, and compile-cache keys. On this 35B-A3B H100 shape, TP1
remains the measured choice.

The validation allocation is two nodes with eight H100 workers per node:
`--nodes=2 --ntasks-per-node=8`. The second value does not request eight
nodes. On the measured cluster, an interactive QoS group limit of eight nodes
was temporarily consumed by other one-node jobs, so exact-2-node probes used
the approved batch partition on the same H100 pool. Queue choice is operational
metadata; never silently change the hardware count or topology.

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

Set only one `CUDA_DEVICE_MAX_CONNECTIONS` value in a run and verify it from
the live training process. Recipe environment dictionaries are defaults:
explicit launcher values can take precedence even when the serialized config
still prints another value. On H100, compare `1` and the recipe default for the
exact overlap/graph shape; connection count can materially change cross-stream
launch ordering and distributed rank drift.

### CPU-side tuning

On GB200 and GB300, CPU affinity and general host-overhead cleanup can move the
needle almost as much as a dispatcher swap. Treat them as first-class tuning
work, not as afterthoughts. On H100, the measured Qwen3.5 GPU-bound path was
neutral under verified per-rank NUMA binding, so remeasure instead of
transferring the GB200 expectation.

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

7. **Do not cargo-cult connection count across sibling models**: `32` supported
   the validated Qwen3 overlap/graph shape, while `1` materially improved the
   measured Qwen3.5 combined schedule. Validate both end-to-end and keep
   no-overlap in the comparison.

8. **Do not infer a TP win from equivalent expert rows**: the measured
   TP2/EP16/ETP1/MBS2/SP Qwen3.5 layout preserved expert work shape but
   regressed step time 17.359% versus TP1.
