---
name: nemo-mbridge-perf-moe-optimization-workflow
description: Systematic workflow for MoE training optimization in Megatron Bridge, based on the Megatron-Core MoE paper. Covers the Three Walls framework, parallel folding, recompute strategy, dispatcher choice, and CUDA-graph bring-up. Use for full MoE throughput sweeps, throughput-regression diagnosis, optimizing MoE throughput, MoE performance tuning, or compute, communication, and memory-wall analysis.
license: Apache-2.0
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
GDN-MoE run, HybridEP improved the native all-to-all baseline, but enabling
plain EP overlap afterward regressed. Keep dispatcher and overlap as separate
A/B dimensions: an overlap stream can contend with GDN, expert, or dispatcher
kernels even when the same knob helps an attention-only MoE.

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
When launching through Slurm containers, `--container-env NAME` only forwards
an existing host value; it does not assign one. A batch submitted from a
non-interactive SSH shell can therefore forward empty `HF_HOME`,
`TILELANG_CACHE_DIR`, or `TORCHINDUCTOR_CACHE_DIR` values and either send every
rank to the Hub or silently recompile every job. Export explicit paths under a
mounted persistent directory inside the batch script, then verify the effective
values in the container. For cached model jobs, run a one-task preflight that
loads config and tokenizer with `local_files_only=True` before launching all
training ranks.
Also remember that `TORCHINDUCTOR_COMPILE_THREADS` is applied per rank. For
eight local training ranks, setting it to 32 creates up to 256 compile workers
per node. Size it from available CPU cores divided by local ranks; persistent
cache does not make per-rank CPU oversubscription harmless during its first
population.

### Qwen3.5 H100 measured campaign

Read [references/qwen35-h100-campaign.md](references/qwen35-h100-campaign.md)
when tuning this model family, interpreting a similar Hopper MoE profile, or
checking whether a proposed kernel, overlap, memory, or topology change was
already tested. The accepted short-run point is about 263.67 model TFLOP/s/GPU
at 22.340925 seconds per step on 16 H100 GPUs, still 8.22% below the 287.305
gate. Treat the reference as workload-specific measured evidence; keep the
workflow here as the reusable decision procedure.

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

7. **FP8 is shape-specific on Hopper**: small experts, GDN projections, and
   quantization overhead can make either blockwise or tensorwise current
   scaling slower than BF16. Treat precision and scaling granularity as
   separate A/Bs and keep only a measured win.

8. **Do not optimize the metric by dropping MoE work**: capacity settings can
    reduce routed work. Use them to bound synchronization overhead only unless
    every configured route is preserved.

9. **Revalidate at the acceptance batch**: report the exact target global
    batch instead of extrapolating from a short screening run.

10. **Shared-expert overlap is a separate concurrency A/B**: dispatcher and EP
    overlap results do not predict it; the extra stream can introduce its own
    compute/communication contention.

11. **Validate the live dispatcher contract**: fused HybridEP permutation can
    require dispatch/combine/preprocessing chunk-size agreement that is lost
    between configuration and runtime buffer construction.

12. **Decompose combined kernel patches**: a gain seen with two fusions can
    disappear when the proposed contribution is isolated.

13. **Trace a knob to its consumer before benchmarking**:
    `high_priority_a2a_comm_stream` applies to combined 1F1B stream creation,
    not standalone HybridEP dispatch.

14. **Require iteration progress after an OOM workaround**: a configuration
    that crosses allocation can still deadlock or stall in asymmetric
    communication state.

15. **Account for PAO parameter remainders before estimating memory savings**:
    BF16 PAO already avoids a redundant full FP32 master-weight copy by default.

16. **Do not rewrite routing semantics to remove metadata synchronization**:
    exact-balanced deterministic routes can exercise a different and unsafe
    fused-dispatch communication pattern. The asynchronous API must preserve
    arbitrary valid routes.

17. **Distinguish dispatcher metadata location from its Python type**:
    dynamically sized HybridEP can return a CPU pinned count tensor after a
    stream synchronization, whereas static nonblocking dispatch can return GPU
    counts. Verify device, dtype, lifetime, overflow, and the consuming grouped
    GEMM rather than assuming a tensor-shaped result is sync-free.

18. **Check the architecture guard at the kernel entry point**: a public
    device-offset wrapper can exist on Hopper while its selected grouped-GEMM
    mode remains sm100-only. A small forward/backward/graph probe is cheaper
    and more reliable than a full-model launch.

19. **Audit non-tensor state across partial graph boundaries**: scoped graphs
    can export dispatcher tensors while leaving Python integers, handles, and
    reset logic eager. Any value that defines a communication or GEMM buffer
    shape must remain valid from graph replay through eager expert compute and
    combine, and the next replay must not inherit a reset dynamic state.

20. **Rebaseline after removing a synchronization wall**: an optimization
    that helped a dynamic dispatcher can become neutral after static
    nonblocking dispatch removes the same CPU wait. Re-run the isolated A/B;
    do not multiply historical speedup ratios from overlapping mechanisms.

21. **Assign cache paths before forwarding them into a container**:
    `--container-env HF_HOME` or `--container-env TILELANG_CACHE_DIR` does not
    create a value. Explicitly export mounted persistent Hugging Face, TileLang,
    and TorchInductor cache directories in non-interactive Slurm jobs. Verify
    config/tokenizer with a local-only preflight before a multi-rank launch and
    verify compiler-cache values before attributing cold compilation to a
    kernel change.

22. **Treat high-priority communication as an isolated diagnostic**: on the
    Qwen3.5 grouped-MM overlap regression, the first steady sample changed only
    from 82,700.3 to 82,668.3 ms when high-priority A2A was enabled. A neutral
    priority A/B rules out simple normal-priority stream starvation; profile
    the combined schedule before tuning dispatcher SM reservations.

23. **Budget compile workers per node, not per command**:
    `TORCHINDUCTOR_COMPILE_THREADS` is instantiated by every local rank. Divide
    the node's usable CPU concurrency by `ntasks-per-node`; otherwise eight
    ranks at 32 threads can create 256 compile workers and turn cache
    population into a CPU and shared-filesystem bottleneck.

24. **Use kernel count to separate extra work from schedule inflation**:
    unchanged kernel count plus larger dispatch/NCCL unions, idle gaps, and
    event-sync time points to rank skew and collective rendezvous rather than
    duplicated model computation.

25. **Allocator API time is a hypothesis, not a verdict**: isolate only
    `PYTORCH_CUDA_ALLOC_CONF` and require iteration-2 progress. Native allocation
    can remove expandable-segment VMM calls yet perform worse because of
    fragmentation and delayed reuse.

26. **Do not fix cross-stream retirement by keeping everything alive**:
    compare first-step peak memory and require iteration 2. On the measured
    Qwen3.5 schedule, combine-input retention added about 5.8 GiB and consumed
    the headroom needed for optimizer-state materialization.

27. **Retire on the owner stream only after an explicit dependency**:
    this preserved the memory benefit and improved the measured Qwen3.5
    overlap stall by about 18%. The matched profile reduced event-sync time by
    65%, major VMM call counts by about 25%, and dispatch by 22%, establishing
    the mechanism. Because it remained nearly 3x slower than no overlap,
    preserve it as diagnostic evidence and profile the next scheduler
    dependency before recipe adoption.

28. **Verify launch-order environment values in the live PID**:
    `CUDA_DEVICE_MAX_CONNECTIONS=1` improved the measured Qwen3.5 owner-release
    schedule by about 49% throughput versus 32 connections, with unchanged peak
    allocation. The serialized recipe still carries its default, so process
    environment is the source of truth. Keep the no-overlap run in the decision
    table: a large overlap-relative gain can still be an end-to-end loss.

29. **Measure useful intersection after fixing rank drift**:
    connection count 1 cut NCCL by 95% but also cut useful HybridEP/expert
    intersection by 93%. Continue with the smallest incremental concurrency
    sweep and reject any setting that restores rendezvous inflation faster than
    it restores useful overlap.

30. **Separate a packaged FP8 training wrapper from its kernel API**:
    TorchAO MXFP8 can be SM100-only while the same Hopper PyTorch build exposes
    lower-level tensorwise or rowwise scaled grouped GEMM. Probe the exact
    forward, dgrad, and 2D-by-2D wgrad contracts before either rejecting the
    hardware path or attempting a full model.

31. **Require the offset dtype after every metadata transform**:
    grouped-MM offsets must remain CUDA `int32`; a `cumsum` without an explicit
    dtype can promote `int32` expert counts and fail before the kernel. Check
    device, dtype, monotonicity, and static-tail semantics in the primitive
    probe.

32. **Do not replace dynamic host splits with coarse fixed expert slots**:
    on the measured H100 shape, fully device-side repacking into fixed
    2,304-token slots was slower than the already losing variable-split TE path
    and later stopped during graph capture. Removing a host synchronization is
    not a win when it multiplies padded expert work.

33. **Profile shared-expert overlap as its own stream**:
    identify the stream's kernel names, sum, primary-stream intersection, and
    dispatcher intersection. A positive end-to-end result can still hide only
    a minority of shared work; the measured Qwen3.5 side stream overlapped the
    primary stream for about 30.8% of its duration.

34. **Stream priority cannot enlarge dependency windows**:
    after shared-expert overlap improved Qwen3.5 by 0.44%, changing only its
    stream to high priority regressed 0.18%. If the trace shows explicit waits
    before FC1/FC2 or combine, optimize those legal windows rather than assuming
    a higher-priority queue can create new concurrency.

35. **Inspect the scaled-GEMM dispatch table, not just its enum**:
    the measured PyTorch build advertised tensorwise scaling but registered no
    tensorwise grouped-GEMM implementation. Its rowwise Hopper path was
    numerically healthy but 7.39x/3.52x slower than BF16 FC1/FC2 even with
    cached weight scales. Gate on a real-shape cached-forward probe before
    spending an allocation on autograd or a full model.

36. **Separate Slurm node count from workers per node and QoS usage**:
    `--nodes=2 --ntasks-per-node=8` requests two eight-GPU nodes, not eight
    nodes. An interactive QoS `GrpNodes=8` error can mean other users already
    consume the partition-wide eight-node pool. Record the exact allocation and
    use an approved batch partition on the same hardware pool when interactive
    is saturated; do not change the benchmark's node count to work around QoS.

37. **Include repack, padding, gather, and backward in fixed-shape GEMM gates**:
    on the measured Qwen3.5 shape, padded batched GEMM was 2.06x slower in
    forward and 2.56x slower through backward despite exact outputs. A faster
    isolated dense GEMM cannot justify a model experiment when its required
    layout contract loses at the complete primitive boundary.
