---
name: nemo-mbridge-perf-moe-dispatcher-selection
description: Choose the right MoE token dispatcher, including alltoall, DeepEP, HybridEP, or experimental NCCL EP, for the hardware, EP degree, and optimization stage. Summarizes DSV3, Qwen3, Qwen3-Next, and VLM bring-up patterns. Use for dispatcher selection, dispatcher-related regressions or crashes, alltoall versus DeepEP, HybridEP, MoE dispatcher, flex backend, or EP-dispatcher selection.
license: Apache-2.0
---

# MoE Dispatcher Selection Guide

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-dispatcher-selection/card.yaml

## Quick Decision

### By hardware

| Hardware | First choice | Why |
|---|---|---|
| H100 | A/B DeepEP and HybridEP after proving both runtime paths | Runtime compatibility and workload shape can outweigh the platform default; NCCL EP is experimental and its static fast path is not available on Hopper |
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
| Qwen3.5 35B-A3B text | HybridEP on the measured 16×H100 stack | About 20% faster than the matching native all-to-all baseline |
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

The experimental `ncclep` flex backend has a separate build gate. Import
`transformer_engine.pytorch.ep` from the exact training container and require
`EpBuffer`, `ep_bootstrap`, `ep_dispatch`, `ep_combine`, and `ep_finalize`.
Those symbols require Transformer Engine's NCCL-EP extension. In the pinned
TE 2.17 source, the build control is `NVTE_WITH_NCCL_EP`: the extension is on
by default when an SM90-or-newer architecture is targeted, and
`NVTE_WITH_NCCL_EP=0` disables it. A source build also requires the recursive
NCCL submodule and NCCL 2.30.4 or newer. Do not confuse this with the similarly
named MCore feature switches, and do not assume that updating MCore or the TE
source pin changes the immutable runtime image.
For the Qwen3.5 H100 campaign, a read-only inventory of the exact training
image found TE 2.15 and no `transformer_engine.pytorch.ep` module, even though
the project lock points at a later source commit containing the extension. The
import probe was cancelled before allocation because the immutable-image
inventory had already failed the capability gate. Record this as an unavailable
container endpoint, not an NCCL EP performance result.

On Hopper, NCCL EP's dynamic-shape mode is the supported path. It narrows the
receive buffer using `tokens_per_expert.sum().item()`, so it introduces a D2H
synchronization and serializes the 1F1B expert-communication overlap boundary.
That makes a non-overlapped dispatcher A/B valid, but it prevents claiming the
static/overlapped fast path. Static shape needs SM100+, the TE operation fuser,
and `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`; the current manager also rejects its
nominal symmetric-memory zero-copy option.

Package import and model construction are necessary but not sufficient. Require
the target multi-node topology to complete a real dispatch and combine before
accepting any timing. On a 16×H100 Qwen3.5 text candidate, DeepEP imported,
constructed the model, entered the training loop, and exposed device-resident
expert counts to an experimental Hopper grouped-MM path. Rank 8 nevertheless
timed out in the first inter-node dispatch (`timeout (dispatch CPU)`) and then
aborted during CUDA illegal-address cleanup. Zero iterations completed, so the
result is a runtime/topology compatibility failure, not a dispatcher throughput
ranking. The DeepEP implementation explicitly requires adaptive routing to be
disabled; verify that cluster setting before retrying.

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

When replacing a HybridEP recipe, clear HybridEP-only rank capacity with a
true YAML/Python null. In CLI overrides, use the runner's `null` spelling; a
literal `None` can remain the string `"None"` and fail MCore validation under
the DeepEP backend.
First confirm the DeepEP package imports in the target container, then require
a complete dispatch/combine and steady iteration. Initialization alone is not
performance evidence.

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

For the detailed Qwen3.5 H100 HybridEP campaign, read
[references/qwen35-h100-hybridep.md](references/qwen35-h100-hybridep.md).
Use it when validating fused-template chunks, static device metadata, grouped
expert backends, graph-state lifetime, or independent SM/block budgets. The
portable rules are:

- log the post-normalization template before interpreting an A/B;
- preserve arbitrary routing, device-count lifetime, and overflow assertions;
- probe the installed grouped-expert backend on real shapes;
- treat graphs, capacity, chunk sizes, and phase resource budgets as separate
  end-to-end A/Bs.

### NCCL EP (experimental)

Select the backend with
`moe_token_dispatcher_type="flex"` and
`moe_flex_dispatcher_backend="ncclep"`. Set
`moe_expert_rank_capacity_factor` explicitly: it sizes the per-rank receive
buffer, and exceeding the budget hard-traps instead of softly dropping routes.
Start with `moe_ncclep_static_shape=false` and
`moe_ncclep_use_symm_mem=false`.

Treat its two shape modes as different hardware paths:

- dynamic shape narrows the received buffer with
  `tokens_per_expert.sum().item()`. It is valid on H100 when the TE NCCL EP
  build is present, but the D2H synchronization serializes the 1F1B overlap
  boundary;
- static shape avoids that narrowing and is the intended overlap/CUDA-graph
  path, but it requires SM100+, the TE operation fuser, and
  `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`. Do not enable it on H100;
- symmetric-memory/zero-copy payload buffers are not implemented in the
  current manager and must remain disabled.

Use a staged gate: exact-container TE-symbol probe, real model-module import,
fresh-cache forward/backward smoke for every JIT/native model kernel, exact
multi-node dispatch/combine correctness, then unprofiled end-to-end training.
The import probe is capability evidence only. A successful bootstrap is still
not a throughput result, and the dynamic Hopper path should be compared
against the same no-overlap HybridEP winner before considering overlap.

Do not reuse a warm JIT cache as the dependency gate for a new container. In a
Qwen3.5 TE 2.17 bring-up, TE's NCCL-EP symbols passed while the container's
prebuilt causal-conv extension had a torch ABI mismatch. After rebuilding it,
a fresh FlashQLA cache exposed that FlashQLA 0.1.2 requires TileLang 0.1.9 and
apache-tvm-ffi 0.1.9 even though the warm torch 2.12 winner had run with the
container's TileLang 0.1.8. Build or install exact package versions in an
isolated environment, assert their import paths, and execute model-shape CUDA
forward and backward before launching all ranks.

On the measured exact-2-node H100 Qwen3.5 path, that gate passed with torch
2.13, CUDA 13.3, TE 2.17, and TileLang 0.1.9. The matched HybridEP control
averaged 24.236950 seconds / 243.043440 model TFLOP/s/GPU over its two
post-compile steps. A minimal NCCL-EP buffer probe then separated single-node
capability from the required topology: eight ranks on one H100 node registered
the symmetric window successfully, but the same shape on 16 ranks across two
nodes failed at the first
`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` call. Setting the
NCCL-contrib recommendation `NCCL_GIN_TYPE=3` was not sufficient. NCCL logged
`SPCX GPUNETIO dlopen failed` and failed
`ncclGinGdakiCreateContext` with error 2 before window setup. The installed
Spectrum-X and DOCA libraries had no missing `ldd -r` dependencies, while the
plugin's embedded GPUNetIO loader was not a dynamic symbol that an external
path probe could call. Treat this as a Spectrum-X/GDAKI runtime-plugin
compatibility failure, not a Bridge backend or capacity result.

The full NCCL-EP candidate was consistent with that isolated boundary: it
reached iteration 0, but its first real dispatch kept every GPU idle until the
backend's 101010 ms timeout; all 16 ranks emitted
`NCCL error 2 at nccl_ep.cc:886`, and the error did not propagate back to
Python. Reject a backend that passes only Python/process-group bootstrap or a
single-node symmetric-window probe. Require a minimal exact-topology window
registration followed by completed multi-node dispatch and combine before
collecting end-to-end timing.

Dispatcher selection also interacts with launch ordering after the SM budget is
chosen. On the same exact 2-node no-overlap path with `num_sms=16`, changing
only `CUDA_DEVICE_MAX_CONNECTIONS` from 32 to 1 reduced the steps 5-8 mean from
23.228 to 22.451 seconds and raised throughput from 253.60 to 262.38 model
TFLOP/s/GPU. The 3.46% step-time gain shows that HybridEP's internal streams
can remain launch-order sensitive even when Bridge's explicit EP overlap is
disabled. Treat the value as a same-workload A/B, not a dispatcher-wide
default or evidence that communication was hidden.

Rebaseline scoped graphs after changing launch ordering. With `num_sms=16`,
connections=1, and rank capacity 1.05 fixed, disabling TE-scoped graphs changed
the steps 5-8 mean from 22.451 to 22.439 seconds (262.38 to about 262.52 model
TFLOP/s/GPU). The 0.053% eager advantage is noise-level, so scoped graphs
provide no measurable benefit after static dispatch and one-connection launch
ordering. Prefer the simpler eager control unless a longer run proves a stable
difference.

The matched no-overlap profile explained why connections=1 won. Relative to
the older connections=32 trace, HybridEP active union fell 8.77% from 7.376 to
6.729 seconds, led by a 13.75% dispatch reduction. Expert/linear union fell
1.58%, but idle gaps increased 15.02% from 4.204 to 4.836 seconds. Kernel count
remained 401,840 and HybridEP/expert intersection remained zero. Connection
count therefore changed serial dispatcher cost and idle scheduling rather than
hiding communication. Use this tradeoff to justify only the adjacent
no-overlap connections=2 A/B; do not infer its outcome from a different
combined-overlap schedule.

That adjacent no-overlap point averaged 22.502 seconds / about 261.78 model
TFLOP/s/GPU over steps 5-8, 0.28% slower than connections=1. The extra
concurrency did not recover the profiled idle penalty. Keep connections=1 for
this dispatcher shape and stop the launch-order sweep.

Do not turn the static rank-capacity factor into an unchecked memory/performance
knob. On that same current winner, reducing only the factor from 1.05 to 1.02
triggered the device-side `HybridEP static rank capacity overflowed and dropped
routed tokens` assertion in the first training iteration. It produced zero
valid throughput samples. Keep 1.05 as the measured safe value for this exact
route distribution and fail closed on any lower factor that overflows.

A matched rank-0 Nsight comparison separated that gain from unrelated kernels.
The grouped/static path reduced expert/linear active union from 8.656 to 8.115
seconds (6.25%), HybridEP active union from 7.723 to 7.376 seconds (4.49%), and
summed host `cudaStreamSynchronize` wait from 7.422 to 1.679 seconds (77.4%).
GDN moved only from 1.355 to 1.334 seconds (1.5%). The remaining expert/linear
and HybridEP regions were still serial, so the next target is their native
nonblocking handoff or overlap rather than another GDN microkernel. Of the 154
remaining host synchronizations, 136 came from `aten::copy_`; 128 were the two
BF16 conversion copies per microbatch in native fused cross-entropy backward.
The isolated Transformer Engine cross-entropy A/B averaged 23,514.2 ms over
steps 5-8 versus 23,516.2 ms with native cross-entropy, a 0.0085% difference
within noise, and MCore warned that the TE implementation had known stability
issues. The host waits overlapped other GPU work and were not recoverable wall
time. Treat profile attribution as a hypothesis until an end-to-end A/B
changes the acceptance metric.

Do not reduce EP solely to keep a HybridEP group inside one NVLink domain
without budgeting the resulting expert data parallel replicas. In a matched
2-node, 16-H100 Qwen3.5-35B-A3B control, changing EP from 16 to 8 made each
HybridEP group node-local but introduced expert-DP=2. Model and optimizer
construction fit at about 41.7 GiB/GPU, and the first iteration completed with
finite loss, `skipped=0`, and `nan=0`. The second iteration then OOMed on all
ranks while creating optimizer state: process usage was about 78.2 GiB/GPU
and each rank needed another 1.89 GiB. Treat EP topology changes as a joint
communication, parameter-replication, optimizer-state, and gradient-reduction
decision; crossing initialization or one iteration is not a memory-fit proof.

Do not substitute `moe_expert_capacity_factor` plus pad-to-capacity unless
token dropping and padded work are part of the intended training semantics.
For a dropless benchmark, verify that every configured route is still executed.

Validate the EP group against the physical ranks per node before launch. Some
HybridEP runtimes require the EP group size to be divisible by the local GPU
count even when the full parallel mesh divides the world size.

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

### NCCL EP

- experimental validation of a TE build compiled with NCCL EP support
- dynamic-shape H100 dispatcher A/Bs where a D2H shape sync is acceptable
- static-shape overlap only on SM100+ with the CuTe DSL fused grouped MLP

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

7. **Fused permutation is a runtime contract**: verify the instantiated
   HybridEP buffer's dispatch/combine/preprocessing chunk sizes before
   attributing a gain or failure to the fusion flag.

8. **Preserve arbitrary routing and metadata lifetime**: do not replace
   force-balance routes or discard the comm-stream `tokens_per_expert` tensor to
   avoid a host synchronization.

9. **NCCL EP source availability is not runtime availability**: require the
   exact container to expose the TE EP module and all five lifecycle symbols.

10. **Do not transfer the NCCL EP static path to Hopper**: SM100+, the TE
    operation fuser, and the CuTe DSL fused grouped MLP are hard prerequisites;
    dynamic H100 mode retains a D2H shape synchronization.

11. **Gate NCCL EP on the exact multi-node GDAKI path**: a one-node symmetric
    window can pass while the same 16-rank shape fails in Spectrum-X/GPUNetIO
    context creation. Check the recommended GIN mode and plugin dependencies,
    but do not launch a training A/B until the exact-topology window,
    dispatch, and combine probes all complete.
