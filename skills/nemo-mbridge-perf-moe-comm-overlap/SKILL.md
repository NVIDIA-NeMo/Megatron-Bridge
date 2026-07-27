---
name: nemo-mbridge-perf-moe-comm-overlap
description: MoE expert-parallel communication overlap in Megatron Bridge. Covers dispatch/combine overlap, flex dispatcher backends, and expert wgrad scheduling. Use for MoE communication-overlap tuning, comm-overlap throughput-regression investigation, overlap_moe_expert_parallel_comm, MoE dispatch overlap, flex dispatchers, DeepEP overlap, or expert-wgrad scheduling.
license: Apache-2.0
---

# MoE Communication Overlap

For the higher-level overview, see:

- @docs/training/communication-overlap.md
- @skills/nemo-mbridge-perf-moe-comm-overlap/card.yaml

## Quick Decision

Use MoE communication overlap when:

- `EP > 1`
- token dispatch or combine time is visible in the profile
- the run is already correct and you are now tuning throughput

Avoid turning it on as an early bring-up step. It is easier to validate after
the dispatcher, routing mode, and recompute plan are already stable.

## Enablement

```python
cfg.comm_overlap.overlap_moe_expert_parallel_comm = True

# Optional: delayed wgrad for additional overlap
cfg.comm_overlap.delay_wgrad_compute = True

# IMPORTANT: disable shared expert overlap when using dispatch overlap
cfg.model.moe_shared_expert_overlap = False
```

### Prerequisites

- `expert_model_parallel_size > 1`
- `num_moe_experts > 1`
- `moe_token_dispatcher_type` must be `"alltoall"` or `"flex"`
- Precision: BF16 or FP16
- If PP is used, VPP (`virtual_pipeline_model_parallel_size`) must be set (non-`None`)

### Flex dispatcher activation

Setting `moe_flex_dispatcher_backend` alone does **not** activate flex dispatch.
You must also set `moe_token_dispatcher_type = "flex"`.

## Recompute And CUDA Graph Interaction

- Full recompute is not a good companion for the overlap path.
- `delay_wgrad_compute` adds further constraints if CUDA-graph scopes include
  attention or MoE-router work.
- In practice, selective recompute is the safer pairing when overlap is enabled.

## Measured Evidence

### HybridEP production-shape validation

A 2026-07-25 controlled Qwen3 30B-A3B pretraining comparison used 16 H100
GPUs, BF16, sequence length 4096, `TP=1`, `PP=1`, `CP=1`, `EP=16`,
`MBS=1`, `GBS=1024`, forced-balanced routing, HybridEP, and Transformer
Engine CUDA-graph scopes `moe_router` and `moe_preprocess`. The only
performance change was plain EP overlap; delayed wgrad stayed disabled.

| Case | Steady window | Step time | Model TFLOPS/GPU |
|---|---:|---:|---:|
| EP overlap off | iterations 5-20 | 24.7138s | 244.039 |
| EP overlap on, search run | iterations 5-20 | 21.0725s | 286.208 |
| EP overlap on, independent validation | iterations 41-50 | 20.9920s | 287.305 |

The independent result reduced step time by 15.059% and increased throughput
by 17.729% over the reproduced baseline. Loss remained finite, no iterations
were skipped or NaN, and rank-0 peak allocated memory was 62.166 GiB.

A same-method rank-0 Nsight Systems comparison captured 463,348 kernels in
each case:

| Profile metric | Overlap off | Overlap on |
|---|---:|---:|
| Communication concurrent with GEMM/attention | 9.079ms | 3,958.997ms |
| Communication time hidden by compute | 0.11% | 36.55% |
| GPU-active interval union | 22.821s | 21.221s |
| HybridEP dispatch-with-permute NVTX | 4.253s | 1.767s |
| HybridEP metadata-preprocess NVTX | 3.109s | 0.670s |

This is direct evidence that the gain came from hiding exposed HybridEP
dispatch/combine work, not from changing the dispatcher, routing, graph
scopes, batch shape, or parallel layout.

### Qwen3.5 production-shape counterexample

A matched 2026-07-26 Qwen3.5-35B-A3B experiment on 16 H100 GPUs used
GBS1024, TP1/PP1/CP1/EP16, static HybridEP, fused dispatcher permutation,
FlashQLA, and a Hopper PyTorch `grouped_mm` expert path with CUDA graphs off.
The no-overlap steps 5-8 averaged 23,516.2 ms (about 250.5 model
TFLOP/s/GPU). Enabling only plain EP overlap raised first-step peak allocation
from 61.39 to 63.15 GiB, then produced an 82,700.3 ms first steady sample
(71.2 model TFLOP/s/GPU). Loss remained finite and skipped/NaN counts were
zero, so this was a scheduling and resource-contention regression rather than
a numerical failure.

The Qwen3 and Qwen3.5 results share hardware, batch, EP degree, HybridEP, and
precision but have different execution maps and expert kernels. Treat those
as part of the benchmark shape and require a matched A/B before enabling
overlap in another model family member.

A matched high-priority-stream control took 82,668.3 ms (71.3 model
TFLOP/s/GPU), versus 82,700.3 ms (71.2) at normal priority. The neutral result
rules out simple starvation of a normal-priority A2A stream; large regressions
still require a combined-schedule profile before tuning dispatcher SM
reservations or enabling delayed wgrad.

The matched profile kept kernel count effectively unchanged (401,840 without
overlap versus 401,776 with overlap), but expanded the capture span from 26.048
to 85.453 seconds and idle gaps from 4.204 to 24.187 seconds. Dispatch active
time grew from 2.802 to 38.912 seconds, NCCL from 0.397 to 16.500 seconds, and
`cudaEventSynchronize` accumulated 12.513 seconds. Roughly 30,000 calls in each
major `cuMem*` VMM category suggested allocator involvement, but an
allocator-only control disproved the simple explanation: `backend:native`
completed iteration 1 in 151.9 seconds and iteration 2 did not complete after
more than six minutes, with ranks near 79.2--80.9 GiB. Keep the expandable
allocator for this shape and investigate cross-stream tensor lifetime/event
dependencies; VMM traffic is correlated evidence, not causality.

Retaining MoE-combine inputs until backward was not a viable lifetime fix.
That isolated control completed one finite iteration in 149.7 seconds, raised
rank-0 peak allocation from about 63.15 to 68.93 GiB, and then OOMed on all 16
ranks when each needed another 1.89 GiB for optimizer state. The schedule needs
early reclamation for capacity. Future fixes must retire storage after an
explicit dependency or otherwise preserve reuse; they cannot simply disable
the release.

The next isolated control used that pattern: combine recorded its completion
event, the compute/owner stream waited on it, and storage was resized there.
The run completed three finite optimizer steps. Iterations 2 and 3 took 67.98
and 64.67 seconds (about 88.9 TFLOPS/GPU mean), versus 82.70 seconds for the
normal `record_stream(comm)` release. First-step peak allocation stayed at
63.15 GiB and iteration-2 peak at 72.03 GiB. Cross-stream retirement therefore
explains part of the regression, but not most of it: no-overlap still ran near
23.52 seconds.

A matched rank-0 profile confirmed causality. Normal and owner-release overlap
traces both launched exactly 401,776 kernels, while owner release reduced the
capture span from 85.453 to 73.024 seconds, idle gaps from 24.187 to 19.916
seconds, dispatch from 38.912 to 30.196 seconds, and NCCL from 16.500 to 14.061
seconds. `cudaEventSynchronize` fell from 8,952 calls / 12.513 seconds to 4,524
calls / 4.346 seconds, and major `cuMem*` call counts fell about 25%. The
benefit was therefore allocator/event and rendezvous contraction, not less
model work. Useful HybridEP/expert intersection also fell from 3.160 to 2.283
seconds, and the remaining dispatch and NCCL walls were still about 10.8x and
35.4x above no overlap. Preserve the evidence, but do not enable this schedule
in a public recipe.

The next launch-order A/B kept that owner-release path fixed and changed only
`CUDA_DEVICE_MAX_CONNECTIONS` from 32 to 1. Iterations 2 and 3 fell to 44.349
and 44.580 seconds (about 132.45 model TFLOP/s/GPU mean) from 67.978 and 64.667
seconds at 32 connections. Iteration-2 peak allocation remained 72.028 GiB and
all numerical checks passed. Verify connection count from the live PID because
recipe dumps can still show their default when an explicit launcher value wins.
The result is a strong H100 scheduler learning, but still a rejected overlap
path: it remained 1.89x slower than the no-overlap winner.

The connection-count profile localized the tradeoff. Relative to owner release
at 32 connections, one connection reduced capture span from 73.024 to 46.742
seconds, dispatch from 30.196 to 11.162 seconds, NCCL from 14.061 to 0.665
seconds, and `cudaEventSynchronize` calls from 4,524 to 426. Major VMM calls
fell another 45--47%. Useful HybridEP/expert intersection also collapsed from
2.283 to 0.156 seconds. One connection fixed most rank-drift inflation by making
the schedule nearly serial; it did not hide communication. If continuing the
experiment, sweep upward one step at a time and keep the no-overlap path as the
end-to-end gate.

That smallest upward sweep was negative. Changing only the connection count
from 1 to 2 made the two steady steps 47.180 and 53.532 seconds, a
50.356-second mean and about 117.0 model TFLOP/s/GPU. This was 13.25% slower by
step time than one connection, while iteration 2 reached 64 allocator retries
and 77.265 GiB reserved. Loss, gradients, skipped-iteration count, and
NaN-iteration count remained healthy. Stop the sweep when this happens:
increasing launch concurrency has started to restore rank/allocator variability,
not useful overlap. Do not assume intermediate values such as 4 or 8 will bridge
two losing endpoints.

Apply the connection-count winner to the no-overlap control before concluding
that it only repairs a broken combined schedule. On the exact 2-node Qwen3.5
control with HybridEP `num_sms=16`, changing only
`CUDA_DEVICE_MAX_CONNECTIONS=32` to `1` reduced the steps 5-8 mean from 23.228
to 22.451 seconds and raised throughput from 253.60 to 262.38 model
TFLOP/s/GPU. This 3.46% step-time gain is real end-to-end improvement on the
faster schedule, but it is still a launch-order result rather than proof of
communication hiding. A matched profile must identify whether the gain comes
from lower idle/rendezvous time or changed kernel concurrency.

That matched profile showed a mixed launch-order effect. Relative to the older
connections=32 no-overlap trace, connections=1 reduced HybridEP active union
8.77% and expert/linear union 1.58%, with unchanged 401,840 kernel count.
However, idle gaps increased 15.02% from 4.204 to 4.836 seconds and useful
HybridEP/expert intersection remained zero. The end-to-end win came from
cheaper serial execution, not communication hiding. This evidence justifies an
adjacent connections=2 A/B on the no-overlap schedule even though connections=2
lost on the distinct owner-release combined-overlap path.

The adjacent no-overlap result was also negative: steps 5-8 averaged 22.502
seconds / about 261.78 model TFLOP/s/GPU, 0.28% slower than connections=1.
Increasing launch concurrency did not recover the profiled idle tradeoff.
Retain connections=1 and stop the sweep rather than extrapolating to 4 or 8.

Switching the owner-release/connections=1 overlap path to
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` was not a fix. All ranks
OOMed in the first fine-grained backward; a representative rank needed another
1.89 GiB with only 9.56 MiB CUDA-free on a 79.11-GiB device. No optimizer step
or performance sample completed. Treat allocator selection as a capacity gate
before a timing dimension, and reduce the combined activation lifetime when
neither native nor stream-ordered allocation transfers.

Selective GDN recompute established the missing lifetime mechanism. On the
same exact schedule, `recompute_modules=["gdn"]` lowered iteration-2 rank-0
peak allocated memory from 72.028 to 59.516 GiB and changed steps 2-3 from a
44.4646-second mean to 25.3615 seconds (42.96% shorter). Loss and gradients
were finite with zero skipped/NaN iterations. It still ran 13.52% slower than
the accepted no-overlap path, so recompute recovered most of the broken
combined schedule without turning it into a winner.

MCore also exposes `model.ep_overlap_early_attn_memory_release=True` for this
lifetime pattern. It moves `pre_dispatch_computation.backward` from after
`moe_combine.forward` to before the next `mlp.forward`, releasing
attention/GDN activations earlier without recomputation. The flag requires EP
overlap and is not automatically a speedup: dispatch backward and combine
forward may become exposed. Use it only after a recompute diagnostic establishes
the lifetime mechanism, then compare end-to-end time, peak allocation,
dispatch/combine unions, idle gaps, and useful communication/compute
intersection.

The exact Qwen3.5-35B-A3B H100 control showed why warmup matters. With the
owner-stream release, one CUDA connection, EP16 HybridEP, and no recompute held
fixed, early release averaged 44.20885 seconds over iterations 2-3, about
133.25 model TFLOP/s/GPU. That was only a 0.575% step-time change from the
44.4646-second broken-overlap control and remained 49.47% below the accepted
263.67-TFLOP/s/GPU path. Peak allocation was 63.152 GiB on cold iteration 1
but returned to the control's 72.028 GiB on iteration 2 after optimizer
warmup. The flag therefore provided neither a durable capacity benefit nor a
material schedule repair on this GDN-heavy shape.

The focused MCore alltoall test
`test_transformer_layer_overlap_early_attn_memory_release` runs four
microbatches through reference and early-release schedules from identical
parameters and compares the captured outputs and parameter gradients. This is
code-level correctness evidence, not throughput evidence. Its containing test
file is marked `flaky_in_dev` for a Transformer Engine 2.17 pybind11 GIL abort,
so the target run must still establish finite loss and gradients with no
skipped or NaN iterations.

### Observe imbalance at the real consumer boundary

For current MCore, prefer `model.log_moe_overload_factor=true` over late
monkeypatches of dispatcher or expert class methods. Fine-grained schedule
plans can retain already-bound callables and bypass class attributes replaced
after construction. The built-in path records immediately after
`dispatch_postprocess`, consumes the returned `tokens_per_expert`, and reports
average, maximum, and cumulative actual-versus-balanced load at step end.

This metric is useful for rank/layer overload diagnosis but has a deliberate
limit: it sums local expert counts into tokens-on-rank. It does not expose the
complete per-expert segment vector needed to replay grouped-GEMM offsets. Use
it to decide whether imbalance is plausible; do not claim that it measured
per-expert shapes.

### Correctness-first alltoall smoke

A 2026-05-18 current-main H100 x16 smoke on Qwen3 30B-A3B mock pretraining
used `EP=16`, `alltoall`, global batch size 1024, CUDA graphs disabled, and
`moe_permute_fusion=false` because the PyTorch 25.11 / TE / Triton stack failed
in Transformer Engine fused permutation in prior bring-up.

Results were directional rather than release-grade:

- no EP overlap: 41.25s steady-state mean over iterations 3-8
- EP overlap: 31.31s steady-state mean over iterations 3-8
- EP overlap plus `delay_wgrad_compute`: 31.20s steady-state mean over
  iterations 3-8

Treat this as evidence that EP overlap can help an inter-node `alltoall` MoE
shape when communication is exposed. It is not proof that delayed wgrad is a
separate win, and it does not validate the fused permutation path. An earlier
2026-05-16 short smoke on the same shape showed the same pattern.

The matched Qwen3.5 grouped-expert result is a counterexample for delayed
wgrad as well. A custom exact backward split made the real-shape grouped-MM
dgrad-only primitive 21.1% shorter than native full backward, with bitwise
matching gradients. Enabling the corresponding
`overlap_dispatch_backward_with_experts_wgrad` schedule on the accepted
connections=1/shared-expert-overlap configuration nevertheless changed the
steps 5-8 mean from 22.341 to 22.489 seconds, a 0.664% regression
(263.67 to about 261.93 model TFLOP/s/GPU). Validate both primitive parity and
the unprofiled end-to-end schedule; a deferrable wgrad does not prove that the
runtime has a useful concurrent window.

## Code Anchors

- Overlap validation: `src/megatron/bridge/training/comm_overlap.py`
- Flex dispatcher backend: `src/megatron/bridge/training/flex_dispatcher_backend.py`
- Config: `src/megatron/bridge/training/config.py`
- Unit tests: `tests/unit_tests/training/test_comm_overlap.py`
- DeepEP tests: `tests/unit_tests/training/test_deepep.py`
- Early-release output/gradient equivalence:
  `3rdparty/Megatron-LM/tests/unit_tests/a2a_overlap/test_schedule_layer_1f1b.py`

## Pitfalls

1. **Shared expert overlap conflict**: `moe_shared_expert_overlap` and
   `overlap_moe_expert_parallel_comm` can conflict. Disable shared expert
   overlap when using the dispatch overlap path.

2. **PP without VPP**: MoE overlap requires VPP when pipeline parallelism is
   active. Without it, the overlap scheduling cannot interleave correctly.

3. **Flex != backend flag**: `moe_flex_dispatcher_backend="deepep"` alone
   does nothing if `moe_token_dispatcher_type` is still `"alltoall"`.

4. **Conservative recipe defaults**: Most public recipes leave MoE overlap
   disabled. You need to explicitly enable it via overrides.

5. **Performance gains are workload-dependent**: overlap helps most when dispatch
   communication is already a visible slice of step time. It is not guaranteed
   to help every small or lightly loaded EP run.

6. **Summed kernel time is not wall time**: concurrent kernels can run longer
   because they contend for SMs or bandwidth, so overlap may increase summed
   per-stream kernel duration while reducing the exposed interval union and
   end-to-end step time.

7. **A positive sibling-model result is not a default**: GDN layers, grouped
   expert kernels, and combined-schedule live ranges can turn a HybridEP
   overlap win into a multi-fold regression at the same nominal topology.
   Check both the first steady step and peak allocation before expanding the
   run.

8. **Use stream priority as a control, not a cure**: a neutral matched
   normal/high-priority A/B rules out simple communication-stream starvation.
   Profile the combined schedule before changing dispatcher SM reservations.

9. **Unchanged kernel count does not mean unchanged schedule cost**: large
   increases in dispatch/NCCL interval unions, idle gaps, and event-sync time
   indicate rank skew and collective rendezvous inflation.

10. **Allocator traces need an allocator-only control**: high `cuMem*` activity
    can be a symptom of fine-grained storage lifetime. Native allocation can
    fragment more severely and make the same overlap schedule slower.

11. **Preserve early reclamation while fixing lifetime ordering**: retaining
    combine inputs until backward can add several GiB and make iteration 2 the
    optimizer-state OOM point. Test dependency-aware retirement instead.

12. **A safer retirement path is not automatically an overlap win**:
    owner-stream release improved the measured Qwen3.5 stall by about 18% and
    preserved memory, but remained nearly 3x slower than no overlap. Re-profile
    before changing another event or stream.

13. **Require profile contraction, not just a faster short run**: the matched
    owner-release trace reduced event-sync time by 65%, major VMM call counts by
    about 25%, and dispatch by 22%, proving the mechanism. It simultaneously
    reduced useful communication/compute intersection and remained far slower
    than no overlap, so it was still rejected.

14. **Benchmark connection count as a launch-order dimension**: on H100,
    `CUDA_DEVICE_MAX_CONNECTIONS=1` can constrain cross-stream scheduling enough
    to reduce rank drift. Compare it with the recipe default on the exact
    dispatcher/graph shape and retain no-overlap as the acceptance control.

15. **Separate ordering recovery from overlap recovery**: lower NCCL and event
    time can come from serialization. Always report the comm/compute
    intersection; a near-zero value means the "overlap" schedule is not
    achieving its intended mechanism.

16. **Allocator backends must pass a multi-step capacity gate**:
    `cudaMallocAsync` can OOM even when expandable segments reaches the
    optimizer step. Do not interpret an allocator startup success as a timing
    result.

17. **Use built-in overload telemetry before monkeypatch logging**:
    fine-grained plans may keep bound callables that bypass late class hooks.
    `log_moe_overload_factor` observes actual post-dispatch rank load, although
    it does not provide per-expert offsets.

18. **Early attention release is an end-to-end A/B, not a memory-only
    decision**: when attention/GDN recompute proves a retained-activation wall,
    `ep_overlap_early_attn_memory_release` can move that backward earlier
    without recompute. The new order can expose dispatch/combine, so require
    step-time, interval-union, and post-optimizer-warmup memory evidence. The
    measured Qwen3.5 A/B was neutral versus its broken overlap control and
    returned to the same 72.028-GiB iteration-2 peak.

## Verification

Look for overlap-related log messages during initialization. The comm overlap
validation in `comm_overlap.py` will raise if prerequisites are not met, so a
clean startup confirms the feature is active.

For a short performance-harness smoke, keep the command shape explicit and vary
only one overlap knob at a time:

```bash
uv run python scripts/performance/run_script.py \
  -m qwen \
  -mr qwen3_30b_a3b \
  --task pretrain \
  -g h100 \
  -c bf16 \
  -ng 16 \
  -gn 8 \
  --max_steps 8 \
  --cuda_graph_impl none \
  --moe_flex_dispatcher_backend None \
  --moe_a2a_overlap false \
  --tokenizer_type NullTokenizer \
  comm_overlap.overlap_moe_expert_parallel_comm=true \
  comm_overlap.delay_wgrad_compute=false \
  model.moe_shared_expert_overlap=false
```

If fused MoE permutation fails during bring-up, add
`model.moe_permute_fusion=false` to separate overlap timing from runtime-stack
validation, then retest with the matched production container.

For performance validation, use an unprofiled steady window as the acceptance
metric. Use a matched Nsight A/B to establish causality:

1. Keep dispatcher, routing, CUDA graphs, batch shape, parallelism, and runtime
   fixed.
2. Toggle only `overlap_moe_expert_parallel_comm`; keep
   `delay_wgrad_compute=false` for the first isolation.
3. Compare communication and compute interval unions and their intersection,
   not only summed kernel durations.
4. Report steady step time, model TFLOPS/GPU, loss finiteness, skipped/NaN
   iterations, and peak allocated memory.
5. If the profile shows large allocator runtime time, repeat the exact shape
   with only `PYTORCH_CUDA_ALLOC_CONF` changed and compare iteration-2 progress
   plus all-rank utilization/memory.
6. After changing storage ownership, compare normal-overlap, modified-overlap,
   and no-overlap traces. Report kernel count, event-sync calls/time, major
   `cuMem*` calls, dispatch/NCCL unions, idle gaps, and useful intersection.
7. When changing `CUDA_DEVICE_MAX_CONNECTIONS`, inspect the live training PID's
   environment and require two finite steady steps; a config dump can show a
   recipe default that was superseded by the launcher.
8. For imbalance diagnosis, add `model.log_moe_overload_factor=true` to a
   diagnostic run and inspect average/max/cumulative overload. Keep it out of
   the accepted timing run, and do not treat its rank totals as per-expert
   segment measurements.
