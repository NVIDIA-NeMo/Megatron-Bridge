# Qwen3.5 H100 overlap measurements

## Contents

- [Scope](#scope)
- [Qwen3.5 counterexample](#qwen35-counterexample)
- [Shared-expert overlap](#shared-expert-overlap-is-a-separate-alternative)
- [Early attention activation release](#early-attention-activation-release)

## Scope

This reference records detailed Qwen3.5-35B-A3B H100 measurements for plain
expert-parallel overlap, shared-expert overlap, delayed expert wgrad, and early
attention activation release. Read it when the target schedule combines GDN,
HybridEP, static grouped experts, or significant activation-lifetime pressure.
Keep the main SKILL.md focused on portable enablement and verification.

## Qwen3.5 counterexample

A 2026-07-26 matched 16-H100 Qwen3.5-35B-A3B experiment showed why the
Qwen3 result must not be copied by model name alone. The graph-free baseline
used GBS1024, TP1/PP1/CP1/EP16, static nonblocking HybridEP, fused
dispatch/permutation, FlashQLA, and an experimental Hopper PyTorch
`grouped_mm` expert path. It averaged 23,516.2 ms over steps 5-8 (about 250.5
model TFLOP/s/GPU).

Changing only `overlap_moe_expert_parallel_comm` to `true` increased the first
step's peak allocation from 61.39 to 63.15 GiB. The first steady sample then
took 82,700.3 ms (71.2 model TFLOP/s/GPU), versus about 23.5 seconds without
overlap, with finite loss and no skipped or NaN iteration. The run was stopped
after that decisive 3.5x regression. The combined schedule changed the
critical path and increased live memory instead of hiding communication.
A second exact A/B enabled `high_priority_a2a_comm_stream` and measured
82,668.3 ms (71.3 model TFLOP/s/GPU) on the same first steady sample. That was
only 0.04% faster than the normal-priority overlap run and ruled out simple
compute-stream starvation of a normal-priority A2A stream.

Treat microbatch count, model execution map, expert backend, and live-memory
pressure as part of the overlap shape. A positive Qwen3 HybridEP result does
not predict Qwen3.5 GDN behavior even at the same hardware, global batch,
parallelism, dispatcher, and precision.

## Shared-expert overlap is a separate alternative

Plain EP overlap and shared-expert overlap are mutually exclusive, and the
failure of one does not predict the other. On the later Qwen3.5 BF16 winner,
enabling only `moe_shared_expert_overlap` reduced the steps 5-8 mean from
22.439 to 22.341 seconds and raised throughput from about 262.52 to 263.67
model TFLOP/s/GPU. The matched trace moved 0.690 seconds of shared FC/SwiGLU
work to a side stream, but only 0.213 seconds overlapped the primary stream and
0.163 seconds overlapped HybridEP.

Treat this as a small measured win, not a reason to maximize concurrency.
Changing connections from 1 to 2 regressed step time 2.67%, while making only
the shared stream high priority regressed 0.18%. Keep the normal-priority
stream and the measured connection count. Stream priority changes which ready
work is favored; it cannot enlarge the legal overlap windows created by FC1,
FC2, dispatch, and combine dependencies.

Delayed grouped wgrad was also rejected on this exact winner. A real-shape
primitive probe split the two grouped expert linears into an exact dgrad path
and materialized wgrad path. All four gradients matched native autograd
bitwise, dgrad-only latency was 21.1% shorter than full backward, and sequential
materialized wgrad added only 1.58%. However, wiring that split into MCore's
`overlap_dispatch_backward_with_experts_wgrad` schedule at connections=1
regressed the exact 2-node steps 5-8 mean from 22.341 to 22.489 seconds
(0.664%) and reduced throughput from about 263.67 to 261.93 model
TFLOP/s/GPU. Primitive critical-path headroom is not an end-to-end win unless
the selected connection count and dependency graph actually execute the
delayed wgrad concurrently with useful communication.

## Early attention activation release

When EP overlap increases peak memory because the overlapped forward allocates
more than the paired backward has freed, MCore provides:

```python
cfg.model.ep_overlap_early_attn_memory_release = True
```

The normal schedule runs `pre_dispatch_computation.backward` (attention plus
the work before MoE dispatch) after `moe_combine.forward`. This flag moves that
backward before the next `mlp.forward`, releasing attention/GDN activations
earlier without activation recompute. It requires
`overlap_moe_expert_parallel_comm=True`.

MCore has a focused alltoall unit test for this ordering in
`3rdparty/Megatron-LM/tests/unit_tests/a2a_overlap/test_schedule_layer_1f1b.py`.
It runs four microbatches through both the reference transformer layer and the
overlap schedule with early release enabled, then compares captured outputs and
parameter gradients. This is source-level correctness coverage, not a
performance result. The whole A2A-overlap file is currently marked
`flaky_in_dev` because Transformer Engine 2.17 can abort with a pybind11 GIL
failure, so retain an end-to-end finite-loss/gradient check in the target
environment.

Treat it as a memory-lifetime A/B, not a general speed flag. The reordered
backward can expose `moe_dispatch.backward` and `moe_combine.forward` that were
previously hidden. It is most justified when an exact recompute control has
already shown that retained attention/GDN activations amplify allocator or
rank-skew stalls; compare peak allocation, dispatch/combine union, idle gaps,
and end-to-end step time.

An exact 2-node Qwen3.5-35B-A3B H100 A/B rejected the flag. With the same
owner-stream combine release, one CUDA connection, EP16 HybridEP, and no
recompute, early release completed finite iterations 2 and 3 in 44.0271 and
44.3906 seconds. The 44.20885-second mean was only 0.575% shorter than the
44.4646-second control and reached about 133.25 model TFLOP/s/GPU, 49.47%
below the accepted 263.67-TFLOP/s/GPU no-EP-overlap path. More importantly,
rank-0 peak allocation was 63.152 GiB on the cold first iteration but returned
to 72.028 GiB after optimizer warmup, exactly matching the control. A cold-step
memory sample is not evidence that the reordered lifetime survives steady
state.
