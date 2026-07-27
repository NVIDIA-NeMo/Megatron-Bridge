# Qwen3.5-35B-A3B H100 measured campaign

## Contents

- [Scope](#scope)
- [Measured campaign record](#measured-campaign-record)

## Scope

This reference records workload-specific 2026 H100 measurements behind the
MoE optimization workflow. Read it when tuning Qwen3.5-35B-A3B, comparing a
similar GDN-MoE profile, or checking whether a candidate was already tested.
Keep the main SKILL.md focused on the reusable workflow and use this document
for detailed A/B outcomes, failure gates, and profiling evidence.

## Measured campaign record


The 2026-07-26 exact-GBS1024 sweep on 16×H100 found one additional end-to-end
win after HybridEP and scoped graphs: fused HybridEP permutation with compatible
runtime chunk sizes reduced step time from 26.081 to 25.007 seconds (4.12%).
Keep the negative controls because individually plausible kernel changes were
not additive:

- tensorwise current-scaling FP8 was 21.14% slower than BF16
- folding Q/K L2 normalization into FlashQLA was 3.45% lower throughput
- a standalone fused GDN RMSNorm+SiLU gate was 0.61% slower
- FlashQLA's default 16 local chunks beat both 8 chunks (2.00% slower) and
  disabled intra-card partitioning (3.10% slower)
- HybridEP 32 SMs beat both 24 SMs (0.48% slower) and 48 SMs (5.71% slower);
  jointly normalized 64-token fused-permutation chunks beat jointly normalized
  32-token chunks (1.46% slower) and 128-token chunks (1.86% slower)
- backward-dispatch/expert-wgrad overlap moved step time by only 0.18% while
  adding about 1 GiB, so treat it as noise until a longer run proves otherwise
- increasing MBS from 1 to 2 exceeded the H100 memory limit; selective
  layernorm recompute did not change that peak, while recomputing both GDN and
  MoE activations fit but erased the larger-MBS throughput opportunity
- 1% optimizer CPU offload crossed the immediate MBS2 allocation failure, but
  both prefix-based and experimental size-aware tensor selection stalled
  before completing iteration 1 with rank-split GPU progress; memory fit is a
  necessary but insufficient acceptance criterion for a distributed recipe
- optimizer-step parameter-gather overlap was not a valid Qwen3 helper
  transfer: both matching config fields resolved to true, but the TP1/PP1/EP16
  path without virtual pipeline parallelism had one model chunk; splitting
  that first chunk from the remaining chunks produced an empty optimizer group
  and failed in construction before iteration 1; require at least two model
  chunks and successful optimizer construction before measuring this overlap
- replacing random force-balance routing with deterministic exact-balanced
  routes was not a valid shortcut around dynamic expert metadata: even a
  cross-rank route-only control that left HybridEP's metadata tensor untouched
  segfaulted during the first backward after fused dispatch
- Transformer Engine's device-metadata grouped-GEMM entry point was not a
  Hopper escape hatch: the discrete-weight training call asserted sm100 or
  newer, and the unfused TE operation API still materialized split sizes with
  `tolist()` in forward and backward
- PyTorch's native BF16 `grouped_mm` was materially different in the same
  container: its SM90 CUTLASS fast path consumed GPU int32 inclusive offsets
  and passed an H100 interface probe covering forward, autograd, zero-token
  experts, and CUDA Graph replay; a 2-node, 16-H100 full-model A/B subsequently
  completed eight optimizer steps with finite loss and gradients, no skipped
  or NaN iterations, improving the steps 5-8 mean from 25,007.2 to 23,516.2 ms
  (235.55 to about 250.5 model TFLOP/s/GPU); because that A/B predated the
  device-side capacity assertion, a later gated run must still prove that no
  routes overflowed
- scoped `moe_preprocess` graphs replay tensor work but not Python-side
  dispatcher assignments: clearing HybridEP's static `num_permuted_tokens`
  after eager combine invalidated the next replay's shape contract, so
  graph-safe integration must preserve non-tensor shape state for the entire
  replay lifetime; preserving it fixed the stall and passed a device-side
  overflow assertion, but the matched graph run averaged 23,600.4 ms versus
  23,516.2 ms eager because static dispatch had already removed the
  synchronization wall that the graphs targeted
- reducing EP from 16 to 8 made each HybridEP group node-local on the 8-GPU
  H100 nodes, but the resulting expert-DP=2 replicated enough expert and
  optimizer state to OOM at iteration 2: construction fit at about
  41.7 GiB/GPU and iteration 1 was finite, while optimizer-state creation
  reached about 78.2 GiB/GPU and needed another 1.89 GiB; require at least two
  optimizer steps before declaring a topology memory-feasible

These results illustrate why a combined patch must be decomposed into
independent A/Bs: an apparent fused-norm gain in a combined experiment did not
reproduce when isolated.

Do not assume that lowering `main_params_dtype` will recover another full
master-weight copy when using BF16 precision-aware optimizer (PAO). PAO enables
`store_param_remainders` by default and stores the missing FP32 master-weight
bits as an int16 remainder. Treat a master-dtype change as a numerical-policy
experiment, not a free memory optimization.

Check a knob's call site before consuming a benchmark slot.
`high_priority_a2a_comm_stream` controls the stream created by the combined
1F1B schedule; it does not reprioritize standalone HybridEP when EP overlap is
disabled. On PP1 without combined 1F1B, changing it is a no-op rather than a
dispatcher A/B.

The matched rank-0 Nsight capture explained the remaining wall. Across a
26.231-second capture span, the GPU kernel-active union was 22.448 seconds.
HybridEP combine, dispatch, and support/RDMA occupied 3.909, 2.845, and 0.968
seconds of active union, while expert/linear GEMMs occupied 8.656 seconds and
GDN kernels only 1.355 seconds. HybridEP and expert GEMMs did not
overlap in that configuration. The trace also recorded 2,775
`cudaStreamSynchronize` calls with 7.422 seconds of summed CPU wait duration,
largely inside HybridEP metadata preprocessing. These durations overlap GPU
work and must not be added to wall time, but they redirect tuning from another
GDN microkernel toward dispatcher metadata, communication, and expert GEMMs.

The grouped-MM/static-dispatch follow-up reduced the kernel-active union from
22.448 to 21.844 seconds. Expert/linear active union fell from 8.656 to 8.115
seconds (6.25%), HybridEP from 7.723 to 7.376 seconds (4.49%), and summed host
stream-sync wait from 7.422 to 1.679 seconds (77.4%); GDN changed by only 1.5%,
from 1.355 to 1.334 seconds. HybridEP and expert/linear work still had zero
measurable overlap, leaving 15.491 seconds of serial active union as the main
target. The remaining 154 host stream synchronizations were no longer
dispatcher-dominated: 136 were `aten::copy_`, including 128 copies attributed
to the two BF16 conversions per microbatch in native fused cross-entropy
backward. The isolated Transformer Engine cross-entropy follow-up averaged
23,514.2 ms over steps 5-8 versus 23,516.2 ms native, only 0.0085% within
noise, and emitted an upstream warning about known TE cross-entropy stability
issues. The host wait was overlapping diagnostic time rather than recoverable
wall time, illustrating why an end-to-end A/B must validate every profile
hypothesis.

The matched plain-overlap profile exposed a different failure mode. Kernel
count stayed effectively unchanged (401,840 without overlap and 401,776 with
overlap), but capture span grew from 26.048 to 85.453 seconds, idle gaps from
4.204 to 24.187 seconds, dispatch active union from 2.802 to 38.912 seconds,
and NCCL active union from 0.397 to 16.500 seconds. Runtime APIs accumulated
12.513 seconds in `cudaEventSynchronize` and roughly 30,000 calls in each major
`cuMem*` VMM category. That combination identifies rank-skew/collective
rendezvous inflation rather than duplicated compute.

Do not infer allocator causality from the VMM trace alone. Changing only
`PYTORCH_CUDA_ALLOC_CONF` to `backend:native` made a finite first iteration
take 151.9 seconds and left iteration 2 incomplete after more than six minutes,
with ranks near 79.2--80.9 GiB and a live idle/busy GPU split. The expandable
allocator remained materially better at about 82.7 seconds per step. Keep the
better allocator and investigate fine-grained storage release plus cross-stream
event dependencies.

The coarse storage-lifetime control was also negative. Keeping MoE-combine
inputs alive until backward made iteration 1 take 149.7 seconds, raised rank-0
peak allocation from about 63.15 to 68.93 GiB, and caused all 16 ranks to OOM
on the next 1.89-GiB optimizer-state allocation. Immediate reclamation is
capacity-critical. A useful scheduler patch must move retirement behind a
correct dependency or reduce its synchronization cost without retaining all
inputs.

Waiting on the combine completion event and retiring the storage on the
compute/owner stream preserved the early-release memory footprint and recovered
part of the stall. Iterations 2 and 3 completed in 67.98 and 64.67 seconds,
versus 82.70 seconds for the normal cross-stream `record_stream` control;
rank-0 peak allocation remained 63.15 GiB on iteration 1 and 72.03 GiB on
iteration 2. This is a useful causal isolation, not an endpoint: the no-overlap
winner still ran near 23.52 seconds.

The matched owner-release profile then closed the attribution loop. Kernel
count stayed exactly 401,776, while capture span fell from 85.453 to 73.024
seconds, GPU idle gaps from 24.187 to 19.916 seconds, HybridEP dispatch from
38.912 to 30.196 seconds, and NCCL from 16.500 to 14.061 seconds.
`cudaEventSynchronize` fell from 8,952 calls / 12.513 seconds to 4,524 calls /
4.346 seconds. `cuMemUnmap`, `cuMemCreate`, and `cuMemMap` call counts each fell
about 25%. This confirms that cross-stream retirement amplified allocator/event
churn and collective rendezvous. It also shows why the change was insufficient:
dispatch and NCCL remained about 10.8x and 35.4x above the no-overlap trace, and
useful HybridEP/expert intersection shrank from 3.160 to 2.283 seconds.

From that improved release path, changing only
`CUDA_DEVICE_MAX_CONNECTIONS=32` to `1` reduced the two steady steps from
67.978 / 64.667 seconds to 44.349 / 44.580 seconds. Throughput rose from about
88.9 to 132.45 model TFLOP/s/GPU while iteration-2 peak allocation remained
72.028 GiB. The live PID environment, rather than the recipe dump, confirmed
the explicit value. Connection count therefore materially controls H100
fine-grained launch ordering and rank drift. The result still failed the
end-to-end decision rule: no-overlap remained 1.89x faster at 23.516 seconds.

The matched profile showed that connection count 1 mainly restored ordering:
NCCL fell from 14.061 to 0.665 seconds, dispatch from 30.196 to 11.162 seconds,
event-sync calls from 4,524 to 426, and major VMM call counts by another
45--47%. But useful HybridEP/expert intersection fell from 2.283 to 0.156
seconds. The remaining 46.742-second capture contained 16.588 seconds of idle
gaps and 11.162 seconds of dispatch. A faster overlap-relative result can
therefore mean "less harmful serialization," not successful communication
hiding.

The next controlled point, `CUDA_DEVICE_MAX_CONNECTIONS=2`, also rejected a
connection-count sweep as the optimization endpoint. It produced 47.180 and
53.532-second steady steps, averaging 50.356 seconds and about 117.0 model
TFLOP/s/GPU. That is a 13.25% step-time regression versus one connection.
Iteration 2 recorded 64 allocator retries and 77.265 GiB reserved, while all
numerical checks stayed finite. More concurrency restored variability without
restoring useful overlap. Once the nearest upward point regresses this way,
return to the faster no-overlap path and target its serialized dispatcher wall
instead of extrapolating a broad connection-count sweep.

The serialized dispatcher still needs resource tuning rather than maximal SM
assignment. Raising only HybridEP `num_sms` from 32 to 108 on that no-overlap
path changed the steps 5--8 mean from 23.516 to 28.212 seconds, a 19.97%
step-time regression (about 250.5 to 208.8 model TFLOP/s/GPU). Numerical checks
remained healthy. "Combined EP overlap disabled" does not imply that HybridEP's
internal persistent kernels and streams cannot contend with neighboring work.
Reducing the same cap from 32 to 20 improved the steps 5--8 mean only from
23.516 to 23.335 seconds (250.5 to 252.45 model TFLOP/s/GPU), a 0.78%
step-time gain. Reducing it from 20 to 16 improved only another 0.46%, to
23.228 seconds and 253.60 model TFLOP/s/GPU. Sweep around the backend default
and measured winner; expect a shallow, diminishing-return low-SM region and do
not jump directly to the hardware-scale preprocessing value. Also establish
the lower boundary: reducing the same Qwen3.5 cap from 16 to 12 regressed the
mean by 9.28%, from 23.228 to 25.385 seconds (253.60 to 232.06 model
TFLOP/s/GPU), because communication became underprovisioned.

Do not conflate the dispatch/combine SM cap with HybridEP preprocessing or
fused-permutation resources. The installed DeepEP 1.2.1+34152ae fused template
used 108 blocks for preprocess, permute, and unpermute, but 16 for dispatch and
combine. Overriding only preprocessing to 32 preserved the other four values
and completed eight numerically healthy exact-2-node steps, yet steps 5--8
averaged 22.434 seconds / about 262.57 model TFLOP/s/GPU. That was 0.418%
slower by step time than the 22.341-second current winner. Treat each resource
field as a separate experimental dimension, print the instantiated template,
and do not transfer a 32-block preprocessing value from another model family
without an end-to-end A/B.
The same warning applies between fields on one model: changing only fused
unpermute from 108 to 32 blocks averaged 23.400 seconds / about 251.73 model
TFLOP/s/GPU over exact 2-node steps 5--8, 4.742% slower than the 22.341-second
winner. Identical live defaults do not imply identical sensitivity; tune
preprocess, permute, and unpermute independently.

After bracketing that dispatcher-SM optimum, re-test launch ordering on the
winning no-overlap configuration. Holding HybridEP `num_sms=16` fixed and
changing only `CUDA_DEVICE_MAX_CONNECTIONS` from 32 to 1 reduced the exact
2-node Qwen3.5 steps 5-8 mean from 23.228 to 22.451 seconds. Throughput rose
from 253.60 to 262.38 model TFLOP/s/GPU, a 3.46% step-time improvement and the
current short-run winner. Unlike the owner-release overlap result, this is an
end-to-end gain on the fastest schedule; unlike a successful overlap result,
it still does not establish communication hiding. Preserve the no-overlap
label and use a matched profile to attribute idle, rendezvous, and kernel
concurrency changes.

Rebaseline CUDA graphs after the launch-order change as well. Holding
`num_sms=16`, connections=1, and static rank capacity 1.05 fixed, eager steps
5-8 averaged 22.439 seconds / about 262.52 model TFLOP/s/GPU versus 22.451
seconds / 262.38 with TE-scoped graphs. The 0.053% step-time difference is
noise-level. The practical learning is not an eager speedup: graphs no longer
provide a measurable benefit after static dispatch and one-connection launch
ordering, so retain the simpler eager control and profile that path.

That current-winner profile found a launch-order tradeoff rather than hidden
work. Connections=1 reduced HybridEP active union 8.77% and expert/linear union
1.58% versus the older connections=32 trace, while kernel count stayed 401,840.
Idle gaps increased 15.02%, from 4.204 to 4.836 seconds, and useful
HybridEP/expert intersection remained zero. The next experiment should
therefore be the adjacent no-overlap connections=2 point: it can test whether
some concurrency recovers the new 0.632-second idle penalty without assuming
that the losing combined-overlap connections=2 result transfers to this
schedule.

The adjacent no-overlap connections=2 run then averaged 22.502 seconds / about
261.78 model TFLOP/s/GPU over steps 5-8, 0.28% slower than connections=1.
Increasing launch concurrency did not recover the 0.632-second profiled idle
penalty. Keep connections=1 and move to a different optimization dimension.

Bracket static dispatcher capacity with correctness guards, not throughput
alone. Holding the current Qwen3.5 winner fixed and reducing only its static
rank-capacity factor from 1.05 to 1.02 triggered the device-side route-overflow
assertion in the first training iteration. Because the run completed zero valid
iterations, it has no performance result. This establishes 1.05 as the measured
safe point for that exact route distribution and rejects 1.02 without allowing
silent work dropping.

Treat mixed-precision FP8 as an end-to-end A/B, not an automatic Hopper win.
On the same exact 2-node Qwen3.5 winner, global hybrid FP8 with tensorwise
scaling and BF16 grouped experts completed eight finite iterations, but steps
5-8 averaged 24.205 seconds / about 243.36 model TFLOP/s/GPU. That regressed
step time 7.87% and throughput 7.30% versus the 22.439-second BF16 control.
The first iteration also took 154.335 seconds and device memory briefly
approached 78.4 GiB/GPU. Keeping experts in BF16 does not isolate the workload
from scaling, compilation, and non-expert FP8 overhead; reject tensorwise for
this shape and test another FP8 recipe only when it has a platform-specific
rationale.

That platform-motivated blockwise follow-up improved memory but not throughput.
Steps 5-8 averaged 24.213 seconds / about 243.28 model TFLOP/s/GPU, a 7.91%
step-time and 7.33% throughput regression versus BF16; its 0.03% step-time
difference from tensorwise was noise-level. Rank-0 first-step max
allocated/reserved memory fell to 63.369/67.228 GiB. When dominant expert GEMMs
remain BF16, changing FP8 scaling can improve headroom without fixing the
end-to-end precision-boundary cost.

Rebaseline graphs separately for that mixed boundary. The Qwen3 26.08.rc2
blockwise-FP8 gain did not transfer to Qwen3.5: adding
`attn,moe_router,moe_preprocess` graphs to the 24.21325-second eager path made
capture iteration 4 take 51.853 seconds and the first two replay steps take
44.8776 and 44.8274 seconds. The bounded exact-2-node job timed out after
iteration 6, so it cannot be accepted as a positive benchmark, but the
44.8525-second replay mean is an 85.24% regression and justifies rejecting the
full scope. When a cross-model transfer reverses this sharply, return to
module-by-module graph scopes rather than widening or rerunning the same stack.

The remaining Hopper grouped-expert alternatives had distinct boundaries.
Native Transformer Engine with variable per-expert splits remained correct but
averaged about 33.18 seconds because forward and backward materialized device
counts on the host. Repacking entirely on-device into fixed 2,304-token expert
slots made the steady samples slower still, about 36.76 seconds, and the run
stopped progressing after TE graph capture. TorchAO's differentiable MXFP8
grouped-MM wrapper was not an H100 alternative: its optimized path asserted
SM100, while its Hopper-compatible mode dequantized back to BF16 before the
grouped GEMM. However, inspect the layers separately before declaring all FP8
grouped GEMM unavailable. The same PyTorch 2.12 container exposed lower-level
`scaled_grouped_mm` and `_scaled_grouped_mm_v2` APIs, but the installed
dispatch table—not the Python enum—defined the usable modes. Tensorwise
appeared in the public scaling enum yet failed both a small exact-2-node probe
and the real Qwen3.5 FC1 shape with `No gemm implementation`. Rowwise
forward was functional on the real shapes and reached about 0.9993 cosine
similarity to BF16, but it was not a throughput candidate. With weight scales
cached, FC1 was 7.39x slower than BF16 and FC2 was 3.52x slower; dynamically
regenerating weight scales made them 10.50x and 5.95x slower. Reject the
primitive before building dgrad/wgrad when the cached-weight forward alone
loses by that margin. The same build exposed the cuBLASLt grouped-GEMM
preference getter but not its setter. A later exact-2-node fail-closed probe
also found no current-main public
`torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm` attribute in NVIDIA
PyTorch `2.12.0a0+0291f960b6.nv26.04`, so the alternative backend could not be
enabled in this container. Probe the installed build rather than inferring API
availability from newer documentation.

Do not stack graph and launch knobs after they are individually neutral.
Optimizer CUDA-graph capture produced finite 22.50--22.67-second early steps
but then stopped progressing during optimizer capture. Removing the measured
20/20 Transformer Engine layernorm SM margins regressed the steps 5-8 mean by
1.41%, to 22.755 seconds. The later intended combine-only 128-to-256 and
128-to-64 trials were not valid kernel A/Bs. The fused-template shim always
selected the unchanged 64-token dispatch chunk and copied it into preprocessing
and combine, making the combine-only environment values inert. Their
22.341/22.563/22.577-second differences are repeat variance, not a chunk
ranking. Keep the measured 20/20 margins, but require a logged
post-normalization template before interpreting any HybridEP chunk sweep. A
corrected full-field A/B then requested 128/128/128 and independently audited
the same-stack Configurer as effective 128/128/128, stages 10/4/2, blocks
108/108, and `valid=True`. Its finite exact-2-node iterations 2--3 averaged
23.5981 seconds / about 249.62 model TFLOP/s/GPU, a 5.627% step-time
regression versus effective 64/64/64. This is the valid final-schedule chunk
comparison; the earlier combine-only numbers remain invalid.

A later exact-2-node candidate kept 64/64/64 and requested only dispatch and
fused-permute stages 12/12 with inflight depths 10/10. Its finite iterations
2--3 averaged 23.463650 seconds / about 251.05 model TFLOP/s/GPU, 5.025% slower
by step time than the winner, so it was rejected. The runtime shim checksum
matched, but the intended post-normalization template marker did not appear.
Use this as evidence not to promote or rerun the candidate, but not as a clean
stage/inflight ranking: requested JIT inputs are not an attributable A/B
without an effective-template record.

Trace duration is not a direct SM-allocation rule. The final Qwen3.5 profile
showed a longer HybridEP combine union than dispatch, but an experimental
split that kept dispatch at 16 SMs and raised only combine to 20 regressed the
exact 2-node steps 5--8 mean from 22.340925 to 22.49995 seconds (about 263.67
to 261.81 model TFLOP/s/GPU), with all numerical gates passing. Use profiles
to nominate an asymmetric-budget A/B, then accept it only on end-to-end
steady-state timing.

Shared-expert overlap was the one later scheduler A/B that improved the final
BF16 path. Enabling only `moe_shared_expert_overlap` at connections=1 reduced
the steps 5-8 mean from 22.439 to 22.341 seconds and raised throughput from
about 262.52 to 263.67 model TFLOP/s/GPU, a 0.44% step-time gain. A matched
profile kept kernel count at 401,840, moved 0.690 seconds of shared FC/SwiGLU
work to a side stream, overlapped 0.213 seconds with the primary stream and
0.163 seconds with HybridEP, reduced main-stream kernel sum 3.10%, and reduced
HybridEP combine 8.71%. Only about 30.8% of the side stream overlapped the
primary stream, so most shared work remained dependency-serialized. Increasing
connections to 2 regressed the mean 2.67% to 22.937 seconds. Raising only the
shared stream's CUDA priority regressed 0.18% to 22.381 seconds. Keep normal
priority, connections=1, and treat explicit dependency windows—not priority—as
the remaining overlap limit.

Re-test graphs after enabling a later overlap winner rather than assuming the
earlier graph A/B transfers. On the same exact 2-node shared-overlap
configuration, TE-scoped attention/router/preprocess graphs averaged 22.365
seconds / about 263.39 model TFLOP/s/GPU over steps 5-8, 0.106% slower than the
22.341-second eager winner. Graph capture made iteration 1 take 150.351 seconds
and emitted AccumulateGrad stream-mismatch warnings. Keep eager: the steady
difference is noise-level, and the graph path adds cold-start and stream
complexity without throughput.

Do not promote a split backward based only on the primitive's apparent
critical-path reduction. For the real Qwen3.5 expert shapes, an exact custom
grouped-linear probe produced bitwise-matching input and weight gradients.
Computing only dgrad on the immediate autograd path was 21.1% shorter than
native full backward; adding both materialized wgrads sequentially was only
1.58% slower than native. The end-to-end MCore
`overlap_dispatch_backward_with_experts_wgrad` run was still negative:
22.489 seconds / about 261.93 model TFLOP/s/GPU over steps 5-8, 0.664% slower
than the 22.341-second shared-overlap winner. At connections=1, the stream and
event schedule did not hide enough wgrad under dispatch-backward to repay its
coordination cost. Treat primitive parity, primitive latency, available
dependency window, and unprofiled end-to-end timing as four separate gates.

Autotune the real dispatcher segmentation and the real eager reference, not
only average tokens or total capacity. On this model,
`torch.compile(mode="max-autotune-no-cudagraphs")` made a balanced
32,768-row expert MLP 28.30% faster in forward and 26.92% faster through
backward than an ordinary PyTorch `silu * value * probability` control, with
all output and gradient cosines above 0.999993. The exact 2-node full model
passed a 34,416-row static budget to the graph but regressed 0.878% to 22.537
seconds / about 261.38 model TFLOP/s/GPU. A follow-up uniform-offset primitive
sweep showed that total rows were not the cause: compiled forward+backward
remained 25.36%, 25.02%, 25.32%, and 25.64% faster at 33,424, 33,760, 34,080,
and 34,416 rows. But production eager training already uses MCore's separately
fused `weighted_bias_swiglu_impl` autograd path, so the original primitive also
changed its reference implementation.

Do not assume reserving SMs from compiler kernels will repair that transfer
gap. On the same fully active, uniformly segmented 34,416-row primitive,
reserving 0, 16, and 24 H100 SMs produced compiled/eager forward+backward
ratios of 0.7500, 0.7596, and 0.7881. With an independent shared MLP on a
normal-priority side stream, concurrent makespan ratios were 0.7599, 0.7607,
and 0.8021. Numerical cosines remained at least 0.99999356, but 16 SMs did not
improve concurrency and 24 regressed, so the upward sweep stopped. Also
separate static buffer length from active routed rows: a uniform sweep whose
final offset reaches the buffer end does not reproduce the inactive tail in a
rank-capacity HybridEP graph. An exact 2-node control with a 34,416-row buffer,
32,768 active rows, and 40-call bursts closed both gaps. Relative to ordinary
eager, compiled forward/backward still appeared 29.32%/23.69% faster. Relative
to the actual fused MCore eager path, compiled single-call forward/backward was
7.05%/5.11% slower. Forty-call compiled bursts were 1.57% faster in forward
and 4.60% faster through backward, but compiled/shared-MLP concurrent makespan
was 7.44% slower. Output matched the fused reference exactly and all gradient
cosines rounded to 1.0. Require the production fused/autograd reference,
capacity, segmentation, repeated-launch behavior, concurrency, and an
end-to-end win before promoting a compiler result.

Fixed capacity also does not make padded batched GEMM automatically preferable
to variable grouped MM. A real-shape H100 primitive probe used 16 local
experts, 2,048 valid rows per expert, 2,304-row capacity, device-side
repack/gather, SwiGLU, router probabilities, and both backward gradients.
Outputs matched exactly, but `torch.bmm` plus 12.5% padding took 1.021 ms
forward and 3.553 ms forward+backward versus 0.495 and 1.389 ms for grouped MM.
That is 2.06x/2.56x slower and raised peak allocation from 1.033 to 1.272 GiB.
Reject the primitive before full training; static GEMM shapes did not repay
padding, data movement, and autograd cost.

Port performance-critical runtime code into the reviewable change before
calling a recipe reproducible. The pinned public-stack control after native
H100 HybridEP alignment averaged 24.70215 seconds / about 238.43 model
TFLOP/s/GPU over exact-2-node steps 5--8. The first reviewable FlashQLA 0.1.2
GDN sample reduced that mean to 23.205475 seconds / about 253.808 model
TFLOP/s/GPU, but its Slurm step exited 0:0 while the enclosing batch exited 9:0
after one rank missed the shutdown window. A pinned-version rerun with the
longer shutdown window completed the config dry-run, all eight training steps,
and focused runtime tests with an overall 0:0 exit. Its steps 5--8 averaged
23.254625 seconds / about 253.271 model TFLOP/s/GPU, a 6.22% throughput gain
over the pinned control, with finite losses and gradients and zero skipped or
NaN iterations. Record training health and terminal health separately; only
the second run is passing short-run evidence.

Fail closed on the exact external kernel version and make its path explicit in
the published launch command. A lazy import with only a helpful error message
does not prove that the measured version was loaded. Check
`flash_qla.__version__ == "0.1.2"` in the reviewable runtime and pass its
container-mounted Python prefix through `PYTHONPATH`; do not rely on an
untracked `sitecustomize.py` or a user-specific source tree. FlashQLA also
needs a shared persistent `TILELANG_CACHE_DIR`, and the grouped-expert path
benefits from a persistent `TORCHINDUCTOR_CACHE_DIR`. A cold-cache allocation
spent its bounded runtime compiling without producing an iteration.

Treat static capacity as a compiler shape, not only a routing scalar. On the
pinned PR stack, changing capacity factor from 1.05 to 1.02 created a new
TorchInductor specialization. Ranks progressed differently while sharing the
new cache and eventually reached an NCCL watchdog before iteration 1, so the
run supplied neither overflow nor timing evidence. Prewarm a new static shape
with isolated rank-local cache writes or an explicit compile barrier before a
distributed A/B; do not interpret a compilation rendezvous failure as a
capacity result.

The current reviewable runtime is still far from the acceptance gate:
253.271 model TFLOP/s/GPU is 11.85% below 287.305 and needs another 13.44%
throughput improvement, to at most 20.4999 seconds per step. Even the
22.340925-second development-stack winner is 8.22% below the gate. Do not spend
a 50-step verification allocation until a shorter exact-topology run crosses
the threshold with terminal exit 0:0.
