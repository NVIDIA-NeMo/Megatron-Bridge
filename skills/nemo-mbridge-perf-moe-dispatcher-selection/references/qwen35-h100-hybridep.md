# Qwen3.5 H100 HybridEP measured campaign

## Contents

- [Scope](#scope)
- [Fused template, static metadata, and backend evidence](#fused-template-static-metadata-and-backend-evidence)

## Scope

This reference records the detailed Qwen3.5-35B-A3B H100 HybridEP evidence
behind the dispatcher-selection skill. Read it when auditing a live fused
template, selecting a Hopper grouped-expert path, or testing HybridEP resource
budgets. Keep the main SKILL.md focused on dispatcher selection and portable
validation rules.

## Fused template, static metadata, and backend evidence

When enabling fused HybridEP permutation, validate the live buffer template
after dispatcher construction. Dispatch, combine, and preprocessing chunk
sizes must remain compatible with the fused path; a config helper accepting
the knobs does not prove that the runtime kept them. On the measured
Qwen3.5-35B-A3B 16×H100 stack, correcting that runtime contract reduced exact
GBS1024 step time from 26.081 to 25.007 seconds (4.12%). The local optimum was
32 communication SMs and jointly normalized 64-token fused-permutation
chunks: 24 and 48 SMs regressed 0.48% and 5.71%, while jointly normalized
32- and 128-token chunks regressed 1.46% and 1.86%. That result is a coupled
runtime-template setting, not a universal combine-only optimum. A later source
audit invalidated the apparent combine-only 128-to-64 comparison on the
static-dispatch/shared-overlap path. Its fused-template compatibility shim
always selected `num_of_tokens_per_chunk_dispatch_api` and copied it into
preprocessing and combine; because dispatch remained at its 64-token default,
both runs instantiated 64/64/64 despite different combine environment values.
The 22.341- versus 22.577-second difference is repeat variance, not evidence
for a combine-chunk setting. Log the post-normalization template and require
the intended field to differ before accepting any chunk A/B. A later
exact-2-node A/B did change all three fused fields: a same-stack Configurer
audit returned effective chunks 128/128/128, stages 10/4/2, blocks 108/108,
and `valid=True`. That run averaged 23.5981 seconds / about 249.62 model
TFLOP/s/GPU over finite iterations 2--3, 5.627% slower by step time than the
effective-64/64/64 winner. Retain 64/64/64 for this final schedule; the result
is workload-specific and does not supersede live-template validation.

A follow-up requested dispatch and fused-permute pipeline stages 12/12 with
inflight depths 10/10 on the same 64/64/64, 16-SM winner. Its finite exact
2-node iterations 2--3 averaged 23.463650 seconds, about 251.05 model
TFLOP/s/GPU, or 5.025% slower by step time than the winner. Reject the
candidate. The staged runtime shim checksum matched, but its post-normalization
template marker did not appear, so the result does not prove that every
requested field reached the compiled kernel. It is evidence against promoting
the candidate, not a clean field-level stage/inflight ranking.

In the matched rank-0 profile, HybridEP combine, dispatch, and support/RDMA
occupied 7.723 seconds of a 22.448-second GPU-active union, with no overlap
against the 8.446-second expert-GEMM union. HybridEP metadata preprocessing
also enclosed most of 2,775 stream synchronizations (7.422 seconds of summed
CPU wait, overlapping GPU work). Treat dynamic metadata and expert-shape
materialization as part of the dispatcher wall, not as generic Python overhead.

Do not remove those synchronizations by substituting a benchmark-only route
layout. A Qwen3.5 H100 control assigned exactly equal expert counts while
spreading each token's top-k routes across EP ranks, kept HybridEP's original
GPU metadata tensor, and still segfaulted during the first backward after fused
dispatch. Likewise, `tokens_per_expert` is produced on the communication stream
and its GPU tensor must remain alive for the asynchronous path. The safe
optimization requires a native nonblocking HybridEP-to-GroupedMLP metadata
contract that supports arbitrary valid routes; a CPU split or deterministic
route shim is not equivalent.

Leave `moe_expert_rank_capacity_factor=None` for the first Hopper run. Static
capacity removes a host-side dynamic-size synchronization, but it also changes
the expert input shape and requires a compatible fused GroupedLinear path.
Validate that combination independently before using it in a benchmark recipe.
In the 2026-07-26 H100 Qwen3.5 experiment, the required Transformer Engine
operation-fuser GroupedMLP path was available only on sm100. Bypassing that
guard reached communication failure and is not a valid Hopper workaround.

NCCL EP has a different availability boundary. The exact immutable training
image contained TE 2.15 and did not expose `transformer_engine.pytorch.ep`,
while the repository's pinned TE 2.17 source did contain `EpBuffer`,
`ep_bootstrap`, `ep_dispatch`, `ep_combine`, and `ep_finalize`. The source
build knob is `NVTE_WITH_NCCL_EP`, enabled by default for SM90-or-newer
targets; `NVTE_WITH_NCCL_EP=0` disables it. Building the extension also needs
the recursive NCCL submodule and NCCL 2.30.4 or newer. Inventory the imported
runtime rather than inferring capability from either the MCore source or lock
file. On H100, only dynamic shape is supported, and its
`tokens_per_expert.sum().item()` narrowing introduces a D2H synchronization.
A non-overlapped NCCL-EP dispatcher A/B is still meaningful, but the static
overlapped path requires SM100+, the TE operation fuser, and
`NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`. The current manager also rejects the
symmetric-memory zero-copy option.

Treat a new dispatcher container as a whole-kernel-stack migration, not only
a Transformer Engine upgrade. On the 2026-07-27 TE 2.17/NCCL-EP bring-up,
`mbridge-260707` exposed all required EP symbols but its preinstalled
`causal_conv1d==1.6.2.post1` extension failed while importing the real
Qwen3Next modeling module. The shared object referenced
`c10::impl::cow::materialize_cow_storage`, which was absent from the
container's torch 2.13 ABI. Rebuilding exact upstream tag commit
`4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` with
`CAUSAL_CONV1D_FORCE_BUILD=TRUE`, no build isolation, and the live torch
CXX11 ABI produced a venv-local extension that passed BF16 CUDA execution.
An `import transformer_engine` probe alone would have missed this blocker.

Warm JIT caches can hide a second incompatibility. The accepted torch 2.12
Qwen3.5 run and the torch 2.13 container both reported TileLang 0.1.8 without
`tilelang.language.async_copy`, yet the old run succeeded because it reused a
compiled FlashQLA cache. A fresh cache retraced FlashQLA 0.1.2 and failed at
`T.async_copy`. FlashQLA 0.1.2's package metadata pins `tilelang==0.1.9` and
`apache-tvm-ffi==0.1.9`; installing those exact packages in the isolated venv
allowed a fresh-cache Qwen3.5-shape forward and backward CUDA smoke to execute
successfully at B1, S4096, 16 key heads, 32 value heads, and head dimension
128. For a new software stack, require both a fresh-cache forward/backward
kernel smoke and a real model-module import before multi-rank timing. A
warm-cache success is performance evidence for that cache, not dependency or
cold-start validation.

Passing that stack gate did not make NCCL EP viable on the measured 2-node
H100 topology. The matched torch 2.13 / CUDA 13.3 / TE 2.17 HybridEP control
completed three iterations; its two post-compile steps averaged 24.236950
seconds / 243.043440 model TFLOP/s/GPU, 8.486779% slower than the accepted
torch 2.12 short-run control.

A buffer-only probe localized the NCCL-EP failure before another training A/B
was attempted. With 256 experts, hidden size 2,048, 4,096 maximum tokens,
34,416 receive rows, 16 communication SMs, dynamic shape, zero-copy disabled,
and `NCCL_GIN_TYPE=3`, eight ranks on one H100 node registered the symmetric
window successfully. The identical shape on 16 ranks across two nodes failed
at the first `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` call in
`nccl_ep.cc:886`. NCCL first logged `SPCX GPUNETIO dlopen failed`, then
`ncclGinGdakiCreateContext` and `ncclGinDevCommSetup` returned error 2. The
Spectrum-X plugin, GPUNetIO host library, and DOCA libraries were present and
had no missing `ldd -r` dependencies. Although the plugin contained
`NCCL_GIN_GPUNETIO_PATH` and loader-name strings, the loader was not exported
as a dynamic symbol, so neither a file nor directory path could be validated
through an external loader probe. Do not infer that an unverified path
assignment repairs the runtime.

The full NCCL-EP candidate was consistent with this isolated boundary. It
completed model construction and Python/process-group bootstrap and entered
iteration 0, but all GPUs remained idle during the first real dispatch. After
the backend's 101010 ms timeout, all 16 ranks reported
`NCCL error 2 at nccl_ep.cc:886`; the error did not propagate to Python, and
no iteration or throughput sample completed. Single-node window success is
only a capability stage. Require exact-topology window registration plus the
first multi-node dispatch/combine to finish before retaining NCCL EP as a
candidate. This failure belongs to the Spectrum-X/GDAKI runtime-plugin
boundary, not the Bridge recipe or dispatcher selection fields.

The runtime contract is more specific than "static shapes are graphable."
With dynamic HybridEP sizing, `dispatch_with_permute(non_blocking=False)`
returns `padded_tokens_per_expert` through pinned CPU memory and synchronizes
the dispatch stream before expert compute. Providing a static
`num_permuted_tokens` switches HybridEP to `non_blocking=True`, keeps the
padded counts on GPU, and requires the caller to retain both the returned
counts and dispatch handle until asynchronous consumers finish. Choose the
rank-capacity headroom from measured routed-count tails, align the budget to
the runtime's BF16 or quantized requirement, and fail the run on any overflow;
the capacity factor is not permission to drop routes.

Do not infer Hopper support from Transformer Engine's `GroupedTensor` Python
types alone. In the measured TE build, the basic `ops.GroupedLinear` still
called `split_sizes.tolist()` in both forward and backward, while
`general_grouped_gemm_for_grouped_tensor` rejected the discrete-weight
training case below sm100. The installed PyTorch `grouped_mm` fast path was a
different implementation: an interface probe on H100 accepted GPU int32
inclusive offsets and passed BF16 forward, native autograd, zero-token expert,
and CUDA Graph replay checks. The follow-up 2-node, 16-GPU full-model run also
completed eight optimizer steps with finite loss and gradients, no skipped or
NaN iterations. Replacing the split-size expert path reduced the steps 5-8
mean from 25,007.2 to 23,516.2 ms and raised throughput from 235.55 to about
250.5 model TFLOP/s/GPU. That A/B predated the device-side overflow assertion,
so it establishes performance and numerical execution, not dropless semantics;
a gated follow-up must fail on any overflow. The path remains an experimental
runtime patch rather than a portable recipe setting.

Do not assume the similarly named scaled grouped-MM API is a drop-in FP8
upgrade. In the same PyTorch 2.12 H100 container, tensorwise scaling was listed
by the Python enum but absent from the native grouped-GEMM dispatch table and
failed the real Qwen3.5 FC1 shape with `No gemm implementation`. Rowwise
forward ran and was numerically close to BF16, but cached-scale FC1 and FC2
were 7.39x and 3.52x slower. The cuBLASLt grouped-backend preference setter
present in newer source was also absent from the installed build. Probe the
real expert shapes and installed backend controls before treating a primitive
name as an available dispatcher/expert pairing.

The fixed-capacity alternative also failed at the complete primitive boundary.
Padding each of 16 experts from 2,048 to 2,304 rows and using `torch.bmm`
produced exact outputs, but device repack/gather plus SwiGLU made forward 2.06x
slower and forward+backward 2.56x slower than the variable grouped-MM path.
Static dispatcher metadata is valuable because it removes synchronization; it
does not imply that every downstream compute kernel should consume padded
slots.

Scoped CUDA Graph replay also has a non-tensor state contract. Capturing
`moe_preprocess` returns selected dispatcher tensor attributes, but it does
not replay Python assignments. If static HybridEP sizing depends on a Python
integer such as `num_permuted_tokens`, preserve that value across the eager
expert/combine phase; clearing it after combine can make the next graph replay
silently re-enter the dynamic-size path. Audit both tensor graph outputs and
the lifetime of shape-defining Python state.

In the matched follow-up, preserving the static budget fixed the distributed
stall: capture returned in 4.21 seconds, all later steps completed, and a
device-side overflow assertion remained clear. However, steps 5-8 averaged
23,600.4 ms (about 249.6 model TFLOP/s/GPU), slightly slower than the 23,516.2
ms eager result. Static dispatch had already removed the CPU synchronization
that made the scoped graphs valuable in the earlier dynamic path, so those
optimizations overlapped rather than added.

HybridEP SM tuning remained necessary even after explicit EP overlap was
disabled. On the same exact 2-node Qwen3.5 grouped/static path, changing only
`moe_flex_dispatcher_num_sms` from 32 to 108 regressed the steps 5--8 mean from
23.516 to 28.212 seconds (about 250.5 to 208.8 model TFLOP/s/GPU). All steps
remained finite. A larger persistent-kernel footprint can still contend with
HybridEP preprocessing, communication streams, or neighboring model work even
when the Bridge combined EP-overlap schedule is off. Do not infer "no overlap"
means "use all SMs"; test the backend default region and the current recipe
point before increasing the cap. Reducing the same cap from 32 to 20 produced a
23.335-second steps 5--8 mean (252.45 model TFLOP/s/GPU), only a 0.78%
step-time improvement. Reducing it again from 20 to 16 produced a
23.228-second mean (253.60 model TFLOP/s/GPU), another 0.46% improvement.
Reducing it further to 12 regressed the mean by 9.28% versus 16, to 25.385
seconds (232.06 model TFLOP/s/GPU). This brackets the coarse optimum: too many
SMs cause contention, but too few underprovision communication. Test nearby
low-SM values before adopting one on another model or runtime.

Do not infer separate dispatch and combine budgets only from active-union
lengths. On the later exact-2-node Qwen3.5 winner, an experimental split kept
dispatch at 16 SMs and raised only combine to 20 because the matched trace
showed a longer combine union. Steps 5--8 instead regressed from 22.340925 to
22.49995 seconds (about 263.67 to 261.81 model TFLOP/s/GPU), a 0.712%
step-time loss. Persistent-kernel occupancy, launch ordering, and neighboring
work determine the end-to-end critical path; profile duration is a candidate
signal, not a sizing formula.

Tune HybridEP preprocessing separately from dispatch/combine and fused
permutation. In DeepEP 1.2.1+34152ae, the fused Qwen3.5 template instantiated
preprocess, permute, and unpermute at 108 blocks each, while dispatch and
combine used 16. Setting only
`moe_hybridep_num_sms_preprocessing=32` left permute/unpermute at 108 and
dispatch/combine at 16. On the exact 2-node current-winner control, that clean
single-variable change averaged 22.434 seconds / about 262.57 model
TFLOP/s/GPU over steps 5--8, a 0.418% step-time regression versus the
22.341-second control. Values copied from another model's performance recipe
are hypotheses, not hardware defaults; inspect the instantiated template,
record every independent block budget, and validate the override end to end.
Reducing only fused-unpermute blocks from 108 to 32 made the same exact
2-node control materially worse: steps 5--8 averaged 23.400 seconds / about
251.73 model TFLOP/s/GPU, a 4.742% step-time regression. Do not extrapolate a
preprocessing A/B to permute or unpermute even when their instantiated defaults
are numerically identical.
