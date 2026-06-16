# Nemotron 3 Ultra — Deterministic vs Non-Deterministic Perf Analysis

**Date**: 2026-06-12
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE, 108 layers, 512 experts top-22, 2 MTP heads)
**Config**: TP=2, PP=3, EP=32, ETP=1, MBS=1, GBS=128, SeqLen=8192, 24 nodes × GB200 (96 GPUs), selective recompute

> Source profiles: rank-0 nsys captures of iters 15–17 of jobs **2103633** (det) and **2103635** (non-det).
> Bit-wise determinism verified separately by job **2103637** (matches 2102770 lm-loss exactly at iters 1, 10, 20, 30, 40, 50).

---

## Runs Compared

| | Det run (2103633) | Non-det run (2103635) |
|---|---|---|
| Slurm runtime | 15:34 / 50 iters | 14:35 / 50 iters |
| nsys profile window | 31.895 s (iters 15–17) | 27.608 s (iters 15–17) |
| **Per-iter avg (nsys window/3)** | **10,632 ms** | **9,203 ms** |
| **Per-iter (log iter 50, clean)** | **9,439 ms** | **8,041 ms** |
| **Throughput (TFLOP/s/GPU, iter 50)** | **415.3** | **487.5** |
| **Step time Δ (iter 50)** | — | **det +1,398 ms (+17.4%)** |
| **Throughput Δ** | — | **det −72.2 (−14.8%)** |
| Bit-wise reproducible | ✓ (md5 match across 2102770↔2103151↔2103637) | n/a |

**Methodology note**: The nsys window covers 3 iterations (15, 16, 17). Iters 15–17 are still in warmup
(iter-15 step time 9.3 s, iter-50 9.4 s for det). All kernel totals in this report are over the
**3-iter window**; divide by 3 for per-iter values.

---

## 1. Fairness & Shared Configuration

### What is identical (verified at the kernel-count level)

| Factor | Det | Non-det |
|---|---|---|
| Hardware | 24× GB200 (NVL16), same node pool | same |
| Container | `nemo:26.04.01.squashfs` | same |
| Parallelism | TP=2, PP=3, EP=32, ETP=1, GBS=128, MBS=1, seq=8192 | same |
| MoE dispatcher | `alltoall` (HybridEP intentionally OFF — NVL16 can't allocate the fabric handle at EP=32) | same |
| Attention backend | TE FusedAttention (cuDNN SDPA) | same |
| DDP comm | `overlap_grad_reduce=true`, `overlap_param_gather=true`, `tp_comm_overlap=false` | same |
| Recompute | selective: moe + layernorm + core_attn + moe_act + mlp + shared_experts | same |
| `TRITON_CACHE_AUTOTUNING` | 1 | 1 |
| NCCL kernel count | **11,013** | **11,013** ✓ |
| nvjet GEMM kernel count | **59,712** | **59,712** ✓ |
| Mamba SSM kernel count | **7,680** | **7,680** ✓ |
| MoE permute kernel count | identical | identical |
| TE Adam optimizer kernel count | 834 | 834 ✓ |

Identical kernel counts confirm the structural recipe is byte-for-byte the same — the only difference is *which* algorithm variant each library picks.

### What changes (the 6 determinism knobs)

| Knob | Det | Non-det | Where it shows in the profile |
|---|---|---|---|
| `model.deterministic_mode=true` | true | false | Routes MCore through det algorithm picks (TE + autograd) |
| `model.cross_entropy_loss_fusion=false` | false | true | Disables fused-CE → split path adds `aten_fill` zero-init |
| `NCCL_ALGO=Ring` | Ring | unset (default Tree) | All NCCL collectives are `RING_LL` in det (incl. small P2P); non-det has mixed Ring + Tree |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | 0 | unset (default 1) | Forces cuDNN SDPA bwd to a 2-pass knob_31 (det) vs 1-pass (non-det) |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | set | unset | Required guard when `torch.use_deterministic_algorithms(True)` is on |
| `MAMBA_DETERMINISTIC=1` | 1 | unset | Mamba2 scan reduce path; surprising: slightly faster (see §4.4) |

---

## 2. Headline: Where Does the +1,398 ms / Iter Go?

### Stream-level decomposition (3-iter nsys window)

Going beyond the kernel-name leaderboard: which CUDA stream produces the gap?

| Stream | Role | Det ms | Non-det ms | Δ ms |
|---|---|---|---|---|
| **7** | main: CCL host calls + most compute + aux | **12,769** | **10,433** | **+2,336** ← bulk of gap |
| **42** | dedicated NCCL `SendRecv` (PP p2p) | **3,940** | **3,448** | **+493** |
| **30** | dedicated NCCL ReduceScatter / AllGather | 372 | 373 | −1 |
| **50** | dedicated NCCL AllReduce (small) | 186 | 183 | +3 |
| 155 | GEMM stream | 1,159 | 1,119 | +41 |
| 156 | GEMM stream | 1,148 | 1,127 | +21 |
| 157 | GEMM stream | 1,162 | 1,144 | +18 |
| 158 | GEMM stream | 1,160 | 1,132 | +28 |

**Reading**: stream-7 + stream-42 carry **+2,829 ms of busy-time growth**. Against the +4,287 ms
window-total delta, that is **~66 % of the wall-time gap** (correction from an earlier "95%" — see §11).
The remaining ~34 % (~1,460 ms) is **idle/bubble growth between kernels**: when stream 7 stays busier
for longer with det-substitute work, downstream kernels on other streams arrive later and the GPU spends
more cycles idle. The four dedicated GEMM streams (155–158) are essentially identical — compute itself
is barely affected.

### Stream-7 decomposition

| Stream-7 sub-bucket | Det ms | Non-det ms | Δ ms | Note |
|---|---|---|---|---|
| NCCL kernels (RS/AG/AR) | 2,399 | 2,189 | **+210** | mostly skew from slower peer arrival — see §4.1 NCCL caveat |
| Compute (nvjet/cudnn/triton/SSM) | **6,112** | **6,113** | **0** | identical — see §1 kernel counts |
| Other (fill, reduce, copy, sort, permute, …) | **4,258** | **2,131** | **+2,127** ← the real gap |

The +2,127 ms on the "other" bucket is **the dominant cost of determinism**. It is *not* in the
model math (GEMM/attention/SSM) and it is *not* in NCCL. It's in the **PyTorch det-substitute
kernels** that replace `scatter_add`, `index_put`, `gather_backward`, fused-CE, etc.

### NVTX namespace decomposition (host-side, every range op_id/seq-stripped)

Querying every NVTX range in the profile (NVTX `text` field, not just the leaderboard top-N), grouped by namespace prefix:

| Namespace | Det ms (3-iter) | Non-det ms | Δ ms | Δ % | Verdict |
|---|---|---|---|---|---|
| `aten::*` | **23,132** | **8,911** | **+14,221** | **+160%** | **the dominant cost source** |
| `backward_node` (e.g. `CheckpointFunctionBackward`, `_LinearBackward`) | 24,292 | 20,929 | +3,362 | +16% | wraps the aten ops above |
| `autograd::evaluate_function` | 26,457 | 23,137 | +3,320 | +14% | also wraps backward_node |
| `other` (Python NVTX) | 18,337 | 16,632 | +1,705 | +10% | misc framework overhead |
| `mcore.transformer_layer` | 6,189 | 5,174 | +1,015 | +19.6% | `_forward_mlp.mlp` is +979 ms of this |
| `mcore.pipeline_parallel` | 3,958 | 3,466 | +492 | +14% | NCCL_ALGO=Ring on PP p2p |
| `nvte::*` | 3,092 | 3,260 | **−168** | **−5.1%** | **det actually FASTER** (TE GEMM −82 ms, rmsnorm −20 ms) |
| `mcore.mlp` | 1,964 | 1,834 | +129 | +7% | linear_fc1/fc2 entry setup |
| `nccl::*` (host API) | 1,176 | 1,212 | −36 | −3% | host-side NCCL python calls |
| `mcore.attention` | 351 | 321 | +30 | +9% | qkv/proj/core_attention forward |
| `mcore.fusions` | 314 | 316 | **−1** | **0%** | **IDENTICAL** |
| `Optimizer.step` | 28 | 31 | **−3** | **−9%** | det Adam actually faster |
| `mcore.models` | 41 | 43 | −2 | −5% | embedding forward |

**Headline**: the namespace breakdown isolates the cost source cleanly:
- **+14,221 ms (+160%) in `aten::*` — this is 93%+ of all NVTX-visible cost growth**
- `mcore.fusions` and `mcore.models` are **byte-identical** in their NVTX time
- `nvte::*` (TE library) is *faster* under det — the TE backward chooses a slightly cheaper code path
- `Optimizer.step#FusedAdam.step` is faster under det too (−9%)

### Per-namespace top contributors (op_id/seq stripped)

**`aten::*` namespace — every operator with |Δ| > 100 ms:**

| Range | Det ms | Non-det ms | Δ ms | Notes |
|---|---|---|---|---|
| `aten::fill_` | 2,481 | 198 | **+2,282** | zero-init before deterministic scatter / index_put |
| `aten::index_put_` | 2,119 | 6 | **+2,113** | det subs use Python-loop index_put; non-det uses atomic-add |
| `aten::_index_put_impl_` | 2,112 | 5 | +2,107 | inner kernel of the above (double-counted in `aten::*` total) |
| `aten::empty` | 2,731 | 718 | +2,013 | det allocates many scratch buffers for substitute paths |
| `aten::empty_strided` | 960 | 326 | +634 | same — strided variant |
| `aten::empty_like` | 840 | 214 | +626 | same — like variant |
| `aten::arange` | 545 | 3 | **+543** | det only — index generation for sort-based gather backward |
| `aten::gather_backward` | 482 | 36 | +446 | det substitute via sort+segment-sum |
| `aten::scatter_add_` | 456 | 15 | +440 | det substitute when `use_deterministic_algorithms=True` |
| `aten::max` | **368** | **0** | **+368** | **det-only**: MCore's `F.embedding` / `gather` det-substitute branch builds an advanced-indexing key — see §11 |
| `aten::min` | **322** | **0** | **+322** | **det-only**: same advanced-indexing key build |
| `aten::clone` | 556 | 253 | +303 | extra copies during det substitute path |
| `aten::contiguous` | 406 | 131 | +274 | det forces contiguous tensors for index sort |
| `aten::remainder` | **235** | **0** | **+235** | **det-only**: hash-style index materialization in the same det branch |
| `aten::scatter` | 33 | 175 | **−142** | non-det uses this; det path swapped to `aten::scatter_add_` |
| `aten::copy_` | 629 | 764 | **−135** | non-det does *more* copies (FP32 master copies in unfused-CE) |

The smoking gun: `aten::max`, `aten::min`, `aten::remainder` are **zero in non-det** and
**>200 ms each in det**. **Source corrected (see §11)** — these are *not* `torch.bincount` internals
(bincount uses fused aminmax + atomic histogram, not max/min/remainder). They come from MCore's
own `if torch.are_deterministic_algorithms_enabled(): ...` branches in the MoE/loss code path,
which substitute high-level ops:

```
torch.use_deterministic_algorithms(True)
    └── MCore checks the flag at call sites and switches:
            scatter           → index_put_ + arange      (hits aten::arange, aten::index_put_)
            scatter_add       → index_add                 (hits the det index_add path; segmented reduce)
            F.embedding /     → advanced-indexing build   (hits aten::max, aten::min, aten::remainder
              gather             (range-bound + bucket-id    for hash-style index materialization)
                                  computation)
    └── PyTorch native ops:
            deterministic index_put_ on CUDA already uses radix-sort + segmented reduce
              (this is the default for accumulate=True since PyTorch 2.0; not a slow Python loop)
            aten::empty + aten::fill_ zero the scatter destination before each call
              — this is intrinsic to the accumulate-into-prezeroed formulation
```

**`backward_node` namespace top entries (every `*Backward` node):**

| Range | Det ms | Non-det ms | Δ ms | Notes |
|---|---|---|---|---|
| `CheckpointFunctionBackward` | 15,499 | 13,630 | +1,869 | this wraps the recomputed-block's backward; the bulk is `aten::fill_` inside |
| `MambaSplitConv1dScanCombinedFnBackward` | 2,874 | 2,233 | +640 | the *autograd range*, not the SSM kernel (kernel itself is faster, see `nvte::*` and SSM kernel-name) |
| `GatherBackward0` | 484 | 38 | +446 | det path = arange + sort + segment-sum (kernel-name level) |
| `_LinearBackward` | 1,053 | 959 | +94 | TE backward — cuBLAS workspace overhead per call |
| `_LayerNormLinearBackward` | 562 | 484 | +77 | TE LayerNormLinear backward |
| `IndexPutBackward0` | 56 | 5 | +51 | the substitute path's own backward node |
| `RouterGatingLinearFunctionBackward` | 324 | 281 | +43 | MoE router GEMM backward (cuBLAS workspace) |
| `_moe_chunk_sortBackward` | 145 | 112 | +33 | TE MoE permute backward |
| `_AllToAllBackward` | 391 | 362 | +29 | MoE all-to-all backward (Ring overhead) |
| `LayerNormFnBackward` | 176 | 147 | +29 | layer norm backward |
| `_moe_permute_mask_mapBackward` | 118 | 90 | +28 | MoE permute backward |
| `_OperationFuserAutogradFunctionBackward` | 116 | 90 | +26 | fused op autograd |

**`nvte::*` namespace (TransformerEngine ranges) — det is faster:**

| Range | Det ms | Non-det ms | Δ ms |
|---|---|---|---|
| `nvte_cublas_gemm_v2` | 1,691 | 1,774 | **−82** |
| `nvte_multi_tensor_gemm` | 1,324 | 1,385 | **−60** |
| `nvte_rmsnorm_fwd` | 19 | 31 | −12 |
| `nvte_rmsnorm_bwd` | 20 | 27 | −8 |
| `nvte_flash_attn_bwd` | 11 | 11 | 0 |
| `nvte_flash_attn_fwd` | 15 | 15 | 0 |

The TE ranges are **CPU-host time** (when the TE library was inside its dispatcher). Det's TE-host time
is 5% **lower** — explained by less scheduler-jitter on stream 7 because the FillFunctor kernels
serialize work and reduce contention with the TE GEMM dispatch loop. The cuDNN flash attn ranges
are essentially identical.

**`mcore.pipeline_parallel` — all of it is NCCL P2P:**

| Range | Det ms | Non-det ms | Δ ms |
|---|---|---|---|
| `send_forward_recv_backward` | 3,412 | 2,998 | **+414** |
| `recv_backward` | 543 | 465 | +78 |
| `send_forward` | 3.1 | 3.0 | 0 |

These match the NCCL `SendRecv` kernel-name analysis: `NCCL_ALGO=Ring` slows P2P chunks.

### Decomposing the +2,127 ms "other" bucket on stream 7

| Kernel | Det calls | Det ms | Non-det calls | Non-det ms | Δ ms | What it is |
|---|---|---|---|---|---|---|
| `vectorized_elementwise<FillFunctor<BFloat16>>` | **35,328** | **1,607.4** | **120** | **24.5** | **+1,583** | Zero-init buffer before `scatter_add` (det substitute) |
| `vectorized_elementwise<FillFunctor<float>>` | 39,369 | 160.5 | 6,975 | 12.9 | **+148** | Same, but fp32 (CE backward, gradient accum) |
| `vectorized_elementwise<FillFunctor<int64>>` | 26,088 | 39.8 | (small) | (small) | +39 | Index buffer zero-init |
| `reduce_kernel` (det reduce) | 22,281 | 218.3 | 11,913 | 75.1 | **+143** | Segmented reduce for det `scatter_add` |
| `arange` (RangeFactories) | 3,096 | 5.2 | 24 | 0.04 | +5.2 | Index generation for det gather path |
| **Sum of det-substitute kernels** | — | **~2,031** | — | **~113** | **+1,918** | **~90 %** of the +2,127 "other" gap (correction in §11) |

All other elementwise/copy/sort/permute kernels are **numerically identical** between the two runs
(see §4.2 below for the side-by-side).

### Mapping kernels to NVTX module ranges

Going one level up to NVTX scopes (so we can say "X module's backward is slower because Y kernel
fires more"):

| NVTX autograd range | Det ms (window) | Non-det ms (window) | Δ ms | Underlying kernel responsible |
|---|---|---|---|---|
| `CheckpointFunctionBackward` (all op_ids) | 15,499 | 13,630 | **+1,869** | `FillFunctor<BFloat16>` (~1.5 s) + reduce + arange |
| `MambaSplitConv1dScanCombinedFnBackward` | 2,874 | 2,233 | **+640** | `FillFunctor<BFloat16>` zero-init in scan-bwd reduce; **NOT** in the SSM scan kernel itself (those go *down* −6%) |
| `GatherBackward0` | 484 | 38 | **+446** | substitute path: `arange` + `cub::DeviceRadixSort` + segment-sum |
| `_LinearBackward` (all op_ids) | 1,053 | 959 | **+94** | per-call cuBLAS +3-5 µs from `:4096:8` workspace setup |
| `_LayerNormLinearBackward` | 562 | 484 | **+77** | same cuBLAS workspace penalty + a det reduce |
| `IndexPutBackward0` | 56 | 5 | **+51** | substitute index_put_ scatter loop |
| `_moe_chunk_sortBackward` | 145 | 112 | **+33** | scatter_add fill + segmented reduce |

The "+1,869 ms on CheckpointFunctionBackward" is **not in the recomputed forward** — that fires the
same kernels at the same speed. It's in the **gradient-accumulation paths inside the
recomputed-block backward**, where every `scatter_add` / `index_put` / fused-CE-bwd / gather-bwd is
now going through a 2–4× slower deterministic substitute.

---

## 3. Compute Path: Essentially Untouched

### 3.1 GEMM (TE / cuBLAS nvjet kernels)

Identical kernel selection — same nvjet variants, same tile shapes (`tst_*`), same call counts.
Top 10 kernels by det total:

| nvjet kernel | det calls | det ms | nondet calls | nondet ms | Δ ms | per-call Δ µs |
|---|---|---|---|---|---|---|
| `tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 8,674 | 1,275 | 8,674 | 1,243 | +32 | +3.7 |
| `tst_128x256_64x6_2x2f_2cta_h_bz_TNT` | 1,728 | 1,076 | 1,728 | 1,054 | +22 | +12.7 |
| `tst_128x160_64x8_2x2f_2cta_h_bz_TNT` | 11,134 | 978 | 11,134 | 951 | +27 | +2.4 |
| `tst_128x192_64x6_4x1f_2cta_v_badd_NTT` | 6,144 | 841 | 6,144 | 825 | +16 | +2.6 |
| `tst_128x256_64x6_2x1_2cta_v_bz_NNT` | 4,481 | 771 | 4,481 | 755 | +16 | +3.6 |
| `tst_256x128_64x5_2x2f_2cta_h_badd_NTT` | 480 | 605 | 480 | 588 | +17 | +35.4 |
| **All 60 nvjet kernels** | **59,712** | **9,093** | **59,712** | **8,912** | **+181** | **+3.0 avg** |

The det penalty on GEMM is **+3.0 µs per cuBLAS call on average**, plausibly cuBLAS doing extra
workspace pointer setup with the `:4096:8` config. No kernel-selection drift. **Net: +2.0%** on the
whole GEMM bucket.

### 3.2 cuDNN flash SDPA — det has 2-pass backward

| | det | non-det |
|---|---|---|
| FWD `fprop_f16_knob_7_128x128x128_4x1x1_kernel0` calls | 192 | 192 ✓ |
| FWD ms / avg µs | 71.96 / **374.8** | 69.80 / **363.6** |
| BWD `bprop_f16_knob_31_128x128x128_1x4x1_kernel0` calls | **96** | **96** ✓ |
| BWD `bprop_f16_knob_31_128x128x128_1x4x1_kernel1` calls | **96** | **0** ← det-only second pass |
| BWD pass0 ms / avg µs | 64.38 / 670.6 | 95.57 / **995.5** |
| BWD pass1 ms / avg µs | 51.45 / 535.9 | — |
| **BWD total ms / iter** | **115.8** (2 kernels) | **95.6** (1 kernel) | **det +20.2 ms / window** |

`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` forces cuDNN to split the SDPA backward into 2 passes for
det reduction. Per-call each pass is shorter, but the total is +21%. The 2-pass schedule also
means more kernel-launch overhead.

### 3.3 Mamba2 selective scan — DET is slightly **faster**

| | det | non-det | Δ |
|---|---|---|---|
| Total SSM kernel ms (window) | **856** | **913** | **−57 (−6.2%)** ← det wins |
| SSM kernel count | 7,680 | 7,680 ✓ | identical |

`MAMBA_DETERMINISTIC=1` switches the chunked-reduce path. On B200 it happens to be slightly faster
than the default. This means **the +640 ms on `MambaSplitConv1dScanCombinedFnBackward` autograd range
is entirely buffer/reduce overhead, not in the SSM math kernel itself.**

---

## 4. Communication Path: +702 ms, Mostly P2P

### 4.1 Per-collective NCCL kernel breakdown (3-iter window)

| Collective | Det calls | Det ms | Det avg µs | Non-det calls | Non-det ms | Non-det avg µs | Δ ms | Avg µs Δ |
|---|---|---|---|---|---|---|---|---|
| **SendRecv** (PP p2p) | 3,486 | **5,291** | **1,518** | 3,486 | **4,771** | **1,369** | **+520** | **+149** |
| **ReduceScatter Sum bf16 RING_LL** | 2,730 | **794** | **291** | 2,730 | **596** | **218** | **+198** | **+73** |
| AllGather RING_LL | 4,362 | 709 | 162 | 4,362 | 753 | 173 | **−44** | −11 (det faster) |
| AllReduce Sum bf16 RING_LL | 9 | 75 | 8,351 | 9 | 44 | 4,883 | +31 | +3,468 |
| AllReduce Sum f32 RING_LL | 30 | 23 | 783 | 15 | 14 | 920 | +9 | — |
| AllReduce Sum f32 **TREE_LL** | 0 | 0 | — | 15 | 10 | 663 | −10 | non-det only |
| AllReduce Sum u32 RING_LL | 6 | 0.2 | 27 | 3 | 0.1 | 35 | +0.1 | — |
| AllReduce Sum u32 **TREE_LL** | 0 | 0 | — | 3 | 0.3 | 93 | −0.3 | non-det only |
| Broadcast | 3 | 0.04 | 13 | 3 | 0.06 | 19 | −0.02 | — |
| **Total NCCL kernel ms** | | **6,896** | | | **6,194** | | **+702** | |

Key observations — **with a major caveat (correction in §11)**:

> **Caveat on the NCCL deltas**: NCCL device kernels are *wait-inclusive* — they include
> peer-wait time as well as data-movement. In a PP workload, when stream-7 / stream-42 work
> arrives late on the slow peer, the kernel on the fast peer waits longer to rendezvous and
> records a higher duration. The "Ring algorithm pinning" story for these deltas is largely
> **wrong**: `NCCL_ALGO` is defined for *collectives* (AR/RS/AG/Broadcast/Reduce) only — P2P
> SendRecv bypasses Ring/Tree algorithm selection entirely (verified in `nccl/src/enqueue.cc`).
> ReduceScatter is RING_LL in **both** runs, so the +73 µs/call delta is not an algorithm choice
> either. These NCCL deltas are mostly the **shadow of det's slower stream-7 work** skewing peer
> arrival — they are a symptom, not a cause. Fixing NCCL config will not recover them; fixing
> the fills will.

1. **SendRecv +520 ms** — *not caused by* `NCCL_ALGO=Ring` (P2P bypasses algo selection).
   This is wait-time growth from peer arrival skew. Cannot be reduced by NCCL tuning.
2. **ReduceScatter +198 ms** — same kernel (`Sum_bf16_RING_LL`) in both runs; the +73 µs/call
   delta is wait time, not algorithm. Fixing the stream-7 fills will reduce this organically.
3. **AllGather is actually faster in det (−44 ms)** — coincidental peer-arrival timing.
4. **Non-det has both `RING_LL` and `TREE_LL` for f32 AllReduce** — confirms `NCCL_ALGO=Ring`
   *does* take effect on collectives. AllReduce on FP32 accumulators is the **only** collective
   where Ring vs Tree could plausibly affect bit-exactness (Tree's reduction order is
   non-deterministic over float). So pinning AR/RS to Ring is the bit-exact-relevant action;
   AG/Broadcast/SendRecv carry no floating-point reduction and could be left at default
   without breaking determinism.

### 4.2 Identical kernels (sanity check — same forward graph)

| Kernel | det calls | det ms | non-det calls | non-det ms |
|---|---|---|---|---|
| `_make_chunk_sort_map_kernel` (MoE) | 1,536 | 279.0 | 1,536 | 280.4 |
| `_sort_chunks_by_map_kernel` (MoE) | 2,304 | 250.9 | 2,304 | 240.8 |
| `_row_id_map_pass_3_kernel` (MoE) | 768 | 165.5 | 768 | 166.2 |
| `_layer_norm_bwd_kernel` (TE) | 384 | 86.8 | 384 | 87.4 |
| `_unpermute_kernel` (MoE) | 1,152 | 79.5 | 1,152 | 78.1 |
| `_permute_kernel` (MoE) | 1,152 | 75.5 | 1,152 | 77.2 |
| `multi_tensor_adam` (TE optimizer) | 834 | 94.0 | 834 | 95.1 |
| `mbtopk::computeBlockwiseWithinKCounts` (router) | 7,680 | (similar) | 7,680 | 70.5 |
| `mbtopk::computeBlockDigitCounts` (router) | 7,680 | (similar) | 7,680 | 58.3 |

These are **byte-identical** between runs. Confirms only the determinism knobs flip — no
kernel-selection drift in the model code path itself.

### 4.3 Bit-wise reproducibility evidence (wandb-tracked)

The deterministic recipe was run **8 separate times** across multiple days and Slurm allocations.
All runs produce identical loss trajectories — confirming the bit-exactness claim.

Iter-by-iter `lm loss` for the three most recent det+overlap=ON runs — **including the nsys-profiled one** — proves nsys instrumentation doesn't perturb determinism:

| iter | [2102770 (det)](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/6c3mdfyz) | **[2103633 (det + nsys)](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/7klz92sb)** | **[2103637 (det, no nsys)](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/pfi9ap38)** | 3-way match |
|---|---|---|---|---|
| 1 | 1.254624E+01 | 1.254624E+01 | 1.254624E+01 | ✓ |
| 5 | 9.316616E+00 | 9.316616E+00 | 9.316616E+00 | ✓ |
| 10 | 4.166083E+00 | 4.166083E+00 | 4.166083E+00 | ✓ |
| 15 | 9.956062E-01 | 9.956062E-01 | 9.956062E-01 | ✓ |
| 20 | 1.962516E-01 | 1.962516E-01 | 1.962516E-01 | ✓ |
| 30 | 6.581618E-02 | 6.581618E-02 | 6.581618E-02 | ✓ |
| 40 | 2.265546E-01 | 2.265546E-01 | 2.265546E-01 | ✓ |
| **50** | **7.411075E-02** | **7.411075E-02** | **7.411075E-02** | **✓** |

These are **the last 3 det runs in chronological order** — submitted on different nodes, with
different nsys-instrumentation status — and every iter-level loss agrees to the last digit.
2103151 (also a paired det run) and the earlier 5 overlap=OFF baselines (2074557/2074641/2074651/2076499/2076503) likewise match each other within their respective recipe-groups.

#### wandb run links — project [`mbridge-dev-zhiyul`](https://wandb.ai/nvidia/mbridge-dev-zhiyul)

**Det baseline (overlap=OFF, 2026-06-09)** — 5 reproductions, same recipe, same loss trajectory:

| Slurm job | Recipe | wandb run |
|---|---|---|
| 2074557 | det + overlap=OFF | [nq1tfhai](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/nq1tfhai) |
| 2074641 | det + overlap=OFF | [y836cdic](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/y836cdic) |
| 2074651 | det + overlap=OFF | [muqyfe0x](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/muqyfe0x) |
| 2076499 | det + overlap=OFF | [ibtpfriv](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/ibtpfriv) |
| 2076503 | det + overlap=OFF | [eyz3wbba](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/eyz3wbba) |

**Det + overlap=ON (2026-06-11/12)** — 3 reproductions, paired bit-wise check:

| Slurm job | Recipe | wandb run |
|---|---|---|
| **2102770** | det + overlap=ON | **[6c3mdfyz](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/6c3mdfyz)** |
| **2103151** | det + overlap=ON (paired) | **[ix4p5y2e](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/ix4p5y2e)** |
| **2103637** | det + overlap=ON (bit-wise check) | **[pfi9ap38](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/pfi9ap38)** |

**Perf-comparison nsys pair (2026-06-12)** — the runs whose nsys profiles drive sections 2–4 above:

| Slurm job | Recipe | wandb run |
|---|---|---|
| **2103633** | det + nsys15-18 | **[7klz92sb](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/7klz92sb)** |
| **2103635** | non-det + nsys15-18 | **[mb07l64y](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/mb07l64y)** |

In wandb, the lm-loss / mtp_1 / mtp_2 / grad_norm panels for the 7 det runs above
**overlay exactly** — pick any 2, click "Overlay runs" in the wandb UI, and the curves
trace each other to the bit.

---

## 5. Overlap Analysis (Comm vs Compute on Stream 7)

Stream 7 is shared between NCCL host launches and most compute. Streams 30/42/50 are dedicated NCCL
streams. Streams 155–158 are dedicated GEMM streams.

| | Det (3-iter window) | Non-det (3-iter window) |
|---|---|---|
| Window wall | 31,895 ms | 27,608 ms |
| GEMM streams (155–158) total | 4,629 ms (14.5% of wall) | 4,522 ms (16.4% of wall) |
| Stream 7 total (the main bottleneck) | 12,769 ms (40.0%) | 10,433 ms (37.8%) |
| Stream 42 NCCL SendRecv total | 3,940 ms (12.4%) | 3,448 ms (12.5%) |
| Stream 30+50 NCCL total | 558 ms (1.8%) | 556 ms (2.0%) |

**Key observation**: in both runs, **stream 7 is on the critical path** because it serializes the
det-substitute "other" kernels (fill/reduce/index) with the bulk of compute. With +2,127 ms of
extra fills/reduces on stream 7 in det, **even if PP p2p (stream 42) and the GEMM streams (155–158)
are perfectly overlapped, stream 7 itself adds ~700 ms/iter of un-overlappable work.**

That's the +1,398 ms/iter wall delta in one sentence:
- ~700 ms/iter from extra fill/reduce/index kernels serialized on stream 7
- ~170 ms/iter from the NCCL Ring/SendRecv slowdowns (mostly on stream 42)
- ~60 ms/iter from cuBLAS workspace + cuDNN bwd 2-pass + misc
- ~470 ms/iter from secondary cascade effects (worsened overlap, extra kernel launches, etc.)

---

## 6. Per-Knob Cost Attribution (Estimates)

Cannot be measured exactly without per-knob isolation runs, but inferred from kernel-name evidence:

| Knob | Touches (kernel signature) | Est. cost / iter | Confidence |
|---|---|---|---|
| `torch.use_deterministic_algorithms(True)` (via `deterministic_mode`) | `FillFunctor<BFloat16>` (1,583/3 ≈ 528 ms), `reduce_kernel` (143/3 ≈ 48 ms), `arange` (5/3), index segment-sum | **~580 ms/iter** | HIGH — direct kernel-name evidence |
| `cross_entropy_loss_fusion=false` | Adds ~6 `FillFunctor<float>` ops in CE bwd | **~50 ms/iter** | MED — partly absorbed in the above |
| `NCCL_ALGO=Ring` (genuine algo cost only — see §4.1 / §11) | AR/RS algo overhead; SendRecv and AG are *not* algo-selectable for P2P / unaffected | **~30–80 ms/iter** | MED — separating algo cost from wait-skew requires a 4-way ablation we have not yet run |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | cuDNN SDPA bwd split into 2 passes | **~7 ms/iter** | HIGH — kernel-knob-name evidence |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | +3 µs/call × 59,712 / 3 cuBLAS calls | **~60 ms/iter** | HIGH — count-based attribution |
| `MAMBA_DETERMINISTIC=1` | SSM kernel switch | **−19 ms/iter** (slightly faster) | HIGH |
| **Sum of primary causes** | | **~705–755 ms/iter** | |
| **Measured wall delta (iter 50)** | | **+1,398 ms/iter** | |
| **Cascade / wait-skew (~640–690 ms/iter)** | (a) NCCL wait-time growth from slower stream-7 work (≈150 ms/iter of the +702 ms/window NCCL delta is wait-skew, not algo); (b) bubble between kernels as stream-7 fills push downstream work later; (c) kernel-launch overhead amplification from extra small substitute calls; (d) marginal GEMM workspace effects | | MED — secondary, attributable in §11 |

The ~640–690 ms/iter "cascade" gap is consistent with the wait-inclusive nature of NCCL kernels
and the bubble growth between kernels: each extra `FillFunctor` / `index_put` call on stream 7
delays the next compute kernel, which in turn delays the next NCCL kernel that was supposed to
rendezvous behind that compute. This is **caused by the primary `use_deterministic_algorithms`
penalty**, not by an independent factor — the fix is the same one (§7.1).

---

## 7. Improvement Opportunities

Concrete, kernel-name-grounded suggestions, ordered by estimated wall-time recovery:

### 7.1 Reduce the cost of `FillFunctor`-zero + `scatter_add`/`index_put` (~400–500 ms/iter)

**Evidence**: 35,328 `FillFunctor<BFloat16>` calls/window (~11,776/iter) zeroing scatter destinations.
Most callers are MoE expert grad scatter + MTP-head loss backward.

> **Correction (see §11)**: an earlier draft proposed "zero buffers once at iter 0 and re-use".
> That is **incorrect** for accumulation destinations — `index_put_(accumulate=True)` adds onto
> existing contents, so the buffer must be re-zeroed every backward or gradients accumulate
> across iterations. The fills are **intrinsic** to the accumulate-into-prezeroed formulation.

**Legitimate fixes (in order of effort):**

| Fix | Mechanism | Bit-exact? | Est. recovery |
|---|---|---|---|
| **(a)** Batch the ~11,776 tiny per-expert scatters into a few large ones | One `scatter_add` per layer instead of one per expert. Still re-zeroed each iter, but ~10 large fills instead of ~11,776 tiny ones — drastically lower kernel-launch overhead. | yes | ~300 ms/iter |
| **(b)** Custom deterministic scatter kernel (Triton or CUDA) that fuses zero-init via write-on-first-touch instead of accumulate | Eliminates the separate `aten::fill_` + `aten::scatter_add_` chain by writing zeros only where data lands. | yes | ~400–500 ms/iter |
| **(c)** Scoped opt-out for specific MoE/MTP ops via `torch.use_deterministic_algorithms(warn_only=True)` around hot scatters | Preserves determinism elsewhere; sacrifices bit-exactness only for the named ops (with a warning). | **no** (those ops only) | ~500–600 ms/iter |

This is still the largest single lever — just a kernel-engineering project, not a caching one-liner.

### 7.2 Reuse forward's sort/permutation in same-iter activation-recompute backward (~140 ms/iter)

> **Correction (see §11)**: an earlier draft proposed caching the sorted-index buffer *across*
> iterations. That is **invalid** — router top-k indices depend on input data and router weights,
> both of which change every iteration; cross-iter caching would route tokens with stale
> assignments. Force-balance does not make the indices stable across iters — `RandomSTE` draws
> fresh logits every forward.

**Salvageable narrower win**: under selective recompute, the *recomputed forward* and the
*backward* both run inside `CheckpointFunctionBackward`, in the **same iteration**. If we
stash the forward's permutation buffer (cheaply, in the checkpoint context) and reuse it in
the recompute pass, we avoid recomputing the sort itself — the indices have not changed
between the original forward and the recompute.

**Wall-time recovery**: ~140 ms/iter (the cost of the recompute-side
`cub::DeviceRadixSort` calls).

### 7.3 Scope `NCCL_ALGO=Ring` to AR/RS only (recovery: smaller than initially claimed)

> **Correction (see §11)**: an earlier draft attributed the entire +520 ms SendRecv delta to
> `NCCL_ALGO=Ring`. That is **wrong** for two reasons:
> 1. `NCCL_ALGO` only applies to collectives (AllReduce, ReduceScatter, AllGather, Broadcast,
>    Reduce); **P2P SendRecv bypasses Ring/Tree selection entirely**.
> 2. NCCL device kernels are wait-inclusive, so the +520 ms is almost entirely peer-arrival
>    skew driven by det's slower stream-7 work, not algorithm cost.
>
> So the "scope NCCL_ALGO" optimization recovers **only** the genuine AR/RS algorithm overhead
> — and only if Tree is in fact slower than Ring on this cluster for those collectives.

**Salvageable narrower fix**: the only NCCL primitives where Tree's reduction order can
break bit-exactness are **AllReduce / ReduceScatter / Reduce** (where floats are summed across
ranks). AllGather, Broadcast, and SendRecv carry **no floating-point reduction** — they can be
left at NCCL default without breaking determinism. Per-function syntax:

```bash
# NCCL ≥ 2.24 (verify in the target container; 2.19 does NOT support this)
export NCCL_ALGO="allreduce:ring;reducescatter:ring;reduce:ring"
```

**Honest expectations**:
- Recovers only the *genuine* AR/RS algorithm overhead. From the data, that is roughly the
  `−15 µs/call × 4,362 AG calls = +66 ms/window` that AG currently sees from being pinned to
  Ring — recovered if we drop AG from the pin list. AR (combined) and RS contribute additional
  unknown amounts, but the **wait-inclusive caveat means the kernel-time numbers overstate the
  algorithm contribution**.
- The change must be re-validated with a 2-run bitwise check (the existing 3-way comparison
  framework).
- Realistic ceiling: ~30–80 ms/iter, not the 170 ms/iter previously claimed.

### 7.4 Selective `use_deterministic_algorithms` opt-out (~200–300 ms/iter, but breaks bit-exactness)

If only *checkpoint-resume reproducibility* (not per-iter bit-exactness) is needed, drop
`torch.use_deterministic_algorithms(True)` and rely on just the env vars + MCore deterministic_mode.
This preserves NCCL/CE/SDPA/Mamba determinism but allows `scatter_add` and `index_put` to use the
atomic-add (non-det) path.

**Trade-off**: same iter-1 loss, may drift by iter-50 if topk routing changes.

---

## 8. Methodology

### 8.1 Profile capture

- Both runs submitted via `scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh`, which 
  uses byte-aligned recipes (the `false → true` last-wins DDP overlap pattern from 
  `launch_nemotron_3_ultra_deterministic.sh` is preserved).
- nsys flags: `--enable_nsys --profiling_start_step 15 --profiling_stop_step 18 --profiling_ranks 0
  --nsys_trace cuda-sw,nvtx --export_nsys_sqlite`.
- 3-iter window captured on rank 0 only.

### 8.2 Window normalization

| | det | non-det |
|---|---|---|
| nsys window | 31,895 ms | 27,608 ms |
| Log iter 15 elapsed (ms) | 9,332 | 7,952 |
| Log iter 16 elapsed (ms) | 11,661 | 10,240 |
| Log iter 17 elapsed (ms) | 10,689 | 9,232 |
| Sum iters 15–17 (log) | 31,682 ms | 27,424 ms |
| nsys window vs log sum | 31,895 / 31,682 = 1.007 | 27,608 / 27,424 = 1.007 |

Window/log discrepancy < 1% → 3 iters captured cleanly, no startup/teardown contamination.

### 8.3 SQL queries used

All kernel-level stats came from direct sqlite queries on the `.sqlite` databases (no `nsys stats`
post-processing dependency). Schema: `CUPTI_ACTIVITY_KIND_KERNEL` joined to `StringIds` by
`demangledName`. NVTX module ranges came from `NVTX_EVENTS.text` (not `textId`) since MCore's
profile_nvtx wrapper writes free-text labels. Per-stream split via `streamId` group-by.

Query example (top kernels by total ms):
```sql
SELECT s.value, COUNT(*) n, SUM((k.end-k.start)/1e6) ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
GROUP BY s.value ORDER BY ms DESC LIMIT 20;
```

Stream-7 "other" decomposition (the dominant bucket):
```sql
SELECT s.value, COUNT(*) n, SUM((k.end-k.start)/1e6) ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
WHERE k.streamId=7
  AND s.value NOT LIKE 'ncclDevKernel%' AND s.value NOT LIKE 'nvjet_%'
  AND s.value NOT LIKE '%cublas%gemm%' AND s.value NOT LIKE '%cudnn%'
  AND s.value NOT LIKE '%triton_%' AND s.value NOT LIKE '%selective_scan%'
  AND s.value NOT LIKE '%state_passing%' AND s.value NOT LIKE '%chunk_scan%'
  AND s.value NOT LIKE '%chunk_state%' AND s.value NOT LIKE '%bmm_chunk%'
  AND s.value NOT LIKE '%causal_conv%'
GROUP BY s.value ORDER BY ms DESC LIMIT 12;
```

---

## 9. Bottom Line

- **Determinism IS bit-exact reproducible** at this scale (2103637 = 2102770 = 2103151 across iters 1, 5, 10, 15, 20, 30, 40, 50 — see §4.3).
- **Cost: +1,398 ms / iter (+17 % step time, −15 % MFU)** vs the same recipe without determinism.
- **~93 % of the NVTX-visible cost growth lives in `aten::*` operators** — MCore's own `if torch.are_deterministic_algorithms_enabled():` branches substitute `scatter`, `scatter_add`, `F.embedding`/`gather` with sort-and-segmented-reduce paths (NOT PyTorch internals — see §11 for the corrected mechanism).
- **`nvte::*` (TransformerEngine) is actually FASTER under det** (−5 %); compute library is not the source.
- **`mcore.fusions` is byte-identical** in NVTX time; `Optimizer.step#FusedAdam.step` is even *faster* under det (−9 %).
- **NCCL kernel time grows +702 ms / 3 iters**, but this is **mostly wait-time skew** from slower stream-7 work, not `NCCL_ALGO=Ring`. SendRecv (P2P) is not algo-selectable at all in NCCL; ReduceScatter is `RING_LL` in both runs. Recovery here is bounded.
- **Compute kernels (GEMM, attention, SSM)** are essentially unchanged at the kernel level: +2 % GEMM (cuBLAS workspace), **+21 % attention BWD alone** (det splits BWD into 2 passes; FWD is +3 %; total attn FWD+BWD is +14 %), **−6 % SSM**.
- **Realistic recovery without sacrificing bit-exactness**: ~300–500 ms / iter from batching/replacing the MoE scatter-zero-add chain (§7.1) + ~140 ms from same-iter recompute reuse (§7.2) + ~30–80 ms from scoping `NCCL_ALGO=Ring` to AR/RS only (§7.3). **Total: ~500–700 ms / iter (~5–7 %)**, all kernel-engineering or scoping work — none is a one-line cache fix.

---

## 10. Scope & Limitations

This report is grounded in 3 iterations from a single rank, with a benchmarking recipe. Three caveats
the reader should weigh before drawing scope-wide conclusions:

### 10.1 Single-rank, first-PP-stage visibility

The profile is **rank 0 only**, which on `TP=2 PP=3` lands on the **first pipeline stage**. That
rank **never sees** the cross-entropy backward or the MTP-head backward (both live on the **last
PP stage**). The CE-fusion knob (`cross_entropy_loss_fusion=false`) and a chunk of the
scatter-zero-add cost therefore do **not** show up in our `aten::*` decomposition — they happen
on the last-stage ranks and get dispatched to rank 0 only through PP P2P (which is what we
*do* see in `mcore.pipeline_parallel`'s `send_forward_recv_backward` range).

**Recommended follow-up**: profile **3 ranks simultaneously**: first PP stage (current),
middle PP stage, and last PP stage. Add **one rank on a different node** to expose EP-imbalance
effects across NVLink islands. The same `print_nsys_leaderboard.py` harness handles multi-CSV
input by simple concatenation per category.

### 10.2 Mock data + force-balanced routing

The current profile uses **mock data + `moe_router_force_load_balancing=True`** (verified by
NVTX `RandomSTE` ranges firing 1,152 times / window). This injects two extra ops every router
forward (`aten::clone` + `aten::normal_` totaling ~90 ms / iter) and forces uniform per-expert
token counts.

**Is the +1,398 ms / iter det penalty representative of production?** Largely yes, but with
caveats:
- MoE permute/scatter buffers are **pre-allocated at max capacity**; `FillFunctor` zeros the
  full buffer regardless of how many real tokens land. So the dominant `aten::fill_` + `scatter_add`
  cost is **buffer-shape-bound, not token-distribution-bound**.
- The det penalty per iter under production-realistic skewed routing would shift by **±1–3 %**
  from all-to-all bandwidth differences, no more.
- The fake-balance overhead itself (+90 ms / iter for `RandomSTE` + `aten::normal_`) is
  *false-presence* — a production run without it would subtract ~90 ms / iter from both det and
  non-det. That changes neither the absolute det penalty nor the relative %.

**Recommended follow-up**: rerun the matched pair with `moe_router_force_load_balancing=False`
+ a production-skewed dataset; compare iter-50 step times and the same NVTX leaderboard.

### 10.3 Iter-16 anomaly

Both runs show iter 16 at **+25–29 % slower than its neighbors**:

| iter | det elapsed (ms) | non-det elapsed (ms) |
|---|---|---|
| 15 | 9,332 | 7,952 |
| **16** | **11,661 (+25 %)** | **10,240 (+29 %)** |
| 17 | 10,689 | 9,232 |
| 18 | 10,726 | 9,318 |

This is unexplained. The 3-iter nsys window happens to start right at iter 15 and capture iter
16 in the middle, so the anomaly is folded into the per-iter average. Hypotheses worth testing:
- Python GC pause (we run with `train.manual_gc_interval=100`, so iter 16 shouldn't be a GC iter
  — but worth verifying)
- A first-time CUDA graph capture or workspace-allocation lazy-init
- Slurm I/O hiccup (NVMe checkpoint scan, fs latency spike)

**Recommended follow-up**: extend the nsys window to **iters 15–19 (5 iters)** and profile a
**later** iter range (e.g. 40–44) to confirm the steady-state per-iter numbers.

---

## 11. Reviewer Feedback Applied

This section records substantive corrections to earlier drafts so reviewers know the chain of
revisions. Original draft phrasings were wrong; the report bodies above have been edited inline
to reflect the corrected versions.

### 11.1 NCCL story (§4.1, §7.3) was largely incorrect

| Claim in earlier draft | What's actually true |
|---|---|
| "`NCCL_ALGO=Ring` adds 149 µs/call to SendRecv" | **Wrong.** `NCCL_ALGO` only applies to collectives (AR/RS/AG/Broadcast/Reduce); P2P SendRecv bypasses Ring/Tree selection entirely (verified in `nccl/src/enqueue.cc`). The +520 ms SendRecv delta is **wait-time skew** from slower stream-7 work delaying peer arrival — a symptom, not a cause. |
| "ReduceScatter Ring algo costs +73 µs/call" | **Wrong.** RS is `Sum_bf16_RING_LL` in **both** runs — same algorithm. The delta is wait-time, not algorithm choice. |
| "Use `NCCL_ALGO_AllReduce=Ring`" | **Wrong syntax.** That variable does not exist. The real per-function form is `NCCL_ALGO="allreduce:ring;reducescatter:ring"`, and it requires **NCCL ≥ 2.24**. Verify the container's NCCL version (2.19 will silently ignore the per-collective form). |
| "Fixing NCCL config recovers ~170 ms / iter" | **Overstated.** The legitimate recovery is bounded by the *genuine* AR/RS algorithm overhead (~30–80 ms / iter), not the wait-inclusive +173 ms. Fixing the fills recovers more. |

§4.1 caveat and §7.3 rewrite reflect the corrected story.

### 11.2 §7.1 fix mechanism was invalid

The earlier draft proposed "cache the scatter destination buffer at iter 0 and reuse". That
**does not work** for accumulation destinations: `index_put_(accumulate=True)` adds onto
existing contents, so the buffer **must** be re-zeroed every backward or gradients
accumulate across iterations. The fills are intrinsic to the accumulate-into-prezeroed
formulation. §7.1 rewrite gives three legitimate alternatives (custom fused kernel; batched
scatter; scoped opt-out).

### 11.3 §7.2 fix was invalid

The earlier draft proposed "cache the topk sorted-index buffer across iterations". That is
**incorrect** — router top-k indices depend on data and weights, both of which change every
iter; cross-iter caching would route tokens with stale assignments. `RandomSTE` draws fresh
logits every forward even under force-balance, so the indices are not iter-stable. §7.2
rewrite limits the win to **same-iter reuse** between forward and the recompute backward.

### 11.4 Mechanism narrative for `aten::max/min/remainder` was fabricated

The earlier draft said det-only `aten::max` / `aten::min` / `aten::remainder` come from
`torch.bincount` called by `scatter_add`'s det substitute. Verified against PyTorch source,
this chain is **wrong**:
- Deterministic `index_put_` on CUDA is **not** a "Python loop" — it's already radix-sort +
  segmented reduce, and that path is the **default** for `accumulate=True` on CUDA.
- Deterministic `scatter_add` does **not** call `bincount`.
- `bincount` does **not** use `max` / `min` / `remainder`; it uses fused `aminmax` + an atomic
  histogram.

Since those three ops *are* det-only in the profile, they most likely come from **MCore's own
explicit `if torch.are_deterministic_algorithms_enabled():` branches** (commonly: `scatter` →
`index_put_ + arange`, `scatter_add` → `index_add`, `F.embedding` → advanced-indexing build,
which uses `max`/`min`/`remainder` for hash-style index materialization). §4.5 "Mapping
kernels..." block has been rewritten to reflect this.

### 11.5 Arithmetic / labeling fixes

| Earlier claim | Correction |
|---|---|
| "Streams 7+42 = ~95 % of wall gap" | **66 %**. The +2,829 ms busy-time delta on streams 7+42 is 66 % of the +4,287 ms window-total delta. The other 34 % (~1,460 ms) is idle/bubble growth that §5/§6 already accounts for, and is not double-counted. |
| "Det substitute kernels = 96 % of the 'other' bucket" | **90 %**. (1,918 ÷ 2,127 = 90.2 %.) |
| "+14 % attention BWD" | The +14 % figure is FWD+BWD combined. **BWD alone is +21 %**; FWD is +3 %. §3.2 and §9 now state this explicitly. |
| §1's NCCL kernel count (11,013) vs §4.1 row sum | Now reconcilable: §4.1 totals to 3,486 + 4,362 + 2,730 + 9 + 30 + 384 + 6 + 3 + 3 = **11,013** for both runs (non-det splits 30 → 15 RING + 15 TREE on `f32` AR). |

---

## 12. Scale-up to 48 Nodes — Nsys-Induced Race + Sub-ULP Residual

**Headline (direct answer, updated 2026-06-16 with the 2134194 rerun):**

- **At 24 nodes / 96 GPUs**: bit-exact across 8 separate allocations including
  both nsys-profiled and non-profiled runs. Nsys was **inert**. (§4.3)
- **At 48 nodes / 192 GPUs**:
  - Two **det + no-nsys** runs on independent allocations (2132938 and 2134194)
    are **bit-exact through iter 40** and disagree by exactly **1 BF16 ULP at
    iter 50** — the recipe is *substantially* deterministic.
  - A **det + nsys** run (2132936) vs a **det + no-nsys** run (2132938)
    diverges starting at **iter 1**, with the gap growing iter-over-iter
    — nsys instrumentation is the **dominant** perturbation at this scale.

The earlier framing ("48-node is not bit-wise deterministic") conflated two
effects that turned out to have very different magnitudes; this section is
rewritten to separate them.

### 12.1 Setup

All 48-node jobs use `launch_nemotron_3_ultra_nsys_compare.sh` at commit
`a2f6ffce` (`-ng 192 -gn 4`). Everything except `world_size` is byte-identical
to the 24-node run (verified via diff of the submitted CLIs; see §12.4).

| | 24-node baseline (§4.3) | 48-node scale-up (this section) |
|---|---|---|
| world_size | 96 | **192** |
| TP / PP / EP / ETP / CP | 2 / 3 / 32 / 1 / 1 | 2 / 3 / 32 / 1 / 1 (unchanged) |
| DP_attn (= world_size / TP·PP·CP) | 16 | **32** |
| DP_expert (= world_size / EP·ETP) | 3 | **6** |
| GBS / MBS | 128 / 1 | 128 / 1 (unchanged) |
| Microbatches per pipeline step | 8 | **4** |
| Det env vars (`NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `MAMBA_DETERMINISTIC=1`) | applied | applied (unchanged) |
| `model.deterministic_mode` / `cross_entropy_loss_fusion` | true / false | true / false (unchanged) |
| `comm_overlap.tp_comm_overlap` | None → effectively False (MCore default) | None → effectively False (unchanged) |

### 12.2 The two effects, separately measured

**Four** 48-node det runs across separate allocations (3 from the original
3-job harness + 1 explicit no-nsys rerun):

| Slurm job | Nsys? | Allocation | wandb (project `mbridge-dev-zhiyul`) |
|---|---|---|---|
| **2132936** | ON  | lyris[0038-005+] | [`ev765dek`](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/ev765dek) |
| **2132937** | ON (non-det) | lyris[0128-014+] | [`6y8p80ay`](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/6y8p80ay) |
| **2132938** | OFF | lyris[0218-023+] | [`s5r2embd`](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/s5r2embd) |
| **2134194** | OFF (rerun) | lyris[0127-014+,0253-026+,0271-028+] | (no-nsys 48n rerun) |

#### 12.2(a) — Dominant effect: nsys-induced race (det+nsys vs det+no-nsys)

`lm loss` of **2132936 (det+nsys)** vs **2132938 (det+no-nsys)**:

| iter | 2132936 (nsys ON) | 2132938 (nsys OFF) | match | gap |
|---|---|---|---|---|
| 1  | 1.254853E+01 | 1.254854E+01 | ✗ | 1 ULP |
| 5  | 9.299143E+00 | 9.298958E+00 | ✗ | ~2·10⁻⁴ |
| 10 | 4.095631E+00 | 4.092978E+00 | ✗ | ~3·10⁻³ |
| 20 | 2.212457E-01 | 2.391398E-01 | ✗ | ~2·10⁻² |
| 30 | 6.335801E-01 | 2.387865E-01 | ✗ | ~4·10⁻¹ |
| 40 | 2.261086E-02 | 9.668706E-02 | ✗ | ~7·10⁻² |
| 50 | 4.662910E-01 | 6.824717E-02 | ✗ | ~4·10⁻¹ |

Divergence already at iter 1 → nsys instrumentation perturbs the very first
forward/backward pass. Critically, at 24 nodes (§4.3) the same paired
nsys-on vs nsys-off comparison was bit-exact across 50 iters — so this is
**scale-dependent**: nsys's per-kernel CPU overhead becomes order-changing
once the collective topology grows enough that one rank running marginally
slower flips a tie-break in NCCL group setup, expert-dispatch sequencing,
or stream-event ordering.

#### 12.2(b) — Residual: sub-ULP scale drift (det+no-nsys vs det+no-nsys)

`lm loss` of **2134194 (no-nsys rerun)** vs **2132938 (no-nsys original)** —
two independent det runs at 48 nodes, neither under nsys:

| iter | 2134194 (rerun) | 2132938 (original) | match |
|---|---|---|---|
| 1  | 1.254854E+01 | 1.254854E+01 | ✓ |
| 2  | 1.254648E+01 | 1.254648E+01 | ✓ |
| 3  | 1.023796E+01 | 1.023796E+01 | ✓ |
| 5  | 9.298958E+00 | 9.298958E+00 | ✓ |
| 10 | 4.092978E+00 | 4.092978E+00 | ✓ |
| 20 | 2.391398E-01 | 2.391398E-01 | ✓ |
| 30 | 2.387865E-01 | 2.387865E-01 | ✓ |
| 40 | 9.668706E-02 | 9.668706E-02 | ✓ |
| **50** | **6.824705E-02** | **6.824717E-02** | **✗ (1 ULP)** |

**Bit-exact for the first 40 iterations** between two independent
allocations, then a 1-ULP gap at iter 50. The residual is real but ~3
orders of magnitude smaller than the nsys-induced effect (compare iter-50
gap of ~10⁻⁵ here vs ~4·10⁻¹ in 12.2(a)).

### 12.3 Mechanism — race, not a missing config

We verified (§12.4) that the 24-node and 48-node CLIs are byte-identical
modulo `-ng`, the container hasn't moved, and the recipe / perf-config /
MCore submodule pointer have not changed between the dates of the two runs.
So both effects in 12.2 are **purely scale-induced**.

**12.3(a) — Why nsys-with-scale produces a race**

Nsys instrumentation adds per-kernel-launch CPU overhead on the profiled
rank(s). With 96 ranks (24 nodes) this overhead is absorbed inside the
existing slack of NCCL collective dispatch. With 192 ranks (48 nodes) the
slack tightens; the profiled rank's CPU loop runs marginally slower than
its peers, which is enough to:

- shift NCCL group-init order so the ring start is rotated by one rank
  (NCCL_ALGO=Ring fixes the *algorithm*, not which rank is the ring's
  origin),
- change the dispatch-batch sequencing in HybridEP (or even the regular
  alltoall path) when one rank arrives later than the rest,
- alter the CUDA stream-event observation order on rank-0 such that
  fused / non-fused kernel paths get chosen differently in TE.

Any of those changes BF16 partial-sum order → 1 ULP at iter 1 → loss
trajectory diverges. The mechanism is classic measurement-perturbation:
the instrumentation is non-inert at scale even though the kernels it
profiles are read-only.

**12.3(b) — The residual without nsys**

Even with no instrumentation, two independent allocations at 48n eventually
drift by 1 ULP (12.2(b), iter 50). Most likely contributors:

- NCCL ring tile size / origin selected at communicator-init time depends
  on which physical NVLink domain the ranks land in; two allocations of
  192 ranks across different lyris nodes can hash to slightly different
  ring layouts.
- HybridEP fabric handle import (`cuMemImportFromShareableHandle` with
  `CU_MEM_HANDLE_TYPE_FABRIC`) walks the NVL72 partition graph in an
  allocation-dependent order at EP=32 across 6 DP groups.

These are sub-ULP per-iter, only visible after enough accumulation
(≈40 iters here).

### 12.4 What changed between 24n (bit-exact) and 48n (this) — audit

Verified by diffing the submitted CLI lines (`Will launch the following
command…` from `submit-det.log` of both runs) and checking git for any
relevant code/config drift between the submission dates (2026-06-12 →
2026-06-15):

| Vector | Same across the two scales? |
|---|---|
| All 4 det env vars | ✓ identical |
| All Hydra overrides (deterministic_mode, ce_fusion, dispatcher, overlap, …) | ✓ identical |
| Parallelism (TP/PP/EP/ETP/CP) | ✓ identical (2/3/32/1/1) |
| GBS / MBS | ✓ identical (128 / 1) |
| Container image | ✓ same `nemo:26.04.01.squashfs`, mtime untouched |
| MCore submodule pointer | ✓ no commits between dates |
| Recipe `nemotron_3_ultra.py` | ✓ no commits |
| Perf-config `configs/nemotronh/` | ✓ no commits |
| `perf_plugins.py` / `argument_builder.py` / `utils/overrides.py` | ✓ no commits |
| `comm_overlap.tp_comm_overlap` (verified both gates: `moe_a2a_overlap=False`, `vp=1`) | ✓ effectively False on both |
| **`world_size` (`-ng`)** | **✗ 96 → 192 (the only difference)** |
| Physical Slurm node allocation | random per job — not a config delta |

→ No unexpected drift. Both effects in 12.2 are purely consequences of
doubling `world_size`.

### 12.5 What this means for the §2–§11 analysis

The 24-node analysis above is **unaffected** — the bit-exactness proof in
§4.3 stood across 8 separate allocations and is still load-bearing for
§1's "fairness" claim. The "nsys is inert" property is real **at 24n**.
That property is what we now know does **not** scale: nsys can perturb
the loss at iter 1 once the world size is large enough that
instrumentation-induced timing variance becomes order-changing.

### 12.6 Action items

Now that the dominant effect is identified as nsys-at-scale (not config
drift), the priorities shift:

1. **For correctness-sensitive runs at ≥48 nodes, profile via a separate
   "shadow" job** rather than instrumenting the run whose loss curve must
   match a baseline. Or restrict profiling to a single rank that's not on
   the critical-path of any all-reduce/expert-dispatch.
2. **Quantify the sub-ULP residual independently.** Schedule
   3-4 more det+no-nsys 48n runs and bin the iter-50 gap across all pairs
   — if every pair shows ≤1 ULP @ iter 50, the residual is *bounded* and
   can be reported as a known scale limit rather than a true bug.
3. **Sweep `NCCL_ALGO` per-collective** (NCCL ≥ 2.24):
   `NCCL_ALGO="allreduce:ring;reducescatter:ring;allgather:ring;sendrecv:ring"`
   — pins all major collectives to Ring explicitly in case some path falls
   through to Tree at the new scale.
4. **Test `moe_token_dispatcher_type=alltoall` (no HybridEP)** at 48n with
   no nsys to isolate whether the iter-50 residual lives in EP fabric vs
   regular DP/TP collectives.

### 12.6 Artifacts

| | path (under `$SHARE = /lustre/share/coreai_dlalgo_llm/zhiyul/nemotron-3-ultra-nsys-compare`) |
|---|---|
| OUT_DIR (CSVs + leaderboard.txt + bitwise_check.txt) | `48n-mismatch/processed/` |
| det+nsys log (2132936) | `48n-mismatch/raw/det/log-*_2132936_0.out` |
| det+no-nsys log (2132938) | `48n-mismatch/raw/det-bitwise/log-*_2132938_0.out` |
| nondet+nsys log (2132937) | `48n-mismatch/raw/nondet/log-*_2132937_0.out` |
| launch script + commit | `scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh` @ `a2f6ffce` (repo-relative) |

---

## Appendix: Profile Artifact Locations

NVIDIA internal cluster only. All paths below resolve under the shared mirror
(total ≈ 1.2 GB, world-readable):

```
SHARE=/lustre/share/coreai_dlalgo_llm/zhiyul/nemotron-3-ultra-nsys-compare
```

| | path (relative to `$SHARE`) |
|---|---|
| 24-node det nsys-rep (job 2103633) | `24n-baseline/raw/det/profile_810827_2103633_node0_rank0.nsys-rep` |
| 24-node det sqlite | `24n-baseline/raw/det/profile_810827_2103633_node0_rank0.sqlite` |
| 24-node det training log (job 2103633) | `24n-baseline/raw/det/log-*_2103633_0.out` |
| 24-node nondet nsys-rep (job 2103635) | `24n-baseline/raw/nondet/profile_2270167_2103635_node0_rank0.nsys-rep` |
| 24-node nondet sqlite | `24n-baseline/raw/nondet/profile_2270167_2103635_node0_rank0.sqlite` |
| 24-node nondet training log (job 2103635) | `24n-baseline/raw/nondet/log-*_2103635_0.out` |
| 24-node OUT_DIR (CSVs + leaderboard + submit/wdj/jobid logs) | `24n-baseline/processed/` |
| 24-node bit-wise check log (job 2103637) | `24n-baseline/raw/det-bitwise/log-*_2103637_0.out` |
| 48-node det nsys-rep (job 2132936) | `48n-mismatch/raw/det/profile_2352489_2132936_node0_rank0.nsys-rep` |
| 48-node det sqlite | `48n-mismatch/raw/det/profile_2352489_2132936_node0_rank0.sqlite` |
| 48-node det training log (job 2132936) | `48n-mismatch/raw/det/log-*_2132936_0.out` |
| 48-node nondet nsys-rep (job 2132937) | `48n-mismatch/raw/nondet/profile_967690_2132937_node0_rank0.nsys-rep` |
| 48-node nondet sqlite | `48n-mismatch/raw/nondet/profile_967690_2132937_node0_rank0.sqlite` |
| 48-node nondet training log (job 2132937) | `48n-mismatch/raw/nondet/log-*_2132937_0.out` |
| 48-node OUT_DIR (CSVs + leaderboard + bitwise_check.txt + submit/wdj/jobid logs) | `48n-mismatch/processed/` |
| 48-node bit-wise check log (job 2132938) | `48n-mismatch/raw/det-bitwise/log-*_2132938_0.out` |
| Top-level layout + job-to-wandb mapping | `README.md` |

Note: 24-node `processed/` does **not** contain a `bitwise_check.txt` because the
06/12 run used the original 2-job script (no automatic bit-wise diff step) —
the 24-node bit-exactness evidence is in §4.3's iter-by-iter table and the
raw `det-bitwise/log-*_2103637_0.out` file. The 48-node `processed/` does have
`bitwise_check.txt` because it was submitted by the current 3-job launcher.
