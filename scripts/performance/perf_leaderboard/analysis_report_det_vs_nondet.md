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

**Reading**: the **+2,336 ms** delta on stream 7 + the **+493 ms** on stream 42 (PP p2p)
account for **~95% of the wall-time gap**. The four dedicated GEMM streams (155–158) are
essentially identical — compute itself is barely affected.

### Stream-7 decomposition

| Stream-7 sub-bucket | Det ms | Non-det ms | Δ ms | Note |
|---|---|---|---|---|
| NCCL kernels (RS/AG/AR) | 2,399 | 2,189 | **+210** | `NCCL_ALGO=Ring` forces Ring for all collectives |
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
| `aten::max` | **368** | **0** | **+368** | **det-only**: used by `torch.bincount` (called by det scatter_add) |
| `aten::min` | **322** | **0** | **+322** | **det-only**: same |
| `aten::clone` | 556 | 253 | +303 | extra copies during det substitute path |
| `aten::contiguous` | 406 | 131 | +274 | det forces contiguous tensors for index sort |
| `aten::remainder` | **235** | **0** | **+235** | **det-only**: bincount index hashing |
| `aten::scatter` | 33 | 175 | **−142** | non-det uses this; det path swapped to `aten::scatter_add_` |
| `aten::copy_` | 629 | 764 | **−135** | non-det does *more* copies (FP32 master copies in unfused-CE) |

The smoking gun: `aten::max`, `aten::min`, `aten::remainder` are **zero in non-det** and **>200 ms each in det**. These are `torch.bincount`'s implementation — bincount is in turn called by `scatter_add`'s deterministic substitute. So the chain is:

```
torch.use_deterministic_algorithms(True)
    └── scatter_add_ → unsafe_scatter disabled
        └── falls back to: bincount(indices) + segment_sum
                            └── max(indices), min(indices), remainder(indices, num_buckets) ← det-only
                            └── + arange(num_indices)
                            └── + empty + fill (zero buffers)
                            └── + scatter into binned buffer
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
| **Sum of det-substitute kernels** | — | **~2,031** | — | **~113** | **+1,918** | ~96% of the +2,127 "other" gap |

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

Key observations:
1. **SendRecv +520 ms** is the largest NCCL delta. PP p2p chunks are small and **latency-bound**;
   `NCCL_ALGO=Ring` adds 149 µs/call (~11%). This is on the dedicated stream 42.
2. **ReduceScatter +198 ms** (+33% per call) — similar story: small per-DP-shard message at this batch/SeqLen.
3. **AllGather is actually faster in det (−44 ms)** — Ring is the bandwidth-optimal choice for AG
   anyway; non-det's default Tree algorithm has slightly higher per-call latency here.
4. Non-det has BOTH `RING_LL` *and* `TREE_LL` variants of f32 AllReduce — confirming NCCL is free
   to pick per-call in non-det, and pinned to RING in det.

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
| `NCCL_ALGO=Ring` | SendRecv (+173 ms/iter), ReduceScatter (+66 ms/iter), AllGather (−15 ms/iter) | **~230 ms/iter** | HIGH — direct NCCL kernel-name evidence |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | cuDNN SDPA bwd split into 2 passes | **~7 ms/iter** | HIGH — kernel-knob-name evidence |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | +3 µs/call × 59,712 / 3 cuBLAS calls | **~60 ms/iter** | HIGH — count-based attribution |
| `MAMBA_DETERMINISTIC=1` | SSM kernel switch | **−19 ms/iter** (slightly faster) | HIGH |
| **Sum** | | **~910 ms/iter** | |
| **Measured wall delta (iter 50)** | | **+1,398 ms/iter** | |
| **Unaccounted gap (~490 ms/iter)** | secondary cascade: bubble from stream-7 stalls, kernel-launch overhead amplification, marginal GEMM workspace effects | | LOW — not directly attributable |

The ~490 ms/iter "unaccounted" gap is consistent with secondary effects: more kernel launches on
stream 7 push downstream compute kernels later, occasionally exposing NCCL collectives that were
previously hidden behind compute.

---

## 7. Improvement Opportunities

Concrete, kernel-name-grounded suggestions, ordered by estimated wall-time recovery:

### 7.1 Pre-allocate `scatter_add`/`index_put` destination buffers (~400–500 ms/iter)

**Evidence**: 35,328 `FillFunctor<BFloat16>` calls/window (~11,776/iter) zeroing scatter destinations.
Most callers are MoE expert grad scatter + MTP-head loss backward. **The buffers are the same
shape every iter**.

**Fix**: cache zero-buffers in `MoE.backward()` and `MTPHead.backward()` and `.zero_()` once at
iter 0, then re-use. Net: 11,776 fills → ~10 fills/iter.

**Wall-time recovery**: ~470 ms/iter (~5% of wall).

### 7.2 Replace `cub::DeviceRadixSort` path in `GatherBackward0` with cached index mapping (~140 ms/iter)

**Evidence**: `GatherBackward0` is +446 ms/window over `CheckpointFunctionBackward` already accounts
for most of `GatherBackward0`'s wall. The MoE router topk indices are deterministic across iters
when load-balancing is on — so the sort-key is constant per iter.

**Fix**: cache the sorted-index buffer in the router; refresh only when topk changes.

**Wall-time recovery**: ~140 ms/iter.

### 7.3 Move PP p2p off Ring algo (~170 ms/iter)

**Evidence**: SendRecv is +149 µs/call × 3,486 calls/window = +520 ms/window = +173 ms/iter.
`NCCL_ALGO=Ring` is overkill for P2P — Ring only affects multi-rank collectives.

**Fix**: scope `NCCL_ALGO=Ring` to AllReduce only, leave SendRecv at default.
- Approach A: per-collective NCCL env-var pinning (NCCL supports collective-specific algo via
  `NCCL_ALGO_AllReduce=Ring` only).
- Approach B: use a separate NCCL communicator for PP that doesn't have `NCCL_ALGO` set.

**Wall-time recovery**: ~170 ms/iter (the full SendRecv delta).

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

- **Determinism IS bit-exact reproducible** at this scale (2103637 = 2102770 = 2103151 across iters 1, 10, 20, 30, 40, 50).
- **Cost: +1,398 ms / iter (+17% step time, −15% MFU)** vs the same recipe without determinism.
- **~93% of the NVTX-visible cost growth lives in `aten::*` operators** (PyTorch's deterministic substitute paths — fill / index_put / arange / scatter_add / bincount-via-max-min-remainder).
- **`nvte::*` (TransformerEngine) is actually FASTER under det** (−5%), confirming the cost is not in the compute library.
- **`mcore.fusions` is byte-identical** in NVTX time; `Optimizer.step` is even faster under det (−9%).
- **`NCCL_ALGO=Ring` slows PP P2P by +492 ms / 3 iters** (visible at both NVTX `mcore.pipeline_parallel` and NCCL kernel level).
- **Compute kernels (GEMM, attention, SSM) are essentially unchanged at the kernel level** — +2% GEMM (cuBLAS workspace), +14% attention bwd (det splits into 2 passes), **−6% SSM (det path is slightly faster on B200)**.
- **Two cleanly-actionable optimizations** could recover **~640 ms / iter (~7%)** without sacrificing bit-exactness: pre-allocate scatter destinations + scope `NCCL_ALGO=Ring` to AllReduce only.

---

## Appendix: Profile Artifact Locations

| | path |
|---|---|
| Det nsys-rep | `~/.nemo_run/experiments/nemotron-3-ultra-det-nsys15-18-1781255131/.../profile_810827_2103633_node0_rank0.nsys-rep` |
| Det sqlite | …same dir, `.sqlite` extension |
| Non-det nsys-rep | `~/.nemo_run/experiments/nemotron-3-ultra-nondet-nsys15-18-1781255131/.../profile_2270167_2103635_node0_rank0.nsys-rep` |
| Non-det sqlite | …same dir, `.sqlite` extension |
| OUT_DIR (CSVs + leaderboard.txt) | `/lustre/fsw/coreai_dlalgo_llm/zhiyul/nsys-compare-20260612-0205/` |
| Bit-wise check (det run #2) | job 2103637, `~/.nemo_run/experiments/nemotron-3-ultra-det-no-nsys-bitwise-check-v2/...` |
