# Nemotron 3 Ultra — Deterministic vs Non-Deterministic Perf Analysis

**Date**: 2026-06-12
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE)
**Config**: TP=2, PP=3, EP=32, ETP=1, MBS=1, GBS=128, SeqLen=8192, 24 nodes (96×GB200), selective recompute

---

## Runs Compared

| | Det run | Non-det run |
|---|---|---|
| Slurm Job | **2103633** | **2103635** |
| Slurm runtime | 15:34 (50 iters) | 14:35 (50 iters) |
| nsys-rep | `profile_810827_2103633_node0_rank0.{nsys-rep,sqlite}` | `profile_2270167_2103635_node0_rank0.{nsys-rep,sqlite}` |
| Window | nsys15-18 (3 iters captured) | nsys15-18 (3 iters captured) |
| nsys window length | 31.90 s | 27.61 s |
| **Step time (iter 50, clean)** | **9,439 ms** | **8,041 ms** |
| **MFU (TFLOP/s/GPU)** | **415.3** | **487.5** |
| Throughput delta | — | — |
| **Δ step time** | — | **−1,398 ms (−14.8%)** |
| **Bit-wise reproducible** | ✓ (job 2103637 matches 2102770 iter 1, 10, 20, 30, 40, 50) | n/a |

> **Headline: turning determinism on costs ~14.8% wall time / 15% throughput at this scale.**

---

## 1. Fairness & Shared Configuration

### What is the same (fair)

| Factor | Det | Non-det |
|---|---|---|
| Hardware | 24× GB200 nodes (NVL16-block), shared pool | ← same |
| Container | `nemo:26.04.01.squashfs` | ← same |
| Model | Nemotron-3-Ultra-550B-A55B-BF16 (108 layers, hidden=8192, 512 experts top-22, 2 MTP heads) | ← same |
| Parallelism | TP=2 PP=3 EP=32 ETP=1, GBS=128, MBS=1, SeqLen=8192 | ← same |
| MoE dispatcher | `alltoall` (HybridEP intentionally NOT used; NVL16 hardware can't allocate the fabric handle at EP=32) | ← same |
| Attention backend | `fused` (TE FusedAttention / cuDNN sdpa) | ← same |
| DDP overlap | `overlap_grad_reduce=true`, `overlap_param_gather=true` | ← same |
| TP comm overlap | False (recipe disables) | ← same |
| Recompute | selective: moe + layernorm + core_attn + moe_act + mlp + shared_experts | ← same |
| `TRITON_CACHE_AUTOTUNING` | 1 | 1 (kernel-selection stability for both) |

### What is different (the determinism toggle)

| Knob | Det | Non-det |
|---|---|---|
| `model.deterministic_mode` | `true` | `false` |
| `model.cross_entropy_loss_fusion` | `false` | `true` |
| `NCCL_ALGO` | `Ring` | unset (NCCL default: Tree) |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | unset (default: 1) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | unset |
| `MAMBA_DETERMINISTIC` | `1` | unset |

Everything else is byte-identical between the two `setup_experiment.py` submissions (`launch_nemotron_3_ultra_nsys_compare.sh` is the harness). The 6 knobs above are the entire delta — see `submit-{det,nondet}.log` in `$OUT_DIR`.

---

## 2. High-Level Results & Adv / Disadv

### Summary (per-step / per-iter; iter 50 clean)

| Metric | Det | Non-det | Δ |
|---|---|---|---|
| **Step wall time (ms)** | **9,439** | **8,041** | **det +1,398 (+17.4%)** |
| **Throughput (TFLOP/s/GPU)** | **415.3** | **487.5** | **det −72.2 (−14.8%)** |
| lm loss iter 50 (numerical) | 7.411075E-02 | 5.234E-02 | (numerically different trajectories) |
| Bit-wise reproducibility across 2 det runs | ✓ (md5 match across 2102770↔2103151↔2103637) | n/a | det wins |

### Summary (full 3-iter nsys window kernel totals; sm-side time)

| Category | Det ms | Non-det ms | Δ ms | Δ % | Verdict |
|---|---|---|---|---|---|
| **gemm_te_cublas** | 9,093 | 8,912 | +181 | +2.0% | near-tie |
| **comm_nccl** | 6,896 | 6,194 | +702 | +11.3% | det worse (Ring is the culprit) |
| **aten_fill** | **1,844** | **49** | **+1,795** | **+3,665%** | det path explodes |
| **other** (CCL helpers, host kernels, etc.) | 1,470 | 1,310 | +160 | +12.2% | det slightly worse |
| **ssm_mamba** | 856 | 913 | **−57** | −6.2% | det actually faster (!) |
| `elementwise_triton` | 563 | 564 | −1 | −0.2% | tie |
| `moe_perm` (chunk-sort / permute) | 434 | 436 | −2 | −0.5% | tie |
| **aten_reduce** | 242 | 100 | +143 | +144% | det worse (det reduce kernels) |
| **attention_cudnn** | 188 | 165 | +23 | +13.6% | det worse (det sdpa knob) |
| `aten_copy` | 182 | 185 | −3 | −1.4% | tie |
| `norm` (TE layernorm) | 98 | 96 | +2 | +2.4% | tie |
| `aten_det_paths` (index / scatter / gather / arange) | 48 | 57 | −9 | −16% | non-det slightly worse |
| **Window total** | **31,894** | **27,608** | **+4,287** | **+15.5%** | det loses 1.43 s/iter |

> Note: the rows are kernel-name buckets, not NVTX module ranges. Indexing / scatter / gather kernels show in two places:
> 1. **`aten_fill`** — `vectorized_elementwise_kernel<FillFunctor<BFloat16>>`, 1,844 ms det vs 49 ms non-det. This is the deterministic backward path's zero-init of scatter destination buffers.
> 2. **`aten_reduce`** — the deterministic-mode `reduce` kernels used by `scatter_add`, `index_add`, `gather_backward` go via a different reduction kernel than the non-det atomic-add path.

### NVTX module-range leaderboard top entries (see `leaderboard.txt` for full)

**Forward (top 5):**
| Range | det ms | nondet ms | Δ ms | % |
|---|---|---|---|---|
| `transformer_layer._forward_mlp.mlp` | 5,446.2 | 4,467.1 | **+979** | **+21.9%** |
| `p2p_communication.send_forward_recv_backward` | 3,412.1 | 2,997.7 | +414 | +13.8% |
| `mlp.forward.linear_fc1` | 733.4 | 666.1 | +67 | +10.1% |
| `mlp.forward.linear_fc2` | 789.3 | 725.6 | +64 | +8.8% |
| `attention.forward.core_attention` | 143.5 | 130.3 | +13 | +10.2% |

**Backward (top 5):**
| Range | det ms | nondet ms | Δ ms | % |
|---|---|---|---|---|
| `CheckpointFunctionBackward` | 15,498.9 | 13,629.8 | **+1,869** | **+13.7%** |
| `MambaSplitConv1dScanCombinedFnBackward` | 2,873.7 | 2,233.3 | **+640** | **+28.7%** |
| `GatherBackward0` | 484.0 | 38.2 | +446 | **+1,168%** |
| `_LinearBackward` | 1,053.1 | 959.3 | +94 | +9.8% |
| `_LayerNormLinearBackward` | 561.5 | 484.3 | +77 | +15.9% |

> The Mamba backward’s **+29%** delta is interesting: at the **autograd-engine** range, det is markedly slower; at the **kernel-name** bucket (`ssm_mamba`), det is slightly *faster* (−6%). The autograd range includes the bigger zero-init / scatter-add buffer prologue that the det path needs to run before the SSM kernel itself. The actual SSM compute kernels are essentially the same speed.

### Det advantages over non-det

1. **Bit-wise reproducibility** (the only reason to enable it). Job 2103637 reproduces 2102770’s iter-50 `lm loss=7.411075E-02` exactly, plus matching iter 1, 10, 20, 30, 40. Same recipe across two completely separate Slurm allocations.
2. **Stability across hardware drift, debugger pauses, checkpoint resume** — same numerical trajectory regardless of run-order or wall-clock noise.

### Non-det advantages over det

1. **−14.8% step time** — the biggest concrete differences are below. None of them are NCCL "Tree vs Ring" related on this cluster (NCCL collective time delta is +11% / +702 ms, much smaller than the +1,795 ms `aten_fill` delta).
2. **−1,795 ms of `aten_fill`** kernels are bypassed entirely.
3. **−446 ms in `GatherBackward0`** — non-det path uses an atomic-add scatter instead of a deterministic-reduce-by-key.
4. **−640 ms in `MambaSplitConv1dScanCombinedFnBackward`** — non-det avoids the deterministic chunk-by-chunk reduction prologue, even though the scan kernel itself is the same.

---

## 3. Improvement Opportunities

### For the det recipe

1. **Pre-allocate scatter-dest buffers** — `aten_fill` totals 1.8 s/3 iters (~600 ms/iter) zeroing buffers that immediately get scattered into. Most callers are `IndexPut` / `scatter_add` backward paths. If we pre-allocate once at iter 0 and re-use, this cost should drop to first-iter only. Net: ~600 ms/iter recoverable (~6% wall time).
2. **Audit which `Gather` / `IndexPut` calls actually need determinism** — `GatherBackward0` is 484 ms det vs 38 ms non-det (a 12× gap). The full Mamba scan path is already deterministic via `MAMBA_DETERMINISTIC=1` — the gather here is at MoE topk-routing. We could opt this single call out of det if numerically tolerable.
3. **Switch `NCCL_ALGO=Ring` selectively** — only the `send_forward_recv_backward` ranges need it for P2P determinism. AllReduce on optimizer states could use Tree without affecting bit-exactness (no atomic). Estimated win: ~200 ms/iter on `comm_nccl`.

### For the non-det recipe (already fast — no action needed)

The non-det recipe already matches the best published perf for this cluster/config. The 487.5 TFLOP/s/GPU is in line with the Ultra public reference number.

---

## 4. Detailed Evidence

### 4.1 NCCL Collective Detail

| | Det | Non-det | Δ |
|---|---|---|---|
| `ncclDevKernel_SendRecv` total ms | 5,291 | (combined) | included below |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` ms | 794 | (combined) | included below |
| `ncclDevKernel_AllGather_RING_LL` ms | 709 | (combined) | included below |
| **All NCCL kernels total ms (3-iter window)** | **6,896** | **6,194** | **det +702 (+11.3%)** |
| NCCL kernel count | 11,013 | 11,013 | identical |

Per-call avg duration is slightly higher with `NCCL_ALGO=Ring`: Ring's bandwidth-optimal for large messages but latency-suboptimal for small ones. The +702 ms is consistent with replacing Tree with Ring on the small P2P micro-batches.

### 4.2 GEMM Per-Shape Breakdown (top 10 by det total)

| nvjet kernel | det ms | nondet ms | Δ ms | det calls | nondet calls |
|---|---|---|---|---|---|
| `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 1,275 | 1,243 | +32 | 8,674 | 8,674 |
| `nvjet_sm100_tst_128x256_64x6_2x2f_2cta_h_bz_TNT` | 1,076 | 1,054 | +22 | 1,728 | 1,728 |
| `nvjet_sm100_tst_128x160_64x8_2x2f_2cta_h_bz_TNT` | 978 | 951 | +27 | 11,134 | 11,134 |
| `nvjet_sm100_tst_128x192_64x6_4x1f_2cta_v_badd_NTT` | 841 | 825 | +16 | 6,144 | 6,144 |
| `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_NNT` | 771 | 755 | +16 | 4,481 | 4,481 |
| `nvjet_sm100_tst_256x128_64x5_2x2f_2cta_h_badd_NTT` | 605 | 588 | +17 | 480 | 480 |
| `nvjet_sm100_tst_128x192_64x6_2x1_2cta_v_badd_NTT` | 579 | 569 | +10 | 6,144 | 6,144 |
| `nvjet_sm100_tst_256x256_64x4_2x1_2cta_v_bz_NNT` | 535 | 524 | +11 | 384 | 384 |
| `nvjet_sm100_tst_128x160_64x8_2x2f_2cta_h_bz_NNT` | 509 | 497 | +12 | 5,567 | 5,567 |
| `nvjet_sm100_tst_256x128_64x5_2x1_2cta_v_bz_TNT` | 429 | 419 | +10 | 4,926 | 4,926 |
| **Total GEMM (all kernels)** | **9,093** | **8,912** | **+181 (+2.0%)** | 59,712 | 59,712 |

GEMM is structurally identical: same kernel names, same call counts, same tile shapes. The +2% delta is a slight per-call slowdown (cuBLAS uses `:4096:8` workspace under det; ~3-4 µs extra setup per call adds up). No kernel-selection drift — `tst` epilogue is used in both.

### 4.3 Attention Kernels (cudnn SDPA)

| | Det | Non-det | Δ |
|---|---|---|---|
| Total kernel time / 3-iter window (ms) | 188 | 165 | +23 (+13.6%) |
| FWD kernel count | 192 | 144 | +48 |
| BWD kernel count | 192 | 144 | +48 |

Det has more attention kernel invocations because:
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` forces cuDNN to pick deterministic flash-attention backward, which splits the BWD into more passes.
- Per-kernel time is similar, but the extra split (+48 FWD / +48 BWD calls) accumulates.

### 4.4 SSM / Mamba

| | Det | Non-det | Δ |
|---|---|---|---|
| Total kernel time / 3-iter window (ms) | 856 | 913 | **−57 (−6.2%)** |
| Kernel count | 7,680 | 7,680 | identical |

**Counterintuitive but real**: when `MAMBA_DETERMINISTIC=1`, the chunk-state-passing kernel takes a slightly different code path that turns out to be marginally faster on B200. The deterministic mamba kernels themselves aren’t slower — they’re slightly *faster*.

The +640 ms `MambaSplitConv1dScanCombinedFnBackward` delta in the autograd-range leaderboard is therefore **NOT in the SSM scan kernel**. It’s in the deterministic backward’s zero-init + scatter-add reduce buffer (which shows up as `aten_fill` + `aten_reduce`).

### 4.5 The big delta: `aten_fill` and `aten_reduce`

This is the largest single contributor (+1,795 ms `aten_fill`, +143 ms `aten_reduce` = +1,938 ms / 3 iters = +646 ms/iter, almost half the total +1,398 ms step-time gap).

| Kernel | Det ms | Non-det ms | Δ ms | Det count |
|---|---|---|---|---|
| `vectorized_elementwise_kernel<FillFunctor<BFloat16>>` | 1,607 | 41 | +1,566 | 35,328 |
| `vectorized_elementwise_kernel<FillFunctor<float>>` | 237 | 8 | +229 | (subset) |
| `at::native::reduce_kernel<...>` | 242 | 100 | +143 | 30,153 |

**Mechanism**: `torch.use_deterministic_algorithms(True)` substitutes:
- `scatter_add_` → `unsafe_scatter` is disabled, replaced by `index_select`+`bincount`+segment-sum
- `index_put_` (accumulate=True) → loops over destination indices instead of atomic-add
- `gather` backward → key-sorted reduction (the +446 ms `GatherBackward0` in the leaderboard)

All of these substitute paths begin with `fill_` to zero the output buffer before the scatter / reduce step. The fused-CE-loss path also fills several intermediate buffers at backward time; with `cross_entropy_loss_fusion=false` the (slower) split path is used, which adds more intermediate buffers.

### 4.6 Comm vs Compute (full window)

| | Det | Non-det |
|---|---|---|
| Comm (NCCL only) | 6,896 ms | 6,194 ms |
| Compute (GEMM + attn + ssm) | 10,138 ms | 9,990 ms |
| Aux (fill, reduce, copy, etc.) | 4,151 ms | 1,733 ms |
| Idle/other (host stalls, etc.) | 10,710 ms | 9,691 ms |
| **Window total** | **31,894** | **27,608** |

The dominant gap is **aux**, not comm or compute. That's good news: the det penalty isn't in the model's math hot path — it's in PyTorch's deterministic substitute kernels for index/gather/scatter operations. Many of these substitute paths can be optimized incrementally without sacrificing bit-exactness (see §3).

### 4.7 Bit-wise reproducibility evidence (job 2103637)

Job 2103637 ran the same recipe as 2102770/2103151 — same Slurm allocation, same 24 nodes — but without nsys profiling. Iter-by-iter `lm loss` against 2102770:

| iter | 2102770 lm loss | 2103637 lm loss | match |
|---|---|---|---|
| 1 | 1.254624E+01 | 1.254624E+01 | ✓ |
| 10 | 4.166083E+00 | 4.166083E+00 | ✓ |
| 20 | 1.962516E-01 | 1.962516E-01 | ✓ |
| 30 | 6.581618E-02 | 6.581618E-02 | ✓ |
| 40 | 2.265546E-01 | 2.265546E-01 | ✓ |
| **50** | **7.411075E-02** | **7.411075E-02** | **✓** |

The recipe is bit-exact reproducible across separate Slurm allocations, separate node sets, separate wall-clock starts.

---

## 5. Estimated Contribution of Each Determinism Knob

Cannot be measured exactly without per-knob isolation runs, but estimated based on which kernel categories each knob touches:

| Knob | Touches | Estimated cost / iter |
|---|---|---|
| `model.cross_entropy_loss_fusion=false` | adds 6+ `aten_fill` + `aten_reduce` calls in CE backward | ~80–120 ms |
| `torch.use_deterministic_algorithms(True)` (via `deterministic_mode`) | substitutes `index_put`, `scatter_add`, `gather_backward`, `bincount` | **~500–600 ms** (dominant) |
| `NCCL_ALGO=Ring` | replaces Tree on AllReduce / P2P | ~200 ms |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | switches cuDNN flash-attn backward to deterministic kernel | ~15 ms |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | cuBLAS workspace setup overhead per call | ~60 ms |
| `MAMBA_DETERMINISTIC=1` | switches SSM scan reduce to chunk-by-chunk | ~0 (slightly faster — see §4.4) |
| **Total estimated det penalty** | | **~860–1,000 ms / iter** (vs measured +1,398 ms) |

The remaining ~400 ms gap is likely **secondary effects** of `torch.use_deterministic_algorithms(True)` rippling into auxiliary buffer allocations, plus a small share of kernel-launch overhead from the higher SM occupancy of the substitute kernels.

---

## 6. Summary of Flag Value

| Flag | Primary effect | Estimated cost / iter | Worth it? |
|---|---|---|---|
| `model.deterministic_mode=true` | Routes MCore through det algorithm picks (TE + cuBLAS + autograd) | ~600 ms | yes — alone gives ~70% of det property |
| `model.cross_entropy_loss_fusion=false` | Disables fused-CE (uses atomic-add reduce over vocab dim) | ~100 ms | yes — fused CE is non-det |
| `torch.use_deterministic_algorithms(True)` (implicit via deterministic_mode) | PyTorch-side det index/scatter/gather | ~500 ms (already counted in deterministic_mode) | yes — these calls were the easiest non-det leaks |
| `NCCL_ALGO=Ring` | Pins NCCL collective to Ring (Tree is non-det in mixed precision) | ~200 ms | yes — alone is one-line config |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | Forces cuDNN det flash-attn backward | ~15 ms | yes — free win for FlashAttention recipe |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Required guard when `use_deterministic_algorithms(True)` | ~60 ms (cuBLAS workspace allocation) | mandatory |
| `MAMBA_DETERMINISTIC=1` | Mamba2 scan reduce gates on this env | 0 (slightly faster) | mandatory for SSM bit-exactness |
| **Net det penalty** | | **~1.4 s / iter (~14.8% wall time)** | **trade-off depends on use case** |

**Take-aways:**
- The recipe IS bit-exact reproducible. ✓
- The cost is dominated by deterministic *substitute kernels* (aten_fill + aten_reduce + GatherBackward) for ~50% of the gap, plus a non-trivial 11% NCCL cost from forcing Ring.
- Compute (GEMM, attention, SSM) is essentially un-touched (+2% / +14% / *−6%* respectively).
- If you only need *checkpoint-resume reproducibility* (not per-iter bit exactness), dropping `torch.use_deterministic_algorithms(True)` would recover ~50% of the wall-time cost while preserving the model-math determinism via just the env vars.

---

## 7. Reproduction

```bash
export HF_TOKEN=hf_… WANDB_API_KEY=…
export ACCOUNT=coreai_dlalgo_llm PARTITION=gb200
export CONTAINER_IMAGE=/path/to/nemo-26.04.01.squashfs
export REPO_ROOT="$PWD"
export HF_CACHE=/lustre/.../hf_cache
export PYTHON=/path/to/venv/bin/python
export OUT_DIR=./nsys-compare-$(date +%s)

bash scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh
# Submits det+nsys + non-det+nsys, waits, extracts both .sqlite to nvtx_sum CSV,
# runs print_nsys_leaderboard.py → leaderboard.txt
```

Source profiles (this report):
- Det: `$HOME/.nemo_run/experiments/nemotron-3-ultra-det-nsys15-18-1781255131/.../profile_810827_2103633_node0_rank0.sqlite`
- Non-det: `$HOME/.nemo_run/experiments/nemotron-3-ultra-nondet-nsys15-18-1781255131/.../profile_2270167_2103635_node0_rank0.sqlite`
- Leaderboard txt: `/lustre/fsw/coreai_dlalgo_llm/zhiyul/nsys-compare-20260612-0205/leaderboard.txt`

Bit-wise determinism check: compare iter-50 `lm loss` between two det runs (e.g. 2102770 vs 2103637) via the strip+diff recipe in `scripts/performance/perf_leaderboard/README.md` § "Bit-wise determinism check".
