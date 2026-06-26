# Nemotron 3 Ultra — Deterministic vs Non-Deterministic Perf Analysis @ 3072 GPUs

**Date**: 2026-06-26
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE, 108 layers, 512 experts top-22, 2 MTP heads)
**Config**: TP=2, PP=3, EP=32, ETP=1, CP=1, MBS=1, **GBS=4096** (auto-scaled 128×32), SeqLen=8192,
**768 nodes × GB200 = 3072 GPUs**, selective recompute

> This is the 3072-GPU companion to [`analysis_report_det_vs_nondet.md`](analysis_report_det_vs_nondet.md)
> (the 24-node baseline + §12 48-node observations). The **kernel-level cost decomposition** below
> reproduces structurally at 3072 GPUs. The **determinism behavior does not** — see §5, which is the
> load-bearing new finding and is written observations-only, no premature root-cause (same discipline
> as the parent doc's §12).
>
> Source profiles: nsys captures of iters 15–17 on **ranks 0 / 1536 / 3071** of jobs **3590404** (det)
> and **3590455** (non-det). Determinism pair: **3590404** (det+nsys) vs **3590484** (det, no-nsys).

---

## Runs Compared

| | Det + nsys (3590404) | Non-det + nsys (3590455) | Det no-nsys (3590484) |
|---|---|---|---|
| wandb (`mbridge-dev-zhiyul`) | [7pcz6l7o](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/7pcz6l7o) | [x0cipvil](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/x0cipvil) | [x2i6b7ev](https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/x2i6b7ev) |
| Steady-state step time (iter 50) | **10,112 ms** | **8,849 ms** | 9,987 ms |
| Throughput (TFLOP/s/GPU, iter 50) | **387.7** | **443.0** | 392.6 |
| **Step time Δ (det+nsys vs nondet+nsys)** | — | **det +1,263 ms (+14.3%)** | — |
| **Throughput Δ** | — | **det −55.3 (−12.5%)** | — |
| Bit-wise reproducible | see §5 (does **not** match the no-nsys run) | n/a | see §5 |

**Headline det penalty at 3072 GPUs: +1,263 ms / iter (+14.3% step time, −12.5% MFU)** vs the same
recipe without determinism. The iter-50 numbers are all **post-window** (nsys capture ends at iter 18),
so none carries active-profiling cost — the +14.3% is a clean det-vs-nondet signal. The small det+nsys
vs det-no-nsys gap at iter 50 (10,112 vs 9,987 ms, ~1.3%) is **run-to-run / allocation variation**, not
nsys overhead (these are separate jobs on different node allocations; ~1% step-time spread is normal).

**Methodology note**: nsys window = iters 15–17 (3 iters, `profile_step_start=15 / end=18`). All
leaderboard kernel totals in §2 are **3-iter window** totals; divide by 3 for per-iter values. Same
convention as the parent doc. **§2 is rank-0 only** — see the caveat at the head of §2.

---

## 1. Fairness & Shared Configuration

### What changes (the 6 determinism knobs) — identical set to the 24n run

| Knob | Det | Non-det |
|---|---|---|
| `model.deterministic_mode` | true | false |
| `model.cross_entropy_loss_fusion` | false | true |
| `NCCL_ALGO` | Ring | unset (default) |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | 0 | unset |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | unset |
| `MAMBA_DETERMINISTIC` | 1 | unset |

### Verified non-default config in the run (from `ConfigContainer.yaml`)

| Field | Value | Determinism relevance |
|---|---|---|
| `cross_entropy_loss_fusion` | **false** | fused-CE off → native unfused CE on both main and MTP heads |
| `cross_entropy_fusion_impl` | `te` | **stored-but-unused** (fusion is off) — a red herring, not active |
| `mtp_num_layers` | 2 | MTP head reuses `compute_language_model_loss` → same unfused CE path |
| `grad_reduce_in_fp32` | false | gradient reduce-scatter accumulates in **BF16** |
| `reduce_scatter_with_fp32_accumulation` | false | RS not promoted to FP32 |
| `average_in_collective` | false | SUM + explicit scaling (not in-collective AVG) |
| `use_megatron_fsdp` / `use_custom_fsdp` | false / false | dist-optimizer path, `data_parallel_sharding_strategy='optim_grads_params'` |
| `cuda_graph_impl` / `enable_cuda_graph` | none / false | **CUDA graphs OFF** → nsys `--cuda-graph-trace=node` is inert |
| `nccl_ub` | false | no symmetric-memory / NVLS-UB collective active |
| `bucket_size` | 512,000,000 | dense-DDP grad bucketing. **Not set by the recipe** — MCore computes `max(40M, 1M × DP_attn)` (`distributed_data_parallel.py:68`). DP_attn=512 → 512M |
| NCCL version | **2.29.3+cuda13.1** | ≥ UBR v2.27 → reduce precision can be NVLink/IB-domain-dependent |

---

## 2. Perf Cost Decomposition (det+nsys vs non-det+nsys, 3-iter window)

Reproduced from `nsys-det-3run-3072gpu/leaderboard.txt`. The story is **structurally identical to the
24-node analysis**: the determinism cost is in MCore's deterministic-substitute `aten::*` ops, not in
the model math.

> **Rank-0 caveat**: `leaderboard.txt` is built from **rank 0 only** (verified: `nsys-det.csv` is
> byte-identical to `nsys-det-rank0.csv`). On `TP=2 PP=3`, rank 0 is the **first pipeline stage** and
> **never runs the CE backward or the MTP-head backward** (those live on the last stage, rank 3071).
> So §2 *understates* the total det cost — the CE/MTP scatter-add substitutes show up only on
> last-stage ranks and reach rank 0 only via PP P2P. The per-rank CSVs in §3 confirm real cross-rank
> asymmetry (`GatherBackward0`: rank0 2,145 µs vs rank3071 1,064 µs). Treat §2 as the first-stage view,
> not the whole model.

### Forward — mcore module ranges (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `transformer_layer._forward_mlp.mlp` | 5,556.2 | 4,334.5 | **+1,221.7** | **+28.2** |
| `p2p_communication.send_forward_recv_backward` | 4,149.7 | 3,649.5 | +500.1 | +13.7 |
| `p2p_communication.send_forward` | 63.9 | 151.9 | −88.0 | −57.9 |
| `p2p_communication.recv_backward` | 109.2 | 193.5 | −84.3 | −43.6 |
| `mlp.forward.linear_fc1` | 763.8 | 686.3 | +77.5 | +11.3 |
| `mlp.forward.linear_fc2` | 832.2 | 755.5 | +76.8 | +10.2 |
| `self_attention` | 510.9 | 471.5 | +39.4 | +8.4 |
| `attention.forward.core_attention` | 147.7 | 131.8 | +16.0 | +12.1 |
| `mcore.fusions.fused_weighted_squared_relu.forward` | 233.9 | 234.8 | −0.9 | −0.4 |
| `Optimizer.step#FusedAdam.step` | 2.9 | 2.7 | +0.1 | +5.2 |

### Backward — autograd engine ranges (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `CheckpointFunctionBackward` | 15,647.8 | 13,455.5 | **+2,192.4** | +16.3 |
| `MambaSplitConv1dScanCombinedFnBackward` | 2,851.1 | 2,193.7 | +657.4 | +30.0 |
| `GatherBackward0` | 474.2 | 36.1 | +438.1 | **+1,212** |
| `_LinearBackward` | 1,068.7 | 957.0 | +111.7 | +11.7 |
| `_LayerNormLinearBackward` | 575.6 | 497.5 | +78.1 | +15.7 |
| `RouterGatingLinearFunctionBackward` | 326.2 | 264.7 | +61.5 | +23.3 |
| `IndexPutBackward0` | 55.4 | 4.4 | +51.0 | **+1,148** |
| `_moe_chunk_sortBackward` | 145.1 | 112.3 | +32.8 | +29.2 |
| `_AllToAllBackward` | 395.4 | 363.5 | +32.0 | +8.8 |
| `LayerNormFnBackward` | 175.6 | 145.2 | +30.5 | +21.0 |

### Op-level — aten / kernels (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Note |
|---|---|---|---|---|
| `aten::fill_` | 2,490.3 | 195.0 | **+2,295.3** | zero-init before deterministic scatter / index_put |
| `aten::index_put_` | 2,126.1 | 4.4 | **+2,121.6** | det substitute for scatter |
| `aten::_index_put_impl_` | 2,118.2 | 3.7 | +2,114.4 | inner kernel of the above |
| `aten::empty` | 2,750.6 | 681.9 | +2,068.8 | scratch buffers for substitute paths |
| `aten::is_nonzero` | 2,573.7 | 1,900.0 | +673.8 | **host sync** (new at scale — see note) |
| `aten::_local_scalar_dense` | 2,605.8 | 1,940.1 | +665.6 | **host sync** (`.item()` pull) |
| `aten::item` | 2,610.1 | 1,944.8 | +665.3 | **host sync** |
| `aten::empty_like` | 835.7 | 200.7 | +635.0 | substitute scratch |
| `aten::arange` | 538.4 | 2.4 | +535.9 | det-only — index generation |
| `aten::gather_backward` | 471.9 | 34.0 | +437.9 | det substitute via sort+segment-sum |
| `aten::scatter_add_` | 446.5 | 14.8 | +431.7 | det substitute |
| `aten::max` | 370.3 | 0 | +370.3 | **det-only** — advanced-indexing key build |
| `aten::min` | 320.5 | 0 | +320.5 | **det-only** — same |
| `aten::remainder` | 240.2 | 0 | +240.2 | **det-only** — hash-style index materialization |

**Reading** — matches the parent doc's §2 mechanism exactly: `aten::fill_` + `index_put_` + `empty` +
`arange` + `scatter_add_` + det-only `max`/`min`/`remainder` are MCore's
`if torch.are_deterministic_algorithms_enabled():` substitute branches. Compute (GEMM/attention) and
`mcore.fusions` are near-identical; `Optimizer.step` is even marginally slower only by noise.

> **New at 3072 that wasn't prominent at 24n**: the host-sync trio `aten::is_nonzero` /
> `aten::_local_scalar_dense` / `aten::item` each grow ~+665 ms (+34%) under det. These are CPU↔GPU
> scalar pulls (e.g. loss `.item()` for logging, nan/skip checks) **present in both runs** (~1,900 ms
> non-det → ~2,600 ms det). A host sync blocks until the preceding GPU work completes, so this is most
> likely **wait-inclusive skew** — det's slower surrounding kernels make the same sync wait longer —
> not extra det-specific work. Same wait-inclusive caveat the parent doc raised for NCCL deltas (§4.1).
> Cause not attributed to a specific call site; recorded as an observation.

---

## 3. Multi-Rank Capture (rank 0 / 1536 / 3071)

Per the original scaling request, the 3072-GPU profile captured **three ranks** — first (0), middle
(1536), and last (3071) — to expose pipeline-stage and NVLink-island asymmetry. CSVs:

| Rank | PP stage (TP=2,PP=3) | det CSV | nondet CSV |
|---|---|---|---|
| 0 | first | `nsys-det-rank0.csv` | `nsys-nondet-rank0.csv` |
| 1536 | middle | `nsys-det-rank1536.csv` | `nsys-nondet-rank1536.csv` |
| 3071 | last | `nsys-det-rank3071.csv` | `nsys-nondet-rank3071.csv` |

The last-stage rank (3071) runs the **CE backward and MTP-head backward** that rank 0 (§2) never sees.
Cross-rank decomposition of these CSVs is a §7 follow-up.

---

## 4. Compute / Comm Path

`leaderboard.txt` has NVTX ranges + aten ops, **not** nvjet/cuDNN/SSM kernel counts — so the claims below
are the parent doc's kernel-level conclusions **carried over and checked for consistency** with the 3072
NVTX ranges, not re-verified at this scale (the per-kernel cross-check is a §7 follow-up, recoverable from
the 933k-row CSVs).

- **GEMM / attention** — `_forward_mlp.mlp` (+28%), `linear_fc1/fc2` (+11%) are *consistent with* the
  cuBLAS `:4096:8` workspace penalty + downstream wait-skew measured at 24n; kernel-selection drift at
  3072 is unverified.
- **PP P2P** (`send_forward_recv_backward` +500 ms; `send_forward`/`recv_backward` *negative*) — wait-time
  skew, not algorithm cost: P2P SendRecv bypasses `NCCL_ALGO` selection entirely (code-grounded, parent
  §11.1). This one holds firmly at 3072.
- **Mamba** — `MambaSplitConv1dScanCombinedFnBackward` +657 ms (+30%) is the autograd range; at 24n the
  SSM scan kernel itself was −6% (the cost is buffer/reduce overhead).

---

## 5. Determinism Observations @ 3072 GPUs (observations-only)

This is the load-bearing new section. **The single det pair on record at 3072 GPUs is not bit-identical**
— the 24n "bit-exact, nsys-inert" result does not carry over to this pair. (Whether *determinism itself*
fails at 3072 is a separate, weaker claim the data can't yet settle — see §5.2.)

### 5.1 The datapoints (`bitwise_check.txt`)

det+nsys (3590404) vs det no-nsys (3590484), same recipe, same 768-node allocation window:

| iter | det+nsys lm loss | det no-nsys lm loss | match |
|---|---|---|---|
| 1 | 1.254725E+01 | 1.254725E+01 | ✓ |
| 2 | 1.254715E+01 | 1.254715E+01 | ✓ |
| 3 | 1.024112E+01 | 1.024113E+01 | ✗ (~1·10⁻⁶) |
| 5 | 9.311169E+00 | 9.311040E+00 | ✗ |
| 10 | 4.123835E+00 | 4.122399E+00 | ✗ |
| 20 | 4.817369E+00 | 8.658229E-01 | ✗ (gross) |
| 30 | 9.573289E-02 | 2.048100E-01 | ✗ |
| 40 | 2.204890E-02 | 7.341090E-02 | ✗ |
| 50 | 1.860187E-01 | 2.204380E-02 | ✗ |

**Shape**: identical at iters 1–2, first disagreement at **iter 3 at ~1·10⁻⁶ relative**, then chaotic
amplification (grossly different by iter 20). Grad-norm trajectories diverge in lockstep (det+nsys
iter-20 gnorm 36.6 vs det no-nsys 6.6).

### 5.2 What this does and does NOT establish

- **Does**: at 3072 GPUs, a det+nsys run and a det no-nsys run are not bit-identical. Contrast 24n
  (8 runs bit-exact incl. nsys-profiled) and 48n (5/7 runs bit-exact through iter 50).
- **Does NOT**: isolate the cause. The pair differs in exactly **one** variable — nsys ON/OFF — but
  there is **no matched no-nsys vs no-nsys control at 3072** (at 48n the matching cohort contained both
  nsys and no-nsys runs; here only one of each exists). So the iter-3 divergence is equally consistent with:
  1. nsys perturbs kernel-launch timing → reorders the **async DP-overlap** collectives
     (`overlap_grad_reduce=true`, `overlap_param_gather=true`) → different BF16 reduce-scatter
     accumulation order → different bits, **or**
  2. the 3072 config is simply **not bit-deterministic run-to-run** even without nsys (async overlap +
     BF16 reduce-scatter at 512-way DP).

  The iter-3 onset is **before** the nsys capture window (15–18), which means active data collection
  is not the cause; if nsys is responsible it is via the profiler being attached process-wide, not via
  the capture itself.

### 5.3 Mechanisms ruled IN / OUT by code + config (no speculation)

- **OUT — CUDA graphs**: off (`cuda_graph_impl: none`), so `--cuda-graph-trace=node` is inert.
- **OUT — fused CE / MTP atomic**: `cross_entropy_loss_fusion=false`; the `te` impl is stored-but-unused.
  MTP reuses the same unfused path.
- **OUT — fused MoE aux-loss atomic**: `moe_router_fusion=false` (the atomicAdd path is not invoked).
- **OUT — symmetric-memory / NVLS collective**: `nccl_ub=false`, FSDP off; all reduce ops are stock
  NCCL collectives → all honor `NCCL_ALGO=Ring`.
- **IN (candidate) — async DP-overlap reordering**: overlap is ON; this is the only path by which a
  timing perturbation can change reduction order. Requires the Level-1 test below to confirm.
- **IN (candidate) — NCCL 2.29 domain-dependent reduce precision**: ≥ UBR v2.27, so reduction can be
  high-precision on NVLink hops and not on IB hops — topology-dependent and not pinned by `NCCL_ALGO`.

### 5.4 Warning scan (both rank-0 logs)

No genuine determinism fault in either log: **no** `"does not have a deterministic implementation"`,
**no** `falling back`, **no** `CUBLAS_WORKSPACE` error. `deterministic_mode: True` engaged clean. The
only flagged non-default line, `cross_entropy_fusion_impl: 'te'`, is inert (fusion off). All other
warnings are packaging deprecations (pynvml, `TORCH_NCCL_AVOID_RECORD_STREAMS`, `torch_dtype`).

---

## 6. Scale Audit: 24n → 48n → 3072 GPUs

| Vector | 24n (96) | 48n (192) | 3072 |
|---|---|---|---|
| world_size | 96 | 192 | **3072** |
| DP_attn (= ws / TP·PP·CP) | 16 | 32 | **512** |
| DP_expert (= ws / EP·ETP) | 3 | 6 | **96** |
| microbatches / pipeline step | 8 | 4 | 8 |
| `bucket_size` (dense DDP) | 40 M | 40 M | **512 M** |
| TP/PP/EP/ETP/CP | 2/3/32/1/1 | same | same |
| GBS / MBS | 128 / 1 | 128 / 1 | **4096 / 1** (auto-scaled) |
| 6 det knobs + overlap ON | applied | applied | applied |
| Bit-exact across runs? | ✓ 8/8 | ◑ 5/7 (1 last-digit, 1 outlier) | ✗ (only det pair on record disagrees; no no-nsys control) |

**This is NOT a clean world_size-only scale-up.** Two *independent* things move from 48n → 3072:
1. **DP width** (and `bucket_size`, which is a *function* of it): DP_attn 32 → 512, DP_expert 6 → 96.
   `bucket_size` is not independent — MCore sets it `max(40M, 1M × DP_attn)`
   (`distributed_data_parallel.py:68`), so 40 M → 512 M is a deterministic consequence of the DP-width
   change, not a separate knob. (This is also why 24n and 48n share 40 M: both floor at the 40 M minimum,
   DP_attn 16 and 32 → 16 M / 32 M < 40 M.)
2. **GBS**: 128 → **4096** — a *genuinely independent* confound. The 3072 launcher applies
   `gbs_scaling_factor`; the 48n datapoints in the parent doc ran fixed GBS=128 (verified:
   `global_batch_size: 4096` in the 3072 `ConfigContainer.yaml`). Larger GBS = more tokens reduced per
   step + different microbatch/accumulation structure.

So while bit-exact reproducibility degrades across the series (24n ✓ → 48n mostly ✓ → 3072 ✗), the runs
differ in **DP width (⇒ bucket_size) AND global batch size** at once. We **cannot** isolate DP width as
the driver — that is exactly why §7's controls hold everything else fixed and vary one knob at a time.

---

## 7. Follow-up Tests (the discriminating experiment first)

Ordered; the first one settles §5.2's ambiguity and needs **zero code changes**.

1. **Two no-nsys det runs at 3072, same node set** — the missing control. Diff loss bits.
   - Match → 3072 is deterministic; nsys (via timing→overlap reordering) is the cause →
     profile and bit-check in separate jobs.
   - Diverge → nsys is innocent; async overlap + BF16 reduce-scatter is non-deterministic at this
     DP width → go to test 2/3.
2. **Overlap OFF** (`ddp.overlap_grad_reduce=false`, `ddp.overlap_param_gather=false`), no-nsys pair.
   Removes the async interleaving — highest-yield single knob if test 1 shows inherent non-determinism.
3. **FP32 reduce-scatter** (`ddp.reduce_scatter_with_fp32_accumulation=true`), no-nsys pair. Shrinks
   amplitude so any residual order diff stays sub-bit.
4. **Per-collective NCCL pin** (NCCL ≥ 2.24, confirmed 2.29 here):
   `NCCL_ALGO="allreduce:ring;reducescatter:ring"` + `NCCL_NVLS_ENABLE=0` + `NCCL_PROTO=Simple`.
5. **`NCCL_DEBUG=WARN`** (or `INFO` on rank 0 only) on the next det run to capture NCCL's actual
   algo/channel/UB selection instead of inferring Ring.

---

## 8. Artifacts

| | path |
|---|---|
| Processed (CSVs + leaderboard.txt + bitwise_check.txt + submit/wdj/jobid logs) | `nsys-det-3run-3072gpu/` (repo-relative) |
| Multi-rank det CSVs | `nsys-det-3run-3072gpu/nsys-det-rank{0,1536,3071}.csv` |
| Multi-rank nondet CSVs | `nsys-det-3run-3072gpu/nsys-nondet-rank{0,1536,3071}.csv` |
| det+nsys log (3590404) | `~/.nemo_run/experiments/nemotron-3-ultra-det-nsys15-18-1782432263/.../log-*_3590404_0.out` |
| nondet+nsys log (3590455) | `~/.nemo_run/experiments/nemotron-3-ultra-nondet-nsys15-18-1782432263/.../log-*_3590455_0.out` |
| det no-nsys log (3590484) | `~/.nemo_run/experiments/nemotron-3-ultra-det-bitwise-check-1782432263/.../log-*_3590484_0.out` |
| Authoritative run config | `.../configs/ConfigContainer.yaml` (per experiment) |
| Launch script | `scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh` |

> **Bottom line**: the determinism *cost* decomposition at 3072 GPUs reproduces the 24n analysis at the
> **aten-op level** (det penalty +14.3% step / −12.5% MFU, dominated by MCore's deterministic-substitute
> `aten::fill_` / `index_put_` / `scatter_add_` / `arange` / det-only `max`/`min`/`remainder`) — read as
> a **rank-0 / first-PP-stage** view (§2 caveat). The determinism *guarantee* does **not** reproduce:
> the single det pair on record diverges from iter 3. Whether that is nsys or inherent non-determinism
> at this scale is **not established** — and the scale series is confounded by GBS + bucket_size, not
> just DP width (§6). Test 1 in §7 (two no-nsys runs, everything else fixed) is the one experiment that
> settles the nsys question.
