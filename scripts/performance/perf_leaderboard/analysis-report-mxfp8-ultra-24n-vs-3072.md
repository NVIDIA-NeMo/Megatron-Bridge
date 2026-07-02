# Nemotron 3 Ultra — MXFP8 Perf & Determinism: 24 nodes vs 3072 GPUs

**Date**: 2026-07-02
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE, 108 layers, 512 experts top-22, 2 MTP heads)
**Precision**: MXFP8 (`-c fp8_mx` → `bf16_with_mxfp8_mixed`: fp8=`e4m3`, fp8_recipe=`mxfp8`,
`fp8_param_gather=True`, `reuse_grad_buf_for_mxfp8_param_ag=True`)
**Config**: TP=2, PP=3, EP=32, ETP=1, CP=1, MBS=1, SeqLen=8192; **24n** = 96 GPUs / GBS=128,
**3072** = 768 nodes / GBS=4096 (auto-scaled 128×32). Both arms: `moe_router_fusion=true`, alltoall
dispatcher, selective recompute **without `mlp`** (§1/§5).

> **MXFP8 counterpart** to the bf16 studies: [`analysis-report-det-vs-nondet-routerfusion-24n.md`](analysis-report-det-vs-nondet-routerfusion-24n.md)
> (24n) and [`analysis_report_det_vs_nondet_3072gpu.md`](analysis_report_det_vs_nondet_3072gpu.md) (3072).
> Same harness, same recipe knobs, precision swapped to MXFP8.
>
> Source profiles: rank-0 (24n) and ranks 0/1536/3071 (3072) nsys captures of iters 15–17. Launchers:
> `launch_nemotron_3_ultra_mxfp8_compare.sh` (24n, `NGPUS=96`) / `launch_nemotron_3_ultra_mxfp8_3072.sh`
> (3072). Each submits det+nsys, non-det+nsys, and two no-nsys det runs.

---

## Headline findings

1. **Determinism is scale-dependent: MXFP8 is bit-exact at 24n but NOT at 3072.** Two independent no-nsys
   det allocations match to the last printed digit at 24n; at 3072 they diverge **from iter 1**. bf16 did
   the same at 3072 — so it is a **scale effect, not an fp8 effect** (§5).
2. **Determinism (det-on vs det-off) step penalty grows with scale: ~+3% (24n) → ~+10% (3072)** (bf16 was
   +14.3% at 3072) (§3).
3. **nsys inflates MXFP8 step time for the WHOLE run (~15–20%), not just the profiled window** — unlike
   bf16, which recovers post-window. Use **no-nsys** runs for true throughput (§4).
4. **True (no-nsys) MXFP8 throughput ≈ 447 TFLOP/s/GPU (24n), ≈ 416–459 (3072)** — competitive with bf16
   (~432); MXFP8 is *not* slower once the nsys tax is removed (§4).

---

## Runs Compared

All jobs: `partition=batch`, account `nemotron_sw_pre`, container `nemo:26.04.01`, wandb project
`nvidia/mbridge-dev`. iter-50 = steady state.

### 24 nodes / 96 GPUs (two independent sweeps: v4, v5)

| arm | v4 job | v4 step / TFLOP | v5 job | v5 step / TFLOP |
|---|---|---|---|---|
| det + nsys (det ON) | 3785758 | 10,603 ms / 369.7 | 3802447 | 10,468 ms / 374.5 |
| non-det + nsys (det OFF) | 3785807 | 10,261 ms / 382.1 | 3802509 | 10,186 ms / 384.9 |
| det, **no-nsys** | 3785847 | **8,735 ms / 448.8** | 3802706 | 9,063 ms / 432.6 |
| det, **no-nsys #2** | 3785920 | **8,756 ms / 447.8** | 3802928 | 8,760 ms / 447.5 |

### 3072 GPUs / 768 nodes

| arm | job | step / TFLOP | wandb |
|---|---|---|---|
| det + nsys (det ON) | 3805747 | 10,532 ms / 372.2 | [xcawi9gg](https://wandb.ai/nvidia/mbridge-dev/runs/xcawi9gg) |
| non-det + nsys (det OFF) | 3806186 | 9,572 ms / 409.6 | [354uwraj](https://wandb.ai/nvidia/mbridge-dev/runs/354uwraj) |
| det, **no-nsys** | 3806542 | 9,413 ms / 416.5 | [6u6erncz](https://wandb.ai/nvidia/mbridge-dev/runs/6u6erncz) |
| det, **no-nsys #2** | 3807782 | 8,539 ms / 459.1 | [bb8tjk0z](https://wandb.ai/nvidia/mbridge-dev/runs/bb8tjk0z) |

(3072 no-nsys #2 first attempt 3806800 failed at distributed init — transient Gloo/TCP node fault, §9.)

---

## 1. Fairness & Shared Configuration

### What changes (the determinism knobs — the A/B axis)

| Knob | det ON | det OFF |
|---|---|---|
| `model.deterministic_mode` | true | false |
| `model.cross_entropy_loss_fusion` | false | true |
| `NCCL_ALGO` | Ring | unset |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | 0 | unset |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | unset |
| `MAMBA_DETERMINISTIC` | 1 | unset |
| `train.fill_uninitialized_memory` | false (opt-out) | n/a |

### Verified non-default config in the run (from the run's own config dump, job 3805747)

| Field | Value | Note |
|---|---|---|
| `mixed_precision.fp8_recipe` | `mxfp8` | + `fp8_param_gather=True`, `reuse_grad_buf_for_mxfp8_param_ag=True` |
| `cuda_graph_impl` | `none` | **no CUDA graphs** |
| `cuda_graph_scope` | `[]` | |
| `moe_a2a_overlap` | `False` | **no EP all-to-all overlap** |
| `delay_wgrad_compute` | `False` | no W/D-split overlap |
| `overlap_grad_reduce` / `overlap_param_gather` | `True` / `True` | DP overlap on (best recipe) |
| `moe_router_fusion` | `True` | held constant both arms |
| `recompute_granularity` | `selective` | |
| `recompute_modules` (effective) | `[moe, layernorm, core_attn, moe_act, shared_experts]` | **`mlp` dropped** (§5) |
| `moe_token_dispatcher_type` | `alltoall` | HybridEP off |

**Recipe parity:** knob-for-knob identical to the bf16 "best recipe"
(`launch_nemotron_3_ultra_deterministic.sh`) **except** (a) `-c fp8_mx`, and (b) the forced `mlp` drop from
`recompute_modules` (§5). Like the bf16 baseline it uses **no CUDA graphs and no EP-comm overlap** — so
these are *baseline-recipe* fp8 numbers, apples-to-apples with the bf16 study, **not** the tuned fp8
ceiling (see §7.1).

---

## 2. Perf Cost Decomposition (det+nsys vs non-det+nsys, 3-iter nsys window)

Rank-0 NVTX buckets, ranked by `|det − nondet|`. **Rank-0 caveat:** on `TP=2 PP=3`, rank 0 is the first
pipeline stage and never runs the CE / MTP-head backward, so this understates total det cost — first-stage
view only. Positive Δ = det slower.

### 3072 GPUs — top buckets

Forward (mcore module ranges):

| Range | det ms | nondet ms | Δ ms | Δ% |
|---|---|---|---|---|
| `_forward_mlp.mlp` | 5,515.1 | 5,447.6 | +67.5 | +1.2 |
| `p2p.send_forward` | 97.2 | 53.1 | +44.1 | +83 |
| `mlp.forward.linear_fc1` | 1,330.2 | 1,299.5 | +30.6 | +2.4 |
| `p2p.recv_backward` | 69.1 | 43.8 | +25.3 | +58 |
| `mlp.forward.linear_fc2` | 1,124.7 | 1,106.7 | +18.0 | +1.6 |
| `attention.self_attention` | 556.7 | 540.0 | +16.8 | +3.1 |

Backward (autograd engine ranges):

| Range | det ms | nondet ms | Δ ms | Δ% |
|---|---|---|---|---|
| `MambaSplitConv1dScanCombinedFnBackward` | 2,655.5 | 2,236.2 | **+419.3** | +18.8 |
| `_CheckpointFunctionBackward` | 16,448.1 | 16,745.7 | −297.7 | −1.8 |
| `_GroupedLinearBackward` | 2,145.6 | 2,320.1 | −174.5 | −7.5 |
| `_LayerNormLinearBackward` | 725.4 | 693.8 | +31.6 | +4.6 |
| `_AllToAllBackward` | 374.3 | 358.2 | +16.1 | +4.5 |

Op-level (aten / NCCL):

| Range | det ms | nondet ms | Δ ms | Δ% |
|---|---|---|---|---|
| `record_param_comms` | 1,619.1 | 1,397.0 | +222.1 | +15.9 |
| `aten::sum` | 725.7 | 528.4 | +197.3 | +37.3 |
| `_CheckpointFunction` | 7,749.7 | 7,595.5 | +154.2 | +2.0 |
| `aten::zeros` | 209.2 | 78.4 | +130.8 | +167 |
| `c10d::barrier` | 146.3 | 35.2 | +111.1 | +316 |
| `DaoAILab::_causal_conv1d_bwd_cpp` | 140.9 | 40.4 | +100.5 | +249 |

### 24 nodes (v4) — top buckets (for contrast)

| Range | Δ ms | Δ% |
|---|---|---|
| `_CheckpointFunctionBackward` | **+1,329.6** | +8.1 |
| `MambaSplitConv1dScanCombinedFnBackward` | +609.9 | +28.7 |
| `_forward_mlp.mlp` (fwd) | +88.2 | +1.6 |
| `mlp.forward.linear_fc1` | +103.6 | +8.1 |
| `aten::sum` | +236.2 | +47.5 |
| `aten::zeros` | +143.1 | +191 |
| `DaoAILab::_causal_conv1d_bwd_cpp` | +105.6 | +275 |

**Reading.** The det-cost signature is consistent at both scales in the *substitution* buckets: the
**Mamba selective-scan backward** (+18–29%) and its **`causal_conv1d_bwd`** (+249–275%) run the
deterministic scan path; the **non-fused cross-entropy** shows as `aten::sum` (+37–47%) and `aten::zeros`
/ `aten::fill_` (+167–191%, zero-init before deterministic scatter). Some large buckets **flip sign
between scales** (e.g. `_CheckpointFunctionBackward` is +1,330 ms at 24n but −298 ms at 3072, and
`_GroupedLinearBackward` flips negative at 3072): at 3072 the non-det arm was globally faster (9,572 vs
10,532), so most buckets are det-heavy, but a few invert under nsys-overhead + node-placement skew — do
not over-read individual flipped buckets. The *net* det penalty is §3; this is the where-det-differs map.

---

## 3. Determinism step penalty (det-on vs det-off), nsys-matched

Both arms are nsys-instrumented (matched overhead → the delta is meaningful; absolute step time carries
the nsys tax, §4):

| scale | det+nsys | nondet+nsys | Δ step | Δ throughput |
|---|---|---|---|---|
| 24n (v4) | 10,603 ms | 10,261 ms | **+342 ms (+3.3%)** | −12.4 (−3.2%) |
| 24n (v5) | 10,468 ms | 10,186 ms | **+282 ms (+2.8%)** | −10.4 (−2.7%) |
| **3072** | 10,532 ms | 9,572 ms | **+960 ms (+10.0%)** | −37.4 (−9.1%) |

**The determinism penalty grows with scale (~+3% at 24n → ~+10% at 3072)** — consistent with bf16
(+14.3% at 3072). Larger EP all-to-all + P2P over more nodes makes the deterministic path
(Ring all-reduce, non-fused CE, deterministic Mamba scan) relatively costlier at scale.

---

## 4. nsys overhead is whole-run for MXFP8 (unlike bf16)

Per-iteration det+nsys vs det-no-nsys ratio (nsys capture window = iters 15–18):

| phase | 24n v4/v5 ratio | note |
|---|---|---|
| pre-window (3–8) | ~1.08–1.12× | already elevated |
| window (15–18) | 1.5–1.67× | active capture |
| flush (19) | 3.6–4.0× | nsys serializing captured data |
| **post-window (45–50)** | **~1.16–1.20×** | **does NOT recover** |

Replicated across **v4 and v5**. Contrast bf16: post-window recovers to the no-nsys baseline
(iter-50 ≈ pre-window), so "iter-50 is a clean datapoint" holds for bf16 but **not** MXFP8. Cause:
`nsys profile` keeps CUPTI/CUDA-API tracing attached the whole run (the window only gates what is
*saved*); MXFP8 issues many more small ops/step (per-tensor cast-to-fp8, amax, scale updates), so the
per-kernel tracing overhead compounds to ~15–20% and persists.

**Consequences:**
- **True MXFP8 throughput** = the **no-nsys** runs: 24n ≈ 447 TFLOP/s (faster than bf16 ~432), 3072 ≈
  416–459. MXFP8 is not slower once the nsys tax is removed.
- Never compare a bf16 nsys iter-50 against an fp8 nsys iter-50 — the tax is asymmetric.
- Model-TFLOP/s is dtype-blind (fixed algorithmic FLOPs ÷ step time), so it gives fp8 no credit for the
  2× fp8 peak; judge fp8 by wall-clock step time, not Model-TFLOP/s.

---

## 5. Determinism observations (bit-exact @ 24n, diverges @ 3072)

Cleanest test = two **no-nsys** det allocations vs each other (no nsys confound), lm loss:

### 24n (v4: 3785847 vs 3785920) — BIT-EXACT ✓
All sampled iters (1/2/3/5/10/20/30/40/50) match to the last printed digit (iter 50 = `4.168909E+00`
both). det+nsys vs det+no-nsys also matched exactly — nsys does not perturb the math at 24n.

### 3072 (3806542 vs 3807782) — DIVERGES ✗
| iter | no-nsys | no-nsys #2 |
|---|---|---|
| 1 | 1.254497E+01 | **1.254498E+01** ← differ at iter 1 |
| 10 | 6.116176E+00 | 6.116106E+00 |
| 50 | 3.675130E-02 | **3.351456E-02** (~9% apart) |

### What this does / does NOT establish
- **Does:** MXFP8 `deterministic_mode` is **not** bit-reproducible across allocations at 3072; a 1-ULP
  iter-1 difference amplifies chaotically. det+nsys vs det+no-nsys shows the same (matches to iter 5,
  diverges from iter 10).
- **Does NOT:** say fp8 is the cause. bf16 diverged identically at 3072 (companion §5). It is a **scale**
  effect (larger reductions / more ranks perturb the low-order bits), ruled in by the bf16 parallel.
- **Ruled OUT:** the recipe (24n with the *same* recipe is bit-exact), and nsys (the no-nsys×2 pair
  diverges on its own).

---

## 6. Scale audit: 24n → 3072

| | 24n (96 GPU) | 3072 (768 nodes) |
|---|---|---|
| Determinism (no-nsys ×2) | **bit-exact ✓** | **diverges from iter 1 ✗** |
| det-on-vs-off penalty (nsys) | ~+3% | ~+10% |
| True throughput (no-nsys) | ~447 TFLOP/s | ~416–459 TFLOP/s |
| nsys overhead (whole-run) | ~15–20% | ~15–20% |
| bf16 counterpart determinism | bit-exact | diverges (companion §5) |

Both precisions lose bit-exact determinism between 24n and 3072; MXFP8 offers no determinism advantage at
scale. The bf16 companion also observed a 48n partial-divergence midpoint — an MXFP8 48n point would fill
the curve (§7.3).

---

## 7. Improvement opportunities / follow-up

### 7.1 Tuned fp8 perf recipe (biggest lever, not yet applied)
This run uses `cuda_graph_impl=none` + no EP overlap (baseline recipe). A *tuned* MXFP8 recipe should add
**TE-scoped CUDA graphs** (`mamba, attn, moe_router, moe_preprocess` — the biggest GB200 host-overhead
lever) and **EP-comm overlap** (`moe_a2a_overlap` / `--overlap-moe-expert-parallel-comm`), mirroring the
Nemotron Super NVFP4 config and dsv3's GB200 MXFP8 (1048 TFLOP/s). Expected to close much of the det
penalty (which is partly host/launch overhead) and lift absolute throughput.

### 7.2 Quantify + remove the nsys tax
The ~15–20% whole-run nsys overhead (§4) is fp8-specific. For clean fp8 perf numbers, standardize on
**no-nsys** runs; only use nsys for the relative det-vs-nondet decomposition. Worth confirming whether a
lighter `--nsys_trace` (drop `cuda-sw`, keep `nvtx`) reduces the persistent tax.

### 7.3 Discriminating experiments
- **MXFP8 at 48n** — fill the determinism scale curve between the bit-exact 24n and divergent 3072
  (bf16 saw partial divergence at 48n).
- **Matched no-nsys fp8-vs-bf16 at 3072** — the fair throughput comparison (both no-nsys, same GPU count),
  to state fp8's wall-clock win/loss at scale without the nsys or Model-TFLOP confounds.

### 7.4 `mlp` recompute alternative
`mlp` was dropped from `recompute_modules` because fp8 + dense-MLP recompute crashes on the `padding_mask`
kwarg (§5). If memory later gets tight at larger batch/seq, re-add via full-granularity recompute or
fine-grained offloading rather than the `mlp` module under fp8.

---

## 8. Methodology
- nsys window = iters 15–18 (`profile_step_start=15 / end=18`); §2 totals are the 3-iter window, rank-0.
- Step time / throughput = iter-50 steady state. **For MXFP8, iter-50 under nsys is NOT clean** (§4) —
  no-nsys runs are used for true throughput; nsys runs only for the matched det-vs-nondet delta and §2.
- Determinism = exact string match of printed lm loss (6 sig figs) across allocations.
- Leaderboard tooling: `extract_nsys_csv.py` + `print_nsys_leaderboard.py` (rank-0 NVTX, top-20 by |Δ|).

## 9. Scope & limitations
- **Rank-0, first-PP-stage** visibility only (§2 understates CE/MTP-head det cost).
- **No-nsys throughput spread** at 3072 (416.5 vs 459.1) is partly **node-placement variance** across the
  two allocations, not a determinism signal — treat as a range.
- **Single sweep at 3072**; 24n has two (v4/v5). Mock data + force-balanced routing.
- Baseline recipe (no CUDA graphs / EP overlap) — not the tuned fp8 ceiling (§7.1).
- 3072 no-nsys #2 needed a resubmit after a transient init failure (§ below).

### 3072 no-nsys #2 init failure (transient)
First attempt (3806800) died at 3:07, pre-iter-1: `ProcessGroupGloo(...)` at
`gloo/transport/tcp/device.cc:99 rv=-2` (Gloo TCP device / name-resolution failure) while
`initialize_model_parallel` built the CP Gloo group, across many ranks → node **nvl72002-T01** `TASK
FAILURE` → step cancelled → TCPStore-heartbeat / NCCL-shm cascade. Not a config bug (3 sibling arms fine;
resubmit 3807782 on different nodes ran clean). Pattern: transient node/network flake during TCP
rendezvous at 768-node scale → retry-on-different-nodes.

## 10. Artifacts
- Launchers: `scripts/performance/launch_nemotron_3_ultra_mxfp8_compare.sh` (24n),
  `launch_nemotron_3_ultra_mxfp8_3072.sh` (3072).
- OUT_DIRs: `mxfp8-compare-24node-v4/`, `-v5/`, `mxfp8-compare-3072gpu/` (each: `leaderboard.txt`,
  `bitwise_check.txt`, `jobid-*.txt`, `wdj-*.txt`, `submit-*.log`).
- Jobs — 24n v4: 3785758/3785807/3785847/3785920; v5: 3802447/3802509/3802706/3802928.
  3072: 3805747/3806186/3806542/3807782 (3806800 failed, replaced by 3807782).
- wandb project `nvidia/mbridge-dev` (3072 run IDs in the Runs Compared table).

---

## Bottom line
MXFP8 Ultra trains cleanly at both scales and is throughput-competitive with bf16 (no-nsys ~447 TFLOP/s at
24n), but **bit-exact determinism only holds at 24n** — at 3072 it diverges from the first step, exactly
as bf16 does, so MXFP8 gives no determinism advantage at scale. When benchmarking MXFP8: separate the nsys
tax (use no-nsys runs), compare wall-clock (not Model-TFLOP/s), and note these are baseline-recipe numbers
— the tuned fp8 recipe (CUDA graphs + EP overlap, §7.1) is the next step.
