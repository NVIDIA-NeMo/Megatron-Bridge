# Nemotron 3 Ultra — MXFP8 Perf & Determinism: 24 nodes vs 3072 GPUs

**Date**: 2026-07-02
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE, 108 layers, 512 experts top-22, 2 MTP heads)
**Precision**: MXFP8 (`-c fp8_mx` → `bf16_with_mxfp8_mixed`: fp8=`e4m3`, fp8_recipe=`mxfp8`,
`fp8_param_gather=True`, `reuse_grad_buf_for_mxfp8_param_ag=True`)
**Config**: TP=2, PP=3, EP=32, ETP=1, CP=1, MBS=1, SeqLen=8192; **24n** = 96 GPUs / GBS=128,
**3072** = 768 nodes / GBS=4096 (auto-scaled 128×32). Both arms: `moe_router_fusion=true`, alltoall
dispatcher, selective recompute **without `mlp`** (see §5).

> Companion to the bf16 studies: [`analysis-report-det-vs-nondet-routerfusion-24n.md`](analysis-report-det-vs-nondet-routerfusion-24n.md)
> and [`analysis_report_det_vs_nondet_3072gpu.md`](analysis_report_det_vs_nondet_3072gpu.md). This is the
> **MXFP8** counterpart, comparing 24n vs 3072.
>
> Launcher: `scripts/performance/launch_nemotron_3_ultra_mxfp8_compare.sh` (24n, `NGPUS=96`) and
> `launch_nemotron_3_ultra_mxfp8_3072.sh` (3072). Each submits 4 arms: det+nsys, non-det+nsys, and two
> no-nsys det runs (reproducibility check). Reservation `sla_res_nemotron_sw_pre` / qos `hero-res` for 3072.

---

## Headline findings

1. **Determinism is scale-dependent: MXFP8 is bit-exact at 24n but NOT at 3072.** Two independent no-nsys
   det allocations match to the last printed digit at 24n, but at 3072 they diverge **from iter 1**. This
   mirrors the bf16 result (bf16 also lost determinism at 3072) — so it is a **scale effect, not an fp8
   effect**.
2. **The determinism (det-on vs det-off) step penalty grows with scale: ~+3% at 24n → ~+10% at 3072**
   (bf16 was +14.3% at 3072).
3. **nsys instrumentation inflates MXFP8 step time for the WHOLE run (~15–20%), not just the profiled
   window** — unlike bf16, which recovers post-window. Use **no-nsys** runs for true throughput.
4. **True (no-nsys) MXFP8 throughput ≈ 447 TFLOP/s/GPU at 24n and ≈ 416–459 at 3072** — competitive with
   bf16 (~432), i.e. MXFP8 is *not* slower once the nsys tax is removed.

---

## 1. Runs compared

All jobs on `partition=batch`, account `nemotron_sw_pre`, container `nemo:26.04.01`, wandb project
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

(3072 no-nsys #2 first attempt job 3806800 failed at distributed init — transient Gloo/TCP node fault on
nvl72002-T01, before iter 1; resubmitted as 3807782 on different nodes. See §6.)

---

## 2. Determinism — bit-exact at 24n, NOT at 3072

The cleanest test is **two no-nsys det allocations vs each other** (no nsys confound). lm loss:

### 24n (v4: 3785847 vs 3785920) — BIT-EXACT ✓
Every sampled iter (1/2/3/5/10/20/30/40/50) matches to the last printed digit (iter 50 = `4.168909E+00`
in both). det+nsys vs det+no-nsys also matched exactly — nsys does not perturb the math at 24n.

### 3072 (3806542 vs 3807782) — DIVERGES ✗
| iter | no-nsys (3806542) | no-nsys #2 (3807782) |
|---|---|---|
| 1 | 1.254497E+01 | **1.254498E+01** ← differ at iter 1 |
| 5 | 1.170400E+01 | 1.170399E+01 |
| 10 | 6.116176E+00 | 6.116106E+00 |
| 20 | 9.673671E-02 | 9.651586E-02 |
| 50 | 3.675130E-02 | **3.351456E-02** (~9% apart) |

A 1-ULP difference at **iter 1** amplifies chaotically to ~9% by iter 50. det+nsys vs det+no-nsys shows
the same (matches to iter 5, diverges from iter 10). **MXFP8 deterministic_mode does not give bit-exact
reproducibility across allocations at 3072** — the same behavior bf16 showed at 3072 (companion §5). It
is a scale effect, not fp8-specific.

---

## 3. Perf — determinism (det-on vs det-off) penalty, nsys-matched

Comparing the two **nsys** arms (matched instrumentation, so the delta is meaningful even though the
absolute step time carries nsys overhead — §4):

| scale | det+nsys | nondet+nsys | Δ step | Δ throughput |
|---|---|---|---|---|
| 24n (v4) | 10,603 ms | 10,261 ms | **+342 ms (+3.3%)** | −12.4 (−3.2%) |
| 24n (v5) | 10,468 ms | 10,186 ms | **+282 ms (+2.8%)** | −10.4 (−2.7%) |
| **3072** | 10,532 ms | 9,572 ms | **+960 ms (+10.0%)** | −37.4 (−9.1%) |

**The determinism step penalty grows with scale (~+3% at 24n → ~+10% at 3072)** — consistent with bf16
(which was +14.3% at 3072). Larger EP all-to-all + P2P over more nodes makes the deterministic
(NCCL_ALGO=Ring, non-fused CE, deterministic Mamba scan) path relatively costlier at scale.

---

## 4. nsys overhead is whole-run for MXFP8 (unlike bf16)

Per-iteration step time, det+nsys vs det-no-nsys (nsys capture window = iters 15–18):

| phase | 24n v4/v5 ratio (nsys/no-nsys) | note |
|---|---|---|
| pre-window (3–8) | ~1.08–1.12× | already elevated |
| window (15–18) | 1.5–1.67× | active capture |
| flush (19) | 3.6–4.0× | nsys serializing captured data |
| **post-window (45–50)** | **~1.16–1.20×** | **does NOT recover** |

Replicated across **v4 and v5**. Contrast bf16: post-window fully recovers to the no-nsys baseline
(iter-50 ≈ pre-window), so bf16's "iter-50 is a clean datapoint" assumption holds — but **for MXFP8 it
does not**. Cause: `nsys profile` keeps CUPTI/CUDA-API tracing attached the whole run (the window only
gates what is *saved*); MXFP8 issues many more small ops/step (per-tensor cast-to-fp8, amax, scale
updates), so the per-kernel tracing overhead compounds to ~15–20% and persists.

**Consequences:**
- For **true MXFP8 throughput**, use the **no-nsys** runs (24n ≈ 447 TFLOP/s; 3072 ≈ 416–459). At 24n
  that (~447) is *faster* than bf16 (~432) — MXFP8 is not slower once the nsys tax is removed.
- Never compare a bf16 nsys iter-50 against an fp8 nsys iter-50 — the nsys tax is asymmetric (small in
  bf16, ~20% in fp8) and would unfairly penalize fp8.
- Model-TFLOP/s is dtype-blind (fixed algorithmic FLOPs ÷ step time), so it gives fp8 no credit for the
  2× fp8 peak; judge fp8 by wall-clock step time / tokens-per-sec, not Model-TFLOP/s.

---

## 5. Config note — `mlp` dropped from recompute for MXFP8

`recompute_modules` = `[moe, layernorm, core_attn, moe_act, shared_experts]` (the base config's `mlp`
is removed). Under fp8/fp4, the dense-layer MLP recompute path calls
`te_checkpoint(self.mlp, …, padding_mask=padding_mask)`, and this container's `torch.utils.checkpoint`
rejects the `padding_mask` kwarg → `ValueError: Unexpected keyword arguments: padding_mask` at the iter-1
forward (bf16 is unaffected — it binds the kwarg via `functools.partial`). Dropping `mlp` skips that path;
MoE experts still recompute via `moe`/`moe_act`. The MoE router-padding flag is **not** the cause (it is
at its default `False`). Observed at 24n before the fix: jobs 3774666/3774680/3780523/3780565 all failed
before iter 1.

---

## 6. 3072 no-nsys #2 init failure (transient, not the recipe)

First attempt (job 3806800) died at 3:07, before iter 1: `ProcessGroupGloo(...)` at
`gloo/transport/tcp/device.cc:99 rv=-2` (Gloo TCP device / name-resolution failure) while
`initialize_model_parallel` created the CP Gloo group, across many ranks. Node **nvl72002-T01** hit
`TASK FAILURE` → Slurm cancelled the step → TCPStore-heartbeat + NCCL-shm-cleanup cascade. Not a
config bug: the 3 sibling arms (identical config) initialized fine, and the resubmit (3807782) on
different nodes ran clean. Pattern: **transient node/network flake during TCP rendezvous at 768-node
scale** — retry-on-different-nodes is the fix.

---

## 7. Bottom line

| | 24 nodes (96 GPU) | 3072 GPU (768 nodes) |
|---|---|---|
| Determinism (no-nsys×2) | **bit-exact ✓** | **diverges from iter 1 ✗** |
| det-on-vs-off step penalty (nsys) | ~+3% | ~+10% |
| True throughput (no-nsys) | ~447 TFLOP/s/GPU | ~416–459 TFLOP/s/GPU |
| nsys overhead (whole-run) | ~15–20% (persistent) | ~15–20% (persistent) |

MXFP8 Ultra trains cleanly at both scales and is throughput-competitive with bf16, but **bit-exact
determinism only holds at small scale** — at 3072 it diverges from the first step (as bf16 does), so
MXFP8 offers no determinism advantage at scale. When benchmarking MXFP8, always separate the nsys tax
(use no-nsys runs) and compare wall-clock, not Model-TFLOP/s.
