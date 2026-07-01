# Nemotron 3 Ultra — Det vs Non-Det Perf Analysis (router-fusion + fill-uninit, 24 nodes)

**Date**: 2026-06-30
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE, 108 layers, 512 experts top-22, 2 MTP heads)
**Config**: TP=2, PP=3, EP=32, ETP=1, MBS=1, GBS=128, SeqLen=8192, 24 nodes × GB200 (96 GPUs), selective recompute

> **What's new vs the 2026-06-12 report** (`analysis_report_det_vs_nondet.md`): this run adds two
> settings and one bug fix, then re-checks determinism *and* perf:
> 1. `model.moe_router_fusion=true` (MoE router kernel fusion)
> 2. `torch.utils.deterministic.fill_uninitialized_memory = False` (drops the NaN-fill that
>    `use_deterministic_algorithms(True)` turns on — recovers throughput, does not change results)
> 3. Fixed a scoping bug in `_validate_and_apply_deterministic_mode` (an in-function
>    `import torch.utils.deterministic` had rebound `torch` as a local → `UnboundLocalError` that
>    crashed every `deterministic_mode=true` run before iter 1; now imported `as _torch_det`).
>
> Source: rank-0 nsys captures (iters 15–17) of jobs **3706179** (det) and **3706214** (non-det);
> determinism verified by two independent no-nsys det allocations **3706236** / **3706255**.

---

## Runs Compared

| | det+nsys (3706179) | nondet+nsys (3706214) | det no-nsys #1 (3706236) | det no-nsys #2 (3706255) |
|---|---|---|---|---|
| **Role** | **perf comparison** | **perf comparison** | **determinism check** | **determinism check** |
| Slurm runtime | 18:55 / 50 iters | 18:47 / 50 iters | 17:59 / 50 iters | 17:59 / 50 iters |
| Bit-wise reproducible | ✓ | n/a | ✓ | ✓ |
| wandb | [gbzxks86](https://wandb.ai/nvidia/mbridge-dev/runs/gbzxks86) | [e2i1o8wg](https://wandb.ai/nvidia/mbridge-dev/runs/e2i1o8wg) | [6d54qxmy](https://wandb.ai/nvidia/mbridge-dev/runs/6d54qxmy) | [j95nentq](https://wandb.ai/nvidia/mbridge-dev/runs/j95nentq) |

> **Methodology.** Perf is compared **nsys-vs-nsys only** (det+nsys vs nondet+nsys — both instrumented,
> apples-to-apples; windowed means in §3). The **no-nsys** runs are used **only** for the determinism
> cross-check (§1); their step times are *not* put against nsys runs, since nsys adds ~150 ms/iter which
> would confound the perf comparison. With that separation, det and non-det perf are expected to be close.

---

## 1. Determinism verdict — BIT-EXACT ✓

Two cross-checks (lm loss at iters 1, 2, 3, 5, 10, 20, 30, 40, 50), all match exactly:

| Diff | Result |
|---|---|
| det+nsys vs det+no-nsys | **all iters match** — nsys instrumentation does not perturb results |
| **det no-nsys #1 vs #2** (two independent allocations) | **all iters match** — reproducible across allocations |

**Extended to 4 independent allocations** (jobs 3706236, 3706255, 3707706, 3707730): all four are
bit-identical at every sampled iter (e.g. iter 1 = `1.254623E+01`, iter 50 = `1.342692E+00`).
Determinism is confirmed across 4 separate node placements, not just a paired check.

Sampled values (identical across all three det runs):

| iter | lm loss |
|---|---|
| 1 | 1.254623E+01 |
| 10 | 4.177998E+00 |
| 20 | 5.008514E-01 |
| 30 | 2.259073E-01 |
| 40 | 5.154770E-02 |
| 50 | 1.342692E+00 |

**Conclusion: `moe_router_fusion=true` + `fill_uninitialized_memory=False` preserve bit-exactness.**
The earlier concern that router fusion (a kernel fusion, like the intentionally-disabled
`cross_entropy_loss_fusion`) might break determinism is **not** borne out for this recipe at 24 nodes.

---

## 2. Configuration

### Identical between det and non-det
Hardware (24× GB200, same pool), container `nemo:26.04.01`, parallelism (TP2/PP3/EP32/ETP1,
GBS128, MBS1, seq8192), MoE dispatcher `alltoall` (HybridEP intentionally OFF), TE FusedAttention,
`overlap_grad_reduce=true` / `overlap_param_gather=true`, selective recompute
(`moe, layernorm, core_attn, moe_act, mlp, shared_experts`), **`moe_router_fusion=true`**.

### Determinism knobs (det only)
`deterministic_mode=true`, `cross_entropy_loss_fusion=false`, `NCCL_ALGO=Ring`,
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `MAMBA_DETERMINISTIC=1`,
and (new) `torch.utils.deterministic.fill_uninitialized_memory=False`.

---

## 3. Perf — det vs non-det (nsys-only), within noise

**Method:** compare **nsys runs only** — det+nsys vs nondet+nsys, both instrumented, using the mean
step time over iters 20–49 (robust to single-iter jitter). No-nsys runs are deliberately excluded from
this comparison — they are the determinism set (§1), and nsys adds ~150 ms/iter, so mixing them would
confound perf. (For reference only, the no-nsys det step time is very stable, ~8891 ms mean across 4
allocations — but that number is **not** compared against the instrumented non-det runs.)

Across 1 det and **3 non-det** nsys allocations:

| arm | windowed mean step time (iters 20–49) |
|---|---|
| det (job 3706179) | 9158.2 ms |
| non-det #1 (3706214) | 9101.4 ms |
| non-det #2 recheck-a (3712367) | 9196.5 ms |
| non-det #3 recheck-b (3712403) | 9000.4 ms |

The three non-det allocations span **9000–9197 ms (~2.2%)**, and **det (9158) sits *inside* that
range.** So the det-vs-non-det difference is **within cross-allocation noise** — no determinism cost
(or savings) is resolvable here. Two earlier readings were undersampling artifacts and are retracted:
- "det ~2% *faster*" came from **iter 50 only**, where the v2 non-det spiked to 9251 ms (its
  recheck-a/b iter-50 were 9090 / 8981 — no spike, confirming the spike was a fluke).
- "det ~0.6% *slower*" came from a single det-vs-single-nondet windowed comparison, now swamped by the
  ~2% spread seen across non-det allocations alone.

**Honest conclusion:** at this sample size the determinism step-time cost is **indistinguishable from
noise (≲1% vs a ~2% cross-allocation spread)**. `fill_uninitialized_memory=False` removes a known
det-only overhead (its purpose) and plausibly shrank the +17% penalty from the 2026-06-12 report, but
**resolving a real cost would need many matched runs on both arms**. Treat §4 as the qualitative
"where det differs" map, not a net-cost number.

---

## 4. Where det and non-det differ (NVTX module-range leaderboard, iters 15–17 window)

Positive delta = det slower. These are summed over the 3-iter nsys window on rank 0.

### Forward — top |Δ|
| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `p2p_communication.recv_backward` | 2007.6 | 1616.8 | **+390.9** | +24.2 |
| `p2p_communication.send_forward_recv_backward` | 6224.5 | 5921.3 | +303.2 | +5.1 |
| `transformer_layer._forward_mlp.mlp` | 3910.6 | 3776.2 | +134.4 | +3.6 |
| `mlp.forward.activation` | 471.2 | 490.5 | −19.3 | −3.9 |
| `mlp.forward.linear_fc2` | 761.4 | 771.5 | −10.1 | −1.3 |
| `attention.forward.self_attention` | 476.9 | 484.2 | −7.3 | −1.5 |

### Backward — top |Δ|
| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `MambaSplitConv1dScanCombinedFnBackward` | 2552.4 | 2201.8 | **+350.6** | +15.9 |
| `CheckpointFunctionBackward` | 12709.2 | 12867.4 | −158.1 | −1.2 |
| `RouterGatingLinearFunctionBackward` | 277.3 | 263.1 | +14.2 | +5.4 |
| `_GroupedLinearBackward` | 1186.8 | 1183.6 | +3.2 | +0.3 |
| `EmbeddingBackward0` | — | 14.3 | −14.3 | — |

### Op-level — top |Δ|
| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `aten::zeros` | 203.6 | 80.6 | +123.0 | +152 |
| `aten::sum` | 681.6 | 560.3 | +121.3 | +21.6 |
| `DaoAILab::_causal_conv1d_bwd_cpp` | 132.6 | 44.4 | +88.2 | +199 |
| `aten::fill_` | 206.4 | 142.9 | +63.5 | +44.4 |
| `aten::item` / `_local_scalar_dense` | ~128 / 123 | ~81 / 77 | +47 / +47 | +58 / +61 |

**Reading:** det's overhead is concentrated in **P2P communication** (`NCCL_ALGO=Ring` forces even
small P2P onto Ring) and the **Mamba selective-scan backward** + its `causal_conv1d_bwd`
(deterministic scan path). The extra `aten::zeros`/`fill_`/`item` reflect the non-fused
cross-entropy path (`cross_entropy_loss_fusion=false`). Some compute paths are equal-or-slightly
faster under det, but this table is a *qualitative* "where det differs" map — it does **not** net out
to a step-time number (see §3: the det-vs-non-det total is not quantifiable from one non-det sample).

---

## 5. Scope & limitations

- **Lighter than the 2026-06-12 report.** This is an NVTX module-range + iter-level analysis, not the
  full stream-7 / per-NCCL-kernel SQL decomposition. For that depth, the `.sqlite` artifacts are
  under each run's experiment dir; reuse the queries in `analysis_report_det_vs_nondet.md §8`.
- **Single-rank, first-PP-stage** visibility; **mock data + force-balanced routing**; **iters 15–17**
  are still warmup for the leaderboard window, while the headline uses the iter-50 log line.
- **1 det vs 3 non-det windowed samples** show det sitting *inside* the non-det ~2% cross-allocation
  spread, so the det-vs-non-det *cost* is **not resolvable** here (see §3). The solid results are
  (a) bit-exact determinism across 4 independent allocations, and (b) tightly reproducible det step
  time (~8891 ms no-nsys). Fully quantifying the cost needs many matched runs on both arms.

---

## 6. Artifacts

- Reports: `nsys-compare-24node-routerfusion-v2/{leaderboard.txt, bitwise_check.txt}`
- Job IDs: det=3706179, nondet=3706214, det-no-nsys=3706236 / 3706255; extra det (4-way check) = 3707706 / 3707730;
  nondet+nsys recheck = 3712367 / 3712403
- Code under test: `model.moe_router_fusion=true` (launcher) + `fill_uninitialized_memory=False`
  (`src/megatron/bridge/training/config.py`, aliased-import fix)
