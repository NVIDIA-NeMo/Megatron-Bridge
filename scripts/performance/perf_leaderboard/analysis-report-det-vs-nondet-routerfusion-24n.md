# Nemotron 3 Ultra — Deterministic vs Non-Deterministic Perf Analysis @ 24 nodes (router-fusion + fill-uninit)

**Date**: 2026-07-01
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE, 108 layers, 512 experts top-22, 2 MTP heads)
**Config**: TP=2, PP=3, EP=32, ETP=1, CP=1, MBS=1, GBS=128, SeqLen=8192, 24 nodes × GB200 (96 GPUs),
selective recompute. **New vs the baseline report**: `model.moe_router_fusion=true` (both arms) and the
`fill_uninitialized_memory` option (`false` on the det arm; a scoping-bug fix in
`_validate_and_apply_deterministic_mode` was also required — see §5).

> Companion to [`analysis_report_det_vs_nondet.md`](analysis_report_det_vs_nondet.md) (24n baseline)
> and [`analysis_report_det_vs_nondet_3072gpu.md`](analysis_report_det_vs_nondet_3072gpu.md).
>
> Source profiles: rank-0 nsys captures of iters 15–17 of jobs **3706179** (det+nsys) and **3706214**
> (non-det+nsys). Determinism verified by no-nsys det jobs **3706236 / 3706255 / 3707706 / 3707730**
> (4 independent allocations).

---

## Runs Compared

| | Det + nsys (3706179) | Non-det + nsys (3706214) | Det no-nsys (3706236) |
|---|---|---|---|
| wandb (`mbridge-dev`) | [gbzxks86](https://wandb.ai/nvidia/mbridge-dev/runs/gbzxks86) | [e2i1o8wg](https://wandb.ai/nvidia/mbridge-dev/runs/e2i1o8wg) | [6d54qxmy](https://wandb.ai/nvidia/mbridge-dev/runs/6d54qxmy) |
| Steady-state step time (iter 50) | **9,041 ms** | **9,252 ms** | 8,891 ms |
| Throughput (TFLOP/s/GPU, iter 50) | **433.6** | **423.7** | 440.9 |
| **Step time Δ (median, 3 runs/arm)** | — | **det −25 ms (−0.3%)** — within noise (§3.1) | — |
| Bit-wise reproducible | ✓ (matches no-nsys, §4) | n/a | ✓ |

**Headline: at 24 nodes there is no resolvable determinism step-time penalty — the det/non-det
difference is within run-to-run noise.** With **3 samples per arm**, the median iter-50 step time is
det+nsys **9,065 ms** vs non-det+nsys **9,090 ms** (**Δ −0.3%**, det marginally faster — i.e.
indistinguishable). Each arm's own spread (det 9,041–9,140 ms; non-det 8,981–9,252 ms) is *larger* than
the gap. This is the expected outcome after `fill_uninitialized_memory=false` removes the det-only
NaN-fill overhead — and it contrasts sharply with the 3072-GPU companion's clean **+14.3%** det penalty
(that penalty does not appear at 24n). See §3 for the multi-sample analysis.

**Methodology note**: nsys window = iters 15–17 (`profile_step_start=15 / end=18`). All iter-50 numbers
are **post-window**, so none carries active-profiling cost — the det-vs-nondet step time is a clean
signal. The det+nsys vs det-no-nsys gap (9,041 vs 8,891 ms, ~1.7%) is run-to-run / nsys-attach
variation, not capture cost. Leaderboard totals in §2 are **3-iter window** totals (divide by 3 for
per-iter); **§2 is rank-0 (first PP stage) only**.

---

## 1. Fairness & Shared Configuration

### What changes (the determinism knobs)

| Knob | Det | Non-det |
|---|---|---|
| `model.deterministic_mode` | true | false |
| `model.cross_entropy_loss_fusion` | false | true |
| `NCCL_ALGO` | Ring | unset (default) |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | 0 | unset |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | unset |
| `MAMBA_DETERMINISTIC` | 1 | unset |
| `train.fill_uninitialized_memory` | **false** (opt-out) | n/a (det block not entered) |

### Identical between arms (not a determinism variable)

`model.moe_router_fusion=true` (**both** arms — held constant so it is not a confound), MoE dispatcher
`alltoall` (HybridEP off), TE FusedAttention (cuDNN SDPA), `overlap_grad_reduce=true` /
`overlap_param_gather=true`, selective recompute (`moe, layernorm, core_attn, moe_act, mlp,
shared_experts`), `TRITON_CACHE_AUTOTUNING=1`, container `nemo:26.04.01`, same 24-node GB200 pool.

---

## 2. Perf Cost Decomposition (det+nsys vs non-det+nsys, 3-iter window)

From `nsys-compare-24node-routerfusion-v2/leaderboard.txt`. Positive Δ = det slower.

> **Rank-0 caveat**: `leaderboard.txt` is rank 0 only. On `TP=2 PP=3`, rank 0 is the **first pipeline
> stage** and never runs the CE / MTP-head backward (those live on the last stage), so §2 *understates*
> total det cost. Treat as the first-stage view.

### Forward — mcore module ranges (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `p2p_communication.recv_backward` | 2,007.6 | 1,616.8 | **+390.9** | +24.2 |
| `p2p_communication.send_forward_recv_backward` | 6,224.5 | 5,921.3 | +303.2 | +5.1 |
| `transformer_layer._forward_mlp.mlp` | 3,910.6 | 3,776.2 | +134.4 | +3.6 |
| `mlp.forward.activation` | 471.2 | 490.5 | −19.3 | −3.9 |
| `mlp.forward.linear_fc2` | 761.4 | 771.5 | −10.1 | −1.3 |
| `attention.forward.self_attention` | 476.9 | 484.2 | −7.3 | −1.5 |

### Backward — autograd engine ranges (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `MambaSplitConv1dScanCombinedFnBackward` | 2,552.4 | 2,201.8 | **+350.6** | +15.9 |
| `CheckpointFunctionBackward` | 12,709.2 | 12,867.4 | −158.1 | −1.2 |
| `RouterGatingLinearFunctionBackward` | 277.3 | 263.1 | +14.2 | +5.4 |
| `_GroupedLinearBackward` | 1,186.8 | 1,183.6 | +3.2 | +0.3 |
| `EmbeddingBackward0` | — | 14.3 | −14.3 | — |

### Op-level — aten / kernels (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Note |
|---|---|---|---|---|
| `aten::zeros` | 203.6 | 80.6 | +123.0 | zero-init before deterministic scatter |
| `aten::sum` | 681.6 | 560.3 | +121.3 | unfused-CE reduction |
| `DaoAILab::_causal_conv1d_bwd_cpp` | 132.6 | 44.4 | +88.2 | Mamba deterministic conv1d bwd |
| `aten::fill_` | 206.4 | 142.9 | +63.5 | det substitute zero-init |
| `aten::item` / `_local_scalar_dense` | ~128 / 123 | ~81 / 77 | +47 / +47 | host syncs (wait-inclusive) |

**Reading**: det's cost concentrates in **P2P comm** (`NCCL_ALGO=Ring` forces small P2P onto Ring —
though P2P SendRecv bypasses algo selection, so this is largely wait-skew), the **Mamba selective-scan
backward** + `causal_conv1d_bwd` (deterministic scan path), and the non-fused cross-entropy
`zeros`/`fill_`/`sum` ops. Compute (GEMM/attention) and `mcore.fusions` are near-identical. This is the
qualitative "where det differs" map; it does **not** net out to a step-time number (see §3).

---

## 3. Step Time: Within Run-to-Run Noise at 24n

Following the 3072-GPU doc's convention (iter-50 steady-state, post-window), but with **multiple
allocations** because the per-iter step time is noisy on this comm-heavy recipe.

### 3.1 iter-50 step time across allocations (3 samples per arm)

| arm | iter-50 samples (ms) | median |
|---|---|---|
| det + nsys | 9,041 (3706179) / 9,065 (3745946) / 9,140 (3746206) | **9,065** |
| non-det + nsys | 9,252 (3706214) / 9,090 (3712367) / 8,981 (3712403) | **9,090** |

median det+nsys (9,065) vs median non-det+nsys (9,090) → **det −25 ms (−0.28%)** — indistinguishable.
Det samples span 9,041–9,140 (~1.1%); non-det span 8,981–9,252 (~3.0%); the two arms overlap heavily.

### 3.2 why the spread is ~2% (it's not determinism)

Per-iter step time (iters 20–49) within single non-det runs already swings **330–494 ms** (std up to
~100 ms), *larger* than the between-run mean gap. Each allocation lands on different nodes
(`recheck-a` on nvl72008/089 ran ~200 ms high and noisy; `recheck-b` on nvl72030/055 ran low and tight,
std 35). The recipe is comm-bound (P2P + all-to-all over shared InfiniBand), so fabric contention +
stragglers produce the jitter. **All three non-det runs are identical config** and still spread ~2% —
so a <1% det-vs-nondet effect cannot be resolved here.

### 3.3 conclusion

At 24 nodes the determinism step-time cost is **within noise** (contrast 3072's clean +14.3%). The
likely reason it is so small here is `fill_uninitialized_memory=false`, which removes the det-only
NaN-fill overhead; §2 shows the residual det tax (comm/Mamba) is offset by equal/faster compute.
Resolving a sub-1% cost would need many matched runs per arm.

---

## 4. Determinism — BIT-EXACT across 4 allocations ✓

Unlike the 3072-GPU pair (which diverges from iter 3), the 24n deterministic recipe is **bit-identical**
across four independent no-nsys allocations (3706236 / 3706255 / 3707706 / 3707730) and the det+nsys
run (3706179) — nsys instrumentation does not perturb results.

| iter | lm loss (identical across all det runs) |
|---|---|
| 1 | 1.254623E+01 |
| 10 | 4.177998E+00 |
| 20 | 5.008514E-01 |
| 30 | 2.259073E-01 |
| 40 | 5.154770E-02 |
| 50 | 1.342692E+00 |

**Conclusion**: `moe_router_fusion=true` + `fill_uninitialized_memory=false` **preserve bit-exactness**
at 24 nodes. Router fusion (a kernel fusion, like the intentionally-disabled `cross_entropy_loss_fusion`)
does **not** break determinism here.

---

## 5. Config / Code Notes

- **`fill_uninitialized_memory` is now an option** (`TrainingConfig.fill_uninitialized_memory`, default
  `True` = torch's own behavior). The det arm sets it `false` to drop the NaN-fill overhead
  (reproducibility unaffected). It only takes effect in deterministic mode.
- **Scoping-bug fix**: an in-function `import torch.utils.deterministic` had rebound `torch` as a
  function-local, throwing `UnboundLocalError` at `torch.use_deterministic_algorithms(True)` and
  crashing every `deterministic_mode=true` run before iter 1. Fixed by aliasing the import
  (`import torch.utils.deterministic as _torch_det`).

---

## 6. Artifacts

| | path |
|---|---|
| Processed (leaderboard.txt + bitwise_check.txt + submit logs) | `nsys-compare-24node-routerfusion-v2/` |
| det+nsys log (3706179) | `~/.nemo_run/experiments/nemotron-3-ultra-det-nsys15-18-1782886365/.../log-*_3706179_0.out` |
| nondet+nsys log (3706214) | `~/.nemo_run/experiments/nemotron-3-ultra-nondet-nsys15-18-1782886365/.../log-*_3706214_0.out` |
| Determinism (no-nsys) jobs | 3706236 / 3706255 / 3707706 / 3707730 |
| nondet+nsys rechecks | 3712367 / 3712403 (step-time noise, §3) |
| det+nsys rechecks (median, §3.1) | 3745946 / 3746206 |
| Launch script | `scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh` |

> **Bottom line**: at 24 nodes with `moe_router_fusion=true` + `fill_uninitialized_memory=false`,
> determinism is **bit-exact across 4 allocations**, and the determinism **step-time penalty is within
> run-to-run noise** (median det ≈ non-det; contrast the 3072-GPU companion's +14.3%). §2 is the
> rank-0 first-stage cost map (comm + Mamba + unfused-CE), not a net step-time number.
