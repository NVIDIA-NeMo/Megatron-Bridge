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

Perf comparison uses **nsys runs only** (both arms instrumented — apples-to-apples), **3 samples per
arm**, iter-50 steady-state (post-nsys-window, so no profiling cost). Reported as the **median** of the 3.

| | Det + nsys | Non-det + nsys |
|---|---|---|
| jobs (3 samples/arm) | 3706179 / 3745946 / 3746206 | 3706214 / 3712367 / 3712403 |
| wandb (primary) | [gbzxks86](https://wandb.ai/nvidia/mbridge-dev/runs/gbzxks86) | [e2i1o8wg](https://wandb.ai/nvidia/mbridge-dev/runs/e2i1o8wg) |
| **Median step time (iter 50)** | **9,065 ms** | **9,090 ms** |
| **Median throughput (TFLOP/s/GPU)** | **432.5** | **431.3** |
| **Step time Δ (det − nondet)** | — | **−25 ms (−0.3%)** — within noise (§3) |
| **Throughput Δ (det − nondet)** | — | **+1.2 (+0.3%)** — within noise (§3) |
| Bit-wise reproducible | ✓ (§4) | n/a |

**Headline: at 24 nodes there is no resolvable determinism penalty — det vs non-det is within
run-to-run noise.** Median over 3 nsys runs/arm: step time **9,065 ms (det)** vs **9,090 ms (non-det)**
→ **Δ −25 ms (−0.3%)**; throughput **432.5** vs **431.3 TFLOP/s/GPU** → **Δ +1.2 (+0.3%)**. Both deltas
are *smaller than each arm's own run-to-run spread* (det 9,041–9,140 ms; non-det 8,981–9,252 ms), so the
determinism cost is indistinguishable from noise at this scale (§3).

**Methodology note**: nsys window = iters 15–17 (`profile_step_start=15 / end=18`); iter-50 is
**post-window**, so it carries no active-profiling cost — a clean det-vs-nondet signal. **Only the nsys
runs are used for the step comparison** — the no-nsys det runs are used *solely* for the determinism
check (§4), never as step-time datapoints. §2 leaderboard totals are **3-iter window**, **rank-0 only**.

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

Traced from the nsys profiles of the pairing where **non-det is slightly faster** — the *expected*
det-penalty direction — so the det cost reads as a clean positive signal (not the noise-flipped
"det faster" pairing): **det+nsys job 3745946** (9,065 ms, the det median) vs **non-det+nsys job
3712403** (8,981 ms, ~0.9% faster). Source: `nsys-trace-det-vs-nondet-expected/leaderboard.txt`
(rank 0, 3-iter window). Positive Δ = det slower.

> **Rank-0 caveat**: `leaderboard.txt` is rank 0 only. On `TP=2 PP=3`, rank 0 is the **first pipeline
> stage** and never runs the CE / MTP-head backward (those live on the last stage), so §2 *understates*
> total det cost. Treat as the first-stage view.

### Forward — mcore module ranges (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `p2p_communication.send_forward_recv_backward` | 6,202.9 | 5,791.1 | +411.8 | +7.1 |
| `transformer_layer._forward_mlp.mlp` | 4,044.3 | 3,639.7 | **+404.6** | +11.1 |
| `p2p_communication.recv_backward` | 2,313.4 | 1,945.6 | +367.8 | +18.9 |
| `p2p_communication.send_forward` | 172.0 | 194.1 | −22.1 | −11.4 |
| `mlp.forward.linear_fc2` | 752.3 | 746.8 | +5.5 | +0.7 |
| `attention.forward.self_attention` | 472.5 | 467.1 | +5.4 | +1.2 |

### Backward — autograd engine ranges (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `CheckpointFunctionBackward` | 12,591.4 | 12,251.9 | **+339.5** | +2.8 |
| `MambaSplitConv1dScanCombinedFnBackward` | 2,490.1 | 2,198.7 | +291.3 | +13.3 |
| `_GroupedLinearBackward` | 1,174.8 | 1,204.4 | −29.6 | −2.5 |
| `_LinearBackward` | 941.7 | 959.2 | −17.5 | −1.8 |
| `EmbeddingBackward0` | — | 14.7 | −14.7 | — |

### Op-level — aten / kernels (top by |Δ|)

| Range | det ms | nondet ms | Δ ms | Note |
|---|---|---|---|---|
| `CheckpointFunction` | 5,584.5 | 5,170.0 | +414.5 | recompute wrapper (wait-inclusive) |
| `aten::sum` | 668.5 | 546.6 | +121.9 | unfused-CE reduction |
| `aten::zeros` | 199.8 | 78.1 | +121.6 | zero-init before deterministic scatter |
| `DaoAILab::_causal_conv1d_bwd_cpp` | 130.0 | 44.7 | +85.2 | Mamba deterministic conv1d bwd |

**Reading**: with non-det as the faster arm, det's cost is consistently positive in the hot buckets —
**P2P comm** (`send_forward_recv_backward` +7%, `recv_backward` +19% — largely wait-skew, since P2P
SendRecv bypasses `NCCL_ALGO`), **MLP forward** (+11%), the **Mamba selective-scan backward** +
`causal_conv1d_bwd` (deterministic scan path, +13% / +191%), the recompute `CheckpointFunction` (+8%),
and the non-fused cross-entropy `aten::sum` / `aten::zeros`. GEMM (`linear_fc1/fc2`,
`_GroupedLinearBackward`) and `mcore.fusions` are near-identical or marginally det-faster — the classic
deterministic-substitute pattern (extra `fill`/`zeros`/`sum` + deterministic Mamba scan). This is the
*where-det-differs* map; the *net* step-time delta is still within noise (§3).

---

## 3. Step Time: Within Run-to-Run Noise at 24n

Uses iter-50 steady-state step time (post-nsys-window), with **multiple allocations per arm** because
the per-iter step time is noisy on this comm-heavy recipe.

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

At 24 nodes the determinism step-time cost is **within run-to-run noise** — median Δ **−25 ms (−0.3%)**
step / **+1.2 TFLOP/s (+0.3%)** throughput, both smaller than the ~2–3% per-arm spread. §2 shows *where*
det differs (comm + Mamba + unfused-CE substitutes), offset by equal/faster compute, but the net is
sub-noise. Resolving a sub-1% cost would need many matched runs per arm.

### 3.4 NOT measured here: the improvement from the flags

This report is **det vs non-det**, both carrying `moe_router_fusion=true` (and the det arm
`fill_uninitialized_memory=false`) — it does **not** measure how much those settings *improve* perf
versus a recipe without them. There is **no matched no-flags baseline** in this run, so no
improvement number is claimed. Quantifying that requires a separate 24n run with the flags **off**
(and same GPU count) — a follow-up, not done here. (Do **not** use the 3072-GPU report's numbers as
that baseline — different GPU count, not a matched comparison.)

---

## 4. Determinism — BIT-EXACT across 4 allocations ✓

The 24n deterministic recipe is **bit-identical** across four independent no-nsys allocations
(3706236 / 3706255 / 3707706 / 3707730) and the det+nsys run (3706179) — nsys instrumentation does not
perturb results.

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

Organized under `nsys-det-3run-24node-routerfusion/` (repo-relative; `raw/` are symlinks into the
per-run experiment dirs — only the comparison runs' nsys profiles + logs are kept):

```
nsys-det-3run-24node-routerfusion/
├── processed/
│   ├── jobid-{det,nondet,det-bitwise}.txt   # 3745946 / 3712403 / 3706236
│   ├── wdj-{det,nondet,det-bitwise}.txt      # wandb run names
│   ├── leaderboard.txt                       # §2 trace (rank-0)
│   ├── nsys-det.csv                          # rank-0 NVTX (det+nsys 3745946)
│   └── nsys-nondet.csv                       # rank-0 NVTX (nondet+nsys 3712403)
└── raw/
    ├── det/          log-*_3745946_0.out + profile_*_3745946_node0_rank0.{nsys-rep,sqlite}
    ├── nondet/       log-*_3712403_0.out + profile_*_3712403_node0_rank0.{nsys-rep,sqlite}
    └── det-bitwise/  log-*_3706236_0.out   (no nsys — determinism check)
```

| Role | job | notes |
|---|---|---|
| det + nsys (perf, §2/§3) | 3745946 | det median (9,065 ms); one of 3 det+nsys (3706179 / 3745946 / 3746206) |
| non-det + nsys (perf, §2/§3) | 3712403 | ~0.9% faster; one of 3 nondet+nsys (3706214 / 3712367 / 3712403) |
| det no-nsys (determinism, §4) | 3706236 | one of 4 bit-exact allocations (3706236 / 3706255 / 3707706 / 3707730) |
| Launch script | — | `scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh` |

> **Bottom line**: at 24 nodes with `moe_router_fusion=true` + `fill_uninitialized_memory=false`,
> determinism is **bit-exact across 4 allocations**, and the det-vs-non-det **step-time penalty is within
> run-to-run noise** (median Δ −25 ms / −0.3% step, +1.2 / +0.3% throughput — nsys runs only). §2 is the
> rank-0 first-stage cost map (comm + Mamba + unfused-CE), not a net step-time number. This report does
> not measure the improvement from the flags themselves (no matched no-flags baseline — §3.4).
