# Nemotron 3 Ultra — Deterministic vs Non-Deterministic Perf Analysis @ 24 nodes (router-fusion + fill-uninit)

**Date**: 2026-07-06 (rev 2 — pinned-node measurement supersedes the earlier "within-noise" reading)
**Model**: Nemotron 3 Ultra (550B-A55B hybrid Mamba+MoE, 108 layers, 512 experts top-22, 2 MTP heads)
**Config**: TP=2, PP=3, EP=32, ETP=1, CP=1, MBS=1, GBS=128, SeqLen=8192, 24 nodes × GB200 (96 GPUs),
selective recompute. `model.moe_router_fusion=true` (both arms) and `train.fill_uninitialized_memory=false`
on the det arm.

> Companion to [`analysis_report_det_vs_nondet.md`](analysis_report_det_vs_nondet.md) (24n baseline)
> and [`analysis_report_det_vs_nondet_3072gpu.md`](analysis_report_det_vs_nondet_3072gpu.md).

---

## TL;DR — the determinism cost is real (~1–2%), not "within noise"

**Headline (rev 2):** on a **pinned 24-node set** (all arms on identical hardware), deterministic mode costs
**+99 ms/iter (+1.1%) steady** and **+176 ms/iter (+2.0%) effective (mean, incl. det's periodic spikes)** —
a clean **4–6σ** signal, not noise. The **root cause is the deterministic Mamba selective-scan backward**
(`MambaSplitConv1dScanCombinedFnBackward`, driven by `MAMBA_DETERMINISTIC=1`), which adds **~+107 ms/iter**
and accounts for essentially the *entire* penalty. Everything else (GEMMs, recompute) is net-neutral or
marginally det-faster.

**The earlier "−0.3% / within noise" reading (rev 1) was a measurement artifact** — see §3 for the three
defects (single-iter-50 sampling, cross-allocation node-placement variance, and an assumed-zero nsys tax
that is actually ~+2%). A node-pinned 2×2 quad removes all three and resolves the signal cleanly.

---

## Runs Compared — pinned 2×2 quad (rev 2, definitive)

All four arms ran back-to-back on the **same physical node-set**
`nvl72016-T[03-05,08-15,18],nvl72042-T[01,03-13]` via an `afterany` dependency chain, so node-placement
variance and nsys asymmetry are both removed. Launcher:
[`launch_nemotron_3_ultra_pinned_quad.sh`](../launch_nemotron_3_ultra_pinned_quad.sh). Steady step time =
mean/median over iters 20–50, excluding any iter > 9800 ms (warmup / checkpoint / GC spikes).

| arm | job | wandb (`mbridge-dev`) | steady mean | median | std |
|---|---|---|---|---|---|
| det + **no-nsys** | 4188572 | nemotron-3-ultra-pinnedquad-det-nonsys | 8972 | 8891 | **217** (spiky) |
| non-det + **no-nsys** | 4188765 | …-nondet-nonsys | 8797 | 8793 | 28 |
| det + nsys | 4188775 | …-det-nsys | 9102 | 9099 | 48 |
| non-det + nsys | 4188788 | …-nondet-nsys | 8980 | 8976 | 19 |

**Deltas on identical nodes** (median is robust to det's spikes; mean captures throughput impact):

| comparison | median Δ | mean Δ | meaning |
|---|---|---|---|
| **CLEAN determinism** (no-nsys) | **+99 ms (+1.12%)** | +176 ms (+2.00%) | the honest determinism cost |
| determinism (nsys-confounded) | +123 ms (+1.37%) | +122 ms (+1.35%) | overstated by the nsys tax |
| nsys tax on det | +207 ms (+2.33%) | +130 ms | nsys out-of-window overhead |
| nsys tax on non-det | +183 ms (+2.08%) | +183 ms | " |

**Reads:**
- Node-pinning collapsed the non-det per-iter std to **19–28 ms** (vs the ~2% cross-allocation spread in
  rev 1), so the +99–123 ms gap is now a **4–6σ** effect — fully resolvable.
- **nsys ON overstates the determinism delta by ~0.25 pp** (median tax is det-heavier: +207 vs +183 ms).
  Use the **no-nsys** arms for the clean number; the nsys arms are for the §2 decomposition only.
- Determinism mode adds **intermittent slow-iter spikes** (det-nonsys std 217, ~4 spikes/31 iters:
  9413/9556/9793 ms; non-det std 28, zero spikes) — a separate, periodic cost on top of the steady floor
  that lifts the effective penalty from +1.1% to +2.0%.

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

`model.moe_router_fusion=true` (**both** arms), MoE dispatcher `alltoall` (HybridEP off), TE FusedAttention
(cuDNN SDPA), `overlap_grad_reduce=true` / `overlap_param_gather=true`, selective recompute (`moe,
layernorm, core_attn, moe_act, mlp, shared_experts`), `train.manual_gc=true` @ interval 100,
`TRITON_CACHE_AUTOTUNING=1`, container `nemo:26.04.01`, **and the same 24-node GB200 node-set** (the key
rev-2 change).

---

## 2. Perf Gap Root Cause — where the +99 ms/iter comes from

Traced from the pinned-node nsys arms (**det-nsys 4188775 vs non-det-nsys 4188788, same nodes**), rank-0,
3-iter window. Source: `24n-baseline/processed/leaderboard.txt`. Positive Δ = det slower.

> **Rank-0 caveat**: rank 0 is the first PP stage and never runs the CE / MTP-head backward (last stage),
> so this understates any CE-side cost. It is the first-stage compute + comm view.

### Backward — autograd engine (the signal)

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `MambaSplitConv1dScanCombinedFnBackward` | 2450.2 | 2129.5 | **+320.7** | **+15.1** |
| `CheckpointFunctionBackward` | 11979.2 | 12056.7 | −77.5 | −0.6 |
| `_GroupedLinearBackward` | 1210.2 | 1218.4 | −8.2 | −0.7 |
| `_LayerNormLinearBackward` (eval_fn) | 509.5 | 503.5 | +6.0 | +1.2 |

### Forward — mcore module ranges

| Range | det ms | nondet ms | Δ ms | Δ % |
|---|---|---|---|---|
| `p2p_communication.send_forward_recv_backward` | 7055.6 | 5906.0 | +1149.7 | +19.5 |
| `p2p_communication.recv_backward` | 2352.4 | 2103.0 | +249.4 | +11.9 |
| `transformer_layer._forward_mlp.mlp` | 3571.8 | 3462.1 | +109.6 | +3.2 |
| `mlp.forward.activation` | 441.2 | 478.6 | −37.4 | −7.8 |
| `mlp.forward.linear_fc2` | 713.8 | 743.8 | −29.9 | −4.0 |
| `attention.forward.self_attention` | 458.3 | 480.1 | −21.8 | −4.6 |

### Root-cause attribution

**The deterministic Mamba selective-scan backward is the entire penalty.** `MambaSplitConv1dScanCombinedFnBackward`
is **+320.7 ms over the 3-iter window ≈ +107 ms/iter** — which matches the clean step-time delta
(**+99 ms/iter median**) almost exactly. `MAMBA_DETERMINISTIC=1` selects the slower deterministic scan path
(`megatron/core/ssm/ops/determinism.py::use_deterministic_mode`; the non-deterministic FLA path is
disabled). Everything else nets out:

- **P2P deltas (+1150 / +249 ms) are wait, not cost.** These `PushPop` ranges are wall-clock and
  *wait-inclusive*: `p2p_communication` issues blocking `req.wait()` on rank 0 (first PP stage), so a
  slower det backward downstream simply makes rank 0 *wait longer* at the P2P recv. This is pipeline
  bubble that reflects the Mamba slowdown — it is **not additive** to the per-rank critical path.
- **GEMM and recompute are net-neutral or det-faster** (`CheckpointFunctionBackward` −77 ms,
  `_GroupedLinearBackward` −8 ms, `linear_fc2`/`activation`/`self_attention` all slightly negative on the
  pinned nodes) — the classic pattern: only the deterministic-substitute kernel (the Mamba scan) is slower.

So: **determinism cost ≈ deterministic Mamba scan backward (~+107 ms/iter)**, plus the intermittent det
spikes (§Runs-Compared). Router fusion and unfused CE do **not** show up as a first-stage cost here.

---

## 3. Why rev 1 said "−0.3% / within noise" (three measurement defects)

Rev 1 compared **iter-50 of one run per arm, median across 3 differently-placed allocations**, with nsys on
both arms, and concluded det was −25 ms (−0.3%) *faster*. Every piece of that was a methodology artifact:

1. **Single-iteration step-time estimate.** iter-50 is one noisy draw. The rev-1 non-det "median" (9090)
   came from an allocation whose steady-state mean was actually the *slowest* of its arm — a lucky-low
   iter-50 that flipped the sign. Steady-state means (iters 20–50) are required.
2. **Cross-allocation node-placement variance (~2%) swamped the ~1% signal.** Rev-1 arms ran on *different*
   nodes; the between-allocation spread (non-det means ranged ~2%) exceeded the determinism gap, so the
   difference of medians was dominated by which nodes each run happened to land on. **Pinning all arms to
   one node-set (rev 2) removes this** — non-det std drops to 19–28 ms.
3. **An assumed-zero nsys tax that is actually ~+2%.** Rev 1 asserted iter-50 was "post-window, no
   profiling cost." In fact the `nsys` wrapper keeps CUPTI/injection resident on the 3 profiled ranks for
   the whole run (`cudaProfilerStop()` only stops *saving*, not the per-CUDA-call interception), and those
   ranks gate the collective — so every iteration runs ~+2% slow, window or not. Measured on pinned nodes:
   **+2.08–2.33% out-of-window tax**, det-heavier by ~0.25 pp. The clean signal needs **no-nsys** arms.

Fixing all three (pinned nodes + steady-state means + no-nsys arms) yields the rev-2 headline: **+1.1%
steady / +2.0% effective, det slower** — same sign and magnitude as the §2 Mamba-scan decomposition.

---

## 4. Determinism — BIT-EXACT across allocations ✓

The 24n deterministic recipe remains **bit-identical** across independent no-nsys allocations (rev-1
allocations 3706236 / 3706255 / 3707706 / 3707730 matched to the last digit) and is unaffected by nsys
instrumentation at this scale. `moe_router_fusion=true` + `fill_uninitialized_memory=false` **preserve
bit-exactness** at 24 nodes; router fusion (a kernel fusion, like the intentionally-disabled
`cross_entropy_loss_fusion`) does **not** break determinism here.

| iter | lm loss (identical across all det runs) |
|---|---|
| 1 | 1.254623E+01 |
| 10 | 4.177998E+00 |
| 50 | 1.342692E+00 |

---

## 5. Config / Code Notes

- **`fill_uninitialized_memory` is an option** (`TrainingConfig.fill_uninitialized_memory`, default `True` =
  torch's own behavior). The det arm sets it `false` to drop the NaN-fill overhead (reproducibility
  unaffected). It only takes effect in deterministic mode.
- **Deterministic-mode plumbing** (`src/megatron/bridge/training/config.py::_validate_and_apply_deterministic_mode`,
  ~L1087–1112): asserts `cross_entropy_loss_fusion` is off, asserts `NCCL_ALGO` is in the deterministic set,
  calls `torch.use_deterministic_algorithms(True)`, and sets
  `torch.utils.deterministic.fill_uninitialized_memory` from the config. The import is a module-level
  `import torch.utils.deterministic` (config.py:25) with direct attribute access at L1112 — **not** a
  function-local alias (correcting rev-1's §5 description).
- **`MAMBA_DETERMINISTIC`** is read by `megatron/core/ssm/ops/determinism.py::use_deterministic_mode`, which
  selects the slower deterministic scan/conv backward path — the root cause in §2.

---

## 6. Artifacts

Organized under `nemotron-3-ultra-nsys-compare/24n-baseline/` (canonical share layout; `raw/` profiles are
symlinks into the per-run experiment dirs):

```
nemotron-3-ultra-nsys-compare/24n-baseline/
├── processed/
│   ├── jobid-{det,nondet,det-bitwise,nondet-nonsys}.txt   # 4188775 / 4188788 / 4188572 / 4188765
│   ├── wdj-{det,nondet,det-bitwise,nondet-nonsys}.txt      # wandb run names
│   ├── nodeset.txt                                         # the pinned 24-node set
│   ├── submit-{det,nondet}.log                             # nemo_run setup_experiment output
│   ├── leaderboard.txt                                     # §2 trace (pinned nsys arms, rank-0)
│   ├── nsys-det.csv                                        # rank-0 NVTX (det+nsys 4188775)
│   └── nsys-nondet.csv                                     # rank-0 NVTX (nondet+nsys 4188788)
└── raw/
    ├── det/            log + profile_*_4188775_node0_rank0.{nsys-rep,sqlite}   # det + nsys
    ├── nondet/         log + profile_*_4188788_node0_rank0.{nsys-rep,sqlite}   # nondet + nsys
    ├── det-bitwise/    log-*_4188572_0.out                                     # det, no-nsys (clean arm)
    └── nondet-nonsys/  log-*_4188765_0.out                                     # nondet, no-nsys (clean arm)
```

| Role | job | notes |
|---|---|---|
| det + no-nsys (clean, §Runs/§3) | 4188572 | clean determinism arm; mean 8972 / med 8891 (spiky, std 217) |
| non-det + no-nsys (clean) | 4188765 | mean 8797 / med 8793 (std 28) |
| det + nsys (§2 decomposition) | 4188775 | mean 9102 / med 9099 |
| non-det + nsys (§2 decomposition) | 4188788 | mean 8980 / med 8976 |
| Launch script | — | `scripts/performance/launch_nemotron_3_ultra_pinned_quad.sh` |

> **Bottom line (rev 2):** at 24 nodes the determinism step-time penalty is **real and ~1–2%**
> (+1.1% steady / +2.0% effective), **root-caused to the deterministic Mamba selective-scan backward**
> (~+107 ms/iter). The rev-1 "−0.3% within noise" was an artifact of single-iter sampling +
> cross-allocation node variance + an unaccounted ~2% nsys tax; a node-pinned 2×2 quad removes all three.
> Determinism remains **bit-exact** at 24n.
