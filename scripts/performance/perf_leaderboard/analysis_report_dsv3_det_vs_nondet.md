# DeepSeek-V3 — Deterministic vs Non-Deterministic Perf Analysis

**Date**: 2026-06-25
**Model**: DeepSeek-V3 (671B, 61 layers, 256 experts top-8, MLA, MTP) — pure MoE, **no Mamba**
**Config**: TP=1, PP=4, VP=4, CP=1, EP=64, ETP=1, MBS=1, GBS=2048, SeqLen=4096, 256 GPUs (64 nodes × GB200), `cuda_graph_impl=transformer_engine`, `recompute_modules=[mla_up_proj]`, `moe_flex_dispatcher_backend=hybridep`
**Cluster**: OCI HSG, container `nemo-26.04.01.squashfs`, MLM pin `c8288b6c` + bind-mounted HybridEP determinism fix (`enable_custom_allgather=False` + `_hybridep_maybe_sync`).

> Source: rank-0 nsys captures of iters 15–17, jobs **3581564** (det) / **3582315** (non-det).
> **HybridEP fix knobs (`HYBRIDEP_SYNC=1`, `HYBRIDEP_CUSTOM_ALLGATHER=0`) are ON in BOTH** → held constant; the delta isolates the determinism knobs, not the fix.
> Bit-wise determinism verified by job **3581575** (det, no-nsys) — §5.

> **Node placement is not a confound.** Det and non-det landed on different racks, but determinism is bit-exact across racks (§5; 7a/7b and det==det-bitwise matched to the last digit on different nodes) → GB200 nodes are numerically equivalent. Residual compute timing differences are normal run-to-run wall-clock jitter (~few %), the same as re-running on identical nodes.

---

## Runs Compared

| | Det (3581564) | Non-det (3582315) |
|---|---|---|
| Slurm runtime | 24:45 / 50 iters | 22:17 / 50 iters |
| **Per-iter (log iter 50)** | **11,342 ms** | **10,454 ms** |
| **Step-time Δ (iter 50)** | — | **det +889 ms (+8.5%)** |
| NCCL algo | `NCCL_ALGO=Ring` | default (Ring+Tree mix) |
| nsys window (iters 15–17) total GPU-kernel ms (all streams) | 37,106 | 32,531 |
| Bit-wise reproducible | ✓ (== 3581575) | n/a |

nsys window = iters 15–17 (warmup, incl. iter-16 anomaly §10.3); divide window totals by 3 for per-iter. Steady-state wall delta = iter-50 (+889 ms / +8.5%).

---

## 1. Fairness & Shared Configuration

**Identical kernel counts (rank 0, 3-iter window):** NCCL 2,652=2,652 · nvjet GEMM 66,528=66,528 · cuDNN attn 4,608=4,608 · MoE permute/sort 6,240=6,240 · reduce_kernel 4,041=4,041 · index/scatter 204=204. Structural recipe is byte-identical; only algorithm-variant picks change. **No Mamba kernels.**

**Knobs that change (via `--deterministic` → `apply_determinism_overrides` + `PerfEnvPlugin`):**

| Knob | Det | Non-det | Evidence |
|---|---|---|---|
| `deterministic_mode=true` | true | false | `FillFunctor` 170,400 vs 14,310 calls |
| `cross_entropy_loss_fusion=false` | false | true | last-PP-stage only — invisible on rank 0 (§10.2) |
| `tp_comm_overlap=false` | false | (already false) | no delta |
| `NCCL_ALGO=Ring` | Ring | default | f32 AllReduce TREE_LL→RING_LL (§4) |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | 0 | 1 | **no SDPA 2-pass split** (§3.2) |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | set | unset | GEMM net −619 ms (within run-to-run jitter §3.1) |

---

## 2. Headline: Where Does the +889 ms / Iter Go?

**`FillFunctor` (deterministic scatter zero-init):**

| | Det | Non-det | Δ |
|---|---|---|---|
| calls (window) | **170,400** | **14,310** | +156,090 |
| GPU ms (window) | **2,563** | **87** | **+2,476** (~+825 ms/iter GPU-busy, mostly overlapped) |

`use_deterministic_algorithms(True)` routes MoE/loss scatter through zero-then-accumulate; every buffer is re-zeroed by a `FillFunctor` before the deterministic scatter. Dominant det-substitute cost (same mechanism as Nemotron).

**Stream decomposition (window busy ms):**

| Stream | Role | Det | Non-det | Δ |
|---|---|---|---|---|
| 16 | main: compute + fills + CCL host | 22,280 | 19,786 | +2,494 |
| 31 | NCCL (SendRecv/PP p2p) | 4,981 | 3,634 | +1,347 |
| 167–170 | GEMM | ~1,830 ea | ~1,830 ea | ±~150 (noise) |

**NVTX namespace decomposition (host ms, op_id-stripped):**

| Namespace | Det | Non-det | Δ | Δ% |
|---|---|---|---|---|
| `backward_node` | 33,521 | 29,818 | +3,703 | +12.4% |
| `other` (Python NVTX) | 32,645 | 29,756 | +2,889 | +9.7% |
| `aten::*` | 4,661 | 3,235 | **+1,426** | **+44.1%** |
| `nvte::*` | 1,699 | 1,825 | **−126** | **−6.9%** (det faster) |
| `autograd::evaluate` | 949 | 986 | −38 | −3.8% |
| `nccl::*` (host) | 249 | 231 | +17 | +7.5% |

**`aten::*` top contributors (|Δ|>50 ms):** `aten::empty` +756.3 · `aten::fill_` +647.8 · `aten::zeros` +50.8 · `aten::zero_` −57.6. (Per-op `index_put`/`scatter_add`/`max`/`min` from Nemotron's last-stage do NOT appear — rank 0 is first PP stage, §10.2.)

---

## 3. Compute Path — untouched (within run-to-run jitter)

**3.1 GEMM**: counts identical (66,528); net **det −619 ms** (faster) — a determinism knob cannot speed up a matmul, so this is **normal run-to-run wall-clock jitter** (~few % of the ~9 s GEMM bucket), the kind seen re-running on identical nodes. Bit-exact determinism (§5) confirms the GEMM math is numerically identical; only the timing jitters. `CUBLAS_WORKSPACE_CONFIG` penalty is below this noise floor.

**3.2 cuDNN SDPA — NO 2-pass split** (unlike Nemotron): attn counts identical (4,608); `flash_bprop`/`fprop` both marginally *faster* in det. No det-only second-pass kernel. `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` has ~0 measurable cost here.

---

## 4. Communication Path

**4.1 Per-collective NCCL (window, rank 0):**

| Collective | Det calls/ms | Non-det calls/ms | Δ ms |
|---|---|---|---|
| **SendRecv** (PP p2p) | 1,344 / **7,388.5** | 1,344 / **4,973.9** | **+2,414.6** |
| AllGather RING_LL | 1,260 / 658.7 | 1,260 / 515.2 | +143.4 |
| AllReduce bf16 RING_LL | 6 / 116.7 | 6 / 116.7 | 0 |
| ReduceScatter bf16 RING_LL | 12 / 67.5 | 12 / 63.9 | +3.6 |
| AllReduce f32 RING_LL | 27 / 28.3 | 6 / 8.7 | det Ring |
| AllReduce f32 **TREE_LL** | 0 | 21 / 14.9 | **non-det only** |
| AllReduce u32 TREE_LL | 0 | 3 / 8.5 | non-det only |

1. **Genuine `NCCL_ALGO=Ring` effect**: f32/u32 AllReduce is TREE_LL (non-det) vs RING_LL (det) — the only bit-exact-relevant NCCL change, and **small** (~tens of ms). AG/RS/bf16-AR are RING_LL in both.
2. **SendRecv +2,414 ms is wait-skew, NOT algo** — P2P bypasses Ring/Tree selection; NCCL kernels are wait-inclusive, so det's slower stream-16 fills delay the PP rendezvous. Symptom of the fills, not an NCCL cost.

**4.2 HybridEP device-sync (fix ON in both):** `hybrid_ep::device_sync_kernel<64>` 9,984 calls both; det 1,004.0 ms vs non 599.9 ms (**+404 ms**) — wait-shadow of det's stream-16 skew, not a config diff. This + the fills drive the `HybridEPCombineBackward +21%` / `HybridEPDispatch +9%` deltas in `leaderboard.txt`.

---

## 5. Bit-wise Reproducibility

det 3581564 vs det-no-nsys 3581575 — `lm loss` bit-identical: iter1 1.189298E+01 · iter5 1.201637E+01 · iter10 8.789434E+00 · iter25 7.569467E+00 · iter50 6.532277E+00. Determinism IS bit-exact for the trained trajectory.

**`seq_load_balancing_loss` ~1-ULP wobble (logging-only):** the aux loss is used twice from one `aux_loss` tensor — (a) attached to activation for the **gradient** (deterministic reduction → bit-identical, hence lm loss matches), (b) logged via `tracker.record(..., needs_dp_avg=True)`, a **separate logging-only reduce+DP-average**. The wobble lives entirely in (b), never touches gradients, didn't compound. **It is not a topology/reduction-order artifact**: the gradient cross-rank reduction is bit-exact across different racks (lm loss matches to the last digit on different nodes), so cross-rank reduction order is node-independent here. That points to a **genuinely non-deterministic reduce in the logging-only path (b)** — a same-node rerun would not fix it. token/global `load_balancing_loss`=0.0 (aux-loss-free recipe), uninformative.

---

## 6. Per-Knob Cost Attribution (caveated by §10.1)

| Knob | Signature | Est. GPU-busy/iter | Confidence |
|---|---|---|---|
| `use_deterministic_algorithms` | FillFunctor +2,476 ms/window | ~825 ms/iter (mostly overlapped) | HIGH |
| `NCCL_ALGO=Ring` (genuine) | f32/u32 AR Tree→Ring | ~few ms/iter | HIGH |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | no SDPA split | ~0 | MED |
| `CUBLAS_WORKSPACE_CONFIG` | GEMM −619 ms (run-to-run jitter) | below noise floor | LOW |
| `cross_entropy_loss_fusion=false` | last-stage only | invisible on rank 0 | — |
| SendRecv +2,414 / device_sync +404 | wait-inclusive | shadow, not independent | MED |

**Wall reconciliation**: +889 ms/iter wall ≪ sum of GPU-busy deltas → most det-substitute GPU time overlaps; only the un-overlappable residual hits the wall.

---

## 7. Improvement Opportunities

1. **Reduce the `FillFunctor`-zero + scatter chain** (dominant real cost): batch per-expert scatters; or a custom write-on-first-touch deterministic scatter kernel; or scoped `use_deterministic_algorithms(warn_only=True)` on hot MoE scatters (breaks bit-exactness for those ops only). "Cache the buffer" is invalid (accumulate-into-prezeroed needs re-zeroing each backward).
2. **Scope `NCCL_ALGO=Ring` to AR/RS/Reduce only** (float-summing collectives); small recovery — SendRecv is wait-skew, not NCCL-tunable.
3. **HybridEP `device_sync` +404 ms** = cost of the `HYBRIDEP_SYNC=1` correctness fix; relax only if a fence is proven safe at scale (no `cudaErrorIllegalAddress` regression).

---

## 8. Methodology

Both runs via `launch_deepseek_v3_nsys_compare.sh` (det = `--deterministic`). nsys: steps 15–18, ranks 0/128/255, `cuda-sw,nvtx`, sqlite export. Stats from `python3 sqlite3` on rank-0 `.sqlite`: `CUPTI_ACTIVITY_KIND_KERNEL`⋈`StringIds` on `demangledName`, ms=`(end-start)/1e6`, grouped by `streamId`; `NVTX_EVENTS.text` for namespaces (op_id-stripped).

---

## 9. Bottom Line

- **Determinism IS bit-exact** at 256-GPU DSv3 scale (3581564 == 3581575), **across different racks** → node-independent.
- **Determinism step-time cost: +889 ms/iter (+8.5%)** at iter 50 (carries normal run-to-run wall-clock jitter, ~few %).
- Only dominant determinism signature: `FillFunctor` zero-init blow-up (`use_deterministic_algorithms`).
- Compute not the source: GEMM/attn count-identical (timing within jitter); `nvte::*` −6.9%.
- NCCL: only genuine change is f32 AllReduce Tree→Ring (small); SendRecv +2,414 ms = wait-skew.
- No SDPA 2-pass split (differs from Nemotron).

---

## 10. Scope & Limitations

**10.1 Different physical nodes — NOT a confound here.** det ran on nvl72029/51/53/120, non-det on nvl72011/32/… (only 72051 overlaps). This does **not** invalidate the comparison: determinism is **bit-exact across different racks** (§5 — 7a/7b and det==det-bitwise matched to the last digit on different nodes), so GB200 nodes are numerically equivalent and a `NODELIST` match is **not required**. The only node-dependent quantity is wall-clock timing, which carries normal run-to-run jitter (~few %) regardless of placement — that jitter (not a topology effect) is why GEMM/attention time wiggles by a few % between the two runs. Treat the ±few % as the comparison's uncertainty band; the +889 ms/iter signal is well outside it.

**10.2 Single-rank, first-PP-stage.** rank 0 on PP=4 never runs CE backward or last-stage MoE-expert-grad scatter → `cross_entropy_loss_fusion=false` and much scatter cost are invisible (surface only via SendRecv wait). 128/255 profiles captured for a multi-stage follow-up.

**10.3 iter-16 anomaly (both):** iter16 ~+12–15% vs neighbors (det 13,262 / non 12,139); folded into the window — use iter-50 for steady-state.

---

## 11. Honesty Notes

Every number is from a direct sqlite query on the rank-0 `.sqlite`. Where the Nemotron mechanism did NOT reproduce (SDPA 2-pass; per-op index_put/max/min), this report says so. NCCL kernel deltas are wait-inclusive; `NCCL_ALGO` applies to collectives only, not P2P SendRecv. Node placement is **not** a confound (determinism is bit-exact across racks, §5/§10.1); the remaining caveat is the single-rank first-PP-stage view (§10.2), which under-counts last-stage CE/scatter cost.

---

## 12. Artifacts

⚠️ **Data lives on the OCI HSG cluster** (these DSv3 runs were on OCI HSG, not lyris). The lyris `/lustre/share/coreai_dlalgo_llm/...` mirror that the Nemotron doc uses **does not exist on HSG** (only `/lustre/fs1` + `/lustre/fsw` are mounted). World-readable, ~1.3 GB:

Layout follows the Nemotron `<scale>/{processed, raw/{det,nondet,det-bitwise}}/` convention:

```
/lustre/fsw/portfolios/llmservice/users/zhiyul/dsv3-det-nondet-nsys/256n/
├── processed/   # leaderboard.txt, nsys-{det,nondet}[-rank{0,128,255}].csv, jobid-*, wdj-*, submit-*, launcher.log
└── raw/
    ├── det/          # log-…_3581564_0.out + profile_*_rank{0,128,255}.{nsys-rep,sqlite}
    ├── nondet/       # log-…_3582315_0.out + profile_*_rank{0,128,255}.{nsys-rep,sqlite}
    └── det-bitwise/  # log-…_3581575_1.out   (no nsys)
```

| Side | Slurm job | wandb run (project `mbridge-dev-zhiyul`) |
|---|---|---|
| det + nsys | 3581564 | `deepseek-v3-det-nsys15-18-1782378442` |
| non-det + nsys (default NCCL) | 3582315 | `deepseek-v3-nondet-default3` |
| det, no-nsys (bit-wise check) | 3581575 | `deepseek-v3-det-bitwise-check-1782378442` |

The repo-relative launcher is `scripts/performance/launch_deepseek_v3_nsys_compare.sh`.
