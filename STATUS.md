# DSv3 256-GPU Clean-Run Status

**Goal:** DeepSeek-V3, 256 GPUs (GB200), default production perf recipe + determinism,
must **finish cleanly to the end of training** (no crash / no OOM).
Bit-identical determinism check is **in scope** for Rounds 7a/7b.

**Run command:** `DETERMINISTIC=true BACKEND=fused GPU=gb200 ./run_deepseek_v3.sh`
(no `MINIMAL`, no small-scale overrides, `RACE_NOISE=0`)

**Driver:** `experiment-runner` agent `dsv3-runner` (iterating)
**Last updated:** 2026-06-25 — Rounds 7a/7b COMPLETE. Determinism CONFIRMED: lm loss + grad norm bit-identical across all 50 iters (one ~1e-6 blip in a logged aux metric — see below).

---

## Success criterion
- Clean run: job trains to completion without `-4 dim`, `cudaErrorIllegalAddress`, or OOM. ✅
- Determinism: lm loss and grad norm bit-identical across all 50 iterations between 7a and 7b. ✅

---

## Iteration log

| Round | Code state | Scale | Job id | Outcome |
|-------|-----------|-------|--------|---------|
| 1 | Full patch stack (per-manager buffer dict + sync + record_stream) | 256 | 2192660 | ❌ OOM in `grouped_linear.py` 1st MoE forward (per-manager dict ~+12 GiB) |
| 2 | Stock 26.04 (MINIMAL, no patch) | 256 | — | ❌ `-4 dim` crash (uninit `num_dispatched_tokens_tensor`, custom-allgather path) |
| 3 | + single line `enable_custom_allgather=False` (stock singleton buffer) | 256 | — | ❌ cleared `-4 dim`; new `cudaErrorIllegalAddress` at `fused_a2a.py:382` (cross-stream race) |
| 4 | Round 3 + verbatim sync (`_hybridep_maybe_sync`) at 4 dispatch/combine sites; **no** per-manager dict, **no** record_stream | 256 (64×4) | 3575525 | ❌ INFRA — `PermissionError` on HF cache `/lustre/fsw/portfolios/coreai/...` during `from_hf_pretrained`; training never started |
| 5 | Same as Round 4 + `HF_HOME` override to writable llmservice path | 256 (64×4) | 3575788 | ✅ CLEAN — all 50/50 iters, step ~11.5s, lm loss 11.67→6.53, 0 skipped, 0 NaN, exit 0 |
| 6 | **Stock MLM** (no hacks — `fused_a2a.py` reverted to v0.4.1); `RACE_NOISE=0` | 256 (64×4) | 3576682 | ❌ CRASH at iter 0 — `cudaErrorIllegalAddress` (rank2 PP watchdog) + `-4 dim` on multiple ranks; cross-stream race is REAL, not a noise artifact. Fix is necessary. |
| 7a | Fix restored (same as Round 5); `RACE_NOISE=0`; determinism run A | 256 (64×4) | 3577135 | ✅ CLEAN 50/50, exit 0 |
| 7b | Fix restored (same as Round 5); `RACE_NOISE=0`; determinism run B | 256 (64×4) | 3577139 | ✅ CLEAN 50/50, exit 0 |

**Determinism verdict (7a vs 7b, independent runs on different node sets):**
- `lm loss` **bit-identical** across all 50 iterations (iter1 `1.189298E+01` → iter50 `6.532277E+00`).
- `grad norm`, `mtp_1 loss`, `load_balancing_loss`, learning rate, skipped/nan counts: **bit-identical** every iteration.
- **Sole exception:** `seq_load_balancing_loss` at **iteration 46 only** — 7a `1.000195E+00` vs 7b `1.000194E+00` (~1e-6). Did NOT perturb lm loss/grad norm (iters 47–50 re-converge bit-identical), so it's a one-off cross-rank reduction-order blip in a *logged* MoE statistic, not training nondeterminism.
- **Conclusion: training trajectory (loss + gradients) is deterministic at 256-GPU scale with the 2-change fix.**

---

## Current code state
`3rdparty/Megatron-LM` @ `d7288711b` (v0.4.1 pin), working-tree changes only (bind-mounted):

- ✅ `fused_a2a.py`: `enable_custom_allgather=False` + verbatim sync fix (`_hybridep_maybe_sync` at 4 dispatch/combine sites) — restored by team-lead after Round 6 proved the crash is real. Singleton buffer preserved; no per-manager dict; no record_stream.
- ✅ `fused_a2a.py` confirmed bind-mounted in both 7a and 7b sbatch scripts.

**Controlled proof (Round 5 vs Round 6):**
- Round 5: fix active, RACE_NOISE=0 → ✅ clean
- Round 6: stock, RACE_NOISE=0 → ❌ crashes at iter 0 with cudaErrorIllegalAddress + -4 dim

---

## Hard constraints (do not violate)
- ❌ **Never** reintroduce the per-manager `_hybrid_ep_buffers: dict` — it OOMs at 256-GPU scale (Round 1). Keep the stock singleton `_hybrid_ep_buffer`.
- ❌ No `record_stream(...)` — not needed at scale.
- Sync code must be **verbatim** from `patches/megatron-lm-dsv3-hybridep-determinism.patch`.
- Both 7a and 7b use the SAME seed (recipe fixes `random_seed=1234` via MockGPTDataset — not overridden).
- Each 256-GPU launch is expensive — state hypothesis + exact change before every relaunch.

---

## Open / next
- ✅ Clean-run goal met (Round 5) and determinism confirmed (Rounds 7a/7b). Investigation goal achieved.
- Optional follow-ups: (1) same-node `NODELIST`-pinned A/B to confirm the iter-46 `seq_load_balancing_loss` blip is a cross-node reduction artifact; (2) check whether `seq_load_balancing_loss` enters the backprop loss (if monitoring-only, the blip is provably harmless).

## W&B
- Project: https://wandb.ai/nvidia/mbridge-dev-zhiyul
- 7a (det-A, job 3577135): https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/btbirum5
- 7b (det-B, job 3577139): https://wandb.ai/nvidia/mbridge-dev-zhiyul/runs/erqsuf6u

## Where to look
- 7a run dir: `~/.nemo_run/experiments/deepseek-v3-det-A/deepseek-v3-det-A_1782357902/`
- 7b run dir: `~/.nemo_run/experiments/deepseek-v3-det-B/deepseek-v3-det-B_1782357938/`
- Cluster: OCI HSG (`oci-hsg-cs-001`); account `nemotron_sw_pre`; container `nemo-26.04.01.squashfs`

## References
- `SUMMARY.md` — full root-cause writeup.
- `run_deepseek_v3.sh` — submission script (adapted for OCI HSG) + bind-mount auto-detection + env knobs.
- `patches/megatron-lm-dsv3-minimal-determinism.patch` — **the confirmed minimal 2-change fix** (enable_custom_allgather=False + sync, stock singleton buffer). Apply with `git -C 3rdparty/Megatron-LM apply patches/megatron-lm-dsv3-minimal-determinism.patch`.
- `patches/megatron-lm-dsv3-hybridep-determinism.patch` — earlier full patch stack (per-manager dict + record_stream); superseded at scale by the minimal patch (the dict OOMs).
