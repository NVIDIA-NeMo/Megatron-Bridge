# DSv3 HybridEP + Determinism Investigation (Megatron-Bridge 26.04 / GB200)

Branch: `zhiyul/deterministics_gb200_2604_dsv3`
Submodule base: `NVIDIA/Megatron-LM@d7288711b` (v0.4.1 pin)
Investigation period: Jun 2 – Jun 23, 2026

---

## TL;DR

DeepSeek-V3 with HybridEP token dispatcher crashes (`cudaErrorIllegalAddress`,
"-4 dim" assertion) and is non-deterministic under
`torch.use_deterministic_algorithms(True)` at GB200 production scale (256 GPUs).
This doc captures the root causes that have been identified, the patches that
fix the 8-GPU EP=4 case bit-exactly, and the open problem that still
reproduces at 16-GPU EP=8.

All patches are bind-mounted at runtime by `run_deepseek_v3.sh` — the
submodule pointer is unchanged.

---

## What changed between 26.02 (worked) and 26.04 (broken)

| Item | 26.02 | 26.04 | Impact |
|------|-------|-------|--------|
| `HybridEPBuffer.enable_custom_allgather` default | `False` | `True` | The new custom-allgather path leaves `num_dispatched_tokens_tensor` uninitialised before `executor.cu` blocking-reads it via `.item<int>()`. Reads as `-4` (`0xFFFFFFFC`). |
| DeepEP intranode path | none (single `internode.cu`) | new `intranode.cu` added | Separate `-4 dim` bug surfaces at EP ≥ 8 even with the custom_allgather fix. **Open**. |

---

## Root causes identified

1. **`enable_custom_allgather=True` race** — 26.04's custom-allgather path
   skips populating the `num_dispatched_tokens_tensor` shared-memory buffer
   before the host blocking read on it. The host then reads uninitialised
   memory (`-4` / `0xFFFFFFFC`) and the downstream tensor allocation asserts.
2. **Cross-stream race in `dispatch_with_permute(non_blocking=True)`** — the
   HybridEP runtime writes outputs on its internal comm stream and returns
   before the writes complete. Any consumer kernel launched on the current
   stream races against the still-in-flight writes →
   `cudaErrorIllegalAddress`.
3. **Singleton buffer shared across VPP chunks** — the module-level
   `_hybrid_ep_buffer` global was shared between forward/backward passes of
   different virtual-pipeline chunks under interleaved 1F1B, causing
   handle/state aliasing and "negative dimension" errors. Already fixed in
   the `2602` investigation; carried forward into 26.04.

---

## Fixes (in `patches/megatron-lm-dsv3-hybridep-determinism.patch`)

Applied to `3rdparty/Megatron-LM` (bind-mounted, submodule pointer unchanged):

| File | Fix |
|------|-----|
| `megatron/core/transformer/moe/fused_a2a.py` | (a) Per-manager `_hybrid_ep_buffers: dict` keyed on `buffer_key=id(_HybridEPManager)`. (b) `enable_custom_allgather=False` forced at `HybridEPBuffer` construction. (c) `torch.cuda.synchronize()` after every `dispatch_with_permute(non_blocking=True)` and `combine_with_unpermute` (gated by `HYBRIDEP_SYNC` env, default `1`). (d) `record_stream(current_stream)` on returned tensors, guarded by `isinstance(t, torch.Tensor) and t.is_cuda` (CPU tensors raise `NotImplementedError`). |
| `megatron/core/transformer/moe/token_dispatcher.py` | `_HybridEPManager.__init__` sets `self._buffer_key = id(self)`; threaded through `hybrid_ep_dispatch` / `hybrid_ep_combine` call sites. |
| `megatron/training/training.py` | `_RaceNoiseStreams` per-rank-seeded side-stream GEMM noise generator (mirrors the standalone `tests/unit_tests/determinism/utils.py:RacingStreams`). `train_step` becomes a thin wrapper: `with _maybe_race_noise(): return _train_step_inner(...)`. No-op unless `RACE_NOISE=1`. |

---

## What's verified

- **8-GPU EP=4 (NUM_LAYERS=8, NUM_EXPERTS=32, PP=2, VP=2, GBS=64):**
  jobs `2183359` (RACE_NOISE=0) and `2183360` (RACE_NOISE=1) both completed
  50 iterations **bit-identical** loss curves, with steady-state forward/
  backward overlap. The patch stack is sufficient at this scale.

## What's still open

- **16-GPU EP=8 (NUM_LAYERS=8, NUM_EXPERTS=32, GBS=256):** jobs `2188676`,
  `2188677` still hit the `-4 dim` crash. The `custom_allgather=False`
  fix is necessary but not sufficient — 26.04's new `intranode.cu` path
  has its own uninitialised-tensor bug at EP ≥ 8. This is the production
  blocker. Next step: bisect 26.02 → 26.04 DeepEP changes; the failing
  symbol surfaces in `intranode.cu`'s metadata exchange, not `internode.cu`.

---

## Reproducer (16-GPU EP=8 crash)

```bash
NUM_GPUS=16 PP_SIZE=2 VP_SIZE=2 EP_SIZE=8 TP_SIZE=1 \
NUM_LAYERS=8 NUM_EXPERTS=32 \
FLEX_BACKEND=hybridep GBS=256 \
DETERMINISTIC=true BACKEND=fused GPU=gb200 \
./run_deepseek_v3.sh
```

4× cheaper iteration than the 256-GPU production repro; same crash signature.

---

## Knobs (env vars consumed by `run_deepseek_v3.sh`)

| Var | Default | Effect |
|-----|---------|--------|
| `HYBRIDEP_SYNC` | `1` | `0` to disable the `torch.cuda.synchronize()` calls in `fused_a2a.py`. Use to test whether the cross-stream race fires at 256-GPU too, or is masked by NCCL barriers at scale. |
| `RACE_NOISE` | `0` | `1` activates `_RaceNoiseStreams` around `train_step`. Per-rank seed = `0xC0FFEE + rank`. Paired runs A/B see identical noise per rank → if loss curves diverge, the race is real. |
| `CUDA_LAUNCH_BLOCKING` | `0` | Leave at 0; setting to 1 only serialises everything and doesn't expose the real race. |
| `NUM_LAYERS`, `NUM_EXPERTS`, `PP_SIZE`, `VP_SIZE`, `EP_SIZE`, `TP_SIZE`, `NUM_GPUS`, `GBS`, `FLEX_BACKEND` | (PerfConfig prod) | Small-scale reproducer overrides. |
| `DETERMINISTIC` | `false` | `true` enables `model.deterministic_mode=true`, `NCCL_ALGO=Ring`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, etc. |

---

## Cluster migration replay

The submodule changes are pushed to a personal fork
(`git@github.com:ZhiyuLi-Nvidia/Megatron-LM.git`, branch
`zhiyul/per_manager_hybridep_fix`). The same changes are also bundled as
`patches/megatron-lm-dsv3-hybridep-determinism.patch` for cases where the
fork isn't reachable.

On the new cluster (fork-based, preferred):

```bash
# 1. clone main repo + checkout this branch
git clone -b zhiyul/deterministics_gb200_2604_dsv3 \
    git@github.com:NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge

# 2. init submodule (pulls v0.4.1 pin = d7288711b)
git submodule update --init --recursive

# 3a. (preferred) check out the fix branch from the personal fork:
git -C 3rdparty/Megatron-LM remote add fork \
    git@github.com:ZhiyuLi-Nvidia/Megatron-LM.git
git -C 3rdparty/Megatron-LM fetch fork zhiyul/per_manager_hybridep_fix
git -C 3rdparty/Megatron-LM checkout zhiyul/per_manager_hybridep_fix

# 3b. (fallback) apply the bundled patch instead:
# git -C 3rdparty/Megatron-LM apply ../../patches/megatron-lm-dsv3-hybridep-determinism.patch

# 4. confirm bind-mounts will be detected by run_deepseek_v3.sh
git -C 3rdparty/Megatron-LM diff --name-only d7288711b

# 5. update secrets.sh and PYTHON path at the top of run_deepseek_v3.sh
#    for the new cluster (HF_TOKEN, WANDB_API_KEY, container path).

# 6. small-scale repro to validate the new cluster:
NUM_GPUS=8 PP_SIZE=2 VP_SIZE=2 EP_SIZE=4 NUM_LAYERS=8 NUM_EXPERTS=32 \
FLEX_BACKEND=hybridep GBS=64 \
DETERMINISTIC=true BACKEND=fused GPU=gb200 \
./run_deepseek_v3.sh
```

The bind-mount detection in `run_deepseek_v3.sh` works regardless of
whether the submodule is on the fork branch (committed) or has the patch
applied to the working tree — both produce a non-empty
`git diff --name-only d7288711b`.

The bind-mount detection logic (lines 67–78 of `run_deepseek_v3.sh`) auto-
discovers any modified `.py` under `3rdparty/Megatron-LM/` and mounts it
over the container's editable install at
`/opt/Megatron-Bridge/3rdparty/Megatron-LM/...`.

---

## Files in this PR

- `run_deepseek_v3.sh` — master submission script with env knobs and
  bind-mount auto-detection.
- `patches/megatron-lm-dsv3-hybridep-determinism.patch` — combined
  Megatron-LM patch (per-manager buffer + custom_allgather=False + sync +
  RACE_NOISE wrapper), as fallback if the personal fork isn't reachable.
- `SUMMARY.md` — this doc.

Submodule branch (preferred migration source):

- `git@github.com:ZhiyuLi-Nvidia/Megatron-LM.git` branch
  `zhiyul/per_manager_hybridep_fix` — based on `d7288711b` with two commits:
  the per-manager HybridEP buffer fix and the
  `custom_allgather=False` + sync + `RACE_NOISE` follow-up.

The submodule pointer in this main-repo branch is intentionally **not**
advanced — bind-mount workflow is preserved, leaving the public submodule
hash (`d7288711b`) untouched in `.gitmodules`-recorded state.
