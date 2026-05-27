# DeepSeek-V3 Determinism Investigation

**Date:** 2026-03-19 → 2026-03-23
**Branch:** `zhiyul/deterministics_gb200_2602`
**Status:** Non-determinism reproduced at small scale (16 GPUs, PP=2, VP=2). Root cause unknown — VPP + HybridEP interaction. VP=2 with MoE is non-deterministic; VP=2 with dense layers is deterministic. New diagnostic: `seq_load_balancing_loss` diverges at step 6, pointing to router/forward-pass non-determinism.

---

## Summary

The 1-2 layer debug runs on 8-64 GPUs (PP=1, VP=None) are bit-reproducible.
The full 61-layer production run (PP=4, VP=4, 256 GPUs) is **not** reproducible.
Non-determinism has been reproduced at 16 GPUs (PP=2, VP=2, EP=8) with all production
factors active. Root cause traced to PP-rank-dependent gradient synchronization ordering
in `backward_step_helper_postprocess()` in the interleaved VP schedule.

---

## Environment

| Item | Value |
|------|-------|
| Container (fused attn) | `nemo-25.11-cudnn9.18.0.76.sqsh` |
| cuDNN | 9.18.0.76 |
| Megatron-LM submodule | `b0cc2706d` (25.04-alpha.rc1-1991-gb0cc2706d) |
| GPU | GB200, 4 per node |
| NVLINK_DOMAIN_SIZE | 72 |

### Deterministic-mode env vars (both runs)

```
NCCL_ALGO=Ring
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
NVTE_FUSED_ATTN=1 / NVTE_FLASH_ATTN=0 / NVTE_UNFUSED_ATTN=0
```

### Deterministic-mode model args (both runs)

```
model.deterministic_mode=true
model.cross_entropy_loss_fusion=false
comm_overlap.tp_comm_overlap=false
model.cuda_graph_impl=none        ← CUDA graphs conflict with deterministic gradient hooks
```

---

## Config Comparison: Small-Scale vs Full-Scale

| Parameter | Small-scale (DETERMINISTIC ✅) | Full-scale (NON-DETERMINISTIC ❌) |
|-----------|-------------------------------|----------------------------------|
| **Script** | `run_deepseek_v3_1layer.sh` | `run_deepseek_v3_full.sh` |
| **GPUs** | 8 (2 nodes) | 256 (64 nodes) |
| **TP** | 1 | 1 |
| **PP** | 1 | 4 |
| **VP** | None | 4 |
| **EP** | 8 | 64 |
| **Layers** | 1 or 2 | 61 |
| **first_k_dense_replace** | 0 (all MoE) | 1 (1 dense + 60 MoE) |
| **MBS** | 1 | 1 |
| **GBS** | 8 | 2048 |
| **GAS (=GBS/MBS)** | 8 | 2048 |
| **recompute_modules** | none | `["mla_up_proj"]` |
| **moe_flex_dispatcher_backend** | default | `hybridep` |
| **moe_a2a_overlap** | default | `False` |
| **force_load_balancing** | `true` (1-layer) / `false` (2-layer) | `false` |
| **overlap_grad_reduce** | NOT set (default false) | NOT set from script; **GB200 recipe sets `true`** |
| **cuda_graph_scope** | N/A (none) | base cfg: `["attn","moe_router","moe_preprocess"]`, overridden to `none` |

### Source of production defaults

```
scripts/performance/configs/deepseek/deepseek_workload_base_configs.py
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1:
    num_gpus=256, global_batch_size=2048
    pipeline_model_parallel_size=4
    virtual_pipeline_model_parallel_size=4
    expert_model_parallel_size=64
    moe_flex_dispatcher_backend="hybridep"
    moe_a2a_overlap=False
    recompute_modules=["mla_up_proj"]
    cuda_graph_impl="transformer_engine"   ← overridden to "none" by deterministic mode
    cuda_graph_scope=["attn","moe_router","moe_preprocess"]
```

---

## Local Repo Changes

### `src/megatron/bridge/models/qwen/qwen_provider.py`
Try-except wrapper for `get_transformer_block_with_experimental_attention_variant_spec` import (graceful degradation if not in this Megatron-LM version).

### `src/megatron/bridge/training/config.py`
Try-except wrapper for `ParamGroupOverride` / `ParamKey` imports from `megatron.core.optimizer`.

### `3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`
Added dynamic detection of `non_blocking` parameter support in `HybridEPBuffer.dispatch_with_permute()`:
- Checks via `inspect` whether the installed DeepEP has `non_blocking`
- Falls back gracefully if not supported
- Extended exception handling from `ImportError` to also catch `ValueError`/`TypeError`

---

## Root-Cause Hypotheses (ordered by likelihood)

### H1 — HybridEP `combine_with_unpermute` backward (HIGH)
`HybridEPCombine.backward` in `fused_a2a.py` calls `dispatch_with_permute(hidden=grad_x, ...)` as its backward.
This SM-based scatter-permute likely uses non-deterministic atomics internally.
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` controls TransformerEngine, **not** DeepEP's internal CUDA kernels.
**Key file:** `3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py` ~line 435-447

### H2 — `overlap_grad_reduce=True` + GAS=2048 (HIGH)
The GB200 recipe sets `overlap_grad_reduce=True` (in `deepseek_llm_pretrain.py`), but the 1-layer debug script does **not** set this.
With 2048 gradient accumulation steps, DDP bucket-reduce timing relative to backward pass can differ between identical runs.

### H3 — `mla_up_proj` activation recompute (MEDIUM)
`CheckpointWithoutOutput.discard_output_and_register_recompute()` registers a backward hook that re-runs `qkv_up_proj_and_rope_apply`.
If the RNG tracker is not properly re-seeded during recompute, small numerical differences accumulate.
**Key file:** `3rdparty/Megatron-LM/megatron/core/transformer/multi_latent_attention.py` ~lines 838-843

### H4 — PP=4 / VP=4 interleaved schedule (MEDIUM)
The interleaved 1F1B schedule with VP=4 changes the ordering of AllReduce collectives relative to computation. Even with `NCCL_ALGO=Ring`, the scheduling of collectives across VP chunks may differ.

### H5 — Dense layer + `gradient_accumulation_fusion` (LOW)
The first dense layer (first_k_dense_replace=1) uses `gradient_accumulation_fusion=True` by default, which fuses wgrad accumulation with the scatter-reduce. This has known non-determinism potential.

### H6 — NCCL collective ordering at scale (LOW)
64-node NCCL collectives may exhibit non-deterministic ring ordering under timing pressure not present in 2-node runs.

---

## Experiment Plan

### How to Measure Determinism
Run the **same job twice** (same seed, same node list). Compare loss and grad-norm at each step.
With `logger.log_interval=1` and `logger.throughput_window_size=1`, each step is logged.
A single bit-level mismatch in loss confirms non-determinism.

### All experiments use:
- `DETERMINISTIC=true`, fused attention, all deterministic env vars from above
- Container: `nemo-25.11-cudnn9.18.0.76.sqsh`
- `ACCOUNT=coreai_dlalgo_llm`, `PARTITION=batch`, `GPUS_PER_NODE=4`
- Full Megatron-LM mount: `3rdparty/Megatron-LM:/opt/megatron-lm`

---

### EXP-1: HybridEP dispatcher at small scale (tests H1)

**Hypothesis:** HybridEP combine backward has non-det. atomics.

**Base:** 1-layer baseline (currently deterministic)
**Add:** `moe_flex_dispatcher_backend=hybridep` (and ensure `moe_token_dispatcher_type=flex`)
**GPUs:** 64, PP=1, VP=None, EP=64, GAS=1, 1 layer
**Time:** ~20 min

```bash
# Additional args to add on top of 1-layer deterministic run:
model.moe_token_dispatcher_type=flex
model.moe_flex_dispatcher_backend=hybridep
```

**Expected:** If non-det → HybridEP is root cause. Audit `combine_with_unpermute` backward atomics.
**If det →** rule out H1.

---

### EXP-2: `overlap_grad_reduce` + large GAS at small scale (tests H2)

**Hypothesis:** GAS + overlapped grad reduce causes non-det. DDP bucket ordering.

**Base:** 1-layer baseline
**Add:** GBS=256 (→ GAS=256 with DP=1), `comm_overlap.overlap_grad_reduce=true`
**GPUs:** 64, PP=1, VP=None, EP=64
**Time:** ~20 min

```bash
model.global_batch_size=256
comm_overlap.overlap_grad_reduce=true
```

**Expected:** If non-det → overlap_grad_reduce + GAS is the cause.
**If det →** rule out H2.

---

### EXP-3: `mla_up_proj` recompute at small scale (tests H3)

**Hypothesis:** Activation recompute with `CheckpointWithoutOutput` breaks RNG determinism.

**Base:** 1-layer baseline
**Add:** selective recompute of mla_up_proj
**GPUs:** 64, PP=1, VP=None, EP=64, GAS=1, 1 layer
**Time:** ~20 min

```bash
model.recompute_granularity=selective
"model.recompute_modules=[mla_up_proj]"
```

**Expected:** If non-det → recompute RNG state mishandling. Check RNG bracketing in `CheckpointWithoutOutput`.
**If det →** rule out H3.

---

### EXP-4: PP=4, VP=4, small GAS, no HybridEP, no recompute (tests H4)

**Hypothesis:** VP=4 interleaved schedule itself introduces ordering non-determinism.

**Base:** `run_deepseek_v3_full.sh` with overrides
**Changes:** 8 layers, GAS=1 (GBS=1), no recompute, alltoall dispatcher
**GPUs:** 256, PP=4, VP=4, EP=64
**Time:** ~30 min

```bash
DETERMINISTIC=true ./run_deepseek_v3_full.sh \
    model.num_layers=8 \
    model.global_batch_size=1 \
    model.first_k_dense_replace=0 \
    "model.recompute_modules=[]" \
    model.moe_token_dispatcher_type=alltoall \
    model.cuda_graph_impl=none
```

**Expected:** If non-det → VP=4 schedule is the cause.
**If det →** rule out H4.

---

### EXP-5: Dense layer interaction (tests H5)

**Hypothesis:** Dense layer `gradient_accumulation_fusion` introduces non-det. wgrad scatter-reduce.

**Base:** 1-layer baseline
**Add:** 2 layers, `first_k_dense_replace=1`
**GPUs:** 8, PP=1, VP=None, EP=8, GAS=1
**Time:** ~20 min

```bash
NUM_GPUS=8 DETERMINISTIC=true ./run_deepseek_v3_1layer.sh \
    --num_layers 2 \
    --first_k_dense_replace 1
```

**Expected:** If non-det → dense layer fusion is the cause.
**If det →** rule out H5.

---

### EXP-6: Full-scale with `overlap_grad_reduce=false` (confirmation)

**Hypothesis:** Disabling `overlap_grad_reduce` makes full-scale deterministic.

**Base:** `run_deepseek_v3_full.sh` with `DETERMINISTIC=true` (current broken state)
**Change:** add `comm_overlap.overlap_grad_reduce=false`
**GPUs:** 256
**Time:** ~45 min

```bash
DETERMINISTIC=true ./run_deepseek_v3_full.sh \
    comm_overlap.overlap_grad_reduce=false
```

**Expected:** If now deterministic → confirmed `overlap_grad_reduce` + PP/GAS interaction is root cause.

---

### Decision Tree

```
EXP-1 non-det?  YES → HybridEP combine backward → audit dispatch_with_permute atomics in DeepEP
                NO  ↓
EXP-2 non-det?  YES → overlap_grad_reduce + GAS → disable in deterministic mode
                NO  ↓
EXP-3 non-det?  YES → mla_up_proj CheckpointWithoutOutput → fix RNG bracketing in recompute
                NO  ↓
EXP-4 non-det?  YES → VP=4 schedule itself → inspect interleaved_pipeline_schedule flush ordering
                NO  ↓
EXP-5 non-det?  YES → dense layer gradient_accumulation_fusion → disable in det. mode
                NO  ↓
EXP-6 det.?     YES → confirmed overlap_grad_reduce + PP/VP/GAS combo → that's the fix
                NO  → multi-factor; run EXP-1+EXP-2 combined
```

---

## Recommended Order of Execution

Run EXP-1, EXP-2, EXP-3, EXP-5 in parallel (all small-scale, independent).
Based on results, run EXP-4 and/or EXP-6 as needed.

**Start with EXP-1 and EXP-2** — highest probability hypotheses and cheapest to test.

---

## Key Files

| File | Relevance |
|------|-----------|
| `scripts/performance/configs/deepseek/deepseek_workload_base_configs.py` | Production defaults (PP=4, VP=4, EP=64, recompute, hybridep) |
| `3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py` | HybridEP dispatch/combine backward (H1) |
| `3rdparty/Megatron-LM/megatron/core/transformer/multi_latent_attention.py` | `recompute_up_proj` + `CheckpointWithoutOutput` (H3) |
| `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py` | **ROOT CAUSE**: VP grad-sync ordering (lines 1257–1291) |
| `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py` | Combined 1F1B + MoE overlap schedule |
| `src/megatron/bridge/training/config.py` | Deterministic mode validation + env enforcement |
| `run_deepseek_v3_1layer.sh` | Base script for EXP-1/2/3/5 |
| `run_deepseek_v3_full.sh` | Base script for EXP-4/6 |

---

## Experiment Results (2026-03-21)

| Exp | Config | GPUs | Result |
|-----|--------|------|--------|
| exp1 | HybridEP, PP=1, VP=None | 8 | DETERMINISTIC ✅ |
| exp2 | overlap_grad_reduce + GAS=256, PP=1, VP=None | 8 | DETERMINISTIC ✅ |
| exp3 | mla_up_proj recompute, PP=1, VP=None | 8 | DETERMINISTIC ✅ |
| exp4 | PP=4, VP=4, alltoall, no recompute | 256 | DETERMINISTIC ✅ |
| combo_abc (PP=2, VP=None) | All 3 factors, PP=2, VP=None | 8 | DETERMINISTIC ✅ |
| **combo_abc_pp2 (PP=2, VP=2)** | **All 3 factors + VPP** | **16** | **NON-DETERMINISTIC ❌** |

**Key observation:** Divergence starts at **iteration 6**, iterations 1–5 are bit-exact.
This precisely matches the VP schedule transition from warmup (forward-only) to steady-state 1F1B.

---

## Root Cause: UNKNOWN — Hypothesis Invalidated

### Falsified Hypothesis: VP + grad_sync_func rank offset causes non-det AllReduce ordering

`backward_step_helper_postprocess()` in `schedules.py` applies a PP-rank-dependent offset:

```python
grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
```

The hypothesis was that different PP ranks firing AllReduce at different virtual_microbatch_ids
creates non-deterministic AllReduce launch ordering. **This hypothesis is WRONG.**

**Proof — combo_abc_pp2 topology**: 16 GPUs, PP=2, VP=2, EP=8, TP=1
- `DP = 16 / (PP × EP × TP) = 16 / 16 = 1`
- `dp_cp_group.size() = 1` → `gradient_scaling_factor = 1.0` (no scaling applied)
- `start_grad_sync()` issues `dist.all_reduce(group=1-rank-group)` → **trivially a no-op**
- The entire grad_sync_func path produces **no numerical change whatsoever**
- The rank offset timing is irrelevant — there is no actual AllReduce to order

The `torch.cuda.synchronize()` fix failed for the same reason: it targeted AllReduce ordering,
but there is no AllReduce.

### Why VP alone is Deterministic (exp4 was DETERMINISTIC with VP=4)

**exp4**: PP=4, VP=4, alltoall dispatcher, **no `overlap_grad_reduce`** → `grad_sync_func=None`
→ the rank-offset branch in `backward_step_helper_postprocess()` is never entered.

This observation is still valid, but the significance was misinterpreted: the branch being
skipped doesn't cause determinism because the branch is a no-op with DP=1 anyway.

The grad_sync_func path is activated by `overlap_grad_reduce=True` + `align_grad_reduce=True`
(see `setup.py` lines 333–344), but with DP=1 it does nothing numerically.

### Why Iterations 1–5 are Deterministic

With PP=2, VP=2, GAS=256, `microbatch_group_size_per_vp_stage = 128`:
- Warmup phase (forward-only): ~130 microbatches per training step
- First backward fires around microbatch 130 of step 1
- Steps 1–5 (microbatches 0–1279) complete without divergence
- Step 6 (microbatch 1280+) is where divergence first appears in the loss log

The divergence timing pattern is consistent with first backward pass in steady-state 1F1B,
but does NOT uniquely point to the grad_sync_func path.

### True Root Cause: Likely VP=2 Warmup Window Exposing DeepEP Non-Determinism

#### Step 1: VP=2 is confirmed to be the ONLY changing parameter

| Config | VP | overlap_grad_reduce | hybridep | recompute | Result |
|--------|----|--------------------|----------|-----------|--------|
| exp1 | None | False | ✅ | — | DET ✅ |
| exp2 | None | ✅ | — | — | DET ✅ |
| exp3 | None | — | — | ✅ | DET ✅ |
| exp4 | 4 | — | — | — | DET ✅ |
| combo_abc | **None** | ✅ | ✅ | ✅ | DET ✅ |
| **combo_abc_pp2** | **2** | ✅ | ✅ | ✅ | **NON-DET ❌** |

combo_abc → combo_abc_pp2 changes ONLY `virtual_pipeline_model_parallel_size=2`.

#### Step 2: VP=2 changes warmup window size dramatically

From `get_num_microbatches_and_warmup()` in schedules.py:
```
VP=None, PP=2, rank 0:  num_warmup = (PP - rank - 1) = 1 forward-only mb
VP=2,    PP=2, rank 0:  num_warmup = (PP - rank - 1)*2 + (VP-1)*microbatch_group_size
                       = 2 + 1 * 128 = 130 forward-only mbs
```

With VP=2: **130 consecutive MoE forward passes run before any backward fires**.
With VP=None: **1 forward pass before backward** (no meaningful window).

#### Step 3: VP=2 is deterministic in dense models but not MoE

Dense model: all operations are matmuls, covered by `torch.use_deterministic_algorithms(True)`.

HybridEP / DeepEP: custom CUDA extension (`buffer.dispatch()`, `buffer.combine()`).
**`torch.use_deterministic_algorithms(True)` has NO effect on third-party CUDA extensions.**
If DeepEP kernels use atomics or non-deterministic CUDA operations internally, they are
completely unaffected by PyTorch's determinism flag.

Supporting evidence:
- exp4 used VP=4 + **alltoall** → DETERMINISTIC. Alltoall uses `torch.distributed.all_to_all_single`
  which is a PyTorch collective, fully covered by the determinism flag.
- HybridEP uses DeepEP's custom NVLink/RDMA kernels — NOT covered.

#### Step 4: The window expansion mechanism

**VP=None** (1 warmup mb): if DeepEP has latent non-det in dispatch/combine, the backward
for that 1 microbatch fires almost immediately after its forward. The non-det is isolated
to 1 microbatch's gradient and is too small to cause visible loss divergence.

**VP=2** (130 warmup mbs): 130 forward passes with potentially non-det DeepEP dispatch run
before ANY backward. Activations from all 130 forwards are stored and used in subsequent
backward passes. Non-det in forward activations compounds over 130 layers before correction.

**This explains the divergence pattern at iteration 6**: the first backward fires around
warmup microbatch 130. With log_interval=1 and GAS=256, the first logged step with
backward computation is step 1. But the divergence appears at step 6 — consistent with
numerical differences from non-det MoE dispatch accumulating over steps 1–5 until they
exceed float precision and diverge.

### New Diagnostic: `seq_load_balancing_loss` Aligns with Divergence

**Observation (2026-03-23):** `seq_load_balancing_loss` diverges between runs at exactly the same
iteration (step 6) as `loss` and `grad_norm`.

**What this tells us:**

`seq_load_balancing_loss` is computed in the router's **forward pass** as:
```python
# router.py _apply_seq_aux_loss()
global_tokens_per_expert = routing_map.sum(dim=0)  # no cross-rank reduction with TP=1
aux_loss = switch_load_balancing_loss_func(scores_for_aux_loss, global_tokens_per_expert, ...)
```

For `seq_load_balancing_loss` to differ between runs, the **routing_map** at step 6 must differ.
The routing_map depends on:
1. `logits = router_gating_linear(input, weight)` — `input` comes from `combine_with_unpermute` of
   the preceding MoE layer, `weight` is the router parameter (updated by optimizer)
2. `scores_for_routing = sigmoid(logits) + expert_bias` — `expert_bias` is updated each step from `local_tokens_per_expert`

**Therefore:** the **hidden states flowing into the router** are non-deterministic by step 6.

**Critically: `expert_bias` does NOT affect `seq_load_balancing_loss`.**

`seq_load_balancing_loss` is computed via `compute_routing_scores_for_aux_loss(logits, ...)`,
which uses only `logits` — **no expert_bias**. So the divergence is purely from different
`logits`, which means different `input` hidden states coming from the preceding MoE layer's
`combine_with_unpermute`.

**This is direct evidence that HybridEP `combine_with_unpermute` forward output is
non-deterministic with VP=2.** No further disambiguation experiment is needed for this point.

### Does VPP Change `seq_load_balancing_loss` or Router Behavior?

**Direct effect: NO.** `seq_load_balancing_loss` uses the same formula regardless of VP.
With identical hidden states and weights, it computes identically.

The divergence at step 6 is caused by **different hidden states** reaching the router, not by
any VPP-specific change to the loss formula.

### Root Cause: HybridEP `combine_with_unpermute` Forward Non-Determinism

The `combine_with_unpermute` kernel aggregates expert outputs from multiple GPUs (topk=8 for
DS-V3) via weighted sum. This CUDA reduction kernel is a custom DeepEP extension — **not
covered by `torch.use_deterministic_algorithms(True)`**.

The most likely mechanism: **non-deterministic parallel floating-point reduction** in the
combine kernel. With 8 experts contributing to each token, if the CUDA warp scheduling
produces different summation orders between runs, BF16 non-associativity causes different
results. The differences are sub-BF16-epsilon in steps 1-5 (not visible in logged metrics)
but compound into the weight/hidden-state trajectory until step 6 where routing decisions
visibly change.

**Why VP=None is deterministic but VP=2 is not:** unclear from static analysis alone.
Possible mechanisms:
- `_hybrid_ep_buffer` global singleton has internal stream/event state that differs between
  VP chunks' interleaved uses (2 chunks share 1 buffer with VP=2; VP=None uses it exclusively)
- VP=2 interleaved backward creates a different memory/stream ordering context for the buffer
  that triggers a different CUDA reduction tree path

### Next Experiments Needed

To isolate the HybridEP non-determinism mechanism:

| Experiment | Config change vs combo_abc_pp2 | Tests |
|---|---|---|
| vp_per_layer_buffer | make `_hybrid_ep_buffer` per-layer (not global singleton) | Eliminates buffer sharing between VP chunks |
| vp_no_recompute | remove `mla_up_proj` from recompute_modules | Is VP+recompute a factor? |
| vp_no_overlap | `comm_overlap.overlap_grad_reduce=false` | Is VP+no_sync toggling a factor? |

**Simplest direct test:** add `torch.cuda.synchronize()` before each `dispatch_with_permute`
call in `HybridEPDispatch.forward` and `HybridEPCombine.backward` when in deterministic mode.
If this eliminates non-determinism, it confirms a CUDA stream ordering issue between VP chunks
sharing the buffer.

Run vp_no_recompute and vp_no_overlap in parallel to rule out other factors.

### Fix Verification: Is the current (reverted) change picked up?

**Yes.** The NeMo container has Megatron-LM installed as an **editable install** at `/opt/megatron-lm`.
The `run_experiments.sh` bind-mounts `3rdparty/Megatron-LM:/opt/megatron-lm` (line 34):
```bash
CUSTOM_MOUNTS=",$WORKDIR/$MEGATRON_DIR:/opt/megatron-lm"
```
Python's editable install resolves imports from the mounted directory.
`GitArchivePackager(include_submodules=False)` in `executors.py` does NOT package the submodule —
the bind-mount is the sole delivery mechanism.

---

## Notes

- `model.cuda_graph_impl=none` is **mandatory** in deterministic mode: CUDA graphs conflict with gradient hooks used by deterministic-mode enforcement.
- `cudnn 9.18.0.76` container is required for deterministic fused attention on GB200 (standard 26.02 container does not support it).
- The `3rdparty/Megatron-LM` submodule must be mounted (not diff-patched) because the diff-based approach yields nothing when `BASE_COMMIT==HEAD`.
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` controls TransformerEngine ops only; it does **not** control DeepEP internal CUDA kernels.
- Env vars (`NCCL_ALGO`, `NVTE_*`, `CUBLAS_WORKSPACE_CONFIG`) must be forwarded into the container via `-ce` flag — outer-shell `export` is NOT inherited by nemo_run's sbatch generation.
