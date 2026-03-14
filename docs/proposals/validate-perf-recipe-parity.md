# Perf Recipe Parity Validation Report

> **Date:** 2026-03-14  
> **Cluster:** cw-dfw-cs-001 (tmux session `aa`)  
> **Test script:** `tests/unit_tests/recipes/test_all_perf_equivalence.py`

## Summary

Verified equivalence between the **old perf pipeline** (`scripts/performance/configs/`) and
the **new flat perf recipes** (`src/megatron/bridge/recipes/<family>/<model>_perf.py`)
across all 8 model families.

| | Count |
|---|---|
| Total recipes tested | **255** |
| **PASSED** (old == new) | **107** |
| No old config exists (new-only recipes) | 10 |
| Import error (qwen_vl base recipe missing) | 22 |
| Config mismatch (real diff) | 116 |

---

## Category 1: PASSED (107)

These produce **identical** `ConfigContainer` objects via both paths.

### Llama3 pretrain v1 — 40/40

All 40 Llama3 8B + 70B pretrain recipes across r100/gb300/gb200/b300/b200/h100 × bf16/fp8cs/fp8mx/nvfp4.

### Llama3 pretrain v2 — 18/18

All 18 Llama3 70B pretrain v2 (GBS=256) recipes.

### Llama3.1 405B pretrain v1 — subset pass

gb200 bf16, gb200 fp8mx, b200 bf16/fp8cs/fp8mx, b300 bf16/fp8cs/fp8mx, h100 bf16/fp8cs.

### Llama3.1 405B pretrain v2 — subset pass

gb200 bf16/fp8cs/fp8mx, gb300 fp8cs/fp8mx, h100 bf16/fp8cs.

### DeepSeek V3 — v1 b200/b300 bf16/fp8cs/fp8mx pass

12 configs across b200/b300 × bf16/fp8cs/fp8mx for both v1 and v2.

### Qwen3 MoE — all v1 pass

All 28 Qwen3 235B-A22B + 30B-A3B v1 pretrain configs pass.

### NemotronH — all pass

All 19 NemotronH 56B + Nemotron 3 Nano configs pass.

### Llama3 SFT — h100 bf16 passes

1 config passes.

### Llama3 LoRA — h100 bf16 passes

1 config passes.

---

## Category 2: No old config exists (10)

New flat recipes that have **no corresponding old-path WorkloadBaseConfig**. These are
intentionally new and should be skipped in equivalence tests.

| Recipe | Missing old config |
|---|---|
| `deepseek_v3_pretrain_*_b200_nvfp4` (v1+v2) | `DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4` |
| `deepseek_v3_pretrain_*_b300_nvfp4` (v1+v2) | `DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4` |
| `deepseek_v3_pretrain_*_gb200_nvfp4` (v1+v2) | `DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4` |
| `deepseek_v3_pretrain_*_h100_fp8mx` (v1+v2) | `DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_MX` |
| `deepseek_v3_pretrain_*_h100_nvfp4` (v1+v2) | `DEEPSEEK_V3_PRETRAIN_CONFIG_H100_NVFP4` |

**Action:** No fix needed — these are new recipes. Mark as expected-skip in tests.

---

## Category 3: Import error — Qwen3-VL (22)

All 22 Qwen3-VL recipes fail at import time:

```
cannot import name 'qwen3_vl_30b_a3b_pretrain_config' from
'megatron.bridge.recipes.qwen_vl.qwen3_vl'
```

The `qwen3_vl_perf.py` file imports `qwen3_vl_30b_a3b_pretrain_config` from the base
recipe module, but this function does not exist on the cluster's copy of
`recipes/qwen_vl/qwen3_vl.py`.

**Action:** Either the base recipe hasn't been created yet, or `qwen3_vl_perf.py`
should import from a different module. Need to add `qwen3_vl_30b_a3b_pretrain_config`
to `recipes/qwen_vl/qwen3_vl.py` (or fix the import path).

---

## Category 4: Config mismatch (116)

Real field-level differences between old and new paths. Grouped by pattern:

### 4a. Llama3.1 405B — 10 failures

Affected: gb300 bf16/fp8cs/nvfp4, gb200 fp8cs/nvfp4, b200 nvfp4, b300 nvfp4,
v2 gb300 bf16/nvfp4, v2 gb200 nvfp4.

Common diff fields (from the log):
- `model.use_te_rng_tracker`
- `ddp.grad_reduce_in_fp32` / `mixed_precision.grad_reduce_in_fp32`
- Some nvfp4 configs have additional parallelism diffs

### 4b. Llama3 SFT — 8 failures

All SFT configs except h100_bf16 fail. Likely missing SFT-specific overrides
in the new flat recipes (sequence packing, dataset config, etc.).

### 4c. Llama3 LoRA — 13 failures

All LoRA configs except h100_bf16 fail. Similar pattern to SFT.

### 4d. DeepSeek V3 — 22 failures (excl. 10 missing)

gb200 and gb300 bf16/fp8cs/fp8mx, h100 bf16/fp8cs, plus all v2 variants.

Common diff fields:
- `model.use_te_rng_tracker`
- `ddp.grad_reduce_in_fp32` / `mixed_precision.grad_reduce_in_fp32`

### 4e. Qwen3 MoE v2 — 12 failures

All Qwen3 235B-A22B v2 configs fail. V1 configs pass.

### 4f. Kimi K2 — 12 failures

All Kimi K2 configs fail. Common diffs:
- `model.cuda_graph_scope`: old=`[]` vs new=`'full'`
- `model.use_te_rng_tracker`: old varies vs new differs
- `ddp.grad_reduce_in_fp32`: old=`False` vs new=`True`
- `mixed_precision.grad_reduce_in_fp32`: old=`False` vs new=`True`

### 4g. GPT-OSS 120B — 10 failures

All GPT-OSS configs fail. Common diffs:
- `model.recompute_granularity`: old=`'selective'` vs new=`None`
- `comm_overlap`: old=`None` vs new=`CommOverlapConfig(tp_comm_overlap=False, ...)`
- `model.use_te_rng_tracker` on some configs

---

## Root Cause Analysis

The 116 config mismatches share a few recurring patterns:

### Pattern A: `grad_reduce_in_fp32` and `mixed_precision.grad_reduce_in_fp32`

`_benchmark_common()` sets both to `False`, but some old-path recipes don't set them
(leaving them at default `True`), or some new recipes inherit a different default
from the library recipe.

**Fix:** Audit each family's old-path recipe to check if `grad_reduce_in_fp32=False`
was applied. If the old path left it at default, the new recipe should too (remove
the override from `_benchmark_common` or apply it conditionally).

### Pattern B: `use_te_rng_tracker`

New flat recipes inherit `use_te_rng_tracker` from the library recipe, but the old
path sometimes overrides it. Mismatch direction varies by family.

**Fix:** Explicitly set `use_te_rng_tracker` in each flat recipe to match old path.

### Pattern C: `cuda_graph_scope` (Kimi)

Old path sets `cuda_graph_scope=[]` (empty list), new path sets `'full'`.

**Fix:** Match old-path value in new recipe.

### Pattern D: `comm_overlap` (GPT-OSS)

Old path doesn't set `comm_overlap` (remains `None`), new path creates
`CommOverlapConfig(tp_comm_overlap=False)`.

**Fix:** Don't set `comm_overlap` in GPT-OSS flat recipes if old path doesn't.

### Pattern E: `recompute_granularity` (GPT-OSS)

Old path sets `recompute_granularity='selective'`, new path leaves it at `None`.

**Fix:** Add `cfg.model.recompute_granularity = 'selective'` to GPT-OSS flat recipes.

---

## Next Steps

1. **Fix `_benchmark_common()`** — `grad_reduce_in_fp32=False` should not be applied
   universally. Move it to per-recipe overrides where the old path explicitly set it.

2. **Fix per-family mismatches** — Apply the specific field fixes from Pattern B-E
   above for each family.

3. **Add `qwen3_vl_30b_a3b_pretrain_config`** base recipe (or fix import path).

4. **Mark 10 new-only DeepSeek configs** as expected-skip in the test.

5. **Re-run** full equivalence test after fixes to reach 255/255 (or 245/245 excl.
   10 skips).
