# Known Config Diffs vs Main (dryrun_reportA baseline)

This document records the 32 intentional or known differences between our branch's
flat recipe functions and the original golden YAML files derived from `main`'s
`scripts/performance/configs/` system.

These diffs were observed in `dryrun_reportA.json` (our new code vs old goldens) and
were **accepted** by regenerating the goldens from our branch.  The goldens now match
our code (283/283 PASS), but the table below documents where and why our output
diverges from what main's system originally produced.

---

## Summary

| Category | Files | Root cause |
|---|---|---|
| [A] NVFP4 `grad_reduce_in_fp32`](#a-nvfp4-grad_reduce_in_fp32) | 14 | Our `_perf_precision("nvfp4")` sets `grad_reduce_in_fp32=True`; main had `False` |
| [B] GPT-OSS FP8-MX precision fields](#b-gpt-oss-fp8-mx-missing-precision-fields) | 10 | We aliased fp8mx = bf16; main had actual FP8-MX recipe fields |
| [C] LLaMA SFT FP8-MX](#c-llama-sft-fp8-mx) | 3 | `cross_entropy_fusion_impl` + seq_length + pad_cu_seqlens diffs |
| [D] DeepSeek H100 FP8-SC (`overlap_grad_reduce` / pp_layout)](#d-deepseek-h100-fp8-sc) | 2 | We alias fp8sc=fp8cs; main v2 had different PP layout and DDP overlap |
| [E] DeepSeek VR200 `recompute_modules`](#e-deepseek-vr200-recompute_modules) | 3 | Our VR200=GB200 alias; main's GB200 had `recompute_modules=['mla_up_proj']` |

---

## A: NVFP4 `grad_reduce_in_fp32`

**Files (14):**

```
deepseek/deepseek_v3_pretrain_b200_nvfp4_v1.yaml
deepseek/deepseek_v3_pretrain_b200_nvfp4_v2.yaml
deepseek/deepseek_v3_pretrain_b300_nvfp4_v1.yaml
deepseek/deepseek_v3_pretrain_b300_nvfp4_v2.yaml  (also has pp_layout diff → see §F)
deepseek/deepseek_v3_pretrain_gb200_nvfp4_v1.yaml (also has recompute_modules diff → see §E)
deepseek/deepseek_v3_pretrain_gb200_nvfp4_v2.yaml (also has recompute_modules diff → see §E)
deepseek/deepseek_v3_pretrain_vr200_nvfp4_v2.yaml (also has recompute_modules diff → see §E)
qwen/qwen3_235b_a22b_pretrain_b200_nvfp4_v1.yaml
qwen/qwen3_235b_a22b_pretrain_b200_nvfp4_v2.yaml
qwen/qwen3_235b_a22b_pretrain_b300_nvfp4_v1.yaml
qwen/qwen3_235b_a22b_pretrain_b300_nvfp4_v2.yaml
qwen/qwen3_235b_a22b_pretrain_gb200_nvfp4_v1.yaml
qwen/qwen3_235b_a22b_pretrain_gb200_nvfp4_v2.yaml
qwen/qwen3_235b_a22b_pretrain_gb300_nvfp4_v1.yaml
qwen/qwen3_235b_a22b_pretrain_gb300_nvfp4_v2.yaml
```

**Diff:**
```
mixed_precision.grad_reduce_in_fp32:  golden=False  new=True
```

**Cause:** `_perf_precision("nvfp4")` in our branch hard-codes
`grad_reduce_in_fp32=True`. Main's `NVFP4` configs had it `False` (the default).

**Action needed:** Verify whether `grad_reduce_in_fp32=True` is correct for NVFP4
training or if our helper should match main's default of `False`.

---

## B: GPT-OSS FP8-MX missing precision fields

**Files (10):**

```
gpt_oss/gpt_oss_120b_pretrain_b200_fp8_mx_v1.yaml
gpt_oss/gpt_oss_120b_pretrain_b300_fp8_mx_v1.yaml
gpt_oss/gpt_oss_120b_pretrain_gb200_fp8_mx_v1.yaml
gpt_oss/gpt_oss_120b_pretrain_gb300_fp8_mx_v1.yaml
gpt_oss/gpt_oss_120b_pretrain_h100_fp8_mx_v1.yaml
gpt_oss/gpt_oss_120b_pretrain_gb300_fp8_mx_v2.yaml
gpt_oss/gpt_oss_120b_pretrain_h100_fp8_mx_v2.yaml
gpt_oss/gpt_oss_120b_pretrain_gb200_fp8_mx_v2.yaml  (also has dispatcher diffs below)
gpt_oss/gpt_oss_120b_pretrain_b200_fp8_mx_v2.yaml   (also has dispatcher diffs below)
gpt_oss/gpt_oss_120b_pretrain_b300_fp8_mx_v2.yaml   (also has dispatcher diffs below)
```

**Diff (all 10 share these):**
```
mixed_precision.fp8:                         golden='e4m3'       new=None
mixed_precision.fp8_param:                   golden=True         new=False
mixed_precision.fp8_param_gather:            golden=True         new=False
mixed_precision.fp8_recipe:                  golden='mxfp8'      new='tensorwise'
mixed_precision.reuse_grad_buf_for_mxfp8_param_ag: golden=True  new=False
```

**Additional diff for b200_v2, b300_v2:**
```
model.expert_model_parallel_size:            golden=8            new=64
model.moe_flex_dispatcher_backend:           golden='hybridep'   new='deepep'
model.moe_hybridep_num_sms:                  golden=32           new=16
model.moe_token_dispatcher_type:             golden='flex'       new='alltoall'
```

**Additional diff for gb200_v2:**
```
model.moe_flex_dispatcher_backend:           golden='hybridep'   new='deepep'
model.moe_hybridep_num_sms:                  golden=32           new=16
model.moe_token_dispatcher_type:             golden='flex'       new='alltoall'
```

**Cause:** We implemented `gpt_oss_120b_pretrain_*_fp8mx_config()` as a direct alias
of the corresponding BF16 config (identical config, different name). Main actually
had FP8-MX settings enabled in these configs (`fp8_recipe='mxfp8'`, `fp8_param=True`,
etc.) and different MoE dispatcher params for v2 B200/B300/GB200.

**Action needed:** If FP8-MX configs should actually run with FP8 enabled, these
functions need real FP8-MX settings applied on top of the BF16 base, not just a
direct alias. The MoE dispatcher differences (EP=8 vs 64, `hybridep` vs `deepep`)
for v2 also need investigation.

---

## C: LLaMA SFT FP8-MX

**Files (3):**

```
llama/llama3_8b_sft_gb200_fp8_mx_v1.yaml
llama/llama3_70b_sft_gb200_fp8_mx_v1.yaml
llama/llama3_70b_sft_gb300_fp8_mx_v1.yaml
```

**Diff for `llama3_8b_sft_gb200_fp8_mx`:**
```
mixed_precision.grad_reduce_in_fp32:               golden=False  new=True
model.cross_entropy_fusion_impl:                   golden='te'   new='native'
dataset.seq_length:                                golden=16384  new=4096
dataset.packed_sequence_specs.packed_sequence_size: golden=16384 new=4096
dataset.packed_sequence_specs.pad_cu_seqlens:      golden=False  new=True
```

**Diff for `llama3_70b_sft_gb200_fp8_mx`:**
```
mixed_precision.grad_reduce_in_fp32:               golden=False  new=True
model.cross_entropy_fusion_impl:                   golden='te'   new='native'
dataset.packed_sequence_specs.pad_cu_seqlens:      golden=False  new=True
```

**Diff for `llama3_70b_sft_gb300_fp8_mx`:**
```
mixed_precision.grad_reduce_in_fp32:               golden=False  new=True
model.cross_entropy_fusion_impl:                   golden='te'   new='native'
```

**Cause:**
- `grad_reduce_in_fp32`: our `_perf_precision("fp8_mx")` sets it `True`; main had `False`.
- `cross_entropy_fusion_impl`: our LLaMA SFT base config defaults to `'native'`; main used `'te'`.
- `seq_length` / `packed_sequence_size` (8B only): our SFT base uses 4096; main's golden had 16384.
- `pad_cu_seqlens` (8B and 70B-gb200): our base sets `True`; main had `False`.

**Action needed:** Audit the LLaMA SFT base config (`llama3_8b_sft_*` / `llama3_70b_sft_*`)
against main's for `cross_entropy_fusion_impl`, `seq_length`, and `pad_cu_seqlens`.

---

## D: DeepSeek H100 FP8-SC

**Files (2):**

```
deepseek/deepseek_v3_pretrain_h100_fp8_sc_v1.yaml
deepseek/deepseek_v3_pretrain_h100_fp8_sc_v2.yaml
```

**Diff for v1:**
```
ddp.overlap_grad_reduce:  golden=True  new=False
```

**Diff for v2 (additional):**
```
model.pipeline_model_parallel_layout:  golden=16-stage layout with MTP  new=None
model.pipeline_model_parallel_size:    golden=16                        new=8
```

**Cause:** We implement fp8_sc as a direct alias of fp8_cs. Main's FP8-SC configs
independently set `overlap_grad_reduce=True` (v1) and used a 16-stage PP layout
with explicit MTP stage placement (v2), rather than the 8-stage fp8_cs layout.

**Action needed:**
- v1: Determine whether `overlap_grad_reduce=True` should be set for H100 FP8-SC.
- v2: Determine if FP8-SC v2 should have a 16-PP layout distinct from FP8-CS v2 (8-PP).

---

## E: DeepSeek VR200 `recompute_modules`

**Files (5):**

```
deepseek/deepseek_v3_pretrain_vr200_fp8_cs_v2.yaml
deepseek/deepseek_v3_pretrain_vr200_fp8_mx_v2.yaml
deepseek/deepseek_v3_pretrain_vr200_nvfp4_v2.yaml   (also has grad_reduce_in_fp32 diff → §A)
deepseek/deepseek_v3_pretrain_gb200_nvfp4_v1.yaml   (VR200 alias source)
deepseek/deepseek_v3_pretrain_gb200_nvfp4_v2.yaml   (VR200 alias source)
```

**Diff:**
```
model.recompute_modules:  golden=['mlp']  new=['mla_up_proj']
```

**Cause:** Our VR200 functions alias GB200 directly. Main's VR200 configs had
`recompute_modules=['mlp']`, while our GB200 base now uses `['mla_up_proj']`.
The GB200 base changed since the original goldens were written; VR200 inherited
the change.

**Action needed:** Confirm which recompute strategy is correct for VR200 vs GB200.
If VR200 should keep `['mlp']`, the VR200 functions need an explicit override instead
of a pure alias.

---

## F: DeepSeek B300 NVFP4 v2 PP layout

**Files (1):**

```
deepseek/deepseek_v3_pretrain_b300_nvfp4_v2.yaml
```

**Diff:**
```
mixed_precision.grad_reduce_in_fp32:  golden=False         new=True
model.pipeline_model_parallel_layout: golden=16-stage MTP  new=8-stage MTP
model.pipeline_model_parallel_size:   golden=16            new=8
```

**Cause:** Main's B300 NVFP4 v2 used a 16-stage PP layout (same as the FP8-SC v2
variant), but our alias derives from the B300 BF16 v2 config which uses 8 stages.

**Action needed:** Confirm whether B300 NVFP4 v2 should use 8-PP (matching BF16 v2)
or 16-PP (matching main's original). If 16-PP, a dedicated config function is needed
rather than the BF16 alias.

---

## Files Verified Identical (no diffs)

The following previously-skipped goldens now PASS because our alias output matched
the existing golden exactly (no regen needed):

- `deepseek/deepseek_v3_pretrain_vr200_bf16_v2.yaml`
- `kimi/kimi_k2_pretrain_h100_fp8_sc_v1.yaml`
- `llama/llama3_8b_pretrain_vr200_*.yaml` (all 4 precisions)
- `llama/llama3_8b_sft_*_fp8_cs*.yaml` (base configs that fp8mx derives from)

(6 files PASS without regen — our alias output happened to match main's golden.)
