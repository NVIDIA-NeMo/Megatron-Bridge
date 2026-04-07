# Experiment Results — Nemotron Diffusion 3B

> Auto-updated by the agent after each evaluation. Single source of truth for comparing experiments.

## Leaderboard

| Rank | Experiment | Branch | Train Steps | Inference | GSM8k Strict | GSM8k Flex | MBPP | MBPP+ | Avg | Delta vs Baseline | Status | Date |
|------|-----------|--------|------------|-----------|-------------|------------|------|-------|-----|-------------------|--------|------|
| 0 | **baseline** | `main` | 5k | dLLM (NFE=256/512, block=32) | 79.98% | 80.44% | 55.00% | 72.49% | 71.98% | — | ✅ | 2026-04-06 |
| 1 | mask_schedule | `autoresearch-mask-scheduling` | 5k | dLLM (NFE=256/512, block=32) | 81.27% | 81.80% | 55.20% | 70.90% | 72.29% | +0.31% | ❌ | 2026-04-06 |
| 2 | gentle_mask | `autoresearch-gentle-mask-schedule` | 5k | dLLM (NFE=256/512, block=32) | 77.56% | 78.77% | 55.60% | 70.11% | 70.51% | -1.47% | ❌ | 2026-04-07 |

> **Avg** = mean of (GSM8k Strict, GSM8k Flex, MBPP, MBPP+). An experiment is **positive** if Avg exceeds baseline Avg (71.98%) by ~0.5% (i.e., Avg >= ~72.5%).

## Inference Config Reference
Default eval settings from `eval_megatron.sh`:
- **Mode:** dLLM, block_length=32, temperature=0.0
- **NFE (diffusion_steps):** equals max_new_tokens per task — GSM8k: 256, MBPP/MBPP+: 512
- **Shots:** GSM8k: 8, MBPP/MBPP+: 3

## Key Takeaways

_Updated as experiments complete. Captures high-level learnings to guide future ideas._

- **What works:** Mask scheduling improves GSM8k (+1.3%) — curriculum helps reasoning tasks.
- **What doesn't work:** Mask scheduling hurts both MBPP+ and (with gentler settings) GSM8k. Neither aggressive (min=0.1, warmup=2500) nor gentle (min=0.3, warmup=1000) improves overall Avg.
- **Surprising findings:** The aggressive schedule (#2) helped GSM8k but the gentle one (#5) hurt it. Mask scheduling is unreliable.

## Detailed Results

### baseline
- **Branch:** `main`
- **Checkpoint:** `ministral_3b/iter_0005000` (5k training steps)
- **Inference:** dLLM, block_length=32, temp=0.0, NFE=max_new_tokens
- **Metrics:**
  - GSM8k (8-shot CoT, strict): **79.98%**
  - GSM8k (8-shot CoT, flexible): **80.44%**
  - MBPP (pass@1): **55.00%**
  - MBPP+ (pass@1): **72.49%**
  - **Avg: 71.98%**
- **Notes:** Default settings, no modifications.

---

### mask_schedule
- **Branch:** `autoresearch-mask-scheduling`
- **Checkpoint:** `ministral_3b_v2/iter_0005000` (5k training steps)
- **Inference:** dLLM, block_length=32, temp=0.0, NFE=max_new_tokens
- **Change:** Linear mask ratio warmup from 0.1→1.0 over first 2500 iterations
- **Metrics:**
  - GSM8k (8-shot CoT, strict): **81.27%** (+1.29%)
  - GSM8k (8-shot CoT, flexible): **81.80%** (+1.36%)
  - MBPP (pass@1): **55.20%** (+0.20%)
  - MBPP+ (pass@1): **70.90%** (-1.59%)
  - **Avg: 72.29%** (+0.31%)
- **Notes:** GSM8k improved but MBPP+ regressed. Net Avg below 0.5% threshold. A gentler warmup (min_ratio=0.3 or shorter warmup) may preserve GSM8k gains without hurting code.

---

### gentle_mask
- **Branch:** `autoresearch-gentle-mask-schedule`
- **Checkpoint:** `ministral_3b_gentle_mask_v1/iter_0005000` (5k training steps)
- **Inference:** dLLM, block_length=32, temp=0.0, NFE=max_new_tokens
- **Change:** Linear mask ratio warmup from 0.3→1.0 over first 1000 iterations
- **Metrics:**
  - GSM8k (8-shot CoT, strict): **77.56%** (-2.42%)
  - GSM8k (8-shot CoT, flexible): **78.77%** (-1.67%)
  - MBPP (pass@1): **55.60%** (+0.60%)
  - MBPP+ (pass@1): **70.11%** (-2.38%)
  - **Avg: 70.51%** (-1.47%)
- **Notes:** Worse than baseline across the board. Mask scheduling direction abandoned.

---
_New experiment results are appended below by the agent._

