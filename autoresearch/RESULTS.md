# Experiment Results — Nemotron Diffusion 3B

> Auto-updated by the agent after each evaluation. Single source of truth for comparing experiments.

## Leaderboard

| Rank | Experiment | Branch | Train Steps | Inference | GSM8k Strict | GSM8k Flex | MBPP | MBPP+ | Avg | Delta vs Baseline | Status | Date |
|------|-----------|--------|------------|-----------|-------------|------------|------|-------|-----|-------------------|--------|------|
| 0 | **baseline** | `main` | 5k | dLLM (NFE=256/512, block=32) | 79.98% | 80.44% | 55.00% | 72.49% | 71.98% | — | ✅ | 2026-04-06 |

> **Avg** = mean of (GSM8k Strict, GSM8k Flex, MBPP, MBPP+). An experiment is **positive** if Avg exceeds baseline Avg (71.98%) by ~0.5% (i.e., Avg >= ~72.5%).

## Inference Config Reference
Default eval settings from `eval_megatron.sh`:
- **Mode:** dLLM, block_length=32, temperature=0.0
- **NFE (diffusion_steps):** equals max_new_tokens per task — GSM8k: 256, MBPP/MBPP+: 512
- **Shots:** GSM8k: 8, MBPP/MBPP+: 3

## Key Takeaways

_Updated as experiments complete. Captures high-level learnings to guide future ideas._

- **What works:** _(none yet)_
- **What doesn't work:** _(none yet)_
- **Surprising findings:** _(none yet)_

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
_New experiment results are appended below by the agent._

