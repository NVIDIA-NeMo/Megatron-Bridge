# Experiment Results — Nemotron Diffusion 3B

> Auto-updated by the agent after each evaluation. Single source of truth for comparing experiments.


## Leaderboard — 32 denoising steps per block (AR-equivalent)

| Rank | Experiment | Checkpoint | MBPP | MBPP+ | GSM8k Strict | GSM8k Flex | Avg | Date |
|------|-----------|-----------|------|-------|-------------|------------|-----|------|
| 1 | **mask_schedule** | `ministral_3b_v2_iter5k/iter_0005000` | 55.20% | 70.90% | 81.27% | 81.80% | 72.29% | 2026-04-09 |
| 2 | **baseline** | `ministral_3b_v3/iter_0005000` | 56.40% | 69.31% | 79.53% | 80.52% | 71.44% | 2026-04-07 |
| 3 | **gentle_mask** | `gentle_mask/iter_0005000` | 55.60% | 69.84% | 77.56% | 78.77% | 70.44% | 2026-04-09 |

## Leaderboard — 16 denoising steps per block

| Rank | Experiment | Checkpoint | MBPP | MBPP+ | GSM8k Strict | GSM8k Flex | Avg | Date |
|------|-----------|-----------|------|-------|-------------|------------|-----|------|
| 1 | **mask_schedule** | `ministral_3b_v2_iter5k/iter_0005000` | 26.00% | 34.92% | 67.17% | 68.39% | 49.12% | 2026-04-09 |
| 2 | **baseline** | `ministral_3b_v3/iter_0005000` | 20.00% | 30.16% | 66.11% | 67.32% | 45.90% | 2026-04-07 |
| 3 | **gentle_mask** | `gentle_mask/iter_0005000` | 20.40% | 31.75% | 63.08% | 64.75% | 44.99% | 2026-04-09 |

## Leaderboard — 1 denoising step per block

| Rank | Experiment | Checkpoint | MBPP | MBPP+ | GSM8k Strict | GSM8k Flex | Avg | Date |
|------|-----------|-----------|------|-------|-------------|------------|-----|------|
| 1 | **mask_schedule** | `ministral_3b_v2_iter5k/iter_0005000` | 0.00% | 0.00% | 0.00% | 0.76% | 0.19% | 2026-04-09 |
| 2 | **gentle_mask** | `gentle_mask/iter_0005000` | 0.00% | 0.00% | 0.00% | 0.61% | 0.15% | 2026-04-09 |
| 3 | **baseline** | `ministral_3b_v3/iter_0005000` | 0.00% | 0.00% | 0.00% | 0.38% | 0.10% | 2026-04-07 |

> **Avg** = mean of (MBPP, MBPP+, GSM8k Strict, GSM8k Flex).

<img src="results_plot.png" width="500"/>

## Inference Config Reference

Default eval settings from `eval_megatron.sh`:
- **Mode:** dLLM, block_length=32, temperature=0.0
- **NFE (diffusion_steps):** equals max_new_tokens per task — GSM8k: 256, MBPP/MBPP+: 512
- **Shots:** GSM8k: 8, MBPP/MBPP+: 3

## Key Takeaways

_Updated as experiments complete. Captures high-level learnings to guide future ideas._

- **Denoising steps matter dramatically:** 32 steps ≈ AR quality (70-72% avg), 16 steps drops to 45-49%, and 1 step is near-random (<0.2%).
- **NFE-quality tradeoff:** Going from 32→16 steps halves compute but loses ~23-25% absolute accuracy.
- **mask_schedule improves over baseline at 32 steps:** +0.85% avg, driven by GSM8k (+1.7%); MBPP is slightly lower (-1.2%). Gap widens at 16 steps (+3.2% avg).
- **gentle_mask underperforms both baseline and mask_schedule:** At 32 steps, gentle_mask is ~1% below baseline on avg and ~2% below mask_schedule, with the gap coming mainly from GSM8k (-2% vs baseline, -3.7% vs mask_schedule).
- **HumanEval and Minerva Math results still pending** for gentle_mask and mask_schedule (v2_iter5k).

_New experiment results are appended below by the agent._
