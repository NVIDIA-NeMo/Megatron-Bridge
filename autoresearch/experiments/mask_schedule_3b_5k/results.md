# Results: mask_schedule_3b_5k

## Experiment
- **Idea:** #2 Mask Scheduling — curriculum mask ratio warmup
- **Branch:** `autoresearch-mask-scheduling`
- **Change:** Linear ramp of max mask ratio from 0.1 to 1.0 over first 2500 iterations
- **Checkpoint:** `ministral_3b_v2/iter_0005000`

## GSM8k (8-shot CoT)
- Baseline Strict: 79.98%
- This experiment Strict: **81.27%** (+1.29%)
- Baseline Flex: 80.44%
- This experiment Flex: **81.80%** (+1.36%)

## MBPP (3-shot)
- Baseline: 55.00%
- This experiment: **55.20%** (+0.20%)

## MBPP+ (3-shot)
- Baseline: 72.49%
- This experiment: **70.90%** (-1.59%)

## Summary
- **Avg:** 72.29% (baseline: 71.98%, delta: **+0.31%**)
- **Verdict:** Below the ~0.5% threshold. Not positive.
- GSM8k improved notably (+1.3%), but MBPP+ regressed (-1.59%), washing out the gains.

## Notes
- The mask schedule warmup clearly affects training dynamics — `num_tokens_dlm` increased from ~215 to ~430 over the first 290 iterations during interactive testing, confirming the curriculum is working.
- The GSM8k improvement suggests the curriculum helps with reasoning tasks, but the MBPP+ regression indicates it may hurt code generation.
- A possible follow-up: try a gentler warmup (min_ratio=0.3 or warmup_iters=1000) to see if we can keep GSM8k gains without MBPP+ regression.
