# Results: gentle_mask_3b_5k

## Experiment
- **Idea:** #5 Cosine Mask Schedule (Gentler Warmup)
- **Branch:** `autoresearch-gentle-mask-schedule`
- **Change:** Linear ramp of max mask ratio from 0.3 to 1.0 over first 1000 iterations
- **Checkpoint:** `ministral_3b_gentle_mask_v1/iter_0005000`

## GSM8k (8-shot CoT)
- Baseline Strict: 79.98%
- This experiment Strict: **77.56%** (-2.42%)
- Baseline Flex: 80.44%
- This experiment Flex: **78.77%** (-1.67%)

## MBPP (3-shot)
- Baseline: 55.00%
- This experiment: **55.60%** (+0.60%)

## MBPP+ (3-shot)
- Baseline: 72.49%
- This experiment: **70.11%** (-2.38%)

## Summary
- **Avg:** 70.51% (baseline: 71.98%, delta: **-1.47%**)
- **Verdict:** Negative. Worse than both baseline and experiment #2.
- The gentler warmup (min_ratio=0.3, warmup=1000) hurt GSM8k more than the aggressive one.
- Mask scheduling in general seems to hurt MBPP+ consistently.
- Conclusion: mask scheduling curriculum is not beneficial for this model/task combination.
