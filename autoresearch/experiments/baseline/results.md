# Results: baseline

## Training
- **Checkpoint:** iter_0005000 (5k training steps)
- **Path:** `/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/ministral_3b_v3/iter_0005000`

## Inference Config
- **Mode:** dLLM (diffusion)
- **Block length:** 32
- **Temperature:** 0.0
- **NFE per task:** GSM8k: 256, MBPP/MBPP+: 512

## Metrics — 32 denoising steps per block

| Benchmark | Score |
|-----------|-------|
| MBPP (pass@1) | 56.40% |
| MBPP+ (pass@1) | 69.31% |
| GSM8k (8-shot CoT, strict) | 79.53% |
| GSM8k (8-shot CoT, flexible) | 80.52% |
| **Avg** | **71.44%** |

Eval results: `/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_eval_results/ministral_3b_v3_sbd32_dllm/seed_42/`

## Metrics — 16 denoising steps per block

| Benchmark | Score |
|-----------|-------|
| MBPP (pass@1) | 20.00% |
| MBPP+ (pass@1) | 30.16% |
| GSM8k (8-shot CoT, strict) | 66.11% |
| GSM8k (8-shot CoT, flexible) | 67.32% |
| **Avg** | **45.90%** |

Eval results: `/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_eval_results/ministral_3b_v3_sbd16_dllm/seed_42/`

## Metrics — 1 denoising step per block

| Benchmark | Score |
|-----------|-------|
| MBPP (pass@1) | 0.00% |
| MBPP+ (pass@1) | 0.00% |
| GSM8k (8-shot CoT, strict) | 0.00% |
| GSM8k (8-shot CoT, flexible) | 0.38% |
| **Avg** | **0.10%** |

Eval results: `/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_eval_results/ministral_3b_v3_dllm/seed_42/`

## Notes
Default 3B model, main branch, 5k training steps. dLLM inference with default settings.
Quality degrades sharply with fewer denoising steps: 32→16 loses ~25% avg, 1 step is near-random.
