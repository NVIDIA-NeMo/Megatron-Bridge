# Results: baseline

## Training
- **Checkpoint:** iter_0005000 (5k training steps)
- **Path:** `/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/ministral_3b/iter_0005000`

## Inference Config
- **Mode:** dLLM (diffusion)
- **Block length:** 32
- **Temperature:** 0.0
- **Diffusion steps (NFE) per task:**
  - GSM8k: 256 (8-shot, max_new_tokens=256)
  - MBPP: 512 (3-shot, max_new_tokens=512)
  - MBPP+: 512 (3-shot, max_new_tokens=512)

## Metrics
| Benchmark | Score |
|-----------|-------|
| GSM8k (8-shot CoT, strict) | 79.98% |
| GSM8k (8-shot CoT, flexible) | 80.44% |
| MBPP (pass@1) | 55.00% |
| MBPP+ (pass@1) | 72.49% |

## Notes
Default 3B model, main branch, 5k training steps. dLLM inference with default settings.
