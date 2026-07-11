# Text-only pretrain validation inventory

Last updated: 2026-07-10

This inventory tracks the text-only subset of the model list in
[PR #4805](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/4805). It is the
execution matrix for [MB-747](https://linear.app/nvidia/issue/MB-747): every
row uses `scripts/training/train.sh`, two 8-GPU H100 nodes, real indexed DCLM,
100 training iterations, and `logger.log_interval=1`.

Vision-language, audio, Omni, and diffusion rows are intentionally excluded.
The machine-readable source of truth is
[`text_pretrain_validation.json`](../../scripts/training/text_pretrain_validation.json).
HF revisions in that file are immutable commit SHAs.

## Reproducible submission

The submission driver renders commands by default. Add `--submit` only after
reviewing the selected commands. Secrets must be inherited by environment
variable name or supplied through a mounted netrc; do not pass token literals.

```bash
uv run --extra recipes python scripts/training/submit_text_pretrain_validation.py \
  --account "$SLURM_ACCOUNT" \
  --partition "$SLURM_PARTITION" \
  --container-image "$CONTAINER_IMAGE" \
  --dataset-path /shared/dclm/dclm_mistral32k_validation_text_document \
  --dataset-cache /shared/dclm/cache \
  --output-root /shared/results/mb747 \
  --hf-home "$HF_HOME" \
  --runtime-venv /shared/venvs/mb747 \
  --env HF_TOKEN \
  --env WANDB_API_KEY \
  --model llama32-1b
```

Omit `--model` to select all rows, or repeat it to submit a size-aware batch.
All W&B runs use project `megatron-bridge-text-pretrain-validation`, group
`mb747-text-pretrain-dclm-20260710`, and a unique model ID as the run name.

## Matrix

`PASS` means all 100 steps used the indexed DCLM corpus and appeared in W&B.
Failures remain in this table with their first causal error; they are not
removed from the scope.

| ID | Family | Architecture | Params | Recipe | Status |
|---|---|---|---:|---|---|
| `ling-flash-2` | Bailing | Ling 2.0 Flash | 100B | `ling_flash_100b_pretrain_16gpu_h100_bf16_config` | RETRY: job 5613704 borderline OOM; expandable segments added |
| `ling-mini-2` | Bailing | Ling MoE V2 / Mini | 16B | `ling_mini_16b_pretrain_8gpu_h100_bf16_config` | [PASS: job 5614323](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/esj00tu8) |
| `deepseek-v2` | DeepSeek | DeepSeek V2 | 235.7B | `deepseek_v2_pretrain_128gpu_h100_bf16_config` | TODO |
| `deepseek-v2-lite` | DeepSeek | DeepSeek V2 Lite | 15.7B | `deepseek_v2_lite_pretrain_8gpu_h100_bf16_config` | [PASS: job 5614279](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/bnpkjx66) |
| `deepseek-v4-flash` | DeepSeek | DeepSeek V4 Flash | 292B | `deepseek_v4_flash_pretrain_32gpu_h100_bf16_config` | TODO |
| `ernie45-21b-a3b` | Ernie | Ernie 4.5 MoE | 21.9B | `ernie45_21b_a3b_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614447 logits-buffer OOM; full recompute added |
| `falcon-h1-500m` | Falcon | Falcon H1 | 0.5B | `falcon_h1_500m_pretrain_1gpu_h100_bf16_config` | [PASS: job 5613718](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/e5hgbepa) |
| `gemma-2b` | Gemma | Gemma | 2.5B | `gemma_2b_pretrain_1gpu_h100_bf16_config` | RETRY: job 5613698 gated-config 401; direct provider added |
| `gemma2-2b` | Gemma | Gemma 2 | 2.6B | `gemma2_2b_pretrain_2gpu_h100_bf16_config` | [PASS: job 5613705](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/03w4wcsy) |
| `gemma3-1b` | Gemma | Gemma 3 | 1B | `gemma3_1b_pretrain_1gpu_h100_bf16_config` | [PASS: job 5613697](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/a62ht0v9) |
| `gemma4-26b-a4b` | Gemma | Gemma 4 26B-A4B MoE | 26.5B | `gemma4_26b_a4b_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614455 RMSNorm OOM; full recompute added |
| `gemma4-31b` | Gemma | Gemma 4 31B dense | 32.7B | `gemma4_31b_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614970 used unsupported PP=2; changed to TP=8/PP=1 |
| `glm45-355b` | GLM | GLM-4.5 | 358.3B | `glm45_355b_pretrain_128gpu_h100_bf16_config` | TODO |
| `glm47-355b` | GLM | GLM-4.7 | 358.3B | `glm47_355b_pretrain_16gpu_h100_bf16_config` | TODO |
| `glm47-flash-31b` | GLM | GLM-4.7-Flash | 31.2B | `glm47_flash_31b_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614789 allocator exhaustion; full recompute added |
| `gpt-oss-20b` | GPT-OSS | GPT-OSS 20B | 21.5B | `gpt_oss_20b_pretrain_16gpu_h100_bf16_config` | [PASS: job 5614349](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/h9y0hlvc) |
| `gpt-oss-120b` | GPT-OSS | GPT-OSS 120B | 120.4B | `gpt_oss_120b_pretrain_64gpu_h100_bf16_config` | RETRY: job 5614979 main-param OOM; validation changed to PP=2/EP=8 |
| `hy3-preview-base` | HY V3 | Hy3 preview-Base | 298.8B | `hy3_299b_pretrain_16gpu_h100_bf16_config` | TODO |
| `llama2-7b` | Llama | Llama 2 | 6.7B | `llama2_7b_pretrain_2gpu_h100_bf16_config` | RETRY: job 5613750 gated-config 401; direct provider added |
| `llama3-8b` | Llama | Llama 3 | 8B | `llama3_8b_pretrain_2gpu_h100_bf16_config` | RETRY: job 5614256 gated-config 401; direct provider added |
| `llama31-8b` | Llama | Llama 3.1 | 8B | `llama31_8b_pretrain_2gpu_h100_bf16_config` | [PASS: job 5614263](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/gry9jhfg) |
| `llama32-1b` | Llama | Llama 3.2 | 1.2B | `llama32_1b_pretrain_1gpu_h100_bf16_config` | [PASS: job 5613679](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/3w8ubwh7) |
| `llama33-70b` | Llama | Llama 3.3 | 70.6B | `llama31_70b_pretrain_32gpu_h100_bf16_config` | RETRY: job 5614973 optimizer-state OOM; BF16 moments added |
| `minimax-m2` | MiniMax | MiniMax-M2 | 456B | `minimax_m2_pretrain_16gpu_h100_bf16_config` | TODO |
| `minimax-m2-5` | MiniMax | MiniMax-M2.5 | 456B | `minimax_m2_5_pretrain_16gpu_h100_bf16_config` | TODO |
| `minimax-m2-7` | MiniMax | MiniMax-M2.7 | 456B | `minimax_m2_7_pretrain_16gpu_h100_bf16_config` | TODO |
| `mistral-7b` | Mistral | Mistral | 7.2B | `mistral_7b_pretrain_2gpu_h100_bf16_config` | [PASS: job 5613796](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/f6agifw6) |
| `mimo-7b` | Xiaomi-MiMo | MiMo | 7.8B | `mimo_7b_pretrain_2gpu_h100_bf16_config` | [PASS: job 5614060](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/vxv6bxye) |
| `mimo-v2-flash` | Xiaomi-MiMo | MiMo-V2-Flash | 309.8B | `mimo_v2_flash_310b_pretrain_16gpu_h100_bf16_config` | TODO |
| `moonlight-16b` | Moonlight | Moonlight | 16B | `moonlight_16b_pretrain_8gpu_h100_bf16_config` | [PASS: job 5614345](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/gefczmtj) |
| `nemotron-h-4b` | Nemotron | Nemotron H | 4.5B | `nemotronh_4b_pretrain_1gpu_h100_bf16_config` | [PASS: job 5613719](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/9yryey3k) |
| `nemotron-nano-9b-v2` | Nemotron | Nemotron Nano v2 | 8.9B | `nemotron_nano_9b_v2_pretrain_2gpu_h100_bf16_config` | [PASS: job 5614274](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/dq1lmioh) |
| `nemotron3-nano` | Nemotron | Nemotron-3 Nano | 31.6B | `nemotron_3_nano_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614962 logits OOM; validation TP increased to 2 |
| `nemotron3-super` | Nemotron | Nemotron-3 Super | 123.6B | `nemotron_3_super_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614983 used unsupported NVFP4; changed to H100 FP8 |
| `llama31-nemotron-nano-4b` | Nemotron | Llama Nemotron | 4.5B | `llama31_nemotron_nano_4b_pretrain_2gpu_h100_bf16_config` | [PASS: job 5613712](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/utr0drdl) |
| `olmoe-7b` | OLMoE | OLMoE | 6.9B | `olmoe_7b_pretrain_8gpu_h100_bf16_config` | [PASS: job 5613795](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/87iw0tzr) |
| `qwen2-7b` | Qwen | Qwen2 | 7.6B | `qwen2_7b_pretrain_2gpu_h100_bf16_config` | [PASS: job 5613798](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/1txppez6) |
| `qwen25-7b` | Qwen | Qwen2.5 | 7.6B | `qwen25_7b_pretrain_2gpu_h100_bf16_config` | [PASS: job 5614046](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/bgh028ce) |
| `qwen3-8b` | Qwen | Qwen3 | 8.2B | `qwen3_8b_pretrain_4gpu_h100_bf16_config` | [PASS: job 5614268](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/diw9fxv9) |
| `qwen3-30b-a3b` | Qwen | Qwen3-MoE | 30.5B | `qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config` | [PASS: job 5614462](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/s759zs24) |
| `qwen3-next-80b-a3b` | Qwen | Qwen3 Next | 81.3B | `qwen3_next_80b_a3b_pretrain_32gpu_h100_bf16_config` | RETRY: job 5614977 Triton backward OOM; validation full recompute added |
| `qwen35-27b` | Qwen | Qwen3.5 dense | 27.8B | `qwen35_27b_pretrain_8gpu_h100_bf16_config` | [PASS: job 5614457](https://wandb.ai/yaoyu/megatron-bridge-text-pretrain-validation/runs/ozws43ul) |
| `qwen35-35b-a3b` | Qwen | Qwen3.5 MoE | 36B | `qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614971 hybrid/dispatcher OOM; full recompute added |
| `sarvam-30b` | Sarvam | Sarvam | 32.2B | `sarvam_30b_pretrain_8gpu_h100_bf16_config` | RETRY: job 5614966 dispatcher OOM; full recompute added |
| `step35-flash` | StepFun | Step-3.5-Flash | 199.4B | `step35_196b_a11b_pretrain_512gpu_h100_bf16_config` | TODO |
