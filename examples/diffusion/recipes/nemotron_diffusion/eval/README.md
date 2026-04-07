# Megatron Evaluation for Nemotron Diffusion Models

Evaluate Megatron-based dLLM (diffusion) and AR (autoregressive) models using [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Files

| File | Description |
|------|-------------|
| `eval_megatron.py` | lm-eval model wrapper (`megatron_dllm`) that loads a Megatron GPTModel via AutoBridge and supports both AR and dLLM generation |
| `eval_megatron.sh` | Launcher script for direct execution or Slurm submission with configurable experiments, tasks, and parallelism |

## Quick Start

### Single GPU

```bash
PYTHONPATH=src:examples:$PYTHONPATH HF_ALLOW_CODE_EVAL=1 \
python examples/diffusion/recipes/nemotron_diffusion/eval/eval_megatron.py \
    --model megatron_dllm \
    --model_args "megatron_load_path=<CHECKPOINT>,hf_model_id=<HF_CONFIG>,tokenizer=<TOKENIZER>,mask_token_id=100,eval_mode=dllm,max_new_tokens=256,max_sequence_length=4096,block_length=32,shift_logits=False,neg_entropy=True,tp=1,pp=1,load_hf_weights=False" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --log_samples \
    --output_path /tmp/eval_results
```

### Using the Launcher Script

```bash
# Direct execution on a GPU node (all tasks sequential)
bash eval_megatron.sh --direct --expts 3b --modes dllm,ar

# Slurm: one job per model/task pair
bash eval_megatron.sh --parallel-tasks --expts 3b --modes dllm --eval-tasks gsm8k_cot,humaneval

# Slurm: one job per model (all tasks sequential within)
bash eval_megatron.sh --parallel-models --expts 3b --modes dllm,ar
```

## Launcher Options

| Option | Default | Description |
|--------|---------|-------------|
| `--direct` | *(default)* | Run directly on the current GPU node |
| `--parallel-tasks` | | Submit one Slurm job per model/task/seed combination |
| `--parallel-models` | | Submit one Slurm job per model (tasks run sequentially) |
| `--expts` | `3b` | Comma-separated experiment names |
| `--modes` | `dllm,ar` | Comma-separated eval modes |
| `--eval-tasks` | all | Comma-separated lm-eval tasks |
| `--seeds` | `42` | Comma-separated random seeds |
| `--gpus` | auto | GPUs per node |
| `--checkpoint` | | Override checkpoint path |
| `--hf-model-id` | | Override HF model/config path |
| `--limit` | | Limit number of eval samples (for testing) |

## GPU / Parallelism Configurations

| TP | GPUs | Launcher | Parallelism |
|----|------|----------|-------------|
| 1 | 1 | `python` | Single GPU |
| 1 | 8 | `accelerate launch` | DP=8 |
| 2 | 8 | `torchrun` | TP=2, DP=4 |

## Supported Tasks

| Task | Few-shot | Max New Tokens |
|------|----------|----------------|
| `gsm8k_cot` | 8 | 256 |
| `humaneval` | 0 | 512 |
| `mbpp` | 3 | 512 |
| `humaneval_plus` | 0 | 512 |
| `mbpp_plus` | 3 | 512 |
| `minerva_math` | 4 | 512 |

## Key Model Arguments

| Argument | Description |
|----------|-------------|
| `eval_mode` | `dllm` for diffusion decoding, `ar` for autoregressive |
| `block_length` | Block size for dLLM generation (default: 32) |
| `steps_per_block` | Denoising steps per block (default: 32) |
| `neg_entropy` | Use negative entropy for confidence scoring |
| `denoising_threshold` | Optional threshold for early stopping |
| `shift_logits` | Shift logits for AR-style loss (set `False` for dLLM) |
| `load_hf_weights` | `True` to load HF weights directly, `False` to load from Megatron checkpoint |
| `cascade_schedule` | Pipe-separated schedule for multi-model cascade decoding |
