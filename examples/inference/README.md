# Megatron Bridge Inference Examples

This directory contains text-generation examples for Megatron Bridge and
Megatron-Core inference.

## Offline Text Generation with Bridge Loading

`scripts/inference/text_generation.py` is the Bridge-backed synchronous
entrypoint. It uses `AutoBridge` for Hugging Face config/model support and
optional Megatron Bridge checkpoint loading, then runs `MegatronLLM.generate`.

```bash
bash examples/inference/run_text_generation.sh --nproc 1 \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --prompt "Megatron Bridge inference is" \
  --max_new_tokens 32
```

For an imported Megatron checkpoint:

```bash
bash examples/inference/run_text_generation.sh --nproc 8 \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --megatron_model_path /path/to/checkpoint/iter_0000000 \
  --tp 8 \
  --prompt "Megatron Bridge inference is"
```

`--hf_model_path` may be omitted when the checkpoint `run_config.yaml` records
`model.hf_model_id`.

By default this script uses dynamic inference. Use `--use-legacy-generation`
when the model-specific example needs the static generation path, for example
for a non-standard attention pattern such as attention sink or sliding-window
attention. Pair it with `--attention-backend local` or
`--attention-backend unfused` when the example requires an unfused/local
attention implementation. `--attention-backend` is applied before the Megatron
model is constructed.

## Model-Specific Examples

The model wrapper scripts show tested arguments for specific model families.
Falcon H1 runs as a one-GPU static generation example because it uses Mamba
layers:

```bash
bash examples/models/falcon_h1/inference.sh
```

Sarvam runs MoE inference with coordinator mode:

```bash
bash examples/models/sarvam/inference.sh
```

Ling/Bailing runs MoE inference with coordinator mode:

```bash
bash examples/models/bailing/inference.sh
```

GPT-OSS runs through static generation with local attention:

```bash
bash examples/models/gpt_oss/inference.sh
```

`examples/models/gpt_oss/inference.sh` uses `--use-legacy-generation` and
`--attention-backend local` because GPT-OSS uses attention behavior that should
run through the static path for this example.

For larger models that need multiple nodes, use the Slurm wrapper for that
model. For example:

```bash
sbatch examples/models/minimax/minimax_m2/slurm_inference.sh
```

Set model-specific environment variables described in each wrapper before
launching, such as `WORKSPACE`, `HF_MODEL_ID`, `MEGATRON_MODEL_PATH`,
`HF_EXPORT_PATH`, `CONTAINER_IMAGE`, and `CONTAINER_MOUNTS`.

## Concurrent Async Generation

`scripts/inference/async_text_generation.py` uses the same Bridge/AutoBridge
model loading and shared inference arguments as the synchronous entrypoint.

```bash
bash examples/inference/run_async_text_generation.sh --nproc 1 \
  --hf_model_path Qwen/Qwen2.5-1.5B \
  --dtype bf16 \
  --prompt "Megatron async inference is" \
  --prompt "Concurrent generation is"
```

The async example uses `MegatronAsyncLLM` in coordinator mode and submits
multiple prompts concurrently from the primary rank.

## OpenAI-Compatible Server

`scripts/inference/openai_server.py` uses the Bridge/AutoBridge model-loading
arguments and serves requests through `MegatronAsyncLLM.serve(...)`.

```bash
bash examples/inference/run_openai_server.sh --nproc 1 \
  --hf_model_path Qwen/Qwen2.5-1.5B \
  --dtype bf16 \
  --host 0.0.0.0 \
  --port 5000
```

After the HTTP server is ready on the primary rank, send OpenAI-compatible
requests to `/v1/completions` or `/v1/chat/completions`.
