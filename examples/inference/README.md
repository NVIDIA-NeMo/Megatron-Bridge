# Megatron Bridge Inference Examples

This directory contains efficient inference examples built on the Megatron-Core
high-level inference APIs from PR #4697.

## Offline Text Generation with Bridge Loading

`text_generation.py` is the Bridge-backed synchronous entrypoint. It uses
`AutoBridge` for Hugging Face config/model support and optional Megatron Bridge
checkpoint loading, then runs `MegatronLLM.generate`.

```bash
examples/inference/run_text_generation.sh --nproc 1 \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --prompt "Megatron Bridge inference is" \
  --max_new_tokens 32
```

For an imported Megatron checkpoint:

```bash
examples/inference/run_text_generation.sh --nproc 8 \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --megatron_model_path /path/to/checkpoint/iter_0000000 \
  --tp 8 \
  --prompt "Megatron Bridge inference is"
```

`--hf_model_path` may be omitted when the checkpoint `run_config.yaml` records
`model.hf_model_id`.

Use `--use-legacy-generation` to run MCore legacy static batching instead of
the default dynamic engine. `--attention-backend` can override the provider
attention backend before the Megatron model is constructed.

## Concurrent Async Generation

`async_text_generation.py` is intentionally direct MCore-style. It does not use
`AutoBridge`; pass normal Megatron training/inference arguments such as
`--load`, tokenizer args, model provider, and parallelism settings.

```bash
examples/inference/run_async_text_generation.sh --nproc 8 \
  --load /path/to/megatron/checkpoint \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen2.5-1.5B \
  --model-provider gpt \
  --bf16 \
  --prompts "Megatron async inference is" "Concurrent generation is"
```

The async example uses `MegatronAsyncLLM` in coordinator mode and submits
multiple prompts concurrently from the primary rank.

## OpenAI-Compatible Server

`openai_server.py` is also direct MCore-style and uses
`MegatronAsyncLLM.serve(...)`.

```bash
examples/inference/run_openai_server.sh --nproc 8 \
  --load /path/to/megatron/checkpoint \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen2.5-1.5B \
  --model-provider gpt \
  --bf16 \
  --host 0.0.0.0 \
  --port 5000
```

After the HTTP server is ready on the primary rank, send OpenAI-compatible
requests to `/v1/completions` or `/v1/chat/completions`.

## Phase 2

The existing `examples/models/*/inference.sh` wrappers are intentionally
unchanged in Phase 1. They can be refactored later to call these entrypoints.
