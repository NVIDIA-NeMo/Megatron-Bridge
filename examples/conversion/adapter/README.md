# Adapter Export & Verification

Scripts for exporting Megatron-Bridge LoRA/DoRA adapter weights to HuggingFace PEFT format and verifying the results.

## Overview

After fine-tuning a model with LoRA (or DoRA) in Megatron-Bridge, the adapter
weights live inside a Megatron distributed checkpoint. The scripts in this
directory let you:

1. **Export** the adapter to a HuggingFace PEFT-compatible directory
   (`adapter_config.json` + `adapter_model.safetensors`).
2. **Verify** the export by loading it with the `peft` library and comparing
   logits against the Megatron checkpoint.
3. **Stream** individual adapter tensors from a Megatron model for inspection
   or custom workflows.

The exported adapter can be loaded with standard HuggingFace tooling:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base, "./my_adapter")
```

## Scripts

### 1. `export_adapter.py` — Checkpoint Export

Converts a Megatron-Bridge PEFT checkpoint to HuggingFace PEFT format. Runs
entirely on CPU — no GPU required.

```bash
uv run python examples/conversion/adapter/export_adapter.py \
    --hf-model-path meta-llama/Llama-3.2-1B \
    --lora-checkpoint /path/to/finetune_ckpt \
    --output ./my_adapter
```

| Argument | Description |
|---|---|
| `--hf-model-path` | HuggingFace model name or local path (architecture + base weights) |
| `--lora-checkpoint` | Path to the Megatron-Bridge distributed checkpoint containing LoRA adapter weights |
| `--output` | Output directory (default: `./my_adapter`) |
| `--trust-remote-code` | Allow custom code from the HuggingFace repository |

**Output structure:**

```
my_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

### 2. `verify_adapter.py` — Export Verification

Loads the exported adapter with the `peft` library and runs verification
checks:

- The PEFT model logits must differ from the base model (adapter has effect).
- When `--lora-checkpoint` is provided, the top-k predicted tokens
  from the PEFT model must match those from the Megatron model with merged
  weights.

Supports CPU-only, single-GPU, and multi-GPU (TP/PP) modes.

```bash
# Quick check (PEFT-only, no Megatron comparison, CPU)
uv run python examples/conversion/adapter/verify_adapter.py \
    --hf-model-path meta-llama/Llama-3.2-1B \
    --hf-adapter-path ./my_adapter \
    --cpu

# Full verification on GPU (single GPU)
uv run python examples/conversion/adapter/verify_adapter.py \
    --hf-model-path meta-llama/Llama-3.2-1B \
    --hf-adapter-path ./my_adapter \
    --lora-checkpoint /path/to/finetune_ckpt/iter_0000020

# Multi-GPU with TP=2
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/adapter/verify_adapter.py \
    --hf-model-path meta-llama/Llama-3.2-1B \
    --hf-adapter-path ./my_adapter \
    --lora-checkpoint /path/to/finetune_ckpt/iter_0000020 \
    --tp 2

# Multi-GPU with PP=4
uv run python -m torch.distributed.run --nproc_per_node=4 \
    examples/conversion/adapter/verify_adapter.py \
    --hf-model-path meta-llama/Llama-3.2-1B \
    --hf-adapter-path ./my_adapter \
    --lora-checkpoint /path/to/finetune_ckpt/iter_0000020 \
    --pp 4
```

| Argument | Description |
|---|---|
| `--hf-model-path` | HuggingFace base model name or path |
| `--hf-adapter-path` | Exported HF PEFT adapter directory |
| `--lora-checkpoint` | *(optional)* Megatron checkpoint iter directory for cross-check |
| `--prompt` | Prompt for the forward pass (default: `"The capital of France is"`) |
| `--top-k` | Number of top tokens to compare (default: `5`) |
| `--tp` | Tensor parallel size (default: `1`) |
| `--pp` | Pipeline parallel size (default: `1`) |
| `--ep` | Expert parallel size (default: `1`) |
| `--cpu` | Run entirely on CPU (no GPU required, TP/PP/EP must be 1) |

### 3. `stream_adapter_weights.py` — Low-Level Adapter Streaming

Demonstrates how to use `AutoBridge.export_adapter_weights` to iterate through
adapter tensors one at a time. Useful for custom export pipelines or debugging.

Requires a GPU (uses NCCL backend).

```bash
# Single GPU
uv run python examples/conversion/adapter/stream_adapter_weights.py \
    --output ./adapters/demo_lora.safetensors

# Multi-GPU (tensor + pipeline parallelism)
uv run python -m torch.distributed.run --nproc_per_node=4 \
    examples/conversion/adapter/stream_adapter_weights.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --output ./adapters/demo_tp2_pp2.safetensors
```

## Programmatic API

The same functionality is available directly through `AutoBridge`:

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")

# One-liner: checkpoint → HF PEFT directory
bridge.export_adapter_ckpt(
    peft_checkpoint="/path/to/finetune_ckpt",
    output_path="./my_adapter",
)

# Or, if you already have a model in memory:
bridge.save_hf_adapter(
    model=megatron_model,
    path="./my_adapter",
    peft_config=lora,
    base_model_name_or_path="meta-llama/Llama-3.2-1B",
)
```
