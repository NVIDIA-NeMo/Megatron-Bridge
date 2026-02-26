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
    --hf-model-id meta-llama/Llama-3.2-1B \
    --megatron-peft-checkpoint /path/to/finetune_ckpt \
    --output-hf-path ./my_adapter
```

| Argument | Description |
|---|---|
| `--hf-model-id` | HuggingFace model name or local path (architecture + base weights) |
| `--megatron-peft-checkpoint` | Path to the Megatron-Bridge distributed checkpoint containing LoRA adapter weights |
| `--output-hf-path` | Output directory (default: `./my_adapter`) |
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
- When `--megatron-peft-checkpoint` is provided, the top-k predicted tokens
  from the PEFT model must match those from the Megatron model with merged
  weights.

```bash
# Quick check (PEFT-only, no Megatron comparison)
uv run python examples/conversion/adapter/verify_adapter.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --hf-adapter-path ./my_adapter

# Full verification (compares against Megatron checkpoint)
uv run python examples/conversion/adapter/verify_adapter.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --hf-adapter-path ./my_adapter \
    --megatron-peft-checkpoint /path/to/finetune_ckpt/iter_0000020
```

| Argument | Description |
|---|---|
| `--hf-model-id` | HuggingFace base model name or path |
| `--hf-adapter-path` | Exported HF PEFT adapter directory |
| `--megatron-peft-checkpoint` | *(optional)* Megatron checkpoint iter directory for cross-check |
| `--prompt` | Prompt for the forward pass (default: `"The capital of France is"`) |
| `--top-k` | Number of top tokens to compare (default: `5`) |

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
