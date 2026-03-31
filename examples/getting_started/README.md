# Getting Started

A numbered sequence of examples to help you learn Megatron Bridge step by step.

| # | Script | What it does | GPU required |
|---|--------|-------------|-------------|
| 0 | [00_convert.py](00_convert.py) | HF → Megatron → HF round-trip conversion | Yes (1 GPU) |
| 1 | [01_pretrain.py](01_pretrain.py) | Pretraining with a recipe (mock data) | Yes (1+ GPUs) |
| 2 | [02_finetune_lora.py](02_finetune_lora.py) | LoRA fine-tuning with SFT recipe | Yes (1+ GPUs) |

## Running

```bash
# Conversion (single process)
python examples/getting_started/00_convert.py

# Training (requires torchrun for distributed setup)
torchrun --nproc-per-node=1 examples/getting_started/01_pretrain.py
torchrun --nproc-per-node=1 examples/getting_started/02_finetune_lora.py
```

If using `uv`, prefix commands with `uv run`.

## Next Steps

- Browse [conversion examples](../conversion/) for multi-GPU and advanced conversion flows
- See [model-specific examples](../models/) for per-model recipes and configs
- Read the [Recipe Usage Guide](../../docs/recipe-usage.md) for customizing training recipes
