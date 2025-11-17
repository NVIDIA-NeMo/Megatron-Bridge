# Gemma 3

[Google's Gemma 3](https://huggingface.co/collections/google/gemma-3-release) is a family of lightweight, state-of-the-art open models built on the same research and technology used to create Gemini models. The Gemma 3 architecture builds on the transformer decoder framework with enhancements including pre-normalization with RMSNorm, GeGLU activations, Rotary Positional Embeddings (RoPE), and hybrid attention patterns (sliding window and global attention).

Gemma 3 models are designed for a wide range of text generation tasks and are available in multiple sizes to suit different computational budgets.

Gemma family models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

### Text-Only Models
- **Gemma 3 1B** (`google/gemma-3-1b-it`): Compact 1B parameter model optimized for efficiency
  - 26 layers, 1152 hidden size
  - 8 attention heads, 2 query groups (GQA)
  - Sequence length: 131,072 tokens
  - Ideal for single-GPU deployment

All models support a sequence length of 131,072 tokens and use hybrid attention patterns (sliding window + global).

## Model Architecture Features

Gemma 3 introduces several architectural innovations:

- **Hybrid Attention Pattern**: Alternates between global and local sliding window attention for efficient long-context processing
- **GeGLU Activation**: Uses gated linear units with GELU activation for improved performance
- **RMSNorm**: Layer normalization without mean centering for faster computation
- **Rotary Embeddings**: Separate RoPE configurations for local and global attention layers
  - Local attention: Uses sliding window with rotary base 10,000
  - Global attention: Extended rotary base for better long-range dependencies

## Conversion with 🤗 Hugging Face

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: Gemma 3 1B
bridge = AutoBridge.from_hf_pretrained("google/gemma-3-1b-it")
provider = bridge.to_megatron_provider()

# Configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Import HF → Megatron
To import the HF model to your desired Megatron path:
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model google/gemma-3-1b-it \
--megatron-path /models/gemma-3-1b-it
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model google/gemma-3-1b-it \
--megatron-path /results/gemma3_1b/checkpoints/iter_00001000 \
--hf-path ./gemma3-hf-export
```

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
--hf_model_path google/gemma-3-1b-it \
--megatron_model_path /models/gemma-3-1b-it \
--prompt "What is artificial intelligence?" \
--max_new_tokens 100
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.

## Pretrain and Finetune Recipes

- See: [bridge.recipes.gemma](../../apidocs/bridge/bridge.recipes.gemma.md)
- Available recipes:
  - `gemma3_1b_pretrain_config`: Pre-training configuration for Gemma 3 1B
  - `gemma3_1b_finetune_config`: Finetuning configuration with PEFT support (LoRA, DoRA)

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretraining

```python
from megatron.bridge.recipes.gemma import gemma3_1b_pretrain_config

# Create a pretraining configuration
config = gemma3_1b_pretrain_config(
    name="my_gemma3_pretrain",
    data_paths=["path/to/data"],
    train_iters=100000,
    global_batch_size=256,
)
```

### Full Finetuning

```bash
torchrun --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/gemma-3-1b-it \
--recipe gemma3_1b_finetune_config \
train.global_batch_size=64 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/gemma3_1b_finetune
```

Or programmatically:
```python
from megatron.bridge.recipes.gemma import gemma3_1b_finetune_config

config = gemma3_1b_finetune_config(
    name="gemma3_1b_full_finetune",
    pretrained_checkpoint="/models/gemma-3-1b-it",
    peft=None,
    train_iters=1000,
    global_batch_size=64,
)
```

### Parameter-Efficient Finetuning (PEFT) with LoRA

```bash
torchrun --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/gemma-3-1b-it \
--recipe gemma3_1b_finetune_config \
--peft_scheme lora \
train.global_batch_size=128 \
checkpoint.save=$SAVE_DIR/gemma3_1b_lora
```

PEFT options:
- `--peft_scheme`: Set to `lora` for LoRA or `dora` for DoRA. Omit for full finetuning.

Or programmatically:
```python
from megatron.bridge.recipes.gemma import gemma3_1b_finetune_config

# LoRA finetuning
config = gemma3_1b_finetune_config(
    name="gemma3_1b_lora_finetune",
    pretrained_checkpoint="/models/gemma-3-1b-it",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=128,
)
```

### Recommended Configurations

| Model | Mode | TP | PP | Global Batch Size | Learning Rate |
|-------|------|----|----|-------------------|---------------|
| Gemma 3 1B | Full SFT | 1 | 1 | 64-128 | 5e-6 |
| Gemma 3 1B | LoRA/DoRA | 1 | 1 | 128-256 | 1e-4 |

## Examples
- Checkpoint import/export: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- Generate text (HF→Megatron): [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Hugging Face Model Cards

- Gemma 3 1B: https://huggingface.co/google/gemma-3-1b-it

## Related Docs
- Gemma3 Vision-Language Models: [Gemma 3 VL](../vlm/gemma3-vl.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
