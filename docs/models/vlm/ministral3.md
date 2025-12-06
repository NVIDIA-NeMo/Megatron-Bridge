# Ministral 3

[Mistral AI's Ministral 3](https://huggingface.co/collections/mistralai/ministral-3) is a family of edge-optimized vision-language models designed for deployment across various hardware configurations. The Ministral 3 architecture combines a powerful language model with a vision encoder for multimodal understanding.

Ministral 3 models support multimodal tasks including image captioning, visual question answering, OCR, and general vision-language understanding. Despite their compact size, these models deliver strong performance for on-device and edge deployment scenarios.

Ministral family models are supported via the Bridge system with auto-detected configuration and weight mapping.

```{important}
Please update `transformers` version to 5.0.0rc0 in order to use the Ministral 3 models.
```

## Available Models

### Vision-Language Models
- **Ministral 3 3B** (`mistralai/Ministral-3-3B-Base-2512`): 3.4B parameter vision-language model
  - 26 layers, 3072 hidden size
  - 32 attention heads, 8 query groups (GQA)
  - Vision encoder: ~0.4B parameters
  - Recommended: 1 node, 8 GPUs

- **Ministral 3 8B** (`mistralai/Ministral-3-8B-Base-2512`): 8.4B parameter vision-language model
  - 34 layers, 4096 hidden size
  - 32 attention heads, 8 query groups (GQA)
  - Vision encoder: ~0.4B parameters
  - Recommended: 1 node, 8 GPUs

- **Ministral 3 14B** (`mistralai/Ministral-3-14B-Base-2512`): ~14B parameter vision-language model
  - 40 layers, 5120 hidden size
  - 32 attention heads, 8 query groups (GQA)
  - Vision encoder: ~0.4B parameters
  - Recommended: 1 node, 8 GPUs

All models support extended context lengths up to 256K tokens using YaRN RoPE scaling.

## Model Architecture Features

Ministral 3 combines efficient language modeling with multimodal capabilities:

**Language Model Features:**
- **YaRN RoPE Scaling**: Advanced rope scaling for extended context lengths (up to 256K tokens)
- **Grouped Query Attention (GQA)**: Memory-efficient attention mechanism with 8 query groups
- **SwiGLU Activation**: Gated linear units with SiLU activation for improved performance
- **RMSNorm**: Layer normalization without mean centering for faster computation
- **Llama 4 Attention Scaling**: Position-dependent attention scaling for improved long-context handling

**Vision-Language Features:**
- **Vision Encoder**: Pre-trained vision encoder for robust visual understanding
- **Multimodal Projector**: Projects vision features to language model space
- **Flexible Image Handling**: Supports variable resolution images and multiple images per conversation

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF VL model to your desired Megatron path:
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model mistralai/Ministral-3-3B-Base-2512 \
--megatron-path /models/ministral3-3b
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model mistralai/Ministral-3-3B-Base-2512 \
--megatron-path /results/ministral3_3b/checkpoints/iter_0001000 \
--hf-path ./ministral3-hf-export
```

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path mistralai/Ministral-3-3B-Base-2512 \
--megatron_model_path /models/ministral3-3b \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`

## Finetune Recipes

- See: [bridge.recipes.ministral3](../../apidocs/bridge/bridge.recipes.ministral3.md)
- Available recipes:
  - `ministral3_3b_finetune_config`: Finetuning for 3B VL model with PEFT support
  - `ministral3_8b_finetune_config`: Finetuning for 8B VL model with PEFT support
  - `ministral3_14b_finetune_config`: Finetuning for 14B VL model with PEFT support

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Full Finetuning

```bash
torchrun --nproc-per-node=8 examples/recipes/ministral3/finetune_ministral3_vl.py \
--pretrained-checkpoint /models/ministral3-3b \
--dataset-type hf \
train.global_batch_size=32 \
train.train_iters=1000
```

Or programmatically:
```python
from megatron.bridge.recipes.ministral3 import ministral3_3b_finetune_config

# Full finetuning
config = ministral3_3b_finetune_config(
    name="ministral3_3b_full_finetune",
    pretrained_checkpoint="/models/ministral3-3b",
    dataset_type="hf",
    peft=None,
    train_iters=1000,
    global_batch_size=32,
)
```

### Parameter-Efficient Finetuning (PEFT) with LoRA

```bash
torchrun --nproc-per-node=8 examples/recipes/ministral3/finetune_ministral3_vl.py \
--pretrained-checkpoint /models/ministral3-3b \
--peft-scheme lora \
--dataset-type hf \
train.global_batch_size=64 \
train.train_iters=1000
```

PEFT options:
- `--peft-scheme`: Set to `lora` for LoRA or `dora` for DoRA. Omit for full finetuning.

You can also combine PEFT with freeze options:
- `--freeze-language-model`: Freeze the language model
- `--freeze-vision-model`: Freeze the vision encoder
- `--freeze-vision-projection`: Freeze the vision projection layer

Example with freeze options:
```bash
torchrun --nproc-per-node=8 examples/recipes/ministral3/finetune_ministral3_vl.py \
--pretrained-checkpoint /models/ministral3-3b \
--peft-scheme lora \
--freeze-vision-model \
train.global_batch_size=64
```

Programmatic configuration:
```python
from megatron.bridge.recipes.ministral3 import ministral3_3b_finetune_config

# LoRA finetuning
config = ministral3_3b_finetune_config(
    name="ministral3_3b_lora_finetune",
    pretrained_checkpoint="/models/ministral3-3b",
    dataset_type="hf",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=64,
)

# LoRA with vision model frozen
config = ministral3_3b_finetune_config(
    name="ministral3_3b_lora_language_only",
    pretrained_checkpoint="/models/ministral3-3b",
    peft="lora",
    freeze_vision_model=True,
    freeze_vision_projection=True,
)
```

### Recommended Configurations

| Model | Mode | TP | PP | Global Batch Size | Learning Rate | Hardware |
|-------|------|----|----|-------------------|---------------|----------|
| Ministral 3 3B | Full SFT | 1 | 1 | 32-64 | 5e-6 | 8 GPUs |
| Ministral 3 3B | LoRA/DoRA | 1 | 1 | 64-128 | 1e-4 | 8 GPUs |
| Ministral 3 8B | Full SFT | 2 | 1 | 32-64 | 5e-6 | 8 GPUs |
| Ministral 3 8B | LoRA/DoRA | 1 | 1 | 64-128 | 1e-4 | 8 GPUs |
| Ministral 3 14B | Full SFT | 4 | 1 | 16-32 | 5e-6 | 8 GPUs |
| Ministral 3 14B | LoRA/DoRA | 2 | 1 | 32-64 | 1e-4 | 8 GPUs |

**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for larger batch sizes and fewer GPUs.

## Example Datasets

| Dataset | Maker Name | Description |
|---------|------------|-------------|
| [cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | `make_cord_v2_dataset` | OCR receipts: Single-image-text dataset for receipt understanding |
| [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) | `make_medpix_dataset` | Medical VQA: Single-image Q&A for clinical images |
| [The Cauldron (Raven subset)](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | `make_raven_dataset` | Visual reasoning: Multi-image analogical reasoning |

To change the dataset, specify `dataset.maker_name=<maker_name>` in your command.

## Examples
- Checkpoint import/export: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- Generate with VLM (HF→Megatron): [examples/conversion/hf_to_megatron_generate_vlm.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_vlm.py)
- Finetuning script: [examples/recipes/ministral3/finetune_ministral3_vl.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/recipes/ministral3/finetune_ministral3_vl.py)

## Hugging Face Model Cards

- Ministral 3 3B Base: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
- Ministral 3 3B Instruct: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- Ministral 3 8B Base: https://huggingface.co/mistralai/Ministral-3-8B-Base-2512
- Ministral 3 8B Instruct: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
- Ministral 3 14B Base: https://huggingface.co/mistralai/Ministral-3-14B-Base-2512
- Ministral 3 14B Instruct: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

## Related Docs
- Related LLM: [Mistral](../llm/mistral.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

