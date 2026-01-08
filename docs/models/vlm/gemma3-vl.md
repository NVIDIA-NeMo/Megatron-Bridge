# Gemma 3 VL (Vision-Language)

[Google's Gemma 3 VL](https://huggingface.co/collections/google/gemma-3-release) is a family of vision-language models built on the same research and technology used to create Gemini models. The Gemma 3 VL architecture combines the text-generation capabilities of Gemma 3 with a SigLIP vision encoder for robust visual understanding.

Gemma 3 VL models support multimodal tasks including image captioning, visual question answering, OCR, and general vision-language understanding.

Gemma family models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

### Vision-Language Models
- **Gemma 3 VL 4B** (`google/gemma-3-4b-it`): 4B parameter vision-language model
  - 34 layers, 2560 hidden size
  - 16 attention heads, 4 query groups (GQA)
  - Vision encoder: SigLIP with 729M parameters
  - Recommended: 1 node, 8 GPUs
  
- **Gemma 3 VL 12B** (`google/gemma-3-12b-it`): 12B parameter vision-language model
  - 48 layers, 3840 hidden size
  - 24 attention heads, 8 query groups (GQA)
  - Vision encoder: SigLIP with 729M parameters
  - Recommended: 1 node, 8 GPUs
  
- **Gemma 3 VL 27B** (`google/gemma-3-27b-it`): 27B parameter vision-language model
  - 62 layers, 5376 hidden size
  - 32 attention heads, 16 query groups (GQA)
  - Vision encoder: SigLIP with 729M parameters
  - Recommended: 2 nodes, 16 GPUs

All models support a sequence length of 131,072 tokens and use hybrid attention patterns (sliding window + global).

## Model Architecture Features

Gemma 3 VL builds on the Gemma 3 architecture with additional multimodal capabilities:

**Language Model Features:**
- **Hybrid Attention Pattern**: Alternates between global and local sliding window attention for efficient long-context processing
- **GeGLU Activation**: Uses gated linear units with GELU activation for improved performance
- **RMSNorm**: Layer normalization without mean centering for faster computation
- **Rotary Embeddings**: Separate RoPE configurations for local and global attention layers

**Vision-Language Features:**
- **SigLIP Vision Encoder**: Pre-trained vision encoder with 729M parameters for robust visual understanding
- **Multimodal Integration**: Seamless integration of visual and textual information through learned projection layers
- **Flexible Image Handling**: Supports variable resolution images and multiple images per conversation

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF VL model to your desired Megatron path:
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model google/gemma-3-4b-it \
--megatron-path /models/gemma-3-4b-it
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model google/gemma-3-4b-it \
--megatron-path /results/gemma3_vl_4b/checkpoints/iter_00001000 \
--hf-path ./gemma3-vl-hf-export
```

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path google/gemma-3-4b-it \
--megatron_model_path /models/gemma-3-4b-it \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`

## Finetune Recipes

- See: [bridge.recipes.gemma3_vl](../../apidocs/bridge/bridge.recipes.gemma3_vl.md)
- Available recipes:
  - `gemma3_vl_4b_finetune_config`: Finetuning for 4B VL model with PEFT support
  - `gemma3_vl_12b_finetune_config`: Finetuning for 12B VL model with PEFT support
  - `gemma3_vl_27b_finetune_config`: Finetuning for 27B VL model with PEFT support

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Full Finetuning

```bash
torchrun --nproc-per-node=8 run/run_vlm_recipe.py \
--pretrained-checkpoint /models/gemma-3-4b-it \
--recipe gemma3_vl_4b_finetune_config \
--dataset-type hf \
dataset.maker_name=make_cord_v2_dataset \
train.global_batch_size=64 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/gemma3_vl_4b_finetune
```

Or programmatically:
```python
from megatron.bridge.recipes.gemma3_vl import gemma3_vl_4b_finetune_config

# Full finetuning
config = gemma3_vl_4b_finetune_config(
    name="gemma3_vl_4b_full_finetune",
    pretrained_checkpoint="/models/gemma-3-4b-it",
    dataset_type="hf",
    peft=None,
    train_iters=1000,
    global_batch_size=64,
)
```

### Parameter-Efficient Finetuning (PEFT) with LoRA

```bash
torchrun --nproc-per-node=8 run/run_vlm_recipe.py \
--pretrained-checkpoint /models/gemma-3-4b-it \
--recipe gemma3_vl_4b_finetune_config \
--peft_scheme lora \
--dataset-type hf \
dataset.maker_name=make_cord_v2_dataset \
train.global_batch_size=128 \
checkpoint.save=$SAVE_DIR/gemma3_vl_4b_lora
```

PEFT options:
- `--peft_scheme`: Set to `lora` for LoRA or `dora` for DoRA. Omit for full finetuning.

You can also combine PEFT with freeze options:
- `model.freeze_language_model=True`: Freeze the language model
- `model.freeze_vision_model=True`: Freeze the vision encoder
- `model.freeze_vision_projection=True`: Freeze the vision projection layer

Example with freeze options:
```bash
torchrun --nproc-per-node=8 run/run_vlm_recipe.py \
--pretrained-checkpoint /models/gemma-3-4b-it \
--recipe gemma3_vl_4b_finetune_config \
--peft_scheme lora \
model.freeze_language_model=True \
model.freeze_vision_model=False \
checkpoint.save=$SAVE_DIR/gemma3_vl_4b_lora_vision
```

Programmatic configuration:
```python
from megatron.bridge.recipes.gemma3_vl import gemma3_vl_4b_finetune_config

# LoRA finetuning
config = gemma3_vl_4b_finetune_config(
    name="gemma3_vl_4b_lora_finetune",
    pretrained_checkpoint="/models/gemma-3-4b-it",
    dataset_type="hf",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=128,
)

# LoRA with vision model frozen
config = gemma3_vl_4b_finetune_config(
    name="gemma3_vl_4b_lora_language_only",
    pretrained_checkpoint="/models/gemma-3-4b-it",
    peft="lora",
    freeze_vision_model=True,
    freeze_vision_projection=True,
)
```

### Recommended Configurations

| Model | Mode | TP | PP | Global Batch Size | Learning Rate | Hardware |
|-------|------|----|----|-------------------|---------------|----------|
| Gemma 3 VL 4B | Full SFT | 1 | 1 | 32-64 | 5e-6 | 8 GPUs |
| Gemma 3 VL 4B | LoRA/DoRA | 1 | 1 | 64-128 | 1e-4 | 8 GPUs |
| Gemma 3 VL 12B | Full SFT | 4 | 1 | 32-64 | 5e-6 | 8 GPUs |
| Gemma 3 VL 12B | LoRA/DoRA | 1 | 1 | 64-128 | 1e-4 | 8 GPUs |
| Gemma 3 VL 27B | Full SFT | 8 | 2 | 16-32 | 5e-6 | 16 GPUs |
| Gemma 3 VL 27B | LoRA/DoRA | 4 | 1 | 32-64 | 1e-4 | 16 GPUs |

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

## Hugging Face Model Cards

- Gemma 3 VL 4B: https://huggingface.co/google/gemma-3-4b-it
- Gemma 3 VL 12B: https://huggingface.co/google/gemma-3-12b-it
- Gemma 3 VL 27B: https://huggingface.co/google/gemma-3-27b-it

## Related Docs
- Text-Only Models: [Gemma 3](../llm/gemma3.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

