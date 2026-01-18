# GLM-4.5V

[GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V) is a powerful vision-language model built on the GLM-4.5 Air architecture. GLM-4.5V combines a 106B parameter sparse MoE language model with a vision encoder for robust multimodal understanding of images and videos.

GLM-4.5V supports multimodal tasks including image captioning, visual question answering, OCR, video understanding, and general vision-language reasoning. The model leverages Multi-Resolution Rotary Position Embedding (MRoPE) for enhanced spatial understanding.

GLM family models are supported via the Bridge system with auto-detected configuration and weight mapping.

```{important}
Please update `transformers` version to 4.57.1 or higher in order to use the GLM-4.5V model.
```

## Available Models

### Vision-Language Models
- **GLM-4.5V** (`zai-org/GLM-4.5V`): 106B parameter vision-language model (based on GLM-4.5 Air)
  - 46 decoder layers, 4096 hidden size
  - 96 attention heads, 8 query groups (GQA)
  - 128 MoE experts with shared experts
  - ~12B active parameters per token
  - Sequence length: 131,072 tokens
  - Recommended: 4 nodes, 32 GPUs (LoRA/DoRA) or 16 nodes, 128 GPUs (Full SFT)

## Model Architecture Features

GLM-4.5V combines efficient sparse MoE language modeling with multimodal capabilities:

**Language Model Features:**
- **Sparse MoE Architecture**: 128 routed experts with shared experts for efficient parameter usage
- **Grouped Query Attention (GQA)**: Memory-efficient attention with 8 query groups
- **SiLU Gated Linear Unit**: Gated linear units with SiLU activation for improved performance
- **RMSNorm**: Layer normalization without mean centering for faster computation
- **Multi-Resolution RoPE (MRoPE)**: Enhanced position embeddings with sections [8, 12, 12] for improved spatial understanding
- **Extended Context**: Supports up to 131,072 tokens

**Vision-Language Features:**
- **Vision Encoder**: Pre-trained vision encoder for robust visual understanding
- **Multimodal Integration**: Seamless integration of visual and textual information
- **Image and Video Support**: Handles both static images and video inputs
- **Flexible Image Handling**: Supports variable resolution images and multiple images per conversation

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF VL model to your desired Megatron path:
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model zai-org/GLM-4.5V \
--megatron-path /models/glm-45v
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model zai-org/GLM-4.5V \
--megatron-path /results/glm_45v/checkpoints/iter_0001000 \
--hf-path ./glm-45v-hf-export
```

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path zai-org/GLM-4.5V \
--megatron_model_path /models/glm-45v \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`

## Finetune Recipes

- See: [bridge.recipes.glm_vl](../../apidocs/bridge/bridge.recipes.glm_vl.md)
- Available recipes:
  - `glm_45v_finetune_config`: Finetuning for GLM-4.5V model with PEFT support

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Full Finetuning

```python
from megatron.bridge.recipes.glm_vl import glm_45v_finetune_config

# Full finetuning
config = glm_45v_finetune_config(
    name="glm_45v_full_finetune",
    pretrained_checkpoint="/models/glm-45v",
    dataset_type="hf",
    peft=None,
    train_iters=1000,
    global_batch_size=32,
)
```

### Parameter-Efficient Finetuning (PEFT) with LoRA

```python
config = glm_45v_finetune_config(
    name="glm_45v_full_finetune",
    pretrained_checkpoint="/models/glm-45v",
    dataset_type="hf",
    peft='lora',
    train_iters=1000,
    global_batch_size=32,
)
```

PEFT options:
- `--peft-scheme`: Set to `lora` for LoRA or `dora` for DoRA. Omit for full finetuning.

You can also combine PEFT with freeze options:
- `--freeze-language-model`: Freeze the language model
- `--freeze-vision-model`: Freeze the vision encoder
- `--freeze-vision-projection`: Freeze the vision projection layer

Example with freeze options:
```python
from megatron.bridge.recipes.glm_vl import glm_45v_finetune_config

# LoRA finetuning
config = glm_45v_finetune_config(
    name="glm_45v_lora_finetune",
    pretrained_checkpoint="/models/glm-45v",
    dataset_type="hf",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=64,
)

# LoRA with vision model frozen
config = glm_45v_finetune_config(
    name="glm_45v_lora_language_only",
    pretrained_checkpoint="/models/glm-45v",
    peft="lora",
    freeze_vision_model=True,
    freeze_vision_projection=True,
)
```

### Recommended Configurations

| Model | Mode | TP | PP | EP | Global Batch Size | Learning Rate | Hardware |
|-------|------|----|----|-----|-------------------|---------------|----------|
| GLM-4.5V | Full SFT | 1 | 8 | 16 | 16-32 | 5e-6 | 128 GPUs (16 nodes) |
| GLM-4.5V | LoRA/DoRA | 1 | 8 | 4 | 32-64 | 1e-4 | 32 GPUs (4 nodes) |

**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for larger batch sizes and fewer GPUs. The sparse MoE architecture requires Expert Parallelism (EP) for efficient training.

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

- GLM-4.5V: https://huggingface.co/zai-org/GLM-4.5V

## Related Docs
- Related LLM: [GLM 4.5](../llm/glm45.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

