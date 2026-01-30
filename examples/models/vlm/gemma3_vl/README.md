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

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

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

See the [conversion.sh](conversion.sh) script for more examples including:
- Multi-GPU round-trip validation between formats

## Inference

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

See the [inference.sh](inference.sh) script for commands to:
- Run inference with Hugging Face checkpoints
- Run inference with imported Megatron checkpoints
- Run inference with exported Hugging Face checkpoints

**Expected output:**
```
...
Generation step 46
Generation step 47
Generation step 48
Generation step 49
======== GENERATED TEXT OUTPUT ========
Image: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png
Prompt: Describe this image.
Generated: <bos><bos><start_of_turn>user
...
Describe this image.<end_of_turn>
<start_of_turn>model
Here's a description of the image you sent, breaking down the technical specifications of the H100 SXM and H100 NVL server cards:

**Overall:**

The image is a table comparing the technical specifications of two
=======================================
```

## Finetune Recipes

- See: [bridge.recipes.gemma3_vl](../../../../docs/apidocs/bridge/bridge.recipes.gemma3_vl.md)
- Available recipes:
  - `gemma3_vl_4b_finetune_config`: Finetuning for 4B VL model with PEFT support
  - `gemma3_vl_12b_finetune_config`: Finetuning for 12B VL model with PEFT support
  - `gemma3_vl_27b_finetune_config`: Finetuning for 27B VL model with PEFT support

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretrain

Pretraining is not verified for this model.

### Supervised Fine-Tuning (SFT)

See the [sft.sh](sft.sh) script for full parameter fine-tuning with configurable model parallelisms.

W&B report coming soon.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [peft.sh](peft.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism.

W&B report coming soon.

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

## Evaluation

Coming soon.

## Hugging Face Model Cards

- Gemma 3 VL 4B: https://huggingface.co/google/gemma-3-4b-it
- Gemma 3 VL 12B: https://huggingface.co/google/gemma-3-12b-it
- Gemma 3 VL 27B: https://huggingface.co/google/gemma-3-27b-it

## Related Docs

- Text-Only Models: [Gemma 3](../../../../docs/models/llm/gemma3.md)
- Recipe usage: [Recipe usage](../../../../docs/recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../../../docs/training/config-container-overview.md)
- Training entry points: [Entry points](../../../../docs/training/entry-points.md)
