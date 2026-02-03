# Ministral 3 - Vision Language Model

[Mistral AI's Ministral 3](https://huggingface.co/collections/mistralai/ministral-3) is a family of edge-optimized vision-language models designed for deployment across various hardware configurations. The Ministral 3 architecture combines a powerful language model with a vision encoder for multimodal understanding.

Ministral 3 models support multimodal tasks including image captioning, visual question answering, OCR, and general vision-language understanding. Despite their compact size, these models deliver strong performance for on-device and edge deployment scenarios.

Ministral family models are supported via the Bridge system with auto-detected configuration and weight mapping.

```{important}
Please upgrade to `transformers` v5 and upgrade `mistral-common` in order to use the Ministral 3 models.
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
  --hf-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --megatron-path ${WORKSPACE}/models/Ministral-3-3B-Instruct-2512-BF16
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --megatron-path ${WORKSPACE}/models/Ministral-3-3B-Instruct-2512-BF16/iter_0000000 \
  --hf-path ${WORKSPACE}/models/Ministral-3-3B-Instruct-2512-BF16-hf-export
```

## Inference

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --megatron_model_path ${WORKSPACE}/models/Ministral-3-3B-Instruct-2512-BF16/iter_0000000 \
  --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
  --prompt "Describe this image." \
  --max_new_tokens 100
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`

See the [`inference.sh`](./inference.sh) script for commands to:
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
Generated: <s><s>[SYSTEM_PROMPT]You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {today}.
...
[IMG_END]Describe this image.[/INST]The image presents a comparison table of technical specifications between two NVIDIA GPUs: the **H100 SXM** and the **H100 NVL**.

### **FPU Performance (Floating-Point Operations Per Second)**
- **FP64**:
  - H100 SXM: 34 teraFLOPS
  - H100 NVL: 30 teraFLOPS
- **FP64 Tensor
=======================================
```

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

### Pretrain

Pretraining is not verified for this model.

### Supervised Fine-Tuning (SFT)

See the [`sft.sh`](./sft.sh) script for full parameter fine-tuning with configurable model parallelisms.

W&B report coming soon.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [`peft.sh`](./peft.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism.

W&B report coming soon.

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

## Evaluation

Coming soon.

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

