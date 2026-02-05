# Gemma 3 VL Examples

This directory contains example scripts for Gemma 3 VL vision-language models.

For model introduction and architecture details, see the [Gemma 3 VL documentation](../../../../docs/models/vlm/gemma3-vl.md).

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
| Gemma 3 VL 4B | Full SFT | 2 | 1 | 32 | 5e-5 | 8 GPUs |
| Gemma 3 VL 4B | LoRA/DoRA | 2 | 1 | 32 | 2e-4 | 8 GPUs |
| Gemma 3 VL 12B | Full SFT | 4 | 1 | 32 | 5e-5 | 8 GPUs |
| Gemma 3 VL 12B | LoRA/DoRA | 2 | 1 | 32 | 2e-4 | 8 GPUs |
| Gemma 3 VL 27B | Full SFT | 8 | 2 | 32 | 5e-5 | 16 GPUs |
| Gemma 3 VL 27B | LoRA/DoRA | 4 | 1 | 32 | 2e-4 | 8 GPUs |

**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for larger batch sizes and fewer GPUs.

## Evaluation

Coming soon.
