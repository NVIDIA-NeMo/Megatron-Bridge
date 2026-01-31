# Gemma 3 VL - Vision Language Model

This directory contains examples for Gemma 3 Vision Language Model, including checkpoint conversion, inference, and fine-tuning.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

See the [conversion.sh](conversion.sh) script for commands to:
- Import Hugging Face checkpoints to Megatron format
- Export Megatron checkpoints back to Hugging Face format
- Run multi-GPU round-trip validation between formats


## Inference

**See the [inference.sh](inference.sh) script for commands to:
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

## Pretrain

Pretraining is not verified for this model.

## Supervised Fine-Tuning (SFT)

See the [sft.sh](sft.sh) script for full parameter fine-tuning with configurable model parallelisms.

[W&B Report](TODO)

## Parameter-Efficient Fine-Tuning (PEFT)

See the [peft.sh](peft.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism.

[W&B Report](TODO)

## Evaluation

TBD