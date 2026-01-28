# Gemma 3 VL - Vision Language Model

This directory contains examples for Gemma 3 Vision Language Model, including checkpoint conversion, inference, and fine-tuning.

## Checkpoint Conversion

See the [conversion.sh](conversion.sh) script for commands to:
- Import Hugging Face checkpoints to Megatron format
- Export Megatron checkpoints back to Hugging Face format
- Run multi-GPU round-trip validation between formats

## Pretrain

Pretraining is not verified for this model.

## Supervised Fine-Tuning (SFT)

See the [sft.sh](sft.sh) script for full parameter fine-tuning with configurable model parallelisms.

[W&B Report](TODO)

## Parameter-Efficient Fine-Tuning (PEFT)

See the [peft.sh](peft.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism.

[W&B Report](TODO)

## Inference

See the [inference.sh](inference.sh) script for commands to:
- Run inference with Hugging Face checkpoints
- Run inference with imported Megatron checkpoints
- Run inference with exported Hugging Face checkpoints

**Example output:**
```
Describe this image.<end_of_turn>
<start_of_turn>model
Here's a description of the image you sent, breaking down the technical specifications of the H100 SXM and H100 NVL server cards:

**Overall:**

The image is a table comparing the technical specifications of two NVIDIA server cards: the H100 SXM and the H100 NVL. It's designed to highlight the performance differences between the two cards, particularly in terms of compute power and memory.

**Column Breakdown:**

*
=======================================
```
