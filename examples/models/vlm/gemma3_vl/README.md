# Gemma 3 VL - Vision Language Model

This directory contains examples for Gemma 3 Vision Language Model, including checkpoint conversion, inference, and fine-tuning.

## Conversion with 🤗 Hugging Face

See the [conversion.sh](conversion.sh) script for example commands to:
- Importing Hugging Face checkpoints into Megatron format on CPU
- Exporting Megatron checkpoints back to Hugging Face on CPU
- Running a multi-GPU round-trip validation to automatically convert between formats and verify model weights for compatibility between Hugging Face and Megatron-LM checkpoints

```{literalinclude} conversion.sh
:language: bash
```

## Pretrain

Pretraining is not verified for this model.

## Finetune Recipes

See the [sft.sh](sft.sh) script for example commands to:
- Full finetuning with all model parameters

```{literalinclude} sft.sh
:language: bash
```

## PEFT Recipes

See the [peft.sh](peft.sh) script for example commands to:
- Parameter-Efficient Finetuning (PEFT) with LoRA

```{literalinclude} peft.sh
:language: bash
```

You will see the following wandb result when the commands succeed:
```
wandb result
```

## Run Inference

See the [inference.sh](inference.sh) script for example commands to:
- Run inference with Hugging Face checkpoints
- Run inference with converted Megatron checkpoints
- Run inference with trained Megatron checkpoints

```{literalinclude} inference.sh
:language: bash
```

You will see the following output when the commands succeed:

**Inference output:**
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
