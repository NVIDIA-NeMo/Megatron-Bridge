# Quantization Examples

This directory contains example scripts for quantizing Megatron Bridge models using NVIDIA ModelOpt.

## Overview

The quantization examples demonstrate how to:

1. **Quantize pretrained models** - Post-training quantization (PTQ)
2. **Train quantized models** - Quantization-aware training (QAT)
3. **Resume quantized models** - Resume and generate with quantized model
4. **Export quantized models** - Convert to HuggingFace format

## Quick Start

### 1. Post-Training Quantization (PTQ)

Quantize LLM:

```bash
torchrun --nproc_per_node 2 examples/quantization/quantize.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --export-quant-cfg fp8 \
    --tp 2 \
    --megatron-save-path ./llama3_2_1b_fp8
```

Quantize VLM:

```bash
torchrun --nproc_per_node 8 examples/quantization/quantize_vlm.py \
    --hf-model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
    --export-quant-cfg fp8 \
    --megatron-save-path ./Qwen3-VL-30B-A3B-Instruct_fp8 \
    --tp 4 \
    --etp 4 \
    --pp 2
```

### 2. Resume Quantized Model

Resume and generate with quantized LLM:

```bash
torchrun --nproc_per_node 2 examples/quantization/ptq_generate.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --megatron-load-path ./llama3_2_1b_fp8 \
    --tp 2
```

Resume and generate with quantized VLM:

```bash
torchrun --nproc_per_node 8 examples/quantization/ptq_generate_vlm.py \
    --hf-model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
    --megatron-load-path ./Qwen3-VL-30B-A3B-Instruct_fp8 \
    --tp 8 \
    --ep 8 \
    --image-path ./demo.jpeg \
    --prompts "Describe this image."
```

### 3. Quantization-Aware Training (QAT)

Train quantized model (requires a checkpoint from [PTQ](#1-post-training-quantization-ptq)) for better accuracy:

```bash
torchrun pretrain_quantized_llama3_8b.py \
    --nproc_per_node 4 \
    model.tensor_model_parallel_size=4 \
    model.gradient_accumulation_fusion=False \
    checkpoint.pretrained_checkpoint=/models/llama3_8b_fp8_init
```

### 4. Export to HuggingFace

Export unified Huggingface checkpoint:

```bash
torchrun --nproc_per_node 2 examples/quantization/export.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --megatron-load-path ./llama3_2_1b_fp8 \
    --export-dir ./llama3_2_1b_fp8_hf \
    --pp 2 \
    --dtype bfloat16
```
