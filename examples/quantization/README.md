# Quantization Examples

This directory contains example scripts for quantizing Megatron Bridge models using NVIDIA ModelOpt.

> **Note**: These are minimal, reference examples. For more comprehensive, production-quality post-training quantization (PTQ), export, and Quantization Aware Distillation (QAD) scripts, see the NVIDIA Model Optimizer repository: [examples/megatron_bridge/](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge).

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
uv run python -m torch.distributed.run --nproc_per_node 2 examples/quantization/quantize.py \
    --hf-model-id Qwen/Qwen3-0.6B \
    --export-quant-cfg fp8 \
    --tp 2 \
    --megatron-save-path ./qwen3_0_6b_fp8
```

Quantize VLM:

```bash
uv run python -m torch.distributed.run --nproc_per_node 8 examples/quantization/quantize_vlm.py \
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
uv run python -m torch.distributed.run --nproc_per_node 2 examples/quantization/ptq_generate.py \
    --hf-model-id Qwen/Qwen3-0.6B \
    --megatron-load-path ./qwen3_0_6b_fp8 \
    --tp 2
```

Resume and generate with quantized VLM:

```bash
uv run python -m torch.distributed.run --nproc_per_node 8 examples/quantization/ptq_generate_vlm.py \
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
uv run python -m torch.distributed.run pretrain_quantized_llama3_8b.py \
    --nproc_per_node 4 \
    model.tensor_model_parallel_size=4 \
    model.gradient_accumulation_fusion=False \
    checkpoint.pretrained_checkpoint=/models/llama3_8b_fp8_init
```

### 4. Export to HuggingFace

Export a quantized Huggingface checkpoint (packed weights and scales, with the `quantization_config` in
`config.json`) at any parallelism:

```bash
uv run python -m torch.distributed.run --nproc_per_node 2 examples/quantization/export.py \
    --hf-model-id Qwen/Qwen3-0.6B \
    --megatron-load-path ./qwen3_0_6b_fp8 \
    --export-dir ./qwen3_0_6b_fp8_hf \
    --pp 2
```
