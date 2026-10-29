# Megatron Bridge Examples: Conversion Scripts

This directory contains example scripts that demonstrate how to use the Megatron Bridge's AutoBridge functionality for model conversion, loading, and inference. These scripts showcase various capabilities including HuggingFace-Megatron conversion, text generation, vision-language models, and multi-GPU parallelism.

## Available Scripts

### `create_hf_toy_model.py` - Preserve Weights in a Shallow Test Checkpoint

Creates a smaller checkpoint by retaining the first transformer layers from an
existing Hugging Face safetensors checkpoint. Unlike a randomly initialized toy
model, the retained tensors are byte-for-byte identical to the source, which is
useful for conversion, checkpoint, and numerical-parity tests.

The tool accepts either a local checkpoint directory or a Hugging Face model ID,
supports sharded checkpoints, updates the safetensors index, and copies tokenizer
and model metadata. It streams tensor bytes instead of loading the checkpoint into
CPU or GPU memory.

```bash
# Qwen3 example: retain the first four layers
uv run python examples/conversion/create_hf_toy_model.py \
  Qwen/Qwen3-0.6B \
  /tmp/qwen3-0.6b-4layers \
  --num-hidden-layers 4

# Verify that Megatron Bridge can import and export the result
uv run python examples/conversion/hf_megatron_roundtrip.py \
  --hf-model-id /tmp/qwen3-0.6b-4layers \
  --output-dir /tmp/qwen3-roundtrip
```

This utility currently targets decoder-only checkpoints with a top-level
`num_hidden_layers` config field and tensor names containing `layers.<index>`.

### `repair_hf_embedding_rows.py` - Repair Near-Zero Input Embedding Rows

Creates a repaired copy of a local Hugging Face safetensors checkpoint by
rewriting diagnosed near-zero input embedding rows. It is intended for one-off
continued-pretraining cleanup when rare token IDs in the base checkpoint can
produce extreme embedding gradients.

The script scans the input embedding row norms, replaces rows whose L2 norm is
non-finite or at/below `--min-norm` with the matching `lm_head.weight` direction
scaled to the RMS norm of healthy input rows, and writes a manifest with the
affected token IDs. It preserves the HF checkpoint layout and safetensors index.

```bash
uv run python examples/conversion/repair_hf_embedding_rows.py \
  --input-hf-path /models/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --output-hf-path /models/NVIDIA-Nemotron-3-Nano-4B-BF16-repaired \
  --min-norm 1.0e-4 \
  --max-rows 256

# Diagnose only, without writing an output checkpoint
uv run python examples/conversion/repair_hf_embedding_rows.py \
  --input-hf-path /models/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --dry-run
```

By default the tool infers common tensor names such as
`backbone.embeddings.weight` and `lm_head.weight`. Use
`--input-embedding-name` and `--output-embedding-name` if a checkpoint uses
different names.

### 1. `hf_megatron_roundtrip.py` - Two-Way Model Conversion

Demonstrates round-trip conversion between HuggingFace and Megatron-LM model formats.

**Features:**
- Load HuggingFace models and convert to Megatron format
- Save converted models back to HuggingFace format
- Weight verification during conversion

**Usage:**
```bash
# Basic conversion (uses default Llama-3.2-1B)
uv run python examples/conversion/hf_megatron_roundtrip.py

# Convert specific model
uv run python examples/conversion/hf_megatron_roundtrip.py --hf-model-id meta-llama/Llama-3.2-3B

# Save to specific directory
uv run python examples/conversion/hf_megatron_roundtrip.py --hf-model-id meta-llama/Llama-3.2-1B --output-dir ./converted_models
```

**Example Output:**
```
Loading from meta-llama/Llama-3.2-1B ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 (98/98) LlamaBridge
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 1235814400
Converting to HuggingFace ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 (98/98) LlamaBridge
                                     Hugging Face Weights Verification
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Weight Name                                     ┃ Shape          ┃ DType    ┃ Device ┃ Matches Original ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ model.norm.weight                               │ (2048,)        │ bfloat16 │ cuda:0 │        ✅        │
│ model.embed_tokens.weight                       │ (128256, 2048) │ bfloat16 │ cuda:0 │        ✅        │
│ model.layers.0.post_attention_layernorm.weight  │ (2048,)        │ bfloat16 │ cuda:0 │        ✅        │
│ model.layers.0.mlp.gate_proj.weight             │ (8192, 2048)   │ bfloat16 │ cuda:0 │        ✅        │
│ model.layers.0.mlp.up_proj.weight               │ (8192, 2048)   │ bfloat16 │ cuda:0 │        ✅        │
│ model.layers.0.mlp.down_proj.weight             │ (2048, 8192)   │ bfloat16 │ cuda:0 │        ✅        │
...
Saving HF-ckpt in Llama-3.2-1B...
```

### 2. Stable Checkpoint Conversion CLI

Production checkpoint import, export, and round-trip validation use
[`scripts/conversion/convert.sh`](../../scripts/conversion/README.md). The CLI
supports local or Slurm execution and selects a single-process CPU backend or a
distributed GPU backend without requiring users to invoke Python, `torchrun`,
`srun`, or `sbatch` directly.

```bash
./scripts/conversion/convert.sh import \
  --executor local --device cpu \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path ./checkpoints/llama3_2_1b

./scripts/conversion/convert.sh export \
  --executor local --device cpu \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path ./checkpoints/llama3_2_1b/iter_0000000 \
  --hf-path ./exports/llama3_2_1b_hf

./scripts/conversion/convert.sh roundtrip \
  --executor local --device gpu --gpus-per-node 2 \
  --hf-model meta-llama/Llama-3.2-1B \
  --tp 2
```

Use `--executor slurm` with the required account, partition, container, mount,
and resource arguments for multi-node round-trip validation. The launcher
compares the in-memory HF → Megatron → HF result and does not write a
checkpoint. The scripts in this directory remain standalone examples for direct
`torch.distributed.run`, generation, benchmarking, and model comparison.

### 3. `hf_to_megatron_generate_text.py` - Text Generation

Demonstrates text generation using HuggingFace models converted to Megatron format with support for parallel inference.

**Features:**
- Load from HuggingFace or pre-converted Megatron checkpoints
- Multi-GPU support with tensor/pipeline parallelism
- Greedy text generation
- Configurable generation parameters

**Usage:**

**Single GPU generation:**
```bash
# From HuggingFace model
uv run python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --prompt "Hello, how are you?" \
  --max_new_tokens 50

# From Megatron checkpoint
uv run python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --megatron_model_path ./checkpoints/llama3_2_1b \
  --prompt "The future of AI is" \
  --max_new_tokens 30
```

**Multi-GPU generation:**
```bash
# Tensor parallelism
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --prompt "Hello world" \
  --tp 2

# Pipeline parallelism
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path meta-llama/Llama-3.2-1B \
  --prompt "Hello world" \
  --pp 2
```

**Example Output:**
```
Loading from meta-llama/Llama-3.2-1B ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 (98/98) LlamaBridge
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 1235814400
Generation step 0
Step 0: output shape=torch.Size([1, 7, 128256]), var=8.5567
Top 5: [(' I', 21.25), (' Today', 19.875), (' My', 19.125), (' We', 19.125), (' This', 19.125)]
Selected: ' I' (id=358)
Generation step 1
...
Generation step 48
Generation step 49
======== GENERATED TEXT OUTPUT ========
Prompt: Hello, how are you?
Generated: <|begin_of_text|>Hello, how are you? I am a 20 year old girl from the Philippines. I am a very outgoing person and I love to meet new people. I am a very friendly person and I love to make new friends. I am a very outgoing person and I love to
=======================================
```

### 4. `hf_to_megatron_generate_vlm.py` - Vision-Language Generation

Demonstrates vision-language model inference with support for both image and text inputs.

**Features:**
- Support for vision-language models (e.g., Qwen2.5-VL)
- Load images from URLs or local files
- Text-only or multimodal generation
- Multi-GPU support

**Usage:**

**With image input:**
```bash
# Image from URL
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
  --prompt "Describe this image." \
  --max_new_tokens 100

# Local image file
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --image_path ./images/sample.jpg \
  --prompt "What objects do you see in this image?"
```

**Text-only generation:**
```bash
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --prompt "Hello, how are you?" \
  --max_new_tokens 50
```

**Multi-GPU with vision:**
```bash
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --image_path ./images/sample.jpg \
  --prompt "Describe this image." \
  --tp 2
```

**Example Output:**
```
Loading HuggingFace model from: Qwen/Qwen2.5-VL-3B-Instruct
Generation step 0
Generation step 1
...
======== GENERATED TEXT OUTPUT ========
Image: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
Prompt: Describe this image.
Generated: This image shows a cozy indoor scene with a wooden table, some books, a cup of coffee, and warm lighting creating a comfortable reading atmosphere.
=======================================
```

### 5. `list_supported_architectures.py` - Supported Models Reference

Lists all HuggingFace model architectures supported by the AutoBridge system.

**Usage:**
```bash
uv run python examples/conversion/list_supported_architectures.py
```

**Example Output:**
```
🚀 Megatron-Bridge AutoBridge - Supported Models
==================================================

✅ Found 5 supported model architecture(s):

   1. LlamaForCausalLM
   2. Qwen2ForCausalLM
   3. Qwen2_5_VLForConditionalGeneration
   4. Qwen3ForCausalLM
   5. Qwen3MoeForCausalLM

💡 Usage:
   To use any of these models, you can load them with:
   >>> bridge = AutoBridge.from_hf_pretrained('model_name')
   >>> model = bridge.to_megatron_model()

🔍 Model Bridge Details:
   Each model has specific implementation details and configurations.
   Check the src/megatron/bridge/models/ directory for:
   • Model-specific bridge implementations
   • Configuration examples and README files
   • Weight mapping details
   • Architecture-specific optimizations

📚 For more examples, see the examples/bridge/ directory.
```

### 6. `hf_megatron_roundtrip_benchmark.py` - Conversion Benchmarking

Benchmark the HF ↔ Megatron round-trip pipeline without writing checkpoints. The script times both the import (HF tensors → Megatron weights) and export (Megatron weights → HF tensors) phases so you can quickly compare performance across different models or parallelism settings.

**Features:**
- Measures import/export timings only (no checkpoints saved)
- Supports tensor, pipeline, and expert parallelism

**Usage:**
```bash
# Single-node benchmark (default Llama-3.2-1B)
uv run python examples/conversion/hf_megatron_roundtrip_benchmark.py

# Specify a custom model
uv run python examples/conversion/hf_megatron_roundtrip_benchmark.py \
  --hf-model-id meta-llama/Llama-3.2-3B

# Multi-GPU benchmark with expert parallelism
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/conversion/hf_megatron_roundtrip_benchmark.py \
  --hf-model-id Qwen/Qwen3-30B-A3B --tp 1 --pp 1 --ep 8
```

**Example Output:**
```
Benchmarking round-trip for Qwen/Qwen3-30B-A3B
TP=1 | PP=1 | EP=8 | ETP=1 | world_size=8

           HF ↔ Megatron Round-Trip Benchmark
┏━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Stage  ┃ Duration (s) ┃ Description                   ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Import │        19.09 │ HF tensors → Megatron weights │
│ Export │         2.43 │ Megatron weights → HF tensors │
│ Total  │        21.52 │ Import + export               │
└────────┴──────────────┴───────────────────────────────┘
```

### 7. `hf_megatron_roundtrip_multi_gpu.py` - Multi-GPU Model Conversion

Demonstrates model conversion and weight verification on multiple GPUs using distributed training.

**Features:**
- Multi-GPU model conversion
- Distributed weight verification
- Support for tensor/pipeline/expert parallelism
- Save models in both HF and Megatron formats

**Usage:**

**Basic multi-GPU conversion:**
```bash
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id meta-llama/Llama-3.2-1B \
  --tp 2

uv run python -m torch.distributed.run --nproc_per_node=4 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id meta-llama/Llama-3.2-1B \
  --tp 2 --pp 2
```

Hugging Face checkpoint export uses loose key validation by default for
backward compatibility. Add `--strict` to require every source checkpoint
tensor to be written.

**Save in Megatron format:**
```bash
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id meta-llama/Llama-3.2-1B \
  --tp 2 \
  --megatron-save-path ./megatron_checkpoints/llama3_2_1b
```

**Load from existing Megatron checkpoint:**
```bash
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id meta-llama/Llama-3.2-1B \
  --tp 2 \
  --megatron-load-path ./megatron_checkpoints/llama3_2_1b
```

**Example Output:**
```
Loading from meta-llama/Llama-3.2-1B ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 (98/98) LlamaBridge
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 617940992
Tensor parallel size: 2
Pipeline parallel size: 1
Expert parallel size: 1
Expert tensor parallel size: 1

                                     Hugging Face Weights Verification
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Weight Name                                     ┃ Shape          ┃ DType    ┃ Device ┃ Matches Original ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ model.norm.weight                               │ (2048,)        │ bfloat16 │ cuda:0 │        ✅        │
│ model.embed_tokens.weight                       │ (128256, 2048) │ bfloat16 │ cuda:0 │        ✅        │
│ model.layers.0.post_attention_layernorm.weight  │ (2048,)        │ bfloat16 │ cuda:0 │        ✅        │
│ model.layers.0.mlp.gate_proj.weight             │ (8192, 2048)   │ bfloat16 │ cuda:0 │        ✅        │
│ model.layers.0.mlp.up_proj.weight               │ (8192, 2048)   │ bfloat16 │ cuda:0 │        ✅        │
...
Success: All tensors from the original checkpoint were written.
```

### 8. `compare_hf_and_megatron/` - Model Comparison Tools

Advanced tools for comparing outputs between HuggingFace and Megatron models.

#### `compare.py` - Forward Pass Comparison

Compares 1-step generation between HuggingFace and Megatron models with detailed analysis.

**Features:**
- Text and vision-language model comparison
- Multi-GPU comparison support
- Debug hooks for detailed analysis
- Statistical comparison metrics

**Usage:**

**Basic text model comparison:**
```bash
uv run python examples/conversion/compare_hf_and_megatron/compare.py \
  --hf_model_path Qwen/Qwen3-1.7B \
  --prompt "Hello, how are you?"
```

**Vision-language model comparison:**
```bash
uv run python examples/conversion/compare_hf_and_megatron/compare.py \
  --hf_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --model_class Qwen2_5_VLForConditionalGeneration \
  --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
  --prompt "Describe this image."
```

**Multi-GPU comparison:**
```bash
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/compare_hf_and_megatron/compare.py \
  --hf_model_path Qwen/Qwen3-1.7B \
  --prompt "Hello world" \
  --tp 2
```

**With debug hooks:**
```bash
uv run python examples/conversion/compare_hf_and_megatron/compare.py \
  --hf_model_path Qwen/Qwen3-1.7B \
  --prompt "Hello world" \
  --enable_debug_hooks
```

**Example Output:**
```
Processing inputs - Prompt: 'Hello, how are you?', Image: None
Input shape: torch.Size([1, 6])
Pixel values shape: None
=== RUNNING HF MODEL (1-STEP) ===
HF output type: <class 'transformers.modeling_outputs.CausalLMOutputWithPast'>
HF output shape: torch.Size([1, 6, 151936])
HF logits stats - mean: 4.6250, std: 2.5938
HF next token: 358 (' I')
HF Top 5: [(' I', 24.375), (' ', 21.625), (' :', 21.125), (' How', 20.25), (' And', 20.0)]
=== RUNNING MEGATRON MODEL (1-STEP) ===
Megatron output shape: torch.Size([1, 6, 151936])
Megatron logits stats - mean: 4.6507, std: 2.5956
Megatron next token: 358 (' I')
Megatron Top 5: [(' I', 24.5), (' ', 21.625), (' :', 21.125), (' How', 20.375), (' And', 20.125)]
=== COMPARISON ===
Token match: True
Logits diff - max: 0.218750, mean: 0.038388
Cosine similarity: 1.002266
=== COMPARISON COMPLETE ===
```

#### `debugger.py` - Debug Utilities

Provides utilities for deep debugging of model forward passes with detailed logging.

When `--enable_debug_hooks` is enabled, the system generates comprehensive debug logs containing detailed information about neural network module execution during forward and backward passes.

**Generated Files:**
- `hf_debug_fwd_log_<world_size>_rank_<rank>.jsonl`: HuggingFace model forward pass logs
- `megatron_debug_component_<i>_fwd_log_<world_size>_rank_<rank>.jsonl`: Megatron model forward pass logs  
- `debug_bwd_log_<world_size>_rank_<rank>.jsonl`: Backward pass gradient logs

**Log Contents:**
Each log entry captures detailed tensor information for every module:
- **Module Identification**: Hierarchical names (e.g., `"transformer.h.0.attn.c_attn"`)
- **Tensor Fingerprints**: Shape, data type, device, and statistical summaries (min, max, mean, abs_sum)
- **Input/Output Data**: Named parameters and activation values with full statistics
- **Weight Parameters**: Module weights and their statistical properties
- **Gradient Information**: Input and output gradients during backward pass

**Use Cases:**
- **Model Verification**: Compare intermediate results between HuggingFace and Megatron models
- **Numerical Debugging**: Identify divergence points in model conversion

### 9. `adapter/` — LoRA/DoRA Adapter Export & Verification

Scripts for exporting Megatron-Bridge LoRA/DoRA adapter weights to HuggingFace PEFT format and verifying correctness. See [`adapter/README.md`](adapter/README.md) for full details.

| Script | Description |
|---|---|
| `adapter/export_adapter.py` | Export a Megatron PEFT checkpoint to HF PEFT format (CPU-only) |
| `adapter/verify_adapter.py` | Verify exported adapter via logit comparison |
| `adapter/stream_adapter_weights.py` | Stream individual adapter tensors for custom workflows |

**Quick start:**
```bash
# Export
uv run python examples/conversion/adapter/export_adapter.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --megatron-peft-checkpoint /path/to/finetune_ckpt \
    --output-hf-path ./my_adapter

# Verify
uv run python examples/conversion/adapter/verify_adapter.py \
    --hf-model-id meta-llama/Llama-3.2-1B \
    --hf-adapter-path ./my_adapter
```
