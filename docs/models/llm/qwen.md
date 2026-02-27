# Qwen

[Qwen](https://huggingface.co/Qwen) is a family of large language models developed by Alibaba Cloud, including dense models (Qwen2, Qwen2.5, Qwen3) and Mixture-of-Experts models (Qwen3 MoE, Qwen3-Next). The models feature innovations like QK layernorm, Gated-Delta Networks, and Zero-Centered RMSNorm for improved training stability and performance.

Qwen family models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

Megatron Bridge supports the following Qwen model variants:

### Dense Models
- **Qwen2**: 0.5B, 1.5B, 7B, 72B
- **Qwen2.5**: 0.5B, 1.5B, 7B, 14B, 32B, 72B  
- **Qwen3**: 0.6B, 1.7B, 4B, 8B, 14B, 32B

### MoE Models
- **Qwen3 MoE**: 30B (3B activated), 235B (22B activated)
- **Qwen3-Next**: 80B (3B activated)

## Model Architecture Features

### Common Features
- **Pre-normalization**: RMSNorm before each transformer sub-layer
- **SwiGLU Activation**: Gated linear units in the feedforward network
- **Rotary Positional Embeddings (RoPE)**: Relative position encoding
- **Grouped Query Attention (GQA)**: Memory-efficient attention mechanism

### Qwen3-Next-Specific Features
- **Gated-Delta Networks**: Advanced gating mechanism for improved learning
- **Zero-Centered RMSNorm**: Centered normalization for training stability
- **Multi-Token Prediction (MTP)**: Auxiliary training objective

### Qwen3-Specific Features
- **QK Layernorm**: Layer normalization on query and key projections
- **QK Layernorm Weight Decay**: Weight decay applied during training

### Qwen2-Specific Features
- **Bias in QKV**: Bias terms in query, key, value projections

---

## Qwen3-Next

### Conversion with 🤗 Hugging Face

#### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Qwen3-Next-80B-A3B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 8
provider.expert_model_parallel_size = 16

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### Import Checkpoint from HF

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --megatron-path /checkpoints/qwen3_next_80b_megatron
```

#### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

# Export a trained Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/qwen3_next_80b/checkpoints/iter_0000500",
    hf_path="/exports/qwen3_next_80b_hf",
)
```

#### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --megatron_model_path /checkpoints/qwen3_next_80b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2 \
  --pp 8 \
  --ep 16
```

### Recipes

#### Available Recipes
- `qwen3_next_80b_a3b_pretrain_config`: Pre-training for Qwen3-Next-80B-A3B
- `qwen3_next_80b_a3b_finetune_config`: Finetuning for Qwen3-Next-80B-A3B (Full SFT only)

#### Parallelism Configuration

| Model | Mode | TP | PP | EP | Total GPUs | Use Case |
|-------|------|----|----|----|-----------:|----------|
| **Qwen3-Next-80B** | Pretrain | 2 | 8 | 16 | 256 | Pre-training (32 nodes) |
| **Qwen3-Next-80B** | Full SFT | 2 | 8 | 16 | 256 | Full supervised finetuning (32 nodes) |

#### Pre-training Example

```python
from megatron.bridge.recipes.qwen import qwen3_next_80b_a3b_pretrain_config

config = qwen3_next_80b_a3b_pretrain_config(
    name="qwen3_next_80b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_next_80b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # Uses TP=2, PP=8, EP=16 (256 GPUs) automatically
)
```

#### Finetuning Example

```python
from megatron.bridge.recipes.qwen import qwen3_next_80b_a3b_finetune_config

config = qwen3_next_80b_a3b_finetune_config(
    name="qwen3_next_80b_full_sft",
    pretrained_checkpoint="/results/qwen3_next_80b/checkpoints/iter_0500000",
    peft=None,  # Full supervised finetuning
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # Uses TP=2, PP=8, EP=16 (256 GPUs) automatically
)
```

**Note**: PEFT (LoRA/DoRA) finetuning is not currently available for Qwen3-Next models.

### Hugging Face Model Cards

- Qwen3-Next-80B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-Next-80B-A3B-Thinking: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking

---

## Qwen3 MoE

### Conversion with 🤗 Hugging Face

#### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Qwen3-30B-A3B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B")
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### Import Checkpoint from HF

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-30B-A3B \
  --megatron-path /checkpoints/qwen3_30b_a3b_megatron
```

#### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B")

# Export a trained Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/qwen3_30b_a3b/checkpoints/iter_0000500",
    hf_path="/exports/qwen3_30b_a3b_hf",
)
```

#### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen3-30B-A3B \
  --megatron_model_path /checkpoints/qwen3_30b_a3b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --ep 8
```

### Recipes

#### Available Recipes
- `qwen3_30b_a3b_pretrain_config`: Pre-training for Qwen3-30B-A3B (30B parameters, 3B activated)
- `qwen3_235b_a22b_pretrain_config`: Pre-training for Qwen3-235B-A22B (235B parameters, 22B activated)
- `qwen3_30b_a3b_finetune_config`: Finetuning for Qwen3-30B-A3B with PEFT support
- `qwen3_235b_a22b_finetune_config`: Finetuning for Qwen3-235B-A22B with PEFT support

#### Parallelism Configuration

| Model | Mode | TP | PP | EP | Total GPUs | Use Case |
|-------|------|----|----|----|-----------:|----------|
| **Qwen3-30B-A3B** | Pretrain | 1 | 1 | 8 | 8 | Pre-training (single node) |
| **Qwen3-30B-A3B** | Full SFT | 1 | 1 | 8 | 8 | Full supervised finetuning |
| **Qwen3-30B-A3B** | LoRA/DoRA | 1 | 1 | 8 | 8 | PEFT finetuning (single node) |
| **Qwen3-235B-A22B** | Pretrain | 2 | 8 | 32 | 512 | Pre-training (64 nodes) |
| **Qwen3-235B-A22B** | Full SFT | 2 | 8 | 32 | 512 | Full supervised finetuning (64 nodes) |
| **Qwen3-235B-A22B** | LoRA/DoRA | 2 | 8 | 32 | 512 | PEFT finetuning (64 nodes) |

#### Pre-training Examples

**Qwen3-30B-A3B:**

```python
from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config

config = qwen3_30b_a3b_pretrain_config(
    name="qwen3_30b_a3b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_30b_a3b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # Uses TP=1, PP=1, EP=8 (8 GPUs) automatically
)
```

**Qwen3-235B-A22B**

```python
from megatron.bridge.recipes.qwen import qwen3_235b_a22b_pretrain_config

config = qwen3_235b_a22b_pretrain_config(
    name="qwen3_235b_a22b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_235b_a22b",
    train_iters=500_000,
    global_batch_size=4096,
    seq_length=4096,
    # Uses TP=2, PP=8, EP=32 (512 GPUs) automatically
)
```

#### Finetuning Examples

**Full Finetuning (30B):**

```python
from megatron.bridge.recipes.qwen import qwen3_30b_a3b_finetune_config

config = qwen3_30b_a3b_finetune_config(
    name="qwen3_30b_a3b_full_sft",
    pretrained_checkpoint="/results/qwen3_30b_a3b/checkpoints/iter_0500000",
    peft=None,
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # Uses TP=1, PP=1, EP=8 (8 GPUs) automatically
)
```

**LoRA Finetuning (30B):**

```python
from megatron.bridge.recipes.qwen import qwen3_30b_a3b_finetune_config

config = qwen3_30b_a3b_finetune_config(
    name="qwen3_30b_a3b_lora",
    pretrained_checkpoint="/results/qwen3_30b_a3b/checkpoints/iter_0500000",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # Uses TP=1, PP=1, EP=8 (8 GPUs) automatically
)
```

### Hugging Face Model Cards

- Qwen3-30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen3-235B-A22B: https://huggingface.co/Qwen/Qwen3-235B-A22B

---

## Qwen3

### Conversion with 🤗 Hugging Face

#### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Qwen3-8B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B")
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### Import Checkpoint from HF

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-8B \
  --megatron-path /checkpoints/qwen3_8b_megatron
```

#### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B")

# Export a trained Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/qwen3_8b/checkpoints/iter_0000500",
    hf_path="/exports/qwen3_8b_hf",
)
```

#### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen3-8B \
  --megatron_model_path /checkpoints/qwen3_8b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

### Recipes

#### Available Recipes
- **Pretrain recipes**: `qwen3_600m_pretrain_config`, `qwen3_1p7b_pretrain_config`, `qwen3_4b_pretrain_config`, `qwen3_8b_pretrain_config`, `qwen3_14b_pretrain_config`, `qwen3_32b_pretrain_config`
- **Finetune recipes**: `qwen3_600m_finetune_config`, `qwen3_1p7b_finetune_config`, `qwen3_4b_finetune_config`, `qwen3_8b_finetune_config`, `qwen3_14b_finetune_config`, `qwen3_32b_finetune_config` (all with PEFT support)

#### Parallelism Configuration

| Model | Mode | TP | PP | Total GPUs | Use Case |
|-------|------|----|----|------------|----------|
| **Qwen3 (0.6B-4B)** | Pretrain | 1 | 1 | 8 | Pre-training (single node) |
| **Qwen3 (0.6B-4B)** | Full SFT | 1 | 1 | 8 | Full supervised finetuning |
| **Qwen3 (0.6B-4B)** | LoRA/DoRA | 1 | 1 | 8 | PEFT finetuning (single node) |
| **Qwen3 (8B-14B)** | Pretrain | 2 | 1 | 16 | Pre-training (2 nodes) |
| **Qwen3 (8B-14B)** | Full SFT | 2 | 1 | 16 | Full supervised finetuning (2 nodes) |
| **Qwen3 (8B-14B)** | LoRA/DoRA | 1 | 1 | 8 | PEFT finetuning (single node!) |
| **Qwen3-32B** | Pretrain | 4 | 1 | 32 | Pre-training (4 nodes) |
| **Qwen3-32B** | Full SFT | 4 | 1 | 32 | Full supervised finetuning (4 nodes) |
| **Qwen3-32B** | LoRA/DoRA | 2 | 1 | 16 | PEFT finetuning (2 nodes) |

#### Pre-training Example

```python
from megatron.bridge.recipes.qwen import qwen3_8b_pretrain_config

config = qwen3_8b_pretrain_config(
    name="qwen3_8b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_8b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # Uses TP=2, PP=1 (16 GPUs) automatically
)
```

#### Finetuning Examples

**Full Finetuning (8B):**

```python
from megatron.bridge.recipes.qwen import qwen3_8b_finetune_config

config = qwen3_8b_finetune_config(
    name="qwen3_8b_full_sft",
    pretrained_checkpoint="/results/qwen3_8b/checkpoints/iter_0500000",
    peft=None,
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # Uses TP=2, PP=1 (16 GPUs) automatically
)
```

**LoRA Finetuning (8B):**

```python
from megatron.bridge.recipes.qwen import qwen3_8b_finetune_config

config = qwen3_8b_finetune_config(
    name="qwen3_8b_lora",
    pretrained_checkpoint="/results/qwen3_8b/checkpoints/iter_0500000",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # Uses TP=1, PP=1 (8 GPUs) automatically
)
```

### Hugging Face Model Cards

- Qwen3 Collection: https://huggingface.co/collections/Qwen/qwen3

---

## Qwen2 / Qwen2.5

### Conversion with 🤗 Hugging Face

#### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Qwen2.5-7B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-7B")
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

#### Import Checkpoint from HF

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen2.5-7B \
  --megatron-path /checkpoints/qwen25_7b_megatron
```

#### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-7B")

# Export a trained Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/qwen25_7b/checkpoints/iter_0000500",
    hf_path="/exports/qwen25_7b_hf",
)
```

#### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path Qwen/Qwen2.5-7B \
  --megatron_model_path /checkpoints/qwen25_7b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

### Recipes

#### Available Recipes
- **Qwen2 Pretrain**: `qwen2_500m_pretrain_config`, `qwen2_1p5b_pretrain_config`, `qwen2_7b_pretrain_config`, `qwen2_72b_pretrain_config`
- **Qwen2.5 Pretrain**: `qwen25_500m_pretrain_config`, `qwen25_1p5b_pretrain_config`, `qwen25_7b_pretrain_config`, `qwen25_14b_pretrain_config`, `qwen25_32b_pretrain_config`, `qwen25_72b_pretrain_config`
- **Qwen2 Finetune**: `qwen2_500m_finetune_config`, `qwen2_1p5b_finetune_config`, `qwen2_7b_finetune_config`, `qwen2_72b_finetune_config` (all with PEFT support)
- **Qwen2.5 Finetune**: `qwen25_500m_finetune_config`, `qwen25_1p5b_finetune_config`, `qwen25_7b_finetune_config`, `qwen25_14b_finetune_config`, `qwen25_32b_finetune_config`, `qwen25_72b_finetune_config` (all with PEFT support)

#### Parallelism Configuration

| Model | Mode | TP | PP | Total GPUs | Use Case |
|-------|------|----|----|------------|----------|
| **Qwen2/2.5 (0.5B-1.5B)** | Pretrain | 1 | 1 | 8 | Pre-training (single node) |
| **Qwen2/2.5 (0.5B-1.5B)** | Full SFT | 1 | 1 | 8 | Full supervised finetuning |
| **Qwen2/2.5 (0.5B-1.5B)** | LoRA/DoRA | 1 | 1 | 8 | PEFT finetuning (single node) |
| **Qwen2/2.5 (7B-14B)** | Pretrain | 2 | 1 | 16 | Pre-training (2 nodes) |
| **Qwen2/2.5 (7B-14B)** | Full SFT | 2 | 1 | 16 | Full supervised finetuning (2 nodes) |
| **Qwen2/2.5 (7B-14B)** | LoRA/DoRA | 1 | 1 | 8 | PEFT finetuning (single node!) |
| **Qwen2.5-32B** | Pretrain | 4 | 1 | 32 | Pre-training (4 nodes) |
| **Qwen2.5-32B** | Full SFT | 4 | 1 | 32 | Full supervised finetuning (4 nodes) |
| **Qwen2.5-32B** | LoRA/DoRA | 2 | 1 | 16 | PEFT finetuning (2 nodes) |
| **Qwen2/2.5-72B** | Pretrain | 8 | 1 | 64 | Pre-training (8 nodes) |
| **Qwen2/2.5-72B** | Full SFT | 8 | 1 | 64 | Full supervised finetuning (8 nodes) |
| **Qwen2/2.5-72B** | LoRA/DoRA | 4 | 1 | 32 | PEFT finetuning (4 nodes) |

#### Pre-training Example

```python
from megatron.bridge.recipes.qwen import qwen25_7b_pretrain_config

config = qwen25_7b_pretrain_config(
    name="qwen25_7b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen25_7b",
    train_iters=500_000,
    global_batch_size=2048,
    seq_length=4096,
    # Uses TP=2, PP=1 (16 GPUs) automatically
)
```

#### Finetuning Examples

**Full Finetuning (7B):**

```python
from megatron.bridge.recipes.qwen import qwen25_7b_finetune_config

config = qwen25_7b_finetune_config(
    name="qwen25_7b_full_sft",
    pretrained_checkpoint="/results/qwen25_7b/checkpoints/iter_0500000",
    peft=None,
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # Uses TP=2, PP=1 (16 GPUs) automatically
)
```

**LoRA Finetuning (7B):**

```python
from megatron.bridge.recipes.qwen import qwen25_7b_finetune_config

config = qwen25_7b_finetune_config(
    name="qwen25_7b_lora",
    pretrained_checkpoint="/results/qwen25_7b/checkpoints/iter_0500000",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # Uses TP=1, PP=1 (8 GPUs) automatically
)
```

### Hugging Face Model Cards

- Qwen2 Collection: https://huggingface.co/collections/Qwen/qwen2
- Qwen2.5 Collection: https://huggingface.co/collections/Qwen/qwen25

---

## Related Docs
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
