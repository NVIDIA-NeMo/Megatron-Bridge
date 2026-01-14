# GLM 4.5

[GLM 4.5](https://huggingface.co/zai-org/GLM-4.5) is a family of large-scale Mixture-of-Experts (MoE) language models developed by Zhipu AI. Built on the GLM (General Language Model) architecture, GLM 4.5 introduces advanced features including sparse MoE layers, shared expert mechanisms, and Multi-Token Prediction (MTP) for improved training efficiency and performance.

GLM 4.5 models are designed for high-performance text generation and understanding tasks, offering excellent quality with efficient inference through sparse activation. The family includes two variants optimized for different deployment scenarios.

GLM 4.5 models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

### GLM 4.5 355B-A32B
**Full Model:** `zai-org/GLM-4.5` (355B total parameters, 32B active per token)
- **Architecture:**
  - 92 transformer layers (first 3 dense, remaining 89 MoE)
  - 5120 hidden size, 12288 FFN hidden size
  - 96 attention heads, 8 query groups (GQA)
  - 160 experts per MoE layer, top-8 routing with 2.5x scaling factor
  - MoE FFN hidden size: 1536
  - Shared expert intermediate size: 1536
  - Includes QK LayerNorm for improved training stability
- **Context:** 131,072 tokens
- **Vocabulary:** 151,552 tokens
- **Optimizations:**
  - Shared expert overlap for better load balancing
  - Router sigmoid scoring with expert bias
  - MTP (Multi-Token Prediction) with 1 layer, 0.3 scaling factor

### GLM 4.5 Air 106B-A12B
**Full Model:** `zai-org/GLM-4.5-Air` (106B total parameters, 12B active per token)
- **Architecture:**
  - 46 transformer layers (first layer dense, remaining 45 MoE)
  - 4096 hidden size, 10944 FFN hidden size
  - 96 attention heads, 8 query groups (GQA)
  - 128 experts per MoE layer, top-8 routing with 1.0x scaling factor
  - MoE FFN hidden size: 1408
  - Shared expert intermediate size: 1408
  - No QK LayerNorm
- **Context:** 131,072 tokens
- **Vocabulary:** 151,552 tokens
- **Optimizations:**
  - Optimized for reduced memory footprint
  - Suitable for mid-range GPU clusters

Both models use RMSNorm, SiLU activation, gated linear units, and RoPE with 1M base frequency.

## Model Architecture Features

GLM 4.5 introduces several advanced architectural innovations:

- **Mixture-of-Experts (MoE)**: Sparse activation with 160/128 experts, activating only 8 per token for efficient scaling
- **Shared Expert Mechanism**: Dedicated shared experts with overlap for improved load balancing and knowledge transfer
- **Multi-Token Prediction (MTP)**: Predicts multiple future tokens simultaneously for better training efficiency
  - Configurable MTP layers (default: 1 layer)
  - Adjustable loss scaling factor (default: 0.3 for early training, 0.1 for later stages)
- **RoPE with Extended Context**: Rotary embeddings with 1M base frequency for robust long-context modeling
- **Advanced Router Design**:
  - Sigmoid score function for better expert selection
  - Expert bias with configurable update rate
  - Auxiliary loss for load balancing
- **Grouped Query Attention (GQA)**: 8 query groups for efficient attention computation
- **RMSNorm**: Fast layer normalization without mean centering

## Conversion with 🤗 Hugging Face

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: GLM 4.5 Air 106B
bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.5-Air")
provider = bridge.to_megatron_provider()

# Configure parallelism before instantiating the model
# For Air 106B: TP=1, PP=4, EP=8 (32 GPUs)
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 4
provider.expert_model_parallel_size = 8
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Import HF → Megatron
```bash
# Import GLM 4.5 Air model
python examples/conversion/convert_checkpoints.py import \
--hf-model zai-org/GLM-4.5-Air \
--megatron-path /models/glm45-air-106b

# Import GLM 4.5 355B model
python examples/conversion/convert_checkpoints.py import \
--hf-model zai-org/GLM-4.5 \
--megatron-path /models/glm45-355b
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model zai-org/GLM-4.5-Air \
--megatron-path /results/glm45_air/checkpoints/iter_00001000 \
--hf-path ./glm45-air-hf-export
```

### Run Inference on Converted Checkpoint
```bash
python examples/conversion/hf_to_megatron_generate_text.py \
--hf_model_path zai-org/GLM-4.5-Air \
--megatron_model_path /models/glm45-air-106b \
--prompt "Explain quantum computing in simple terms." \
--max_new_tokens 200
```

## Pretrain and Finetune Recipes

### Available Recipes
- Pretrain recipes:
  - `glm45_355b_pretrain_config`: Pre-training for GLM 4.5 355B
  - `glm45_air_106b_pretrain_config`: Pre-training for GLM 4.5 Air 106B
- Finetune recipes:
  - `glm45_355b_finetune_config`: Finetuning for GLM 4.5 355B with PEFT support
  - `glm45_air_106b_finetune_config`: Finetuning for GLM 4.5 Air 106B with PEFT support

### Parallelism Configurations

#### GLM 4.5 355B-A32B
| Mode | TP | PP | EP | Total GPUs | Use Case |
|------|----|----|----|-----------:|----------|
| **Pretrain** | 2 | 8 | 16 | 256 | Full pre-training |
| **PEFT (LoRA/DoRA)** | 2 | 4 | 4 | 32 | Parameter-efficient finetuning |
| **Full SFT** | 2 | 8 | 16 | 256 | Full supervised finetuning |

#### GLM 4.5 Air 106B-A12B
| Mode | TP | PP | EP | Total GPUs | Use Case |
|------|----|----|----|-----------:|----------|
| **Pretrain** | 1 | 4 | 8 | 32 | Full pre-training |
| **PEFT (LoRA/DoRA)** | 1 | 2 | 4 | 8 | Parameter-efficient finetuning (1 node!) |
| **Full SFT** | 1 | 4 | 8 | 32 | Full supervised finetuning |

### Pre-training Example

```python
from megatron.bridge.recipes.glm import glm45_air_106b_pretrain_config

# Create a pre-training configuration
config = glm45_air_106b_pretrain_config(
    name="glm45_air_pretrain",
    data_paths=["path/to/data"],
    train_iters=100000,
    global_batch_size=2048,
    micro_batch_size=1,
    lr=1e-4,
    min_lr=1e-5,
    # MTP configuration
    mtp_num_layers=1,
    mtp_loss_scaling_factor=0.3,  # 0.3 for first 15T tokens, 0.1 after
    # Recompute for memory efficiency
    recompute_granularity="selective",
)
```

### Finetuning Examples

#### Full Finetuning (GLM 4.5 Air)
```python
from megatron.bridge.recipes.glm import glm45_air_106b_finetune_config

config = glm45_air_106b_finetune_config(
    name="glm45_air_full_finetune",
    pretrained_checkpoint="/models/glm45-air-106b",
    peft=None,  # Full finetuning
    train_iters=1000,
    global_batch_size=128,
    micro_batch_size=1,
    finetune_lr=5e-6,
)
```

#### LoRA Finetuning (GLM 4.5 Air)
```python
from megatron.bridge.recipes.glm import glm45_air_106b_finetune_config

config = glm45_air_106b_finetune_config(
    name="glm45_air_lora_finetune",
    pretrained_checkpoint="/models/glm45-air-106b",
    peft="lora",  # or "dora"
    train_iters=1000,
    global_batch_size=128,
    micro_batch_size=1,
    finetune_lr=1e-4,
    # Uses TP=1, PP=2, EP=4 (8 GPUs) automatically
)
```

#### DoRA Finetuning (GLM 4.5 355B)
```python
from megatron.bridge.recipes.glm import glm45_355b_finetune_config

config = glm45_355b_finetune_config(
    name="glm45_355b_dora_finetune",
    pretrained_checkpoint="/models/glm45-355b",
    peft="dora",
    train_iters=1000,
    global_batch_size=128,
    micro_batch_size=1,
    finetune_lr=1e-4,
    # Uses TP=2, PP=4, EP=4 (32 GPUs) automatically
)
```

### Command-Line Training

```bash
# GLM 4.5 Air - LoRA finetuning on single node (8 GPUs)
torchrun --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/glm45-air-106b \
--recipe glm45_air_106b_finetune_config \
peft=lora \
train.global_batch_size=128 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/glm45_air_lora

# GLM 4.5 355B - Full finetuning (256 GPUs)
torchrun --nnodes=32 --nproc-per-node=8 run/run_recipe.py \
--pretrained-checkpoint /models/glm45-355b \
--recipe glm45_355b_finetune_config \
peft=None \
train.global_batch_size=256 \
train.train_iters=1000 \
checkpoint.save=$SAVE_DIR/glm45_355b_full
```

## Advanced Configuration

### Multi-Token Prediction (MTP)
MTP can be configured for improved training efficiency:

```python
config = glm45_355b_pretrain_config(
    name="glm45_with_mtp",
    mtp_num_layers=1,  # Number of MTP prediction layers
    mtp_loss_scaling_factor=0.3,  # 0.3 early training, 0.1 later
    # Set to None or 0 to disable MTP
)
```

### Activation Recomputation
For memory-constrained scenarios:

```python
config = glm45_air_106b_pretrain_config(
    name="glm45_with_recompute",
    recompute_granularity="selective",  # or "full"
    recompute_method="uniform",
    recompute_num_layers=2,
)
```

### Expert Parallelism Tuning
Adjust expert parallelism based on your cluster:

```python
config = glm45_air_106b_pretrain_config(
    name="glm45_custom_parallelism",
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,  # Adjust based on GPU count
    sequence_parallel=True,
)
```

## Examples
- Checkpoint import/export: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- Generate text (HF→Megatron): [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Hugging Face Model Cards
- GLM 4.5 355B: https://huggingface.co/zai-org/GLM-4.5
- GLM 4.5 Air 106B: https://huggingface.co/zai-org/GLM-4.5-Air

## Key Features for Production

1. **Efficient Scaling**: MoE architecture enables 355B parameters with only 32B active per token
2. **Single-Node PEFT**: GLM 4.5 Air can be fine-tuned with LoRA/DoRA on just 8 GPUs
3. **Long Context**: Native 131K token context window with optimized RoPE
4. **Multi-Token Prediction**: Faster convergence through MTP training
5. **Flexible Deployment**: Multiple parallelism strategies for different hardware configurations
6. **Load Balancing**: Shared expert overlap and auxiliary loss for optimal expert utilization

## Related Docs
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

