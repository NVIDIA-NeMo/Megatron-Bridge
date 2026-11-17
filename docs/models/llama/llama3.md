# Llama 3

[Meta's Llama](https://www.llama.com/models/llama-3/) builds on the general transformer decoder framework with some key additions such as pre-normalization, SwiGLU activations, and Rotary Positional Embeddings (RoPE). More information is available in the companion paper ["Llama: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971). With a wide variety of model sizes - Llama has options for every inference budget.

Llama family models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

Megatron Bridge supports the following Llama model variants:

- **Llama 3.2**: 1B, 3B
- **Llama 3**: 8B, 70B (with 8K, 16K, 64K, 128K context variants)
- **Llama 3.1**: 8B, 70B, 405B (with 128K context length)

All models support both pretraining and finetuning with full parameter updates or PEFT methods (LoRA, DoRA).

## Model Architecture Features

- **Pre-normalization**: RMSNorm before each transformer sub-layer for training stability
- **SwiGLU Activation**: Gated linear units in the feedforward network
- **Rotary Positional Embeddings (RoPE)**: Relative position encoding via rotation matrices
- **Grouped Query Attention (GQA)**: Memory-efficient attention mechanism (70B+ models)
- **Extended Context**: Native support for long sequences up to 128K tokens (Llama 3.1)

## Conversion with 🤗 Hugging Face

### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Llama 3.1 8B
bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3.1-8B")
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Import Checkpoint from HF

```bash
./scripts/conversion/convert.sh import \
  --hf-model meta-llama/Meta-Llama-3.1-8B \
  --megatron-path /checkpoints/llama31_8b_megatron
```

### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Export a trained/finetuned Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/llama31_8b/checkpoints/iter_0000500",
    hf_path="/exports/llama31_8b_hf",
)
```

### Run Inference on Converted Checkpoint

```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path meta-llama/Meta-Llama-3.1-8B \
  --megatron_model_path /checkpoints/llama31_8b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

For more details, see [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Recipes

See: [bridge.recipes.llama.llama3](../../apidocs/bridge/bridge.recipes.llama.llama3.md)

### Available Recipes

- **Pretrain recipes**:
  - `llama32_1b_pretrain_config`, `llama32_3b_pretrain_config`: Llama 3.2 (1B, 3B)
  - `llama3_8b_pretrain_config`: Llama 3 8B with 8K context
  - `llama3_8b_16k_pretrain_config`, `llama3_8b_64k_pretrain_config`, `llama3_8b_128k_pretrain_config`: Llama 3 8B with extended context (16K/64K/128K)
  - `llama3_8b_pretrain_2gpu_h100_fp8cs_config`, `llama3_8b_pretrain_2gpu_h100_fp8mx_config`, `llama3_8b_pretrain_2gpu_h100_nvfp4_config`: Llama 3 8B H100 low precision (FP8/MXFP8/NVFP4)
  - `llama3_70b_pretrain_config`, `llama3_70b_16k_pretrain_config`, `llama3_70b_64k_pretrain_config`: Llama 3 70B (8K/16K/64K context)
  - `llama31_8b_pretrain_config`, `llama31_70b_pretrain_config`, `llama31_405b_pretrain_config`: Llama 3.1 (8B/70B/405B, 128K context)
  - `llama31_405b_pretrain_64gpu_gb200_fp8mx_config`: Llama 3.1 405B on 64 GB200 GPUs, 8K context, MXFP8, no recompute

- **SFT recipes**:
  - `llama32_1b_sft_config`, `llama32_3b_sft_config`: Llama 3.2 full SFT
  - `llama3_8b_sft_config`, `llama31_8b_sft_config`: Llama 3/3.1 8B full SFT
  - `llama3_70b_sft_config`, `llama31_70b_sft_config`: Llama 3/3.1 70B full SFT
  - `llama31_405b_sft_config`: Llama 3.1 405B full SFT

- **PEFT recipes** (LoRA, DoRA):
  - `llama32_1b_peft_config`, `llama32_3b_peft_config`: Llama 3.2 PEFT
  - `llama3_8b_peft_config`, `llama31_8b_peft_config`: Llama 3/3.1 8B PEFT
  - `llama3_70b_peft_config`, `llama31_70b_peft_config`: Llama 3/3.1 70B PEFT
  - `llama31_405b_peft_config`: Llama 3.1 405B PEFT

### Parallelism Configurations

#### Llama 3.2 (1B, 3B)
| Model | Mode | TP | PP | Total GPUs | Use Case |
|-------|------|----|----|------------|----------|
| **1B / 3B** | Pretrain | 1 | 1 | 8 | Pre-training (single node) |
| **1B / 3B** | Full SFT | 1 | 1 | 8 | Full supervised finetuning |
| **1B / 3B** | LoRA/DoRA | 1 | 1 | 8 | PEFT finetuning |

#### Llama 3 / 3.1 (8B)
| Model | Mode | TP | PP | CP | Total GPUs | Use Case |
|-------|------|----|----|----|-----------:|----------|
| **8B** | Pretrain | 1 | 1 | 2 | 16 | Pre-training |
| **8B** | Full SFT | 2 | 1 | 1 | 16 | Full supervised finetuning |
| **8B** | LoRA/DoRA | 1 | 1 | 1 | 8 | PEFT finetuning (single node) |

#### Llama 3 / 3.1 (70B)
| Model | Mode | TP | PP | VP | CP | Total GPUs | Use Case |
|-------|------|----|----|----|----|------------|----------|
| **70B** | Pretrain | 4 | 4 | 5 | 2 | 64 | Pre-training |
| **70B** | Full SFT | 8 | 4 | - | 1 | 256 | Full supervised finetuning (32 nodes) |
| **70B** | LoRA/DoRA | 8 | 1 | - | 1 | 8 | PEFT finetuning (single node!) |

#### Llama 3.1 (405B)
| Model | Mode | TP | PP | VP | CP | Total GPUs | Use Case |
|-------|------|----|----|----|----|------------|----------|
| **405B** | Pretrain | 8 | 8 | 2 | 4 | 512 | Pre-training (64 nodes) |
| **405B** | Pretrain MXFP8, 8K | 8 | 8 | 8 | 1 | 64 | GB200, 16 four-GPU nodes |
| **405B** | Full SFT | 8 | 16 | - | 1 | 2048 | Full supervised finetuning (256 nodes) |
| **405B** | LoRA/DoRA | 4 | 8 | 8 | 1 | 256 | PEFT finetuning (32 nodes) |

**Key Features**:
- **Context Parallelism**: Enabled for long context training (16K/64K/128K variants)
- **Sequence Parallel**: Enabled by default for larger models (70B+) for memory efficiency
- **Low Precision Training**: FP8, MXFP8, NVFP4 options available for 8B model
- **Virtual Pipeline**: VP parallelism for 70B and 405B models

### Pre-training Example

#### Llama 3.1 405B on 64 GB200 GPUs

`llama31_405b_pretrain_64gpu_gb200_fp8mx_config()` configures the full 126-layer
model at sequence length 8192 with TP8, PP8, VPP8, CP1, and sequence parallelism.
Place all 16 four-GPU nodes within one NVLink domain. At 64 GPUs, data
parallelism is one; global batch 1536 and microbatch one accumulate 1536
microbatches per optimizer update. Keep this global batch fixed when launching.

The recipe uses MXFP8 with FP8 parameter gather and gradient-buffer reuse,
BF16 gradient accumulation, and FP32 optimizer master parameters and Adam
moments. Activation recomputation and CUDA graphs are disabled. Tensor-parallel
communication overlap and the Transformer Engine RNG tracker are enabled.
There is no optimizer or activation offload. Training schedule, checkpointing,
and numerical checks retain their base training defaults.

Validation in NeMo 26.10.rc10 on 64 GB200 GPUs completed 10 updates on real
WikiText-103 tokenized with the native Llama 3.1 tokenizer. All ranks exited
successfully with finite losses and gradients and no skipped updates. Median
update time after warmup was 289 seconds, approximately 1718 model TFLOP/s/GPU.
Peak allocated/reserved GPU memory across all ranks was 154.6/157.1 GiB.
This validates short-run capacity and stability; long-term convergence and
MXFP8 checkpoint save/resume were not tested. Enabling CUDA graphs exceeded
memory in the tested configuration.

The default dataset is synthetic. For real-text training, supply an indexed
dataset produced with the native Llama 3.1 tokenizer (128256 vocabulary entries):

```python
from megatron.bridge.recipes.llama import llama31_405b_pretrain_64gpu_gb200_fp8mx_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

cfg = llama31_405b_pretrain_64gpu_gb200_fp8mx_config()
cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
cfg.tokenizer.tokenizer_model = "meta-llama/Meta-Llama-3.1-405B"
cfg.dataset.blend = (["/path/to/llama-tokenized-data"], [1.0])
cfg.dataset.split = "9998,1,1"
cfg.checkpoint.save = "/path/to/checkpoints"
cfg.checkpoint.load = cfg.checkpoint.save
pretrain(cfg, forward_step)
```

Launch this script with 64 distributed processes, one per GPU. Dataset prefixes
refer to the corresponding `.bin` and `.idx` files without their extensions.
The recipe is also exported as
`--recipe llama31_405b_pretrain_64gpu_gb200_fp8mx_config` for
`scripts/training/run_recipe.py`.

#### Llama 3 8B

```python
from megatron.bridge.recipes.llama import llama3_8b_pretrain_config

config = llama3_8b_pretrain_config(
    name="llama3_8b_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/llama3_8b",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=8192,
    # Uses TP=1, PP=1, CP=2 (16 GPUs) automatically
)
```

### Finetuning Examples

**Before finetuning**, ensure these environment variables are set:
- `SAVE_DIR`: checkpoint and log saving directory
- `HF_TOKEN`: to download models from HF Hub (if required)
- `HF_HOME`: (optional) to avoid re-downloading models and datasets
- `WANDB_API_KEY`: (optional) to enable WandB logging

#### Full Finetuning (Llama 3 8B)

```python
from megatron.bridge.recipes.llama import llama3_8b_sft_config

cfg = llama3_8b_sft_config(
    name="llama3_8b_full_sft",
    pretrained_checkpoint="/results/llama3_8b/checkpoints/iter_0500000",
    train_iters=1000,
    global_batch_size=64,
    finetune_lr=5e-6,
    # Uses TP=2, PP=1 (16 GPUs) automatically
)
```

#### LoRA Finetuning (8B)

```python
from megatron.bridge.recipes.llama import llama3_8b_peft_config

cfg = llama3_8b_peft_config(
    name="llama3_8b_lora",
    pretrained_checkpoint="/results/llama3_8b/checkpoints/iter_0500000",
    peft_scheme="lora",  # or "dora" for DoRA
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # Uses TP=1, PP=1 (8 GPUs) automatically
)
```

#### LoRA Finetuning (70B)

```python
from megatron.bridge.recipes.llama import llama3_70b_peft_config

cfg = llama3_70b_peft_config(
    name="llama3_70b_lora",
    pretrained_checkpoint="/results/llama3_70b/checkpoints/iter_0500000",
    peft_scheme="lora",
    train_iters=1000,
    global_batch_size=128,
    finetune_lr=1e-4,
    # Uses TP=8, PP=1 (8 GPUs) automatically
)
```

## Hugging Face Model Cards & References

### Hugging Face Model Cards
- Llama 3.2 1B: https://huggingface.co/meta-llama/Llama-3.2-1B
- Llama 3.2 3B: https://huggingface.co/meta-llama/Llama-3.2-3B
- Llama 3 8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B
- Llama 3 70B: https://huggingface.co/meta-llama/Meta-Llama-3-70B
- Llama 3.1 8B: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
- Llama 3.1 70B: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B
- Llama 3.1 405B: https://huggingface.co/meta-llama/Meta-Llama-3.1-405B

### Technical Papers
- Llama: Open and Efficient Foundation Language Models: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- The Llama 3 Herd of Models: [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

## Related Docs
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
