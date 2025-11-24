# Llama Recipes with Megatron Bridge

This guide shows you how to pretrain and finetune Llama models using Megatron Bridge.

## Quickstart

The fastest way to get started with Megatron Bridge pretraining:

```bash
torchrun --nproc_per_node=1 00_quickstart_pretrain.py
```

This runs Llama 3.2 1B pretraining on a single GPU with mock data.

For finetuning, you need a checkpoint in Megatron format. Convert from HuggingFace:

```bash
python ../../conversion/convert_checkpoints.py import \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama32_1b
```

Then run finetuning:

```bash
torchrun --nproc_per_node=1 01_quickstart_finetune.py \
    --pretrained-checkpoint ./checkpoints/llama32_1b
```

This finetunes Llama 3.2 1B using LoRA on the SQuAD dataset.

To use real data, uncomment and modify in the script:

```python
config.data.data_path = "/path/to/your/dataset"
```

## Configuration with YAML

For more complex configurations, use YAML files and command-line overrides:

```bash
torchrun --nproc_per_node=2 02_pretrain_with_yaml.py \
    --config-file conf/llama32_1b_pretrain.yaml
```

Understanding YAML Configuration:

YAML files should be organized into sections that mirror the `ConfigContainer` structure. Each top-level key corresponds to a configuration section (e.g., `data`, `train`, `model`, `optimizer`). Overrides are applied in a nested manner according to the ConfigContainer fields.

Example YAML (`conf/llama32_1b_pretrain.yaml`):

```yaml
# Each section maps to a ConfigContainer field
data:                              # GPTDatasetConfig
  data_path: /path/to/training/data
  sequence_length: 4096

train:                             # TrainingConfig
  train_iters: 100
  global_batch_size: 256

checkpoint:                        # CheckpointConfig
  save: ./checkpoints/llama32_1b
  save_interval: 50

model:                             # Model Provider
  seq_length: 4096                 # Must match data.sequence_length
  tensor_model_parallel_size: 1
  
optimizer:                         # OptimizerConfig
  lr: 0.0003
```

Override from command line using dot notation:

Command-line overrides follow the same pattern as YAML structure. The first part before the dot indicates which subconfig of ConfigContainer to override (e.g., `train`, `model`, `optimizer`), and the part after the dot specifies the field within that subconfig.

```bash
torchrun --nproc_per_node=2 02_pretrain_with_yaml.py \
    --config-file conf/llama32_1b_pretrain.yaml \
    train.train_iters=5000 \
    train.global_batch_size=512 \
    optimizer.lr=0.0002
```

In this example:
- `train.train_iters=5000` → overrides `ConfigContainer.train.train_iters`
- `optimizer.lr=0.0002` → overrides `ConfigContainer.optimizer.lr`

These example scripts are configured to accept overrides in the priority order (highest to lowest):
1. Command-line overrides (dot notation: `section.field=value`)
2. YAML config file (nested structure)
3. Base recipe defaults (from `llama32_1b_pretrain_config()`)

## Multi-Node Training

### Direct Slurm with sbatch

For traditional HPC workflows without NeMo-Run:

```bash
# 1. Configure launch_with_sbatch.sh
# Edit SBATCH directives and script variables at the top

# 2. Submit job
sbatch launch_with_sbatch.sh
```

The `launch_with_sbatch.sh` script shows how to:
- Configure Slurm job parameters
- Set up multi-node torchrun
- Use containers (optional)
- Pass arguments to training scripts

### NeMo-Run

For better job management and remote launching capabilities:

Prerequisites:

```bash
pip install nemo-run
```

From the Slurm cluster (LocalTunnel):

```bash
python 04_launch_slurm_with_nemo_run.py \
    --script 00_quickstart_pretrain.py \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account
```

From your local machine (SSHTunnel):

```bash
python 04_launch_slurm_with_nemo_run.py \
    --script 00_quickstart_pretrain.py \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --ssh-tunnel \
    --host my-cluster.example.com \
    --user myusername \
    --remote-job-dir /home/myusername/nemo-runs
```

With custom config:

```bash
python 04_launch_slurm_with_nemo_run.py \
    --script 03_finetune_with_yaml.py \
    --nodes 1 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --config-file conf/llama32_1b_finetune.yaml
```

## Finetuning

### Quickstart: Finetune with LoRA

Prerequisites: You need a checkpoint in Megatron format. Convert from HuggingFace:

```bash
python ../../conversion/convert_checkpoints.py import \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama32_1b
```

Run finetuning:

```bash
torchrun --nproc_per_node=1 01_quickstart_finetune.py \
    --pretrained-checkpoint ./checkpoints/llama32_1b
```

By default, this:
- Uses LoRA (Low-Rank Adaptation) for efficient finetuning
- Trains on the SQuAD dataset
- Works on a single GPU
- Llama 3.2 1B model

Customize in the script:

```python
# Use your own dataset (JSONL format)
config.data.data_path = "/path/to/your/dataset.jsonl"

# Adjust LoRA hyperparameters
config.peft.dim = 16  # LoRA rank
config.peft.alpha = 32  # LoRA alpha scaling
```

### Configuration with YAML

For more complex finetuning configurations:

```bash
torchrun --nproc_per_node=2 03_finetune_with_yaml.py \
    --config-file conf/llama32_1b_finetune.yaml
```

Example YAML (`conf/llama32_1b_finetune.yaml`):

```yaml
# Each section maps to a ConfigContainer field
data:                              # FinetuningDatasetConfig
  data_path: /path/to/finetuning_dataset.jsonl
  seq_length: 4096

train:                             # TrainingConfig  
  train_iters: 100
  global_batch_size: 128

checkpoint:                        # CheckpointConfig
  pretrained_checkpoint: /path/to/pretrained/checkpoint
  save: ./checkpoints/llama32_1b_finetuned
  save_interval: 50

peft:                             # PEFT (LoRA config)
  dim: 8      # LoRA rank
  alpha: 16   # LoRA alpha

model:                            # Model Provider
  seq_length: 4096                # Must match data.seq_length
  
optimizer:                        # OptimizerConfig
  lr: 0.0001
```

Override from command line using dot notation:

The first part before the dot indicates which ConfigContainer subconfig to override, and the part after specifies the field.

```bash
torchrun --nproc_per_node=2 03_finetune_with_yaml.py \
    --config-file conf/llama32_1b_finetune.yaml \
    peft.dim=16 \
    train.train_iters=2000
```

Here, `peft.dim=16` overrides `ConfigContainer.peft.dim`.

Full finetuning (no LoRA):

```bash
torchrun --nproc_per_node=2 03_finetune_with_yaml.py \
    --peft none \
    train.train_iters=1000
```

### Working with Checkpoints

**Important:** Finetuning requires checkpoints in Megatron format. You cannot use HuggingFace checkpoints directly.

You can obtain Megatron checkpoints by:

1. Converting from HuggingFace
2. Using Megatron checkpoints from your own pretraining runs

Convert HuggingFace checkpoint to Megatron format:

```bash
python ../../conversion/convert_checkpoints.py import \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama32_1b
```

Use the checkpoint:

```bash
# Command line (quickstart scripts)
torchrun --nproc_per_node=1 01_quickstart_finetune.py \
    --pretrained-checkpoint ./checkpoints/llama32_1b

# YAML config (03_finetune_with_yaml.py)
# In conf/llama32_1b_finetune.yaml:
# checkpoint:
#   pretrained_checkpoint: ./checkpoints/llama32_1b
```
