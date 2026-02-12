# Nemotron 3 Examples

This directory contains example scripts for Nemotron 3 language models.

For model introduction and architecture details, see the Nemotron 3 documentation.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

See the [conversion.sh](conversion.sh) script for checkpoint conversion examples.

### Import HF → Megatron

To import the HF model to your desired Megatron path:

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --trust-remote-code
```

### Export Megatron → HF

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000 \
    --hf-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-hf-export
```

### Round-trip Validation

Multi-GPU round-trip validation between formats:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-load-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000 \
    --tp 2 --pp 2 \
    --trust-remote-code
```

## Training Recipes

- See: [bridge.recipes.nemotronh](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py)
- Available recipes:
  - `nemotron_3_nano_pretrain_config`: Pretraining configuration
  - `nemotron_3_nano_finetune_config`: Finetuning configuration with PEFT support

Before training, ensure the following are configured:
1. **Container Image**: Set `CONTAINER_IMAGE` in the SLURM scripts to your container path
2. **Container Mounts**: (optional) Set `CONTAINER_MOUNTS` for data and workspace directories
3. **Environment Variables**:
   - `HF_TOKEN`: to download models from HF Hub (if required)
   - `HF_HOME`: (optional) to avoid re-downloading models and datasets
   - `WANDB_API_KEY`: (optional) to enable WandB logging

All training scripts use SLURM for containerized multi-node training.

### Pretrain

See the [slurm_pretrain.sh](slurm_pretrain.sh) script for pretraining with configurable model parallelisms.

W&B report coming soon.

### Supervised Fine-Tuning (SFT)

See the [slurm_sft.sh](slurm_sft.sh) script for full parameter fine-tuning.

W&B report coming soon.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [slurm_peft.sh](slurm_peft.sh) script for LoRA fine-tuning.

W&B report coming soon.

## Evaluation

Coming soon.
