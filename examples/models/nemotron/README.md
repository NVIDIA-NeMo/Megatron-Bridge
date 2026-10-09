# Nemotron Examples

This directory contains model-specific examples for the Nemotron family:

| Model | Parameters | Active Parameters | Subdirectory |
|-------|-----------|-------------------|--------------|
| Nemotron 3 Nano | 30B | A3B | [nemotron_3_nano/](nemotron_3_nano/) |
| Nemotron 3 Nano Omni | 30B | A3B | [nemotron_3_nano_omni/](nemotron_3_nano_omni/) |
| Nemotron 3 Super | 120B | A12B | [nemotron_3_super/](nemotron_3_super/) |
| Nemotron 3 Ultra | 550B | A55B | [nemotron_3_ultra/](nemotron_3_ultra/) |
| Nemotron 3.5 Lightning | 30B | A3B | [nemotron_3_5_lightning/](nemotron_3_5_lightning/) |
| Nemotron 3.5 Super VL | 120B language model | A12B | [nemotron_3_5_super_vl/](nemotron_3_5_super_vl/) (draft) |

## Path migration

Model examples now live directly under this directory. Update saved commands
and example imports to use the model-specific paths above: the former
`nemotron_3/<variant>` hierarchy is flattened, Lightning and Super VL use
`nemotron_3_5_*`, and Omni uses `nemotron_3_nano_omni`. Old example paths
are intentionally removed. The deprecated Nano v2 VL example script and
YAML overrides have also been removed. Public `megatron.bridge` recipe names and model
conversion entry points are unchanged. Historical release branches retain
their own layouts; use the README from the branch you check out.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

Nano and Super have conversion scripts: [nemotron_3_nano/conversion.sh](nemotron_3_nano/conversion.sh), [nemotron_3_super/conversion.sh](nemotron_3_super/conversion.sh). Ultra has Slurm examples for multi-node conversion, inference, and OpenMath training; see [nemotron_3_ultra/](nemotron_3_ultra/) and [Ultra documentation](../../../docs/models/nemotron/nemotron3-ultra.md).

## Training Recipes

Available recipes:

**Nano** ([source](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py)):
- `nemotron_3_nano_pretrain_config`: Pretraining
- `nemotron_3_nano_sft_config`: Supervised fine-tuning
- `nemotron_3_nano_peft_config`: PEFT with LoRA support

**Super** ([source](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_super.py)):
- `nemotron_3_super_pretrain_config`: Pretraining
- `nemotron_3_super_sft_config`: Supervised fine-tuning
- `nemotron_3_super_peft_config`: PEFT with LoRA support

**Ultra** ([source](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_ultra.py)):
- `nemotron_3_ultra_pretrain_config`: Pretraining
- `nemotron_3_ultra_sft_openmathinstruct2_packed_config`: Packed OpenMathInstruct-2 SFT
- `nemotron_3_ultra_peft_openmathinstruct2_packed_config`: Packed OpenMathInstruct-2 PEFT

Before training, ensure the following are configured:
1. **Container Image**: Set `CONTAINER_IMAGE` in the SLURM scripts to your container path
2. **Container Mounts**: (optional) Set `CONTAINER_MOUNTS` for data and workspace directories
3. **Environment Variables**:
   - `HF_TOKEN`: to download models from HF Hub (if required)
   - `HF_HOME`: (optional) to avoid re-downloading models and datasets
   - `WANDB_API_KEY`: (optional) to enable WandB logging

All training scripts use SLURM for containerized multi-node training.

### Nano

See the SLURM scripts in [nemotron_3_nano/](nemotron_3_nano/): [slurm_pretrain.sh](nemotron_3_nano/slurm_pretrain.sh), [slurm_pretrain_fsdp.sh](nemotron_3_nano/slurm_pretrain_fsdp.sh) - fsdp_dtensor is the ckpt format supported as of now. To convert to torch_dcp format follow this [guide](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/megatron_fsdp#sbatch_checkpoint_convertsh) , [slurm_sft.sh](nemotron_3_nano/slurm_sft.sh), [slurm_peft.sh](nemotron_3_nano/slurm_peft.sh).

### Super

See the SLURM scripts in [nemotron_3_super/](nemotron_3_super/): [slurm_pretrain.sh](nemotron_3_super/slurm_pretrain.sh), [slurm_sft.sh](nemotron_3_super/slurm_sft.sh), [slurm_peft.sh](nemotron_3_super/slurm_peft.sh).

### Ultra

See [nemotron_3_ultra/slurm_inference.sh](nemotron_3_ultra/slurm_inference.sh) for the 4-node inference pattern.
For OpenMath training, use [nemotron_3_ultra/slurm_sft.sh](nemotron_3_ultra/slurm_sft.sh) and
[nemotron_3_ultra/slurm_peft.sh](nemotron_3_ultra/slurm_peft.sh), which default to the current
OpenMath tuning starting points.

## Evaluation

Coming soon.
