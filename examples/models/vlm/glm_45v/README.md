# GLM-4.5V Examples

This directory contains example scripts for GLM-4.5V vision-language model.

For model introduction and architecture details, see the [GLM-4.5V documentation](../../../../docs/models/vlm/glm-45v.md).

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

### Import HF → Megatron
To import the HF VL model to your desired Megatron path:
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model zai-org/GLM-4.5V \
--megatron-path /models/GLM-4.5V
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model zai-org/GLM-4.5V \
--megatron-path /results/glm_45v/checkpoints/iter_00001000 \
--hf-path ./glm-45v-hf-export
```

See the [conversion.sh](conversion.sh) script for more examples including:
- Multi-GPU round-trip validation between formats

## Inference

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path zai-org/GLM-4.5V \
--megatron_model_path /models/GLM-4.5V \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100 \
--trust_remote_code
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`
- GLM-4.5V requires `--trust_remote_code` flag

See the [inference.sh](inference.sh) script for commands to:
- Run inference with Hugging Face checkpoints
- Run inference with imported Megatron checkpoints
- Run inference with exported Hugging Face checkpoints

**Expected output:**
```text
...
Generation step 46
Generation step 47
Generation step 48
Generation step 49
======== GENERATED TEXT OUTPUT ========
Image: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png
Prompt: Describe this image.
Generated: [gMASK]<sop><|user|>
<|begin_of_image|><|image|>...<|end_of_image|>Describe this image.<|assistant|>
<think>The image shows a technical specifications table comparing two NVIDIA GPU models: H100 SXM and H100 NVL. The table is organized with rows representing different technical specifications and columns for each GPU model.

Here's a breakdown of the information presented:

=======================================
```

## Finetune Recipes

- See: [bridge.recipes.glm_vl](../../../../docs/apidocs/bridge/bridge.recipes.glm_vl.md)
- Available recipes:
  - `glm_45v_finetune_config`: Finetuning for GLM-4.5V model with PEFT support

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretraining

Pretraining is not verified for this model.

### Supervised Fine-Tuning (SFT)

Full parameter fine-tuning requires 64 nodes (512 GPUs) with TP=1, PP=8, EP=16.

**Usage:**
```bash
# 1. Edit slurm_sft.sh to configure:
#    - #SBATCH directives (partition, account, etc.)
#    - CONTAINER_IMAGE path

# 2. Submit the job:
sbatch slurm_sft.sh
```

See [slurm_sft.sh](slurm_sft.sh) for the full Slurm job script.

W&B report coming soon.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

LoRA fine-tuning requires 8 nodes (64 GPUs) with TP=1, PP=8, EP=4.

**Usage:**
```bash
# 1. Edit slurm_peft.sh to configure:
#    - #SBATCH directives (partition, account, etc.)
#    - CONTAINER_IMAGE path

# 2. Submit the job:
sbatch slurm_peft.sh
```

See [slurm_peft.sh](slurm_peft.sh) for the full Slurm job script.

W&B report coming soon.


**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for fewer GPUs. Expert parallelism (EP) is essential for efficient training of this MoE model.

### Recommended Configurations

| Model | Mode | TP | PP | EP | Global Batch Size | Learning Rate | Hardware |
|-------|------|----|----|-----|-------------------|---------------|----------|
| GLM-4.5V | Full SFT | 1 | 8 | 16 | 32 | 5e-6 | 512 GPUs (64 nodes) |
| GLM-4.5V | LoRA/DoRA | 1 | 8 | 4 | 32 | 1e-4 | 64 GPUs (8 nodes) |

### Multi-Node Setup with Local Repository

If you are mounting a local Megatron Bridge repository, you must pre-sync the uv cache to avoid race conditions when multiple nodes attempt to sync simultaneously. Follow these steps:

1. **Start a container with your mounts and run `uv sync`:**
   ```bash
   # Start an interactive container with the same mounts you'll use in Slurm
   srun --nodes=1 --ntasks=1 --container-image=/path/to/container.sqsh \
        --container-mounts=/path/to/Megatron-Bridge:/opt/Megatron-Bridge,/shared/uv_cache:/shared/uv_cache \
        --pty bash
   
   # Inside the container, pre-sync to the shared cache
   cd /opt/Megatron-Bridge
   UV_CACHE_DIR=/shared/uv_cache uv sync
   ```

2. **Update the Slurm script with UV_CACHE_DIR and mounts:**
   ```bash
   # In slurm_sft.sh or slurm_peft.sh, set:
   export UV_CACHE_DIR="/shared/uv_cache"
   
   # And configure container mounts:
   CONTAINER_MOUNTS="/path/to/Megatron-Bridge:/opt/Megatron-Bridge,/shared/uv_cache:/shared/uv_cache"
   ```

3. **Submit the job:**
   ```bash
   sbatch slurm_sft.sh   # or slurm_peft.sh
   ```

## Evaluation

Coming soon.
