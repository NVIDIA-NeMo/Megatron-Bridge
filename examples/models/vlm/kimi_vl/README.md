# Kimi-K2.5-VL Full-Model Guide

Step-by-step guide to run the full Kimi-K2.5-VL pipeline (conversion,
inference, comparison) using the full-size model (~1T params,
384 MoE experts, FP8 expert weights). Multi-node SLURM required.

## Prerequisites

```bash
export WORKSPACE=/your/custom/path
```

Ensure the following are available:
- `HF_TOKEN`: to download `moonshotai/Kimi-K2.5` from HuggingFace Hub
- `HF_HOME`: (optional) to cache downloaded models and datasets
- `WANDB_API_KEY`: (optional) to enable WandB logging

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion (HF → Megatron → HF)

The full model requires multi-node Slurm for conversion.

**Import** the full HF checkpoint into Megatron format (multi-GPU):

```bash
srun --mpi=pmix -A <YOUR_ACCOUNT> \
    --partition batch \
    -N4 \
    -t 4:00:00 \
    --container-image=<CONTAINER_IMAGE> \
    --container-mounts=<YOUR_MOUNT> \
    --no-container-entrypoint \
    --no-container-remap-root \
    --exclusive \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    python examples/conversion/convert_checkpoints_multi_gpu.py import \
        --hf-model moonshotai/Kimi-K2.5 \
        --megatron-path ${WORKSPACE}/models/Kimi-K2.5-megatron \
        --tp 8 --ep 8 --pp 4
```

### Round-Trip Verification

Use [slurm_conversion.sh](slurm_conversion.sh) to sweep multiple parallelism
configs (TP, PP, EP) and verify HF ↔ Megatron round-trip conversion:

```bash
sbatch examples/models/vlm/kimi_vl/slurm_conversion.sh
```

Default configs: `TP=2,EP=48` | `TP=2,PP=2,EP=24` | `TP=4,EP=24`.

## Inference

The full model requires multi-node inference. Recommended parallelism:
TP=2, EP=48, PP=1 (48 GPUs, 6 nodes).

Kimi K2.5 VL uses a model-specific generation script that handles PP layout,
pre-expanding image placeholders for pipeline parallelism, and Kimi processor
patching.

```bash
sbatch examples/models/vlm/kimi_vl/slurm_inference.sh
```

See [slurm_inference.sh](slurm_inference.sh) for configuration details.

Note:
- `--trust_remote_code` is required for Kimi-K2.5 models.
- You can optionally pass `--megatron_model_path` to use a pre-converted
  checkpoint (faster startup).
