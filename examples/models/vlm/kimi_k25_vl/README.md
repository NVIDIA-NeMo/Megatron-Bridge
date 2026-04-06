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
sbatch examples/models/vlm/kimi_k25_vl/slurm_conversion.sh
```

Default configs: `TP=2,EP=48` | `TP=2,PP=2,EP=24` | `TP=4,EP=24`.

## Inference

The full model requires multi-node inference. Recommended parallelism:
TP=2, EP=48, PP=1 (48 GPUs, 6 nodes).

Uses the shared VLM generation script
(`examples/conversion/hf_to_megatron_generate_vlm.py`), which auto-detects
Kimi models and handles processor patching, image-token pre-expansion for PP,
and TP sequence padding for MoE.

```bash
sbatch examples/models/vlm/kimi_k25_vl/slurm_inference.sh
```

See [slurm_inference.sh](slurm_inference.sh) for configuration details.

Note:
- `--trust_remote_code` is required for Kimi-K2.5 models.
- Use `--pp_layout` to specify custom pipeline layouts (e.g.
  `--pp_layout "Et*15|t*15|t*16|t*15L"` for PP=4).
- You can optionally pass `--megatron_model_path` to use a pre-converted
  checkpoint (faster startup).

### Expected Inference Output

With the [Qwen-VL demo image](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg)
and prompt `"Describe this image."`, the model enters `<think>` reasoning mode
before producing the final answer. The first 100 generated tokens look like:

```
<think>The user wants me to describe the image. Let me analyze what I see in the image:

1. **Setting**: A beach scene during what appears to be sunset or sunrise
   (golden hour lighting). The ocean is visible in the background with waves.

2. **Main subjects**:
   - A woman sitting on the sand
   - A large dog (looks like a yellow Labrador or Golden Retriever)

3. **The woman**:
   - Long dark hair
```

The model correctly identifies the beach scene, golden hour lighting, the
woman, and the dog breed. Kimi-K2.5 is a thinking model, so the initial output
is always the internal `<think>` reasoning chain before the final response.

## Supervised Fine-Tuning (SFT)

See [slurm_sft.sh](slurm_sft.sh) for full-parameter SFT. The recipe uses:
- **Recipe**: `kimi_k25_vl_sft_config`
- **Parallelism**: TP=2, PP=16, EP=32 (1024 GPUs, 128 nodes)
- **Optimizer**: Distributed Muon with cosine annealing
- **MoE dispatcher**: alltoall with DeepEP backend
- **Recompute**: full, uniform, 1 layer

```bash
sbatch examples/models/vlm/kimi_k25_vl/slurm_sft.sh
```

Before training, ensure the following are configured:
1. **Container Image**: Set `CONTAINER_IMAGE` in the SLURM script
2. **Container Mounts**: (optional) Set `CONTAINER_MOUNTS` for data and workspace
3. **Environment Variables**:
   - `HF_TOKEN`: to download models from HF Hub
   - `HF_HOME`: (optional) to avoid re-downloading models
   - `WANDB_API_KEY`: (optional) to enable WandB logging

Note:
- `--trust_remote_code` is required for Kimi-K2.5 models.
- The recipe uses `--hf_path` to pass the HF model path for reliable config
  loading in multi-node environments.
