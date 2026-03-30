# Kimi-K2.5-VL Full-Model Guide

Step-by-step guide to run the full Kimi-K2.5-VL pipeline (conversion,
inference, comparison, training) using the full-size model (~1T params,
384 MoE experts, FP8 expert weights). Multi-node SLURM required.

## Prerequisites

```bash
export WORKSPACE=/your/custom/path
```

Ensure the following are available:
- `HF_TOKEN`: to download `moonshotai/Kimi-K2.5` from HuggingFace Hub
- `HF_HOME`: (optional) to cache downloaded models and datasets
- `WANDB_API_KEY`: (optional) to enable WandB logging

## Step 1: Download the Full Model

The full model is hosted on HuggingFace. Download it or let the scripts
pull it on-the-fly:

```bash
huggingface-cli download moonshotai/Kimi-K2.5 \
    --local-dir ${WORKSPACE}/models/Kimi-K2.5
```

Alternatively, you can pass `moonshotai/Kimi-K2.5` directly to scripts
and they will download automatically (requires `HF_TOKEN`).

## Step 2: Checkpoint Conversion (HF → Megatron → HF)

**Import** the full HF checkpoint into Megatron format:

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model moonshotai/Kimi-K2.5 \
    --megatron-path ${WORKSPACE}/models/Kimi-K2.5-megatron \
    --trust-remote-code
```

For faster multi-GPU conversion, use `convert_checkpoints_multi_gpu.py` via SLURM:

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

**Export** back to HF format for round-trip verification:

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model moonshotai/Kimi-K2.5 \
    --megatron-path ${WORKSPACE}/models/Kimi-K2.5-megatron/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Kimi-K2.5-hf-export
```

## Step 3: HF vs Megatron Comparison

Compare 1-step forward-pass outputs between HuggingFace and Megatron.
The full model requires multi-node with TP=2, EP=48:

```bash
# Requires 48 GPUs (6 nodes × 8 GPUs)
torchrun --nproc_per_node=8 --nnodes=6 \
    examples/models/vlm/kimi_vl/compare.py \
    --hf_model_path moonshotai/Kimi-K2.5 \
    --trust_remote_code \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --tp 2 --ep 48
```

## Step 4: Inference (HF-to-Megatron Generation)

Run greedy auto-regressive generation through the Megatron model.
Recommended parallelism: TP=2, EP=48, PP=1 (48 GPUs, 6+ nodes).

**Via SLURM** (recommended):

```bash
sbatch examples/models/vlm/kimi_vl/slurm_inference.sh
```

Note:
- `--trust_remote_code` is required for Kimi-K2.5 models.
- You can optionally pass `--megatron_model_path` to use a pre-converted checkpoint (faster startup).


## Step 5: SFT Training

Full training run with explicit parallelism, logging, and checkpoint settings.
Recommended parallelism: TP=2, PP=2, EP=64 (128 GPUs, 16 nodes).

NOTE: sft is not test yet, we will update slurm_sft.sh as soon as we test it.

**Via SLURM** (recommended):

```bash
sbatch examples/models/vlm/kimi_vl/slurm_sft.sh
```

Note: Unlike the toy model, no architecture overrides (hidden_size, ffn_hidden_size,
num_moe_experts, etc.) are needed — the recipe loads the full model architecture
from the HuggingFace config automatically.
