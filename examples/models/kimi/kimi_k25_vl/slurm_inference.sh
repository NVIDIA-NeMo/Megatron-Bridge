#!/bin/bash
# ==============================================================================
# Kimi-K2.5-VL Multi-Node Inference
#
# Full model (~1T params, 384 MoE experts, FP8 expert weights)
# Config: TP=2, EP=48, PP=1 -> 48 GPUs (6 nodes)
#
# Supports both HF checkpoints (on-the-fly conversion) and pre-converted
# Megatron checkpoints (faster startup).
#
# Usage:
#   sbatch slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=kimi-k25-vl-infer
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --exclusive

# ── Paths (edit these for your environment) ──────────────────────────────
MEGATRON_BRIDGE_PATH=""   # Path to Megatron-Bridge repo
CONTAINER_IMAGE=""        # Path to container .sqsh image
DATA_DIR=""               # Path to data directory (mounted as /opt/data)
HF_HOME_DIR=""            # Path to HuggingFace cache directory
UV_CACHE=""               # Path to UV cache directory
# export HF_TOKEN=""      # HuggingFace token (if needed)

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_MOUNTS="${MEGATRON_BRIDGE_PATH}:/opt/Megatron-Bridge,${DATA_DIR}:/opt/data"
WORKDIR="/opt/Megatron-Bridge"

# ── Tokens / Caches ──────────────────────────────────────────────────────
export HF_HOME="${HF_HOME_DIR}"
export UV_CACHE_DIR="${UV_CACHE}"

# ── Model / Parallelism ──────────────────────────────────────────────────
HF_MODEL_PATH="moonshotai/Kimi-K2.5"
TP=2
EP=48
PP=1

# Option: Use pre-converted Megatron checkpoint (faster startup)
# Set to empty string to load from HF directly
MEGATRON_CHECKPOINT=""
# MEGATRON_CHECKPOINT="/path/to/megatron/checkpoint/iter_0000000"

# ── Inference Config ─────────────────────────────────────────────────────
IMAGE_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
PROMPT="Describe this image."
MAX_NEW_TOKENS=200

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================"
echo "Kimi-K2.5-VL Inference"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "TP=$TP PP=$PP EP=$EP (Total GPUs: $((SLURM_JOB_NUM_NODES * 8)))"
echo "======================================"

mkdir -p "${MEGATRON_BRIDGE_PATH}/logs"

MEGATRON_CKPT_ARG=""
if [ -n "$MEGATRON_CHECKPOINT" ]; then
    MEGATRON_CKPT_ARG="--megatron_model_path $MEGATRON_CHECKPOINT"
    echo "Using Megatron checkpoint: $MEGATRON_CHECKPOINT"
else
    echo "Using HF model (on-the-fly conversion): $HF_MODEL_PATH"
fi

CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 15; fi && "
CMD="${CMD}uv run --no-sync python examples/conversion/hf_to_megatron_generate_vlm.py"
CMD="$CMD --hf_model_path $HF_MODEL_PATH"
CMD="$CMD --trust_remote_code"
CMD="$CMD $MEGATRON_CKPT_ARG"
CMD="$CMD --image_path \"$IMAGE_URL\""
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --tp $TP --ep $EP --pp $PP"

echo "Command: $CMD"

srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd $WORKDIR && $CMD"

echo "======================================"
echo "Inference completed"
echo "======================================"
