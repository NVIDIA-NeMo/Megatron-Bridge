#!/bin/bash
# ==============================================================================
# Kimi-K2.5-VL Supervised Fine-Tuning (SFT)
#
# Full model (~1T params, 384 MoE experts, FP8 expert weights)
# Recipe: kimi_k25_vl_sft_config
# Recommended parallelism: TP=4, PP=4, EP=32
#
# Usage:
#   sbatch slurm_sft.sh
#   sbatch --nodes=128 slurm_sft.sh   # override node count
# ==============================================================================

#SBATCH --job-name=kimi-k25-vl-sft
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --exclusive

# ── Paths (edit these for your environment) ──────────────────────────────
MEGATRON_BRIDGE_PATH=""   # Path to Megatron-Bridge repo
CONTAINER_IMAGE=""        # Path to container .sqsh image
DATA_DIR=""               # Path to data directory (mounted as /opt/data)
HF_HOME_DIR=""            # Path to HuggingFace cache directory
UV_CACHE=""               # Path to UV cache directory
OUTPUT_DIR=""             # Path to save checkpoints and logs
# export HF_TOKEN=""      # HuggingFace token (if needed)

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_MOUNTS="${MEGATRON_BRIDGE_PATH}:/opt/Megatron-Bridge,${DATA_DIR}:/opt/data"
WORKDIR="/opt/Megatron-Bridge"

# ── Tokens / Caches ──────────────────────────────────────────────────────
export HF_HOME="${HF_HOME_DIR}"
export UV_CACHE_DIR="${UV_CACHE}"

# ── Model / Training ─────────────────────────────────────────────────────
HF_MODEL_PATH="moonshotai/Kimi-K2.5"
RECIPE="kimi_k25_vl_sft_config"
DATASET_NAME="cord_v2"
SEQ_LENGTH=4096
TRAIN_ITERS=5000
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=1
SAVE_INTERVAL=2000
LOG_INTERVAL=1
WANDB_PROJECT="megatron-bridge-kimi-vl"

# ── Parallelism ──────────────────────────────────────────────────────────
# Recommended: TP=4, PP=4, EP=32 → 128 GPUs (16 nodes)
TP=4
PP=4
EP=32

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTPX_LOG_LEVEL=WARNING

echo "======================================"
echo "Kimi-K2.5-VL SFT Training"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "TP=$TP PP=$PP EP=$EP (Total GPUs: $((SLURM_JOB_NUM_NODES * 8)))"
echo "Recipe: $RECIPE"
echo "Dataset: $DATASET_NAME"
echo "======================================"

mkdir -p "${MEGATRON_BRIDGE_PATH}/logs"

SAVE_DIR="${OUTPUT_DIR}/kimi_k25_vl_sft"

CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$HF_MODEL_PATH \
    model.seq_length=$SEQ_LENGTH \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.expert_model_parallel_size=$EP \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=$SAVE_DIR \
    checkpoint.save_interval=$SAVE_INTERVAL \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=kimi_k25_vl_${DATASET_NAME}_sft \
    dataset.maker_name=make_${DATASET_NAME}_dataset \
    dataset.seq_length=$SEQ_LENGTH"

CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 15; fi && "
CMD="${CMD}uv run --no-sync python scripts/training/run_recipe.py"
CMD="$CMD --recipe $RECIPE"
CMD="$CMD --step_func vlm_step"
CMD="$CMD --hf_path $HF_MODEL_PATH"
CMD="$CMD $CLI_OVERRIDES"

echo "Command: $CMD"

srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd $WORKDIR && $CMD"

echo "======================================"
echo "SFT training completed"
echo "======================================"
