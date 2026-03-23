#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Kimi-K2.5-VL Multi-Node Distributed Inference
# Recommended: TP=1, PP=1, EP=4 (8 GPUs, 1 node) or scale EP as needed.
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set MODEL_PATH and optionally MEGATRON_CHECKPOINT
#   3. Set CONTAINER_IMAGE or use NO_CONTAINER=true for bare metal
#   4. Submit: sbatch slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=kimi-vl-inference
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=8          # Tasks per node (1 per GPU)
#SBATCH --gpus-per-node=8            # GPUs per node
#SBATCH --time=02:00:00              # Max run time
#SBATCH --partition=gpu              # Partition name
#SBATCH --account=my_account         # Account name
#SBATCH --output=logs/kimi_vl_inference_%j.out
#SBATCH --error=logs/kimi_vl_inference_%j.err
#SBATCH --exclusive                  # Exclusive node access

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Workspace directory
WORKSPACE=${WORKSPACE:-/workspace}

# Model configuration
MODEL_NAME=Kimi-K2.5

# Option 1: Use HuggingFace model path (will load and convert on-the-fly)
MODEL_PATH=${WORKSPACE}/${MODEL_NAME}
# MODEL_PATH=moonshotai/Kimi-K2.5  # Or use HF Hub path

# Option 2: Use pre-converted Megatron checkpoint (faster)
MEGATRON_CHECKPOINT=""
# MEGATRON_CHECKPOINT=${WORKSPACE}/models/${MODEL_NAME}/iter_0000000

# Inference configuration
IMAGE_PATH="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
PROMPT="Describe this image."
MAX_NEW_TOKENS=1000

# Parallelism configuration
# Full model recommended: TP=1, PP=16, EP=32 (64 GPUs, 8 nodes)
# Adjust to match your allocation: max(TP, EP) × PP = total GPUs
TP=1      # Tensor Parallelism
PP=1      # Pipeline Parallelism
EP=4      # Expert Parallelism (MoE)

# Container configuration (required for SLURM pyxis)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional, space-separated)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /workspace:/workspace"

# Set to true to run without container (bare metal)
NO_CONTAINER=false

# ==============================================================================
# Environment Setup
# ==============================================================================

# NCCL optimizations
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# UV cache on shared filesystem (recommended for multi-node setups)
# Pre-sync once before submitting jobs: UV_CACHE_DIR=/path/to/cache uv sync
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# HuggingFace cache directory (recommended for shared filesystem)
# export HF_HOME="/path/to/shared/HF_HOME"

# Authentication tokens
# export HF_TOKEN="hf_your_token_here"

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Kimi-K2.5-VL Multi-Node Inference"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "Model: $MODEL_NAME"
echo "Parallelism: TP=$TP, PP=$PP, EP=$EP"
echo "======================================"

# Create logs directory
mkdir -p logs

# Calculate total processes
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))
REQUIRED_GPUS=$(( (TP > EP ? TP : EP) * PP ))

# Validate parallelism configuration
if [ $REQUIRED_GPUS -ne $TOTAL_GPUS ]; then
    echo "ERROR: Parallelism mismatch!"
    echo "  max(TP, EP) × PP = max($TP, $EP) × $PP = $REQUIRED_GPUS"
    echo "  Total allocated GPUs = $TOTAL_GPUS"
    echo "  These must be equal!"
    exit 1
fi

MEGATRON_CKPT_ARG=""
if [ -n "$MEGATRON_CHECKPOINT" ]; then
    MEGATRON_CKPT_ARG="--megatron_model_path $MEGATRON_CHECKPOINT"
fi

CMD="uv run --no-sync python examples/models/vlm/kimi_vl/hf_to_megatron_generate_vlm.py \
    --hf_model_path $MODEL_PATH \
    --trust_remote_code \
    $MEGATRON_CKPT_ARG \
    --image_path \"$IMAGE_PATH\" \
    --prompt \"$PROMPT\" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --tp $TP \
    --pp $PP \
    --ep $EP"

# Only rank 0 on each node runs uv sync
SYNC_CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 5; fi"
FULL_CMD="$SYNC_CMD && $CMD"

echo "Executing inference..."
echo "Command: $CMD"
echo "======================================"

# Execute based on container configuration
if [ "$NO_CONTAINER" = true ]; then
    echo "Running without container (bare metal)"
    srun --mpi=pmix bash -c "$FULL_CMD"
else
    # Require container image
    if [ -z "$CONTAINER_IMAGE" ]; then
        echo "ERROR: CONTAINER_IMAGE must be set, or use NO_CONTAINER=true for bare metal."
        exit 1
    fi

    echo "Running with container: $CONTAINER_IMAGE"

    # Build srun command with container
    SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"

    # Add container mounts
    if [ -n "$CONTAINER_MOUNTS" ]; then
        for mount in $CONTAINER_MOUNTS; do
            SRUN_CMD="$SRUN_CMD --container-mounts=$mount"
        done
    fi

    $SRUN_CMD bash -c "$FULL_CMD"
fi

echo "======================================"
echo "Inference completed"
echo "======================================"
