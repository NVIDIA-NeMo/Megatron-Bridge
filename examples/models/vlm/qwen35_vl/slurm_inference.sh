#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# Qwen3.5-VL Multi-Node Distributed Inference
#
# Qwen3.5-397B-A17B is a large MoE model (397B parameters, 17B active)
# Recommended: TP=2, PP=16, EP=64 for full model (1024 GPUs, 128 nodes)
# For smaller setups: TP=1, PP=8, EP=16 (128 GPUs, 16 nodes)
#
# Usage:
#   1. Modify the #SBATCH directives below for your cluster
#   2. Set MODEL_PATH and CHECKPOINT_PATH as needed
#   3. Set CONTAINER_IMAGE or use --no-container-image for bare metal
#   4. Submit: sbatch slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=qwen35v-inference
#SBATCH --nodes=16                   # Number of nodes (128 GPUs = 16 nodes × 8 GPUs)
#SBATCH --ntasks-per-node=8          # Tasks per node (1 per GPU)
#SBATCH --gpus-per-node=8            # GPUs per node
#SBATCH --time=02:00:00              # Max run time (2 hours)
#SBATCH --partition=gpu              # Partition name
#SBATCH --account=my_account         # Account name
#SBATCH --output=logs/qwen35v_inference_%j.out
#SBATCH --error=logs/qwen35v_inference_%j.err
#SBATCH --exclusive                  # Exclusive node access

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Workspace directory
WORKSPACE=${WORKSPACE:-/workspace}

# Model configuration
MODEL_NAME=Qwen3.5-397B-A17B

# Option 1: Use HuggingFace model path (will load and convert on-the-fly)
MODEL_PATH=${WORKSPACE}/models/Qwen/${MODEL_NAME}
# MODEL_PATH=Qwen/${MODEL_NAME}  # Or use HF Hub path

# Option 2: Use pre-converted Megatron checkpoint (faster)
MEGATRON_CHECKPOINT=${WORKSPACE}/models/${MODEL_NAME}_megatron/iter_0000000
# Comment out to use HF model directly
# MEGATRON_CHECKPOINT=""

# Inference configuration
IMAGE_PATH="https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png"
PROMPT="Describe this image."
MAX_NEW_TOKENS=100

# Parallelism configuration for 128 GPUs (16 nodes × 8 GPUs)
# Constraint: TP × PP × EP = Total GPUs
TP=4      # Tensor Parallelism
PP=8      # Pipeline Parallelism
EP=16     # Expert Parallelism (MoE)
# Total: 1 × 8 × 16 = 128 GPUs

# For 1024 GPUs (128 nodes), use:
# TP=2, PP=16, EP=64  (2 × 16 × 64 = 2048 GPUs - too many)
# TP=2, PP=16, EP=32  (2 × 16 × 32 = 1024 GPUs)

# Container configuration (required for SLURM pyxis)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/nemo-framework.sqsh"

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

# For transformers >= 5.2.0 (required for Qwen3.5)
# Run once: uv add "transformers>=5.2.0"

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Qwen3.5-VL Multi-Node Inference"
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
REQUIRED_GPUS=$((TP * PP * EP))

# Validate parallelism configuration
if [ $REQUIRED_GPUS -ne $TOTAL_GPUS ]; then
    echo "ERROR: Parallelism mismatch!"
    echo "  TP × PP × EP = $TP × $PP × $EP = $REQUIRED_GPUS"
    echo "  Total allocated GPUs = $TOTAL_GPUS"
    echo "  These must be equal!"
    exit 1
fi

# Build inference command
CMD="uv run --no-sync python examples/conversion/hf_to_megatron_generate_vlm.py"
CMD="$CMD --hf_model_path $MODEL_PATH"

# Add Megatron checkpoint if specified
if [ -n "$MEGATRON_CHECKPOINT" ]; then
    CMD="$CMD --megatron_model_path $MEGATRON_CHECKPOINT"
fi

# Add inference parameters
CMD="$CMD --image_path \"$IMAGE_PATH\""
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"

# Add parallelism parameters
CMD="$CMD --tp $TP"
CMD="$CMD --pp $PP"
CMD="$CMD --ep $EP"

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
