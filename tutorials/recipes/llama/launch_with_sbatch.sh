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

#SBATCH --job-name=megatron-bridge-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --exclusive

# ==============================================================================
# Direct Slurm Launch with sbatch
#
# This script demonstrates how to launch training directly using sbatch without
# NeMo-Run. This is useful if you prefer traditional HPC workflows or don't want
# to install additional dependencies.
#
# Usage:
#   1. Modify the #SBATCH directives above for your cluster
#   2. Set the TRAINING_SCRIPT and other variables below
#   3. Submit: sbatch launch_with_sbatch.sh
#
# For NeMo-Run based launching (recommended), see 04_launch_slurm_with_nemo_run.py
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Modify these for your setup
# ==============================================================================

# Training script to run (choose one)
TRAINING_SCRIPT="00_quickstart_pretrain.py"
# TRAINING_SCRIPT="01_quickstart_finetune.py"
# TRAINING_SCRIPT="02_pretrain_with_yaml.py"
# TRAINING_SCRIPT="03_finetune_with_yaml.py"

# Optional: YAML config file (for *_with_yaml.py scripts)
CONFIG_FILE=""
# CONFIG_FILE="conf/llama32_1b_pretrain.yaml"
# CONFIG_FILE="conf/llama32_1b_finetune.yaml"

# Optional: Additional CLI overrides (for *_with_yaml.py scripts)
CLI_OVERRIDES=""
# CLI_OVERRIDES="train.train_iters=1000 train.global_batch_size=512"

# Optional: For finetuning scripts, specify checkpoint path
PRETRAINED_CHECKPOINT=""
# PRETRAINED_CHECKPOINT="./checkpoints/llama32_1b"

# Container image (optional, only if using containers)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional, space-separated)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /model:/model"

# ==============================================================================
# Environment Setup
# ==============================================================================

# Set common environment variables
# Optional: Set these if needed
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Megatron Bridge Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Script: $TRAINING_SCRIPT"
echo "======================================"

# Build the command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/${TRAINING_SCRIPT}"

# Build torchrun command
CMD="torchrun"
CMD="$CMD --nproc_per_node=$SLURM_GPUS_PER_NODE"
CMD="$CMD --nnodes=$SLURM_JOB_NUM_NODES"
CMD="$CMD --node_rank=\$SLURM_PROCID"
CMD="$CMD --master_addr=\$(scontrol show hostname \$SLURM_NODELIST | head -n1)"
CMD="$CMD --master_port=29500"
CMD="$CMD $SCRIPT_PATH"

# Add config file if specified
if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config-file $CONFIG_FILE"
fi

# Add pretrained checkpoint if specified (for finetuning)
if [ -n "$PRETRAINED_CHECKPOINT" ]; then
    CMD="$CMD --pretrained-checkpoint $PRETRAINED_CHECKPOINT"
fi

# Add CLI overrides if specified
if [ -n "$CLI_OVERRIDES" ]; then
    CMD="$CMD $CLI_OVERRIDES"
fi

echo "Executing: $CMD"
echo "======================================"

# Execute with or without container
if [ -n "$CONTAINER_IMAGE" ]; then
    # With container
    SRUN_CMD="srun --container-image=$CONTAINER_IMAGE"
    
    # Add container mounts
    if [ -n "$CONTAINER_MOUNTS" ]; then
        for mount in $CONTAINER_MOUNTS; do
            SRUN_CMD="$SRUN_CMD --container-mounts=$mount"
        done
    fi
    
    $SRUN_CMD bash -c "$CMD"
else
    # Without container
    srun bash -c "$CMD"
fi

echo "======================================"
echo "Job completed"
echo "======================================"

