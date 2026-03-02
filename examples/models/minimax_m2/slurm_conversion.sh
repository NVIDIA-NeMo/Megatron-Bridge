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
# MiniMax-M2 Checkpoint Conversion (Multi-Node via Slurm)
#
# MiniMax-M2 (MoE: 256 experts, top-8, ~230GB fp8)
# Use this script when TP * EP * PP > 8 (requires more than one 8-GPU node).
# For single-node (TP * EP * PP <= 8), use conversion.sh instead.
#
# Usage:
#   1. Modify the #SBATCH directives and CONFIGURATION section for your cluster
#   2. Submit: sbatch slurm_conversion.sh
#   3. Submit inference after conversion:
#      sbatch --dependency=afterok:<job_id> slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=minimax-m2-convert
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --account=<your-account>
#SBATCH --output=logs/minimax_m2_convert_%j.out
#SBATCH --error=logs/minimax_m2_convert_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION — edit these for your environment
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
PROJECT_DIR=${PROJECT_DIR:-.}
MODEL_NAME=MiniMax-M2
HF_MODEL_ID=MiniMaxAI/$MODEL_NAME
GPUS_PER_NODE=8

TP=2
EP=8
PP=1

CONTAINER_IMAGE=${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to your container path}
CONTAINER_MOUNTS="/lustre:/lustre,${PROJECT_DIR}:/opt/Megatron-Bridge"
CONTAINER_WORKDIR=/opt/Megatron-Bridge

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# ==============================================================================
# Job Execution
# ==============================================================================

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "======================================"
echo "MiniMax-M2 Checkpoint Conversion"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Parallelism: TP=$TP, EP=$EP, PP=$PP"
echo "Total GPUs: $((TP * EP * PP))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "======================================"

mkdir -p logs

SRUN_CMD="srun --ntasks-per-node=1 --no-container-mount-home \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts=$CONTAINER_MOUNTS"

echo ""
echo "Importing HF -> Megatron checkpoint ..."
$SRUN_CMD bash -c "cd $CONTAINER_WORKDIR && \
    if [ \$SLURM_LOCALID -eq 0 ]; then uv sync; else sleep 10; fi && \
    uv run --no-sync python examples/conversion/convert_checkpoints.py import \
    --hf-model $HF_MODEL_ID \
    --megatron-path ${WORKSPACE}/models/$MODEL_NAME \
    --trust-remote-code"
IMPORT_EXIT=$?
if [ $IMPORT_EXIT -ne 0 ]; then
    echo "ERROR: Import failed (exit $IMPORT_EXIT)"
    exit $IMPORT_EXIT
fi

echo "======================================"
echo "Conversion completed successfully"
echo "======================================"
