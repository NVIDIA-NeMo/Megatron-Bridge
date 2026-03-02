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
# MiniMax-M2 Inference (Multi-Node via Slurm)
#
# MiniMax-M2 (MoE: 256 experts, top-8, ~230GB fp8)
# Use this script when TP * EP * PP > 8 (requires more than one 8-GPU node).
# For single-node (TP * EP * PP <= 8), use inference.sh instead.
#
# Usage:
#   1. Modify the #SBATCH directives and CONFIGURATION section for your cluster
#   2. Run conversion first: sbatch slurm_conversion.sh
#   3. Submit with dependency: sbatch --dependency=afterok:<convert_job_id> slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=minimax-m2-inference
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --account=<your-account>
#SBATCH --output=logs/minimax_m2_inference_%j.out
#SBATCH --error=logs/minimax_m2_inference_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION — edit these for your environment
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
PROJECT_DIR=${PROJECT_DIR:-.}
MODEL_NAME=MiniMax-M2
HF_MODEL_ID=MiniMaxAI/$MODEL_NAME
MEGATRON_CKPT=${WORKSPACE}/models/${MODEL_NAME}/iter_0000000
GPUS_PER_NODE=8
PROMPT="What is artificial intelligence?"
MAX_NEW_TOKENS=100

# MiniMax-M2 needs EP=32 (8 nodes) to fit 256 experts in memory.
# Increasing TP does NOT reduce expert memory — increase EP instead.
TP=2
EP=32
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
echo "MiniMax-M2 Inference"
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
echo "Running inference ..."
$SRUN_CMD bash -c "cd $CONTAINER_WORKDIR && \
    if [ \$SLURM_LOCALID -eq 0 ]; then uv sync; else sleep 10; fi && \
    uv run --no-sync python -m torch.distributed.run \
    --nnodes=\$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path $HF_MODEL_ID \
    --megatron_model_path $MEGATRON_CKPT \
    --prompt '$PROMPT' \
    --max_new_tokens $MAX_NEW_TOKENS \
    --tp $TP --ep $EP \
    --trust-remote-code"

echo "======================================"
echo "Inference completed"
echo "======================================"
