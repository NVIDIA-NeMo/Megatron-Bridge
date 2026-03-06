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
# MiniMax-M2 Conversion Round-Trip Verification (Multi-Node via Slurm)
#
# MiniMax-M2 (MoE: 256 experts, top-8, ~230GB fp8)
# Sweeps multiple parallelism configs (TP,PP,EP) to verify HF <-> Megatron
# round-trip conversion across different GPU layouts.
#
# Each config runs hf_megatron_roundtrip_multi_gpu.py sequentially.
# All configs must use the same total GPU count (GPUS_PER_NODE * NODES).
#
# Usage:
#   1. Modify the #SBATCH directives and CONFIGURATION section for your cluster
#   2. Set CONTAINER_IMAGE to your container path
#   3. Adjust PARALLELISM_CONFIGS for desired TP,PP,EP combos
#   4. Submit: sbatch slurm_conversion.sh
# ==============================================================================

#SBATCH --job-name=minimax-m2-roundtrip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --account=<your-account>
#SBATCH --output=logs/minimax_m2_roundtrip_%j.out
#SBATCH --error=logs/minimax_m2_roundtrip_%j.err
#SBATCH --exclusive

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_NAME=MiniMax-M2
HF_MODEL_ID=MiniMaxAI/$MODEL_NAME
GPUS_PER_NODE=8

# Parallelism configs: "TP,PP,EP" per entry (TP*PP*EP must equal total GPUs)
# EP must divide 256 (number of experts).
PARALLELISM_CONFIGS=("2,1,4" "1,2,4" "2,2,2")

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional; comma-separated for srun --container-mounts)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/lustre:/lustre,/path/to/project:/opt/Megatron-Bridge"

# Container working directory
CONTAINER_WORKDIR=/opt/Megatron-Bridge

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# UV cache on shared filesystem (recommended for multi-node setups)
# Pre-sync once before submitting: UV_CACHE_DIR=/path/to/cache uv sync
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# HuggingFace cache directory (recommended for shared filesystem)
# export HF_HOME="/path/to/shared/HF_HOME"

# Authentication tokens (set these for your environment)
# export HF_TOKEN="hf_your_token_here"

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "MiniMax-M2 Round-Trip Conversion Sweep"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Parallelism configs: ${PARALLELISM_CONFIGS[*]}"
echo "======================================"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set. Please specify a valid container image."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi
echo "SRUN base: $SRUN_CMD"
echo "======================================"

CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP <<< "$CONFIG"
    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    NPROC=$((TP * PP * EP))

    echo ""
    echo "======================================"
    echo "Config $CONFIG_INDEX/${#PARALLELISM_CONFIGS[@]}: TP=$TP, PP=$PP, EP=$EP (nproc=$NPROC)"
    echo "======================================"

    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=${MASTER_PORT:-29500}

    CMD="uv run --no-sync python -m torch.distributed.run \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
        --hf-model-id $HF_MODEL_ID \
        --tp $TP --pp $PP --ep $EP \
        --trust-remote-code"

    echo "Executing: $CMD"
    echo "======================================"

    $SRUN_CMD bash -c "cd $CONTAINER_WORKDIR && \
        if [ \$SLURM_LOCALID -eq 0 ]; then uv sync; else sleep 10; fi && \
        $CMD"
    RUN_EXIT=$?
    if [ $RUN_EXIT -ne 0 ]; then
        echo "ERROR: Config TP=$TP, PP=$PP, EP=$EP failed with exit code $RUN_EXIT"
        exit $RUN_EXIT
    fi
    echo "[OK] Config $CONFIG_INDEX: TP=$TP, PP=$PP, EP=$EP passed"
done

echo ""
echo "======================================"
echo "All ${#PARALLELISM_CONFIGS[@]} configs passed"
echo "======================================"
