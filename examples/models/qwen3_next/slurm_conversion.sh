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
# Qwen3-Next 80B-A3B Checkpoint Conversion
#
# Runs HF -> Megatron import, Megatron -> HF export, and round-trip validation
# on a single node with 8 GPUs. Requires a node with sufficient CPU memory
# (>320GB) for the Megatron -> HF export of the 80B model.
#
# Usage:
#   1. Set CONTAINER_IMAGE to your container path
#   2. Submit: sbatch slurm_conversion.sh
# ==============================================================================

#SBATCH --job-name=qwen3next-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --partition=batch_short
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --output=logs/qwen3_next_convert_%j.out
#SBATCH --error=logs/qwen3_next_convert_%j.err
#SBATCH --exclusive
#SBATCH --mem=0

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional; comma-separated for srun --container-mounts)
CONTAINER_MOUNTS=""

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Qwen3-Next 80B-A3B Conversion Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "======================================"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

$SRUN_CMD bash -c "WORKSPACE=$WORKSPACE ./examples/models/qwen3_next/conversion.sh"

echo "Conversion job completed"
