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
# Pre-pack SFT dataset for packed-sequence training (CPU job).
#
# Run once before submitting slurm_sft.sh with openmathinstruct2_gsm8k.
# Packed files are cached and skipped automatically on subsequent runs.
#
# Usage:
#   1. Set CONTAINER_IMAGE and CONTAINER_MOUNTS below
#   2. Submit: sbatch pack_data_job.sh
# ==============================================================================

#SBATCH --job-name=gpt-oss-pack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --account=my_account
#SBATCH --output=logs/gpt_oss_pack_%j.out
#SBATCH --error=logs/gpt_oss_pack_%j.err

WKDIR="${WKDIR:-}"
WORKSPACE="${WORKSPACE:-/workspace}"

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data,/workspace:/workspace"

# HuggingFace / NeMo cache directories (set to avoid re-downloading)
# export HF_HOME="/path/to/shared/HF_HOME"
# export NEMO_HOME="/path/to/shared/NEMO_HOME"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

$SRUN_CMD bash -c "
    uv run --no-sync python /opt/Megatron-Bridge/scripts/training/pack_sft_data.py \
        --recipe gpt_oss_20b_sft_openmathinstruct2_thinking_packed_config
"
