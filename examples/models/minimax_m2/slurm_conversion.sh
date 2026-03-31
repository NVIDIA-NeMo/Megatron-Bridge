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
# The full model OOMs on a single 8-GPU node — minimum 2 nodes (16 GPUs)
# with EP >= 8.
#
# Sweeps multiple parallelism configs (TP,PP,EP) to verify HF <-> Megatron
# round-trip conversion. Each config runs sequentially.
#
# Usage:
#   1. Fill in CONTAINER_IMAGE, CONTAINER_MOUNTS, and token exports
#   2. Adjust PARALLELISM_CONFIGS if needed
#   3. Submit: sbatch slurm_conversion.sh
# ==============================================================================

#SBATCH --job-name=minimax-m2-roundtrip
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/minimax_m2_roundtrip_%j.log
#SBATCH --exclusive

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/path/to/shared/storage:/mnt/storage,/path/to/project:/opt/Megatron-Bridge"
WORKDIR="/opt/Megatron-Bridge"

# ── Tokens / Caches ──────────────────────────────────────────────────────
# export HF_TOKEN="hf_your_token_here"
# export HF_HOME="/path/to/shared/HF_HOME"
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# ── Parallelism configs: "TP,PP,EP" per entry ────────────────────────────
# TP*PP*EP must equal total GPUs (NODES * GPUS_PER_NODE).
# EP must divide 256 (number of experts).
# Note: TP=2,EP=4 on 8 GPUs (1 node) OOMs — use EP >= 8 with 2+ nodes.
PARALLELISM_CONFIGS=("2,1,8" "1,2,8" "2,2,4")

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_NAME=MiniMax-M2
HF_MODEL_ID=MiniMaxAI/$MODEL_NAME

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "MiniMax-M2 Round-Trip Conversion Sweep"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "Parallelism configs: ${PARALLELISM_CONFIGS[*]}"
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

CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP <<< "$CONFIG"
    CONFIG_INDEX=$((CONFIG_INDEX + 1))

    echo ""
    echo "======================================"
    echo "Config $CONFIG_INDEX/${#PARALLELISM_CONFIGS[@]}: TP=$TP, PP=$PP, EP=$EP"
    echo "======================================"

    # Sync dependencies once per node, then run the roundtrip
    CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 10; fi && "
    CMD="${CMD}uv run --no-sync python examples/conversion/hf_megatron_roundtrip_multi_gpu.py"
    CMD="$CMD --hf-model-id $HF_MODEL_ID"
    CMD="$CMD --tp $TP --pp $PP --ep $EP"
    CMD="$CMD --trust-remote-code"

    echo "Executing: $CMD"

    $SRUN_CMD bash -c "cd $WORKDIR && $CMD"
    RUN_EXIT=$?
    if [ $RUN_EXIT -ne 0 ]; then
        echo "ERROR: Config TP=$TP, PP=$PP, EP=$EP failed (exit $RUN_EXIT)"
        exit $RUN_EXIT
    fi
    echo "[OK] Config $CONFIG_INDEX: TP=$TP, PP=$PP, EP=$EP passed"
done

echo ""
echo "======================================"
echo "All ${#PARALLELISM_CONFIGS[@]} configs passed"
echo "======================================"
