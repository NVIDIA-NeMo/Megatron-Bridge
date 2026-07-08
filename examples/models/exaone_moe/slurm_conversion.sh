#!/usr/bin/env bash
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
# K-EXAONE-236B-A23B Conversion Round-Trip Verification (Slurm)
#
# K-EXAONE is a BF16 MoE model with 128 routed experts and 8 active experts.
# The default TP=1, PP=1, EP=16 configuration uses 16 GPUs across 2 nodes.
#
# Usage:
#   1. Set CONTAINER_IMAGE and, if needed, CONTAINER_MOUNTS.
#   2. Export HF_TOKEN, HF_HOME, and UV_CACHE_DIR on shared storage.
#   3. Create the log directory and submit:
#        mkdir -p logs
#        sbatch examples/models/exaone_moe/slurm_conversion.sh
# ==============================================================================

#SBATCH --job-name=k-exaone-roundtrip
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/k_exaone_roundtrip_%j.log
#SBATCH --exclusive

set -euo pipefail

# -- Container ---------------------------------------------------------------
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"
WORKDIR="${WORKDIR:-/opt/Megatron-Bridge}"

# -- Model / Parallelism -----------------------------------------------------
HF_MODEL_ID="${HF_MODEL_ID:-LGAI-EXAONE/K-EXAONE-236B-A23B}"
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-16}"
ETP="${ETP:-1}"

# -- Environment -------------------------------------------------------------
# Keep HF_HOME and UV_CACHE_DIR on storage shared by all nodes.
# export HF_TOKEN="..."
# export HF_HOME="/path/to/shared/HF_HOME"
# export UV_CACHE_DIR="/path/to/shared/uv_cache"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$CONTAINER_IMAGE" ]]; then
    echo "ERROR: Set CONTAINER_IMAGE to the Enroot .sqsh image path."
    exit 1
fi

WORLD_SIZE="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
MODEL_PARALLEL_SIZE=$((TP * PP * EP))
if [[ "$MODEL_PARALLEL_SIZE" -ne "$WORLD_SIZE" ]]; then
    echo "ERROR: TP*PP*EP=$MODEL_PARALLEL_SIZE must equal allocated tasks=$WORLD_SIZE."
    exit 1
fi
if ((128 % EP != 0)); then
    echo "ERROR: EP=$EP must divide K-EXAONE's 128 routed experts."
    exit 1
fi

SRUN_ARGS=(--mpi=pmix --container-image="$CONTAINER_IMAGE" --no-container-mount-home)
if [[ -n "$CONTAINER_MOUNTS" ]]; then
    SRUN_ARGS+=(--container-mounts="$CONTAINER_MOUNTS")
fi

echo "======================================"
echo "K-EXAONE Round-Trip Conversion"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES | Tasks: $WORLD_SIZE"
echo "Model: $HF_MODEL_ID"
echo "TP=$TP PP=$PP EP=$EP ETP=$ETP"
echo "======================================"

# Warm the shared uv cache once before all ranks enter distributed setup.
srun --nodes=1 --ntasks=1 "${SRUN_ARGS[@]}" \
    bash -c 'cd "$1" && uv sync' bash "$WORKDIR"

srun "${SRUN_ARGS[@]}" \
    bash -c '
        cd "$1"
        uv run --no-sync python examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
            --hf-model-id "$2" \
            --tp "$3" \
            --pp "$4" \
            --ep "$5" \
            --etp "$6" \
            --trust-remote-code
    ' bash "$WORKDIR" "$HF_MODEL_ID" "$TP" "$PP" "$EP" "$ETP"

echo "======================================"
echo "Round-trip conversion completed"
echo "======================================"
