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
# GLM-4.7 Inference (Multi-Node via Slurm)
#
# GLM-4.7 (MoE: 160 experts, top-8, ~358B params)
# Requires at least 4 nodes (32 GPUs) with EP=32.
#
# Megatron inference engine support is not yet available for this architecture.
# This launcher defaults to the older slow sanity-check generation path for now.
#
# Usage:
#   1. Set CONTAINER_IMAGE and token exports; set CONTAINER_MOUNTS if needed
#   2. Create logs/ if your Slurm setup requires the output directory to exist
#   3. Submit: sbatch examples/models/glm47/slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=glm47-inference
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/glm47_inference_%j.log
#SBATCH --exclusive

set -euo pipefail

# Container
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
# CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"
# CONTAINER_MOUNTS="/data:/data,/path/to/local/checkout:/opt/Megatron-Bridge"
WORKDIR="${WORKDIR:-/opt/Megatron-Bridge}"

# Tokens / Caches
# export HF_TOKEN="hf_your_token_here"
# export HF_HOME="/path/to/shared/HF_HOME"
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# Model / Generation
MODEL_NAME="${MODEL_NAME:-GLM-4.7}"
HF_MODEL_ID="${HF_MODEL_ID:-zai-org/$MODEL_NAME}"
PROMPT="${PROMPT:-What is artificial intelligence?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

# Environment
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "GLM-4.7 Inference"
echo "Job: ${SLURM_JOB_ID:-unknown} | Nodes: ${SLURM_JOB_NUM_NODES:-unknown}"
echo "Parallelism: TP=1 PP=1 EP=32"
echo "======================================"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

SRUN_CMD=(srun --mpi=pmix --container-image="$CONTAINER_IMAGE" --no-container-mount-home)
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD+=(--container-mounts="$CONTAINER_MOUNTS")
fi

echo "Warming uv cache"
"${SRUN_CMD[@]}" -N 1 --ntasks=1 bash -c "cd \"$WORKDIR\" && uv sync"

printf -v HF_MODEL_ARG "%q" "$HF_MODEL_ID"
printf -v PROMPT_ARG "%q" "$PROMPT"

CMD="uv run --no-sync python examples/conversion/hf_to_megatron_generate_text.py"
CMD="$CMD --hf_model_path $HF_MODEL_ARG"
CMD="$CMD --prompt $PROMPT_ARG"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --tp 1 --pp 1 --ep 32"

echo "Executing: $CMD"

"${SRUN_CMD[@]}" bash -c "cd \"$WORKDIR\" && $CMD"

echo "======================================"
echo "Inference completed"
echo "======================================"
