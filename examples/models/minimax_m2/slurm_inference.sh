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
# The full model OOMs on a single 8-GPU node.
# Minimum EP=16 (2 nodes) for inference; TP=2,EP=8 OOMs during forward pass.
# Increasing TP does NOT reduce expert memory — increase EP instead.
#
# Usage:
#   1. Fill in CONTAINER_IMAGE, CONTAINER_MOUNTS, and token exports
#   2. Submit: sbatch slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=minimax-m2-inference
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/minimax_m2_inference_%j.log
#SBATCH --exclusive

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/lustre:/lustre,/path/to/project:/opt/Megatron-Bridge"
WORKDIR="/opt/Megatron-Bridge"

# ── Tokens / Caches ──────────────────────────────────────────────────────
# export HF_TOKEN="hf_your_token_here"
# export HF_HOME="/path/to/shared/HF_HOME"
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# ── Model / Parallelism ──────────────────────────────────────────────────
MODEL_NAME=MiniMax-M2
HF_MODEL_ID=MiniMaxAI/$MODEL_NAME
PROMPT="What is artificial intelligence?"
MAX_NEW_TOKENS=100
TP=1
EP=16
PP=1

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "MiniMax-M2 Inference"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "TP=$TP PP=$PP EP=$EP (Total GPUs: $((TP * EP * PP)))"
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

# Sync dependencies once per node, then run inference
CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 10; fi && "
CMD="${CMD}uv run --no-sync python examples/conversion/hf_to_megatron_generate_text.py"
CMD="$CMD --hf_model_path $HF_MODEL_ID"
CMD="$CMD --prompt '$PROMPT'"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --tp $TP --ep $EP"
CMD="$CMD --trust-remote-code"

echo "Executing: $CMD"

$SRUN_CMD bash -c "cd $WORKDIR && $CMD"

echo "======================================"
echo "Inference completed"
echo "======================================"
