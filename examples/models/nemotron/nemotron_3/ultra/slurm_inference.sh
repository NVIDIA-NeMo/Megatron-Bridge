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
# Nemotron 3 Ultra Inference (4 nodes / 32 GPUs via Slurm)
#
# Usage:
#   1. Set CONTAINER_IMAGE, CONTAINER_MOUNTS, and cache/token environment variables.
#   2. Optionally set HF_MODEL_PATH to a local Hugging Face snapshot.
#   3. Submit with: sbatch examples/models/nemotron/nemotron_3/ultra/slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=nemotron-ultra-inference
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/nemotron_ultra_inference_%j.log
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
WORKDIR=${WORKDIR:-/opt/Megatron-Bridge}

HF_MODEL_PATH=${HF_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-}
PROMPT=${PROMPT:-"Solve 2x + 3 = 11. Show the reasoning briefly."}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
KV_CACHE_BUFFER_SIZE_GB=${KV_CACHE_BUFFER_SIZE_GB:-4}
INFERENCE_MOE_TOKEN_DISPATCHER_TYPE=${INFERENCE_MOE_TOKEN_DISPATCHER_TYPE:-nccl}

TP=${TP:-1}
PP=${PP:-4}
EP=${EP:-8}
ETP=${ETP:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

[ -n "${HF_HOME:-}" ] && export HF_HOME
[ -n "${UV_CACHE_DIR:-}" ] && export UV_CACHE_DIR
[ -n "${NEMO_HOME:-}" ] && export NEMO_HOME
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
EXPERT_MESH=$((EP * ETP))
if [ "$EXPERT_MESH" -gt "$TP" ]; then
    STAGE_GPUS=$EXPERT_MESH
else
    STAGE_GPUS=$TP
fi
MIN_GPUS=$((PP * STAGE_GPUS))
if [ "$((TOTAL_GPUS % MIN_GPUS))" -ne 0 ]; then
    echo "ERROR: nodes*GPUS_PER_NODE must be a multiple of PP*max(TP,EP*ETP) for MoE inference."
    echo "TP=$TP PP=$PP EP=$EP ETP=$ETP nodes=$SLURM_JOB_NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE min_gpus=$MIN_GPUS total_gpus=$TOTAL_GPUS"
    exit 2
fi

mkdir -p logs

export HF_MODEL_PATH MEGATRON_MODEL_PATH PROMPT MAX_NEW_TOKENS KV_CACHE_BUFFER_SIZE_GB INFERENCE_MOE_TOKEN_DISPATCHER_TYPE
export TP PP EP ETP GPUS_PER_NODE WORKDIR
[ -n "${COORDINATOR_HOST:-}" ] && export COORDINATOR_HOST

echo "Nemotron 3 Ultra inference"
echo "Job ${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} GPUs/node=${GPUS_PER_NODE} TP=${TP} PP=${PP} EP=${EP} ETP=${ETP}"
echo "HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-<load HF weights directly>}"
echo "KV_CACHE_BUFFER_SIZE_GB=${KV_CACHE_BUFFER_SIZE_GB}"
echo "COORDINATOR_HOST=${COORDINATOR_HOST:-<auto local IP>}"

SRUN_CMD=(srun --mpi=pmix --no-kill --container-image="${CONTAINER_IMAGE}" --no-container-mount-home)
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD+=(--container-mounts="${CONTAINER_MOUNTS}")
fi

"${SRUN_CMD[@]}" bash -c '
set -euo pipefail
cd "$WORKDIR"

rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.pth \
      /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.py 2>/dev/null || true

export PYTHONPATH="$WORKDIR/src:$WORKDIR/3rdparty/Megatron-LM:${PYTHONPATH:-}"

MEGATRON_MODEL_ARGS=()
if [ -n "${MEGATRON_MODEL_PATH:-}" ]; then
    MEGATRON_MODEL_ARGS=(--megatron_model_path "$MEGATRON_MODEL_PATH")
fi

COORDINATOR_ARGS=()
if [ -z "${COORDINATOR_HOST:-}" ]; then
    COORDINATOR_HOST=$(python3 - <<'"'"'PY'"'"'
import socket

print(socket.gethostbyname(socket.gethostname()))
PY
)
fi
COORDINATOR_ARGS=(--coordinator-host "$COORDINATOR_HOST")

uv run --no-sync python scripts/inference/text_generation.py \
    --hf_model_path "$HF_MODEL_PATH" \
    "${MEGATRON_MODEL_ARGS[@]}" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --kv_cache_buffer_size_gb "$KV_CACHE_BUFFER_SIZE_GB" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --use-coordinator \
    "${COORDINATOR_ARGS[@]}" \
    --inference-moe-token-dispatcher-type "$INFERENCE_MOE_TOKEN_DISPATCHER_TYPE" \
    --distributed-timeout-minutes 90
'
