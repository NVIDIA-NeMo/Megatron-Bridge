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
# Nemotron 3 Ultra Distributed GPU Checkpoint Import
#
# Use this path when host RAM is not large enough for CPU checkpoint import
# and the checkpoint must be materialized across GPUs.
#
# Usage:
#   1. Modify the #SBATCH directives for your cluster.
#   2. Set CONTAINER_IMAGE and optional CONTAINER_MOUNTS.
#   3. Submit: sbatch slurm_conversion.sh
# ==============================================================================

#SBATCH --job-name=nemotron-ultra-gpu-import
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/nemotron_ultra_import_%j.log
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
WORKDIR=${WORKDIR:-/opt/Megatron-Bridge}
MODEL_HOME=${MODEL_HOME:-${WORKSPACE:-/workspace}/models/nvidia}

HF_MODEL_PATH=${HF_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${MODEL_HOME}/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-megatron}

TP=${TP:-1}
PP=${PP:-6}
EP=${EP:-8}
ETP=${ETP:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

[ -n "${HF_HOME:-}" ] && export HF_HOME
[ -n "${UV_CACHE_DIR:-}" ] && export UV_CACHE_DIR
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

if [ "$((TP * PP * EP))" -ne "$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))" ]; then
    echo "ERROR: TP*PP*EP must equal nodes*GPUS_PER_NODE for this script."
    echo "TP=$TP PP=$PP EP=$EP nodes=$SLURM_JOB_NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE"
    exit 2
fi

if [ -e "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt" ] || [ -e "${MEGATRON_MODEL_PATH}/latest_train_state.pt" ]; then
    echo "ERROR: target already contains a Megatron checkpoint: ${MEGATRON_MODEL_PATH}"
    exit 3
fi

mkdir -p logs "$(dirname "$MEGATRON_MODEL_PATH")"

MASTER_ADDR=$(python3 - <<'PY'
import os
import re

nodelist = os.environ.get("SLURM_NODELIST", "")
match = re.match(r"([A-Za-z0-9_-]+)\[(\d+)", nodelist)
print(match.group(1) + match.group(2) if match else nodelist.split(",")[0])
PY
)
MASTER_PORT=$((18000 + SLURM_JOB_ID % 40000))
export MASTER_ADDR MASTER_PORT HF_MODEL_PATH MEGATRON_MODEL_PATH TP PP EP ETP GPUS_PER_NODE WORKDIR

echo "Nemotron 3 Ultra distributed GPU import"
echo "Job ${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} GPUs/node=${GPUS_PER_NODE} TP=${TP} PP=${PP} EP=${EP} ETP=${ETP}"
echo "HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH}"

SRUN_CMD=(srun --mpi=pmix --no-kill --container-image="${CONTAINER_IMAGE}" --no-container-mount-home)
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD+=(--container-mounts="${CONTAINER_MOUNTS}")
fi

"${SRUN_CMD[@]}" bash -c '
set -euo pipefail
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src:$WORKDIR/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="$GPUS_PER_NODE"

uv run --no-sync python examples/conversion/convert_checkpoints_multi_gpu.py \
    import \
    --hf-model "$HF_MODEL_PATH" \
    --megatron-path "$MEGATRON_MODEL_PATH" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --torch-dtype bfloat16
'

echo IMPORT_DONE
