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

#SBATCH --job-name=minimax-m3-inference
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:45:00
#SBATCH --partition=batch
#SBATCH --output=logs/minimax_m3_inference_%j.log
#SBATCH --exclusive

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the Megatron-Bridge container}"
: "${CONTAINER_MOUNTS:?Mount shared storage and this repository}"
: "${HF_HOME:?Set HF_HOME to a shared Hugging Face cache}"
: "${UV_CACHE_DIR:?Set UV_CACHE_DIR to a shared uv cache}"

WORKDIR=${WORKDIR:-/opt/Megatron-Bridge}
HF_MODEL_ID=${HF_MODEL_ID:-MiniMaxAI/MiniMax-M3}
PROMPT=${PROMPT:-Explain why the sky appears blue in one concise paragraph.}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-32}
ETP=${ETP:-1}

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR MASTER_PORT

SRUN=(
    srun
    --mpi=pmix
    --container-image="${CONTAINER_IMAGE}"
    --container-mounts="${CONTAINER_MOUNTS}"
    --no-container-mount-home
)

"${SRUN[@]}" --nodes=1 --ntasks=1 bash -lc 'cd "$1" && uv sync --extra te' minimax-m3-sync "${WORKDIR}"

"${SRUN[@]}" bash -lc '
    export RANK="${SLURM_PROCID:?}"
    export WORLD_SIZE="${SLURM_NTASKS:?}"
    export LOCAL_RANK="${SLURM_LOCALID:?}"
    cd "$1"
    uv run --no-sync python examples/conversion/hf_to_megatron_generate_text.py \
        --hf_model_path "$2" \
        --prompt "$3" \
        --max_new_tokens "$4" \
        --apply-chat-template \
        --thinking-mode disabled \
        --tp "$5" \
        --pp "$6" \
        --ep "$7" \
        --etp "$8" \
        --trust-remote-code
' minimax-m3-inference "${WORKDIR}" "${HF_MODEL_ID}" "${PROMPT}" "${MAX_NEW_TOKENS}" "${TP}" "${PP}" "${EP}" "${ETP}"
