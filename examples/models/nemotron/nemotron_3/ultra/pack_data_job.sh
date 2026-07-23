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

#SBATCH --job-name=nemotron-ultra-pack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=cpu
#SBATCH --output=logs/nemotron_ultra_pack_%j.log

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
WORKDIR=${WORKDIR:-/opt/Megatron-Bridge}
HF_MODEL_PATH=${HF_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
RECIPE_NAME=${RECIPE_NAME:-nemotron_3_ultra_sft_openmathinstruct2_packed_config}
SEQ_LENGTH=${SEQ_LENGTH:-4096}

[ -n "${HF_HOME:-}" ] && export HF_HOME
[ -n "${NEMO_HOME:-}" ] && export NEMO_HOME
[ -n "${UV_CACHE_DIR:-}" ] && export UV_CACHE_DIR
export WORKDIR HF_MODEL_PATH RECIPE_NAME SEQ_LENGTH

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

mkdir -p logs

SRUN_CMD=(srun --mpi=pmix --container-image="${CONTAINER_IMAGE}" --no-container-mount-home)
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD+=(--container-mounts="${CONTAINER_MOUNTS}")
fi

"${SRUN_CMD[@]}" bash -c '
set -euo pipefail
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src:$WORKDIR/3rdparty/Megatron-LM:${PYTHONPATH:-}"

uv run --no-sync python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe "$RECIPE_NAME" \
    --seq-length "$SEQ_LENGTH" \
    --hf-path "$HF_MODEL_PATH"
'

echo PACK_DATA_DONE
