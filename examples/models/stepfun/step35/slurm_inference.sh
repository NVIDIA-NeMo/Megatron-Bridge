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
# Step-3.5-Flash Inference (1 node, 8 GPUs)
#
# Runs greedy text generation through the Megatron Bridge text-generation helper.
# By default the script loads the HF checkpoint and converts in memory. Set
# MEGATRON_MODEL_PATH to an imported checkpoint directory to generate from a
# pre-converted Megatron checkpoint instead.
# ==============================================================================

#SBATCH --job-name=step35-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --partition=batch
#SBATCH --output=logs/step35_inference_%j.log
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to your .sqsh container path}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
BRIDGE_PATH=${BRIDGE_PATH:-$(pwd)}
WORKDIR=${WORKDIR:-/opt/Megatron-Bridge}

HF_MODEL=${HF_MODEL:-stepfun-ai/Step-3.5-Flash}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-}
PROMPT=${PROMPT:-"Explain the difference between dense and sparse expert models in one paragraph."}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

export HF_HOME=${HF_HOME:?Set HF_HOME to a shared Hugging Face cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv_cache}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

mkdir -p logs

EXTRA_ARGS=""
if [ -n "${MEGATRON_MODEL_PATH}" ]; then
    EXTRA_ARGS="--megatron_model_path ${MEGATRON_MODEL_PATH}"
fi

SRUN_CMD="srun --mpi=pmix --container-image=${CONTAINER_IMAGE}"
if [ -n "${CONTAINER_MOUNTS}" ]; then
    SRUN_CMD="${SRUN_CMD} --container-mounts=${BRIDGE_PATH}:${WORKDIR},${CONTAINER_MOUNTS}"
else
    SRUN_CMD="${SRUN_CMD} --container-mounts=${BRIDGE_PATH}:${WORKDIR}"
fi

${SRUN_CMD} --no-container-mount-home bash -c "
    cd ${WORKDIR}
    uv sync
    uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 \
        examples/conversion/hf_to_megatron_generate_text.py \
        --hf_model_path '${HF_MODEL}' \
        --prompt '${PROMPT}' \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --tp ${TP} --pp ${PP} --ep ${EP} --etp ${ETP} \
        --trust-remote-code \
        ${EXTRA_ARGS}
"
