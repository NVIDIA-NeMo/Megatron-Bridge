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

# GLM-5 / GLM-5.1 round-trip verification on 8 Slurm nodes (64 GPUs).
# Run this wrapper from a Slurm login node; convert.sh submits one job per
# parallelism configuration and waits for each job by default.
#
# Required:
#   export CONTAINER_IMAGE=/path/to/container.sqsh
#   export SLURM_ACCOUNT=<your-account>
# Optional:
#   export MODEL_NAME=GLM-5.1
#   export CONTAINER_MOUNTS=/shared:/shared,/host/path:/container/path
#   bash "$0" --srun-arg=--mpi=pmix

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the Megatron-Bridge container}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to your Slurm account}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
CONVERT_SH="${CONVERT_SH:-${REPO_ROOT}/scripts/conversion/convert.sh}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
NODES=8
GPUS_PER_NODE=8

MODEL_NAME="${MODEL_NAME:-GLM-5}"
HF_MODEL_ID="${HF_MODEL_ID:-zai-org/${MODEL_NAME}}"
# TP*PP*EP must equal NODES*GPUS_PER_NODE for these data-parallel-free layouts.
# EP must divide 256 (the number of routed experts).
PARALLELISM_CONFIGS=("2,1,32")

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

MOUNT_ARGS=(--mount "${REPO_ROOT}:/opt/Megatron-Bridge")
IFS=',' read -r -a EXTRA_MOUNTS <<< "${CONTAINER_MOUNTS:-}"
for mount in "${EXTRA_MOUNTS[@]}"; do
    if [[ -n "${mount}" ]]; then
        MOUNT_ARGS+=(--mount "${mount}")
    fi
done

ENV_ARGS=()
for name in HF_TOKEN HF_HOME UV_CACHE_DIR TORCH_NCCL_AVOID_RECORD_STREAMS NCCL_NVLS_ENABLE; do
    if [[ -n "${!name:-}" ]]; then
        ENV_ARGS+=(--env "${name}")
    fi
done

for config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP <<< "${config}"
    echo "Submitting ${MODEL_NAME} roundtrip: TP=${TP}, PP=${PP}, EP=${EP}"
    "${CONVERT_SH}" roundtrip \
        --executor slurm --device gpu \
        --nodes "${NODES}" --gpus-per-node "${GPUS_PER_NODE}" \
        --account "${SLURM_ACCOUNT}" --partition "${SLURM_PARTITION}" --time "${TIME_LIMIT}" \
        --container-image "${CONTAINER_IMAGE}" \
        "${MOUNT_ARGS[@]}" \
        "${ENV_ARGS[@]}" \
        --experiment-name "${MODEL_NAME,,}-roundtrip-tp${TP}-pp${PP}-ep${EP}" \
        --hf-model "${HF_MODEL_ID}" \
        --tp "${TP}" --pp "${PP}" --ep "${EP}" \
        "$@"
done
