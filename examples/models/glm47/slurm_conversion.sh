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

# GLM-4.7 round-trip verification on 4 Slurm nodes (32 GPUs).
# Run this wrapper from a Slurm login node; convert.sh submits the job and
# waits for it by default.
#
# Required:
#   export CONTAINER_IMAGE=/path/to/container.sqsh
#   export SLURM_ACCOUNT=<your-account>
# Optional:
#   export CONTAINER_MOUNTS=/shared:/shared,/host/path:/container/path
#   bash "$0" --srun-arg=--mpi=pmix

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the Megatron-Bridge container}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to your Slurm account}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
CONVERT_SH="${CONVERT_SH:-${REPO_ROOT}/scripts/conversion/convert.sh}"

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

"${CONVERT_SH}" roundtrip \
    --executor slurm --device gpu \
    --nodes 4 --gpus-per-node 8 \
    --account "${SLURM_ACCOUNT}" --partition "${SLURM_PARTITION:-batch}" --time 04:00:00 \
    --container-image "${CONTAINER_IMAGE}" \
    "${MOUNT_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    --experiment-name glm47-roundtrip \
    --hf-model zai-org/GLM-4.7 \
    --tp 1 --pp 1 --ep 32 --etp 1 \
    "$@"
