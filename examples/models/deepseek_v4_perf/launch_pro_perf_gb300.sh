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

# Launch the full 61-layer DeepSeek-V4-Pro GB300 MXFP8 performance recipe.
# See README.md for required container dependencies and Megatron-Core setup.

set -euo pipefail

: "${WORKSPACE:?Set WORKSPACE to shared storage for logs and NeMo-Run artifacts}"
: "${MBRIDGE_VENV:?Set MBRIDGE_VENV to a host environment containing nemo_run}"
: "${MCORE_DEV:?Set MCORE_DEV to the prepared Megatron-Core checkout}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT for the target cluster}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION to a GB300 partition}"

MBRIDGE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MCORE_COMMIT=${MCORE_COMMIT:-9d46c924dce3818f2b5f894f7380712c780d1801}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.06.01}
NUM_GPUS=${NUM_GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MAX_STEPS=${MAX_STEPS:-20}
TIME_LIMIT=${TIME_LIMIT:-01:10:00}
CONFIG_VARIANT=${CONFIG_VARIANT:-v1}

if [[ "$(git -C "${MCORE_DEV}" rev-parse HEAD)" != "${MCORE_COMMIT}" ]]; then
    echo "ERROR: MCORE_DEV must be based on ${MCORE_COMMIT}" >&2
    exit 1
fi

EXPERTS_PY="${MCORE_DEV}/megatron/core/transformer/moe/experts.py"
if grep -q 'clamped SwiGLU needs TE >= 2.17.0.dev0' "${EXPERTS_PY}"; then
    echo "ERROR: MCORE_DEV still contains the TE 2.17 version guard." >&2
    echo "       Apply the capability-check patch documented in README.md." >&2
    exit 1
fi

source "${MBRIDGE_VENV}/bin/activate"

export NEMORUN_HOME=${NEMORUN_HOME:-${WORKSPACE}/.nemo_run}
mkdir -p "${NEMORUN_HOME}"

SRC_MOUNT="${MBRIDGE}/src/megatron:/opt/Megatron-Bridge/src/megatron"
MCORE_MOUNT="${MCORE_DEV}/megatron:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron"
MOUNTS="${SRC_MOUNT},${MCORE_MOUNT}"

HF_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ARGS=(--hf_token "${HF_TOKEN}")
fi

DRY_FLAG=()
if [[ -n "${DRY:-}" ]]; then
    DRY_FLAG=(-d)
fi

cd "${MBRIDGE}"

PYTHONPATH="${MBRIDGE}/src:${MBRIDGE}/scripts/performance:${PYTHONPATH:-}" \
python -m scripts.performance.setup_experiment \
    -m deepseek -mr deepseek_v4_pro --task pretrain \
    --num_gpus "${NUM_GPUS}" \
    -a "${SLURM_ACCOUNT}" -p "${SLURM_PARTITION}" \
    -l "${NEMORUN_HOME}" -i "${CONTAINER_IMAGE}" \
    -t "${TIME_LIMIT}" \
    --gpu gb300 -c fp8_mx -cv "${CONFIG_VARIANT}" \
    -gn "${GPUS_PER_NODE}" -ms "${MAX_STEPS}" \
    -E NVTE_CPU_OFFLOAD_V1=1 \
    -E NCCL_NET_PLUGIN=none \
    -E NCCL_NET_GDR_LEVEL=PHB \
    -E NCCL_NET_GDR_C2C=1 \
    -cm "${MOUNTS}" \
    "${HF_ARGS[@]}" \
    "${DRY_FLAG[@]}"
