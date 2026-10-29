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

set -euo pipefail

if [[ $# -gt 1 || (${1:-} != "" && ${1:-} != "--launch") ]]; then
    echo "Usage: $0 [--launch]" >&2
    exit 2
fi

: "${DATACOMP_ROOT:?Set DATACOMP_ROOT to the shared tutorial-data root.}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT.}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION.}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE.}"

bridge_root=${BRIDGE_ROOT:-$(pwd)}
qwen_hf_id=${QWEN_HF_ID:-Qwen/Qwen3.6-35B-A3B}
qwen_hf_revision=${QWEN_HF_REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}
qwen_megatron=${QWEN_MEGATRON:-${DATACOMP_ROOT}/models/qwen3.6-35b-a3b-megatron}
submission_args=(--submission-dry-run)
if [[ ${1:-} == "--launch" ]]; then
    submission_args=()
fi

./scripts/conversion/convert.sh import \
    --executor slurm \
    --device gpu \
    --nodes 1 \
    --gpus-per-node 8 \
    --account "${SLURM_ACCOUNT}" \
    --partition "${SLURM_PARTITION}" \
    --container-image "${CONTAINER_IMAGE}" \
    "${submission_args[@]}" \
    --mount "${DATACOMP_ROOT}" \
    --mount "${bridge_root}:/opt/Megatron-Bridge" \
    --hf-model "${qwen_hf_id}" \
    --hf-revision "${qwen_hf_revision}" \
    --megatron-path "${qwen_megatron}" \
    --torch-dtype bfloat16 \
    --tp 2 --pp 1 --ep 4 --etp 1
