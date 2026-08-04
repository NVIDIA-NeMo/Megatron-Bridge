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
datacomp_energon=${DATACOMP_ENERGON:-${DATACOMP_ROOT}/energon}
qwen_hf_id=${QWEN_HF_ID:-Qwen/Qwen3.6-35B-A3B}
qwen_hf_revision=${QWEN_HF_REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}
qwen_megatron=${QWEN_MEGATRON:-${DATACOMP_ROOT}/models/qwen3.6-35b-a3b-megatron}
submission_args=(--dry-run)
if [[ ${1:-} == "--launch" ]]; then
    submission_args=()
fi

./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 "${submission_args[@]}" \
    --account "${SLURM_ACCOUNT}" \
    --partition "${SLURM_PARTITION}" \
    --container-image "${CONTAINER_IMAGE}" \
    --mount "${DATACOMP_ROOT}" \
    --mount "${bridge_root}:/opt/Megatron-Bridge" \
    --recipe qwen35_vl_35b_a3b_pretrain_mock_config \
    --mode pretrain \
    --dataset energon \
    --step-func qwen3_vl_step \
    --pretrained_checkpoint "${qwen_megatron}/iter_0000000" \
    --max_steps 1000 \
    --save_dir "${DATACOMP_ROOT}/training/qwen3.6-35b-a3b/checkpoints" \
    --save_interval 500 \
    train.global_batch_size=512 \
    train.micro_batch_size=1 \
    dataset.micro_batch_size=1 \
    dataset.path="${datacomp_energon}" \
    dataset.task_encoder.hf_processor_path="${qwen_hf_id}" \
    dataset.task_encoder.hf_processor_revision="${qwen_hf_revision}" \
    dataset.do_validation=true \
    dataset.do_test=false \
    dataset.enable_in_batch_packing=false \
    dataset.defer_in_batch_packing_to_step=true \
    model.hf_model_id="${qwen_hf_id}" \
    model.bos_token_id=248044 \
    model.eos_token_id=248044 \
    checkpoint.hf_source_path="${qwen_hf_id}" \
    checkpoint.load=null \
    logger.save_config_filepath="${DATACOMP_ROOT}/training/qwen3.6-35b-a3b/resolved-config.yaml"
