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

# DeepSeek-V4 import + export with the Bridge.
#
# DSv4 currently requires TP=1; scale via expert and pipeline parallelism (EP, PP).
# The Bridge dispatches FP8 / MXFP4 dequantisation by tensor dtype, so the
# same script works for Flash, Flash-Base, Pro, and Pro-Base.
#
# Override defaults by exporting environment variables before running:
#   WORKSPACE: directory for converted Megatron checkpoints (default: /workspace)
#   MODEL_VARIANT: one of DeepSeek-V4-Flash, DeepSeek-V4-Flash-Base,
#                  DeepSeek-V4-Pro, DeepSeek-V4-Pro-Base
#                  (default: DeepSeek-V4-Flash-Base)
#   HF_MODEL_ID: HuggingFace model ID or local path (default: deepseek-ai/${MODEL_VARIANT})
#   EP: expert-parallel size (default: 4 for Flash, 8 for Pro)
#   PP: pipeline-parallel size (default: 1 for Flash, 4 for Pro)
#   NNODES, NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, MASTER_PORT: torchrun launch overrides
#   UV_RUN_ARGS: extra arguments passed after `uv run` (for example: "--active --no-sync")
#   CONVERSION_EXECUTOR_ARGS: execution options passed to convert.sh. The default
#                  runs locally on WORLD_SIZE GPUs. Set Slurm account, partition,
#                  container, mounts, nodes, and GPUs here for multi-node models.
#   RUN_COMPARE: set to 1 to run the HF/Megatron logits comparison (default: 0)
#   RUN_ROUNDTRIP: set to 1 to run the second import/export round-trip (default: 0)
#
# Defaults below are for GB200 (192 GB). For H100 (80 GB) configs, see README.md.

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash-Base}
HF_MODEL_ID=${HF_MODEL_ID:-deepseek-ai/${MODEL_VARIANT}}
RUN_COMPARE=${RUN_COMPARE:-0}
RUN_ROUNDTRIP=${RUN_ROUNDTRIP:-0}
read -r -a UV_RUN_ARGS_ARRAY <<< "${UV_RUN_ARGS:-}"

if [[ -z "${EP:-}" ]]; then
    case "${MODEL_VARIANT}" in
        DeepSeek-V4-Pro*) EP=8 ;;
        *)                EP=4 ;;
    esac
fi
if [[ -z "${PP:-}" ]]; then
    case "${MODEL_VARIANT}" in
        DeepSeek-V4-Pro*) PP=4 ;;
        *)                PP=1 ;;
    esac
fi
TP=1
WORLD_SIZE=$((TP * PP * EP))
read -r -a CONVERSION_EXECUTOR_ARGS_ARRAY <<< \
    "${CONVERSION_EXECUTOR_ARGS:---executor local --device gpu --gpus-per-node ${WORLD_SIZE}}"

_first_slurm_host() {
    local nodelist=$1
    local prefix entries first

    if [[ "${nodelist}" != *"["* ]]; then
        echo "${nodelist%%,*}"
        return
    fi

    prefix="${nodelist%%[*}"
    entries="${nodelist#*[}"
    entries="${entries%%]*}"
    first="${entries%%,*}"
    first="${first%%-*}"
    echo "${prefix}${first}"
}

NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-1}}
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
        NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
    elif (( NNODES > 1 )); then
        NPROC_PER_NODE=$((WORLD_SIZE / NNODES))
    else
        NPROC_PER_NODE=${WORLD_SIZE}
    fi
fi

if (( NPROC_PER_NODE * NNODES != WORLD_SIZE )); then
    echo "NPROC_PER_NODE (${NPROC_PER_NODE}) * NNODES (${NNODES}) must equal TP*PP*EP (${WORLD_SIZE})." >&2
    exit 1
fi

MASTER_PORT=${MASTER_PORT:-29500}
TORCHRUN=(uv run "${UV_RUN_ARGS_ARRAY[@]}" python -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}")
if (( NNODES > 1 )); then
    NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
    if [[ -z "${MASTER_ADDR:-}" ]]; then
        if [[ -n "${SLURM_NODELIST:-}" ]]; then
            MASTER_ADDR=$(_first_slurm_host "${SLURM_NODELIST}")
        else
            echo "MASTER_ADDR must be set when NNODES=${NNODES}." >&2
            exit 1
        fi
    fi
    TORCHRUN+=(--nnodes "${NNODES}" --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}")
fi

echo "DeepSeek-V4 conversion: MODEL_VARIANT=${MODEL_VARIANT} HF_MODEL_ID=${HF_MODEL_ID}"
echo "Parallelism: TP=${TP} PP=${PP} EP=${EP} WORLD_SIZE=${WORLD_SIZE}"
echo "Launch: NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} NODE_RANK=${NODE_RANK:-0} MASTER_ADDR=${MASTER_ADDR:-local} MASTER_PORT=${MASTER_PORT}"

MEGATRON_DIR="${WORKSPACE}/models/${MODEL_VARIANT}"
EXPORT_DIR="${WORKSPACE}/models/${MODEL_VARIANT}-hf-export"
ITER=iter_0000000

# 1) Import HF -> Megatron (FP8 / MXFP4 dequantised to bfloat16 in-flight)
./scripts/conversion/convert.sh import \
    "${CONVERSION_EXECUTOR_ARGS_ARRAY[@]}" \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_DIR}" \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --torch-dtype bfloat16 \
    --trust-remote-code

# 2) Compare HF and Megatron logits on a short prompt
if [[ "${RUN_COMPARE}" == "1" ]]; then
    "${TORCHRUN[@]}" \
        examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "${HF_MODEL_ID}" \
        --megatron_model_path "${MEGATRON_DIR}" \
        --prompt "Hello, how are you?" \
        --tp ${TP} --pp ${PP} --ep ${EP} \
        --trust-remote-code
fi

# 3) Export Megatron -> HF (round-trip)
./scripts/conversion/convert.sh export \
    "${CONVERSION_EXECUTOR_ARGS_ARRAY[@]}" \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_DIR}/${ITER}" \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --torch-dtype bfloat16 \
    --hf-path "${EXPORT_DIR}" \
    --distributed-save \
    --trust-remote-code

# 4) Round-trip validation (bf16 -> Megatron -> bf16)
# DSv4 HF weights are quantized (FP8/MXFP4), so the first import dequantises
# to bfloat16. A true lossless roundtrip re-imports the exported bf16 checkpoint
# and compares against the first export.
if [[ "${RUN_ROUNDTRIP}" == "1" ]]; then
    ROUNDTRIP_DIR="${WORKSPACE}/models/${MODEL_VARIANT}-roundtrip"
    ./scripts/conversion/convert.sh import \
        "${CONVERSION_EXECUTOR_ARGS_ARRAY[@]}" \
        --hf-model "${EXPORT_DIR}" \
        --megatron-path "${ROUNDTRIP_DIR}" \
        --tp ${TP} --pp ${PP} --ep ${EP} \
        --torch-dtype bfloat16 \
        --trust-remote-code

    ROUNDTRIP_EXPORT_DIR="${WORKSPACE}/models/${MODEL_VARIANT}-roundtrip-export"
    ./scripts/conversion/convert.sh export \
        "${CONVERSION_EXECUTOR_ARGS_ARRAY[@]}" \
        --hf-model "${EXPORT_DIR}" \
        --megatron-path "${ROUNDTRIP_DIR}" \
        --hf-path "${ROUNDTRIP_EXPORT_DIR}" \
        --tp ${TP} --pp ${PP} --ep ${EP} \
        --torch-dtype bfloat16 \
        --distributed-save \
        --trust-remote-code
fi
