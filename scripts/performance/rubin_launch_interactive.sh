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


# NOTE: THIS SCRIPT IS ADDED TEMPORARILY. IT WILL BE REMOVED IN NEAR FUTURE.
#
# Launch Megatron-Bridge training inside an interactive NeMo container.
#
# Prerequisites:
#   You are already inside a container via something like:
#     srun -A rubin -p <PARTITION> --pty \
#       --container-image <CONTAINER_IMAGE> bash
#
# Usage:
#   bash scripts/performance/rubin_launch_interactive.sh
#
# Override any variable before sourcing, e.g.:
#   MODEL_FAMILY=qwen MODEL_RECIPE=qwen3_30b_a3b bash scripts/performance/rubin_launch_interactive.sh

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# User-configurable variables (override any of these before running)
# ──────────────────────────────────────────────────────────────────────
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODEL_RECIPE="${MODEL_RECIPE:-llama3_8b}"
GPU_TYPE="${GPU_TYPE:-r100}"              # recipe to load (h100/b200/gb200/gb300/b300/r100)
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"    # bf16, fp8_cs, fp8_mx, nvfp4
CONFIG_VARIANT="${CONFIG_VARIANT:-v1}"
DATA="${DATA:-mock}"                      # mock | rp2 | squad
MAX_STEPS="${MAX_STEPS:-10}"
NUM_GPUS="${NUM_GPUS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
HF_TOKEN="${HF_TOKEN:-}"

# Parallelism overrides — single-GPU safe defaults.
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
GBS="${GBS:-128}"
CG_IMPL="${CG_IMPL:-none}"
CG_SCOPE="${CG_SCOPE:-full_iteration}"

# Nsys profiling (set ENABLE_NSYS=1 to activate)
ENABLE_NSYS="${ENABLE_NSYS:-0}"
NSYS_PROFILE_START="${NSYS_PROFILE_START:-5}"
NSYS_PROFILE_END="${NSYS_PROFILE_END:-6}"
NSYS_OUTPUT="${NSYS_OUTPUT:-/tmp/nsys_profile}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx}"

# Megatron-Bridge root inside the container
MBRIDGE_ROOT="${MBRIDGE_ROOT:-/opt/Megatron-Bridge}"

# ──────────────────────────────────────────────────────────────────────
# Environment variables (performance + framework)
# ──────────────────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export TORCH_NCCL_HIGH_PRIORITY=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export HF_HUB_OFFLINE=0
export HF_TOKEN="${HF_TOKEN}"

export PYTHONPATH="${MBRIDGE_ROOT}:${MBRIDGE_ROOT}/scripts/performance${PYTHONPATH:+:$PYTHONPATH}"

# ──────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────
if [ ! -f "${MBRIDGE_ROOT}/scripts/performance/run_script.py" ]; then
    echo "ERROR: run_script.py not found at ${MBRIDGE_ROOT}/scripts/performance/run_script.py"
    echo "       Set MBRIDGE_ROOT to the Megatron-Bridge directory inside the container."
    exit 1
fi

cd "${MBRIDGE_ROOT}"

# ──────────────────────────────────────────────────────────────────────
# Hydra-style overrides & nsys setup
# ──────────────────────────────────────────────────────────────────────

# Hydra-style overrides appended after the argparse flags.
# run_script.py passes unknown args through to set_cli_overrides().
HYDRA_OVERRIDES=(
    # Force FP32 reduction to fix NaN grad norm issues on single-GPU runs.
    mixed_precision.grad_reduce_in_fp32=true
    ddp.grad_reduce_in_fp32=true
)

# Build the nsys wrapper command if profiling is enabled
NSYS_PREFIX=()

if [ "${ENABLE_NSYS}" = "1" ]; then
    HYDRA_OVERRIDES+=(
        profiling.use_nsys_profiler=true
        profiling.profile_step_start="${NSYS_PROFILE_START}"
        profiling.profile_step_end="${NSYS_PROFILE_END}"
        "profiling.profile_ranks=[0]"
    )

    NSYS_PREFIX=(
        nsys profile
        -s none
        -t "${NSYS_TRACE}"
        -o "${NSYS_OUTPUT}"
        --force-overwrite true
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
    )
fi

# ──────────────────────────────────────────────────────────────────────
# Build the command (after nsys may have patched MAX_STEPS)
# ──────────────────────────────────────────────────────────────────────
RUN_CMD=(
    scripts/performance/run_script.py
    --model_family_name  "${MODEL_FAMILY}"
    --model_recipe_name  "${MODEL_RECIPE}"
    --gpu                "${GPU_TYPE}"
    --compute_dtype      "${COMPUTE_DTYPE}"
    --config_variant     "${CONFIG_VARIANT}"
    --data               "${DATA}"
    --num_gpus           "${NUM_GPUS}"
    --max_steps          "${MAX_STEPS}"
    --tensor_model_parallel_size   "${TP}"
    --pipeline_model_parallel_size "${PP}"
    --context_parallel_size        "${CP}"
    --micro_batch_size   "${MBS}"
    --global_batch_size  "${GBS}"
    --cuda_graph_impl    "${CG_IMPL}"
    --cuda_graph_scope   "${CG_SCOPE}"
)

echo "============================================================"
echo " Megatron-Bridge Interactive Launch"
echo "============================================================"
echo " Model:     ${MODEL_FAMILY} / ${MODEL_RECIPE}"
echo " GPU type:  ${GPU_TYPE}  (recipe selection only)"
echo " Precision: ${COMPUTE_DTYPE}"
echo " Data:      ${DATA}"
echo " Steps:     ${MAX_STEPS}"
echo " GPUs:      ${NUM_GPUS}  (nproc_per_node=${NPROC_PER_NODE})"
echo " Parallel:  TP=${TP}, PP=${PP}, CP=${CP}"
echo " Batch:     MBS=${MBS}, GBS=${GBS}"
if [ "${ENABLE_NSYS}" = "1" ]; then
echo " Nsys:      ON  (steps ${NSYS_PROFILE_START}-${NSYS_PROFILE_END}, output: ${NSYS_OUTPUT})"
fi
echo " Hydra:     ${HYDRA_OVERRIDES[*]}"
echo "============================================================"

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    echo "Launching with torchrun (nproc_per_node=${NPROC_PER_NODE}) ..."
    "${NSYS_PREFIX[@]}" torchrun --nproc_per_node="${NPROC_PER_NODE}" "${RUN_CMD[@]}" "${HYDRA_OVERRIDES[@]}"
else
    echo "Launching with python (single process) ..."
    "${NSYS_PREFIX[@]}" python "${RUN_CMD[@]}" "${HYDRA_OVERRIDES[@]}"
fi
