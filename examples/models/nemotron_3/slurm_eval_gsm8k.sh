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
# Nemotron-3-Nano-30B GSM8K Evaluation
#
# Uses evaluation_with_nemo_run.py (Evaluator/scripts) with Ray backend.
#
# Usage:
#   1. Set CHECKPOINT / parallelism below
#   2. Submit: sbatch slurm_eval_gsm8k.sh
# ==============================================================================

#SBATCH --job-name=nemotron-nano-eval-gsm8k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --output=logs/nemotron_nano_eval_gsm8k_%j.out
#SBATCH --error=logs/nemotron_nano_eval_gsm8k_%j.err

# ==============================================================================
# CONFIGURATION
# ==============================================================================

export WKDIR="/lustre/fsw/portfolios/coreai/users/weijiac"
export WORKSPACE="/lustre/fsw/portfolios/coreai/users/weijiac/nemo_workspace"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/weijiac/sqsh/nemo_26.02.rc5.sqsh}"

# ── Checkpoint ────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-/lustre/fsw/portfolios/coreai/users/weijiac/nemo_workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000}"
RUN_NAME="nemotron_3_nano_base_ep4"

# ── Parallelism for deployment ─────────────────────────────────────────────────
TP=1
PP=1
EP=4
NUM_GPUS=4

# ── Batch / eval settings ──────────────────────────────────────────────────────
MAX_BATCH_SIZE=8
PARALLELISM=8

# ==============================================================================
# JOB EXECUTION
# ==============================================================================

EVAL_SCRIPT_DIR="${WORKSPACE}/Evaluator/scripts"
EVAL_OUTPUT="${WORKSPACE}/results/eval_gsm8k/${RUN_NAME}"
mkdir -p logs "${EVAL_OUTPUT}"

# Mount output dir to /results (evaluation_with_nemo_run.py hardcodes output_dir="/results/")
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre,${WORKSPACE}/Megatron-Bridge:/opt/Megatron-Bridge,${WORKSPACE}/Megatron-LM:/opt/megatron-lm,${WORKSPACE}/Export-Deploy:/opt/Export-Deploy,${EVAL_OUTPUT}:/results}"

echo "============================================"
echo "Nemotron-3-Nano-30B  |  GSM8K Evaluation"
echo "============================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Parallelism : TP=${TP}  PP=${PP}  EP=${EP}  (${NUM_GPUS} GPUs)"
echo "Batch size  : ${MAX_BATCH_SIZE}"
echo "Output      : ${EVAL_OUTPUT}"
echo "Container   : ${CONTAINER_IMAGE}"
echo "============================================"

srun \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -c "
        export PYTHONPATH=/opt/Export-Deploy:/opt/megatron-lm:/opt/Megatron-Bridge/src:\${PYTHONPATH}
        export LOG_DIR=/results
        cd ${EVAL_SCRIPT_DIR}
        python3 evaluation_with_nemo_run.py \
            --megatron_checkpoint '${CHECKPOINT}' \
            --serving_backend ray \
            --eval_task gsm8k \
            --tensor_parallelism_size ${TP} \
            --pipeline_parallelism_size ${PP} \
            --expert_model_parallel_size ${EP} \
            --devices ${NUM_GPUS} \
            --nodes 1 \
            --batch_size ${MAX_BATCH_SIZE} \
            --parallel_requests ${PARALLELISM} \
            --server_port 8082 \
            --endpoint_type completions \
            --additional_args '--batch_wait_timeout_s 0.1'
    "

EXIT_CODE=$?
echo "============================================"
echo "Job finished with exit code: ${EXIT_CODE}"
echo "Results written to: ${EVAL_OUTPUT}"
echo "============================================"
echo "GSM8K SCORES"
grep "value:" "${EVAL_OUTPUT}/results.yml" 2>/dev/null || echo "results.yml not found"
echo "============================================"
exit ${EXIT_CODE}
