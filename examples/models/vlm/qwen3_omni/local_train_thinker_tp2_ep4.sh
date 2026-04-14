#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

export NUM_GPUS=8
export TENSOR_PARALLEL_SIZE=2
export PIPELINE_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=4
export EXPERT_TENSOR_PARALLEL_SIZE=1
export SEQUENCE_PARALLEL=False
export TRAIN_ITERS=1
export RUN_NAME=qwen3_omni_full_train_tp2_ep4

echo "[info] fixed parallel config: TP=${TENSOR_PARALLEL_SIZE} PP=${PIPELINE_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} EP=${EXPERT_MODEL_PARALLEL_SIZE} ETP=${EXPERT_TENSOR_PARALLEL_SIZE} SP=${SEQUENCE_PARALLEL}"

bash "${SCRIPT_DIR}/local_train_thinker_full.sh"
