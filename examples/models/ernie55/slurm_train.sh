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
# ERNIE 5.5 training with SLURM: fine-tune a Hugging Face checkpoint
#
# Runs an ERNIE 5.5 recipe (megatron.bridge.recipes.ernie) with the ERNIE 5.5 forward step, one srun
# task per GPU in a Pyxis/enroot container. The default recipe is ernie55_sft_64gpu_gb300_bf16_config: the
# production shape (16 nodes x 4 GPUs, TP1 PP1 CP64 EP64, one 262144-token row per micro-batch) starting from the
# weights of ERNIE55_HF_PATH. A 16-node job needs a QoS that admits 16 nodes.
#
# Usage:
#   ERNIE55_HF_PATH=/path/to/ernie55_hf DATA_PATH=/path/to/sft.jsonl \
#   CONTAINER_IMAGE=/path/to/image.sqsh MEGATRON_LM_PATH=/path/to/Megatron-LM CONTAINER_MOUNTS="/path:/path" \
#       sbatch --account=<account> --partition=<partition> --qos=<qos> slurm_train.sh
#
# On H100, the recipe is ernie55_sft_128gpu_h100_bf16_config (megatron.bridge.recipes.ernie.h100.ernie55: 16 nodes
# x 8 GPUs, TP1 PP8 CP1 EP16, packed rows of at most 4096 tokens, about 12 to 13 hours for its 8260 steps); add
# RECIPE and ask for 8 tasks and GPUs per node and for the time:
#   RECIPE=ernie55_sft_128gpu_h100_bf16_config <the variables above> \
#       sbatch --ntasks-per-node=8 --gpus-per-node=8 --time=16:00:00 --account=<account> ... slurm_train.sh
#
# Required env vars:
#   CONTAINER_IMAGE       enroot/squashfs image with Megatron Bridge's dependencies
#   MEGATRON_LM_PATH      Megatron-LM dev checkout at or after f78809f6d (it has the DSv4 _post_o_group_proj hook),
#                         put on PYTHONPATH ahead of any installed megatron-core
#   ERNIE55_HF_PATH       Hugging Face ERNIE 5.5 checkpoint directory (config.json, safetensors, tokenizer files);
#                         the recipe builds the model from its config.json and starts from its weights
#
# Optional env vars:
#   MEGATRON_BRIDGE_PATH  Megatron Bridge checkout (default: /opt/Megatron-Bridge)
#   WORKSPACE             base directory for checkpoints and logs (default: /workspace)
#   RECIPE                recipe function name (default: ernie55_sft_64gpu_gb300_bf16_config)
#   DATA_PATH             SFT JSONL: chat ("messages"), prompt/completion ("input"/"output") or pre-tokenized
#                         ("input_ids"/"labels") rows; synthetic rows when unset
#   HF_TOKENIZER          tokenizer directory for text rows (default: ERNIE55_HF_PATH when it holds tokenizer.json)
#   TRAIN_ITERS           number of training steps (default: the recipe's)
#   EXTRA_OVERRIDES       more recipe overrides, space-separated key=value, appended after the ones above
#                         (e.g. "checkpoint.save_optim=true")
#   GPUS_PER_NODE         GPUs, and tasks, per node (default: the job's --ntasks-per-node, which is 4 here)
#   CHECKPOINT_DIR        checkpoint directory, used to save AND to load, so a resubmission resumes from its last
#                         saved step (default: $WORKSPACE/checkpoints/<RECIPE without _config>); use a new directory
#                         for a fresh run
#   CONTAINER_MOUNTS      space-separated container mounts, e.g. "/data:/data /workspace:/workspace"
#   MASTER_PORT           rendezvous port (default: 29500)
# ==============================================================================

#SBATCH --job-name=ernie55
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --account=my_account
#SBATCH --output=ernie55_%j.out
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-/opt/Megatron-Bridge}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
RECIPE=${RECIPE:-ernie55_sft_64gpu_gb300_bf16_config}
ERNIE55_HF_PATH=${ERNIE55_HF_PATH:-}
DATA_PATH=${DATA_PATH:-}
HF_TOKENIZER=${HF_TOKENIZER:-}
TRAIN_ITERS=${TRAIN_ITERS:-}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
RUN_NAME=${RUN_NAME:-${RECIPE%_config}}
LOG_DIR=${LOG_DIR:-${WORKSPACE}/logs/${RUN_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${WORKSPACE}/checkpoints/${RUN_NAME}}

if [ -z "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi
if [ -z "${MEGATRON_LM_PATH}" ]; then
    echo "ERROR: MEGATRON_LM_PATH must name a Megatron-LM dev checkout at or after f78809f6d."
    exit 1
fi
# The sft recipe fine-tunes the checkpoint (run_recipe.py infers Bridge's finetune loop from its name).
if [ -z "${ERNIE55_HF_PATH}" ]; then
    echo "ERROR: ${RECIPE} fine-tunes a Hugging Face checkpoint: set ERNIE55_HF_PATH to its directory."
    exit 1
fi
if [ -z "${HF_TOKENIZER}" ] && [ -f "${ERNIE55_HF_PATH}/tokenizer.json" ]; then
    HF_TOKENIZER=${ERNIE55_HF_PATH}
fi
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# Recipe overrides forwarded to run_recipe.py.
# checkpoint.load follows checkpoint.save: the recipes would otherwise load from <checkout>/nemo_experiments.
OVERRIDES="checkpoint.save=${CHECKPOINT_DIR} checkpoint.load=${CHECKPOINT_DIR} logger.tensorboard_dir=${LOG_DIR}/tensorboard"
if [ -n "${DATA_PATH}" ]; then
    OVERRIDES="${OVERRIDES} dataset.path=${DATA_PATH}"
    if [ -n "${HF_TOKENIZER}" ]; then
        OVERRIDES="${OVERRIDES} dataset.hf_tokenizer=${HF_TOKENIZER}"
    fi
fi
if [ -n "${TRAIN_ITERS}" ]; then
    OVERRIDES="${OVERRIDES} train.train_iters=${TRAIN_ITERS}"
fi
if [ -n "${EXTRA_OVERRIDES}" ]; then
    OVERRIDES="${OVERRIDES} ${EXTRA_OVERRIDES}"
fi

# ==============================================================================
# Environment (the recipe sets its own launch variables, such as CUDA_DEVICE_MAX_CONNECTIONS=1, at startup)
# ==============================================================================

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}
export MEGATRON_BRIDGE_PATH MEGATRON_LM_PATH RECIPE OVERRIDES ERNIE55_HF_PATH

NUM_NODES=${SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-4}}

echo "======================================"
echo "ERNIE 5.5: ${RECIPE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${NUM_NODES}, GPUs: $((NUM_NODES * GPUS_PER_NODE))"
echo "Checkpoint in: ${ERNIE55_HF_PATH}"
echo "Data: ${DATA_PATH:-synthetic rows}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "======================================"

# ==============================================================================
# Job execution: one task per GPU, each one rank
# ==============================================================================

read -r -d '' INNER_SCRIPT <<'EOF' || true
set -euo pipefail
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
export PYTHONPATH=${MEGATRON_BRIDGE_PATH}/src:${MEGATRON_LM_PATH}:${PYTHONPATH:-}
cd "${MEGATRON_BRIDGE_PATH}"
# shellcheck disable=SC2086  # OVERRIDES is a space-separated list of key=value arguments
python scripts/training/run_recipe.py --recipe "${RECIPE}" --step_func ernie55_step ${OVERRIDES}
EOF

SRUN_ARGS=(
    --nodes="${NUM_NODES}"
    --ntasks-per-node="${GPUS_PER_NODE}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --container-image="${CONTAINER_IMAGE}"
    --no-container-mount-home
    --kill-on-bad-exit=1
    --wait=60
    --output="${LOG_DIR}/${RUN_NAME}_%j.log"
)
if [ -n "${CONTAINER_MOUNTS}" ]; then
    # Pyxis takes one comma-separated list of mounts.
    SRUN_ARGS+=(--container-mounts="${CONTAINER_MOUNTS// /,}")
fi

srun "${SRUN_ARGS[@]}" bash -c "${INNER_SCRIPT}"

echo "======================================"
echo "Job completed"
echo "======================================"
