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
# DeepSeek-V4-Flash full SFT, end to end.
#
# Two phases inside one allocation:
#   1. Import the HF checkpoint into a Megatron checkpoint (FP8/MXFP4 -> bf16).
#   2. Full-parameter SFT on top of the imported weights.
#
# Runs on Hopper (H100/H200) and Blackwell (B200/GB200), with MTP on or off.
# The model is TP=1, PP=4, EP=8 -> 32 GPUs:
#   * GB200 NVL: 4 GPUs/node -> 8 nodes  (the #SBATCH default below)
#   * H100/H200: 8 GPUs/node -> 4 nodes  (set HARDWARE=hopper, --nodes=4,
#                                          --gpus-per-node=8)
#
# Usage:
#   1. Edit the #SBATCH directives for your cluster (account/partition, and the
#      node/gpus-per-node block for your hardware).
#   2. Set CONTAINER_IMAGE to a container that has the DSv4 prerequisites
#      (Megatron-Bridge on a main2dev Megatron-LM, plus fast_hadamard_transform).
#   3. Pick HARDWARE (blackwell|hopper) and MTP (on|off) below.
#   4. sbatch slurm_sft.sh
#
# DSv4 sparse attention (CSA/DSA indexer) rejects packed sequences
# (csa.py asserts packed_seq_params is None), so SFT runs unpacked (SBHD).
# Do not enable dataset.enable_offline_packing.
# ==============================================================================

#SBATCH --job-name=dsv4-sft
#SBATCH --account=my_account
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/dsv4_sft_%j.out
#SBATCH --error=logs/dsv4_sft_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

HARDWARE=${HARDWARE:-blackwell}   # blackwell (B200/GB200) | hopper (H100/H200)
MTP=${MTP:-on}                    # on | off

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash}
HF_MODEL_ID="deepseek-ai/${MODEL_VARIANT}"

# Parallelism for the full Flash model. DSv4 requires TP=1 (MLA TP is not
# supported alongside the hybrid attention path); scale with PP and EP.
TP=1
PP=4
EP=8
CP=1
WORLD_SIZE=$((TP * PP * EP * CP))   # 32

SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-1000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LR=${LR:-5e-6}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
# eval_iters x global_batch_size samples are drawn per eval; keep
# eval_iters * GBS <= your validation/test set size or the eval hangs trying to
# form a batch (a small test set is the usual culprit).
EVAL_ITERS=${EVAL_ITERS:-2}
# The end-of-run test eval (do_test) is off by default — it's optional for SFT
# and is the common hang source on small test sets. Set DO_TEST=true if your
# test split is large enough.
DO_TEST=${DO_TEST:-false}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-1}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge-dsv4-sft}

# Checkpoint saving. Default OFF: the goal here is to get training running, which
# only needs the base weights *loaded*, not new checkpoints written. Each SFT save
# is ~570 GB (full bf16 model), so skipping them keeps the whole run inside ~750 GB
# (HF cache + the imported base) -- comfortable on a 2 TB scratch.
# Set SAVE_CKPT=1 for a real fine-tune that needs to persist weights.
SAVE_CKPT=${SAVE_CKPT:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-300}
KEEP_CKPTS=${KEEP_CKPTS:-1}        # checkpoint.most_recent_k when SAVE_CKPT=1; -1 keeps all

# The imported (base) checkpoint we load from. The bf16 import is ~570 GB; on a
# small (e.g. 2 TB) personal scratch, put MEGATRON_DIR and HF_HOME on shared/project
# storage (read-only base weights, reusable across runs and users).
MEGATRON_DIR=${MEGATRON_DIR:-${WORKSPACE}/models/${MODEL_VARIANT}}
PRETRAINED_CKPT="${MEGATRON_DIR}/iter_0000000"   # save_megatron_model writes here

CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/lustre:/lustre,/workspace:/workspace"

# Pre-stage these on shared storage before submitting (see README):
# export HF_HOME="/home/scratch.${USER}/HF_HOME"   # persists the HF download
# export HF_TOKEN="hf_your_token_here"             # if the repo is gated
# export WANDB_API_KEY="your_wandb_key_here"

# ==============================================================================
# Recipe selection
#
# The SFT recipe is hardware-agnostic: use_fused_mhc=False (the validated, unfused
# mHC path) runs on BOTH Hopper and Blackwell. (The fused mHC kernel currently NaNs
# in SFT — see README Blockers — so there is no separate Blackwell recipe.) HARDWARE
# only affects topology (nodes x gpus-per-node) below. MTP selects the recipe.
# ==============================================================================

case "${MTP}" in
    on)  RECIPE_NAME=deepseek_v4_flash_sft_config ;;
    off) RECIPE_NAME=deepseek_v4_flash_no_mtp_sft_config ;;
    *)   echo "ERROR: MTP must be 'on' or 'off' (got ${MTP})"; exit 1 ;;
esac

case "${HARDWARE}" in
    blackwell|hopper) ;;
    *) echo "ERROR: HARDWARE must be 'blackwell' or 'hopper' (got ${HARDWARE})"; exit 1 ;;
esac

# ==============================================================================
# Topology
# ==============================================================================

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set to a container with the DSv4 prerequisites."
    exit 1
fi

NNODES=${SLURM_JOB_NUM_NODES:-$([ "$HARDWARE" = "hopper" ] && echo 4 || echo 8)}
NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-$([ "$HARDWARE" = "hopper" ] && echo 8 || echo 4)}

if [ $((NNODES * NPROC_PER_NODE)) -ne "$WORLD_SIZE" ]; then
    echo "ERROR: NNODES($NNODES) x NPROC_PER_NODE($NPROC_PER_NODE) != WORLD_SIZE($WORLD_SIZE)."
    echo "       Fix the #SBATCH --nodes / --gpus-per-node block for ${HARDWARE}."
    exit 1
fi

MASTER_PORT=${MASTER_PORT:-29571}
MASTER_ADDR=$(python3 - <<'PY'
import os
import re

nodelist = os.environ.get("SLURM_NODELIST", "")
match = re.match(r"([\w-]+)\[(\d+)", nodelist)
print(match.group(1) + match.group(2) if match else (nodelist.split(",")[0] if nodelist else "localhost"))
PY
)

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

TORCHRUN="python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

RUN_NAME=${MODEL_VARIANT}_sft_${HARDWARE}_mtp-${MTP}_tp${TP}_pp${PP}_ep${EP}_${SLURM_JOB_ID:-manual}
CHECKPOINT_DIR=${WORKSPACE}/results/${RUN_NAME}/checkpoints

# Saving is gated on checkpoint.save (None -> nothing written). Default mode loads
# the base weights and trains without persisting anything.
if [ "$SAVE_CKPT" = "1" ]; then
    SAVE_OVERRIDES="checkpoint.save=$CHECKPOINT_DIR checkpoint.save_interval=$SAVE_INTERVAL checkpoint.most_recent_k=$KEEP_CKPTS"
    SAVE_DESC="$CHECKPOINT_DIR (every $SAVE_INTERVAL it, keep $KEEP_CKPTS)"
else
    SAVE_OVERRIDES="checkpoint.save=null checkpoint.load=null"
    SAVE_DESC="disabled (load-only run; set SAVE_CKPT=1 to persist)"
fi

echo "======================================"
echo "DeepSeek-V4-Flash SFT"
echo "======================================"
echo "Job ID:        ${SLURM_JOB_ID:-manual}"
echo "Hardware:      $HARDWARE  ($NNODES nodes x $NPROC_PER_NODE GPU = $WORLD_SIZE)"
echo "MTP:           $MTP"
echo "Recipe:        $RECIPE_NAME"
echo "Parallelism:   TP=$TP PP=$PP EP=$EP CP=$CP"
echo "HF model:      $HF_MODEL_ID"
echo "Base ckpt:     $PRETRAINED_CKPT"
echo "SFT output:    $SAVE_DESC"
echo "======================================"

# ------------------------------------------------------------------------------
# Phase 1: HF -> Megatron import (skipped if the base checkpoint already exists)
# ------------------------------------------------------------------------------

if [ -e "${PRETRAINED_CKPT}/.metadata" ]; then
    echo "Phase 1: base checkpoint already present at $PRETRAINED_CKPT - skipping import."
else
    echo "Phase 1: importing $HF_MODEL_ID -> $MEGATRON_DIR (TP=$TP PP=$PP EP=$EP)"
    $SRUN_CMD bash -lc "
        set -euo pipefail
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        cd /opt/Megatron-Bridge
        $TORCHRUN scripts/conversion/run_conversion.py import \
            --device gpu \
            --hf-model '$HF_MODEL_ID' \
            --megatron-path '$MEGATRON_DIR' \
            --tp $TP --pp $PP --ep $EP \
            --torch-dtype bfloat16 \
            --trust-remote-code
    "
fi

# ------------------------------------------------------------------------------
# Phase 2: full SFT
#
# No --dataset: the recipe ships an unpacked (SBHD) SQuAD config. To use another
# source, pass `--dataset gsm8k` or `--dataset local-jsonl dataset.dataset_root=<path>`.
# Do not enable `dataset.enable_offline_packing`; DSv4 rejects packed sequences.
# ------------------------------------------------------------------------------

CLI_OVERRIDES=" \
    model.seq_length=$SEQ_LENGTH \
    dataset.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    optimizer.lr=$LR \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    scheduler.lr_decay_iters=$TRAIN_ITERS \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    dataset.do_test=$DO_TEST \
    checkpoint.pretrained_checkpoint=$PRETRAINED_CKPT \
    $SAVE_OVERRIDES \
    checkpoint.load_optim=false \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=$RUN_NAME \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.expert_model_parallel_size=$EP \
    model.context_parallel_size=$CP"

echo "Phase 2: SFT with recipe $RECIPE_NAME"
$SRUN_CMD bash -lc "
    set -euo pipefail
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export NCCL_PXN_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    cd /opt/Megatron-Bridge
    $TORCHRUN scripts/training/run_recipe.py \
        --recipe $RECIPE_NAME \
        --step_func gpt_step \
        $CLI_OVERRIDES
"

echo "======================================"
echo "DeepSeek-V4-Flash SFT job completed"
echo "======================================"
