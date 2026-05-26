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
# DeepSeek-V4-Flash Pretraining
#
# This script runs one DeepSeek-V4-Flash recipe through scripts/training/run_recipe.py.
# Defaults are a DCLM release-candidate run following the GPT-OSS script shape:
# 1000 train iterations, validation every 100 iterations, 10 eval iters.
# Defaults use Megatron-LM PR4894 with model.apply_dsa_kernel_fusion=true.
# The validated release path is seq4096 TP1 PP4 EP8 CP1 with full activation
# recompute; this avoids the unfused CSA activation OOM path.
#
# Release recipes:
#   Adam BF16:
#     sbatch --job-name=dsv4-adam-bf16 --export=ALL,RECIPE_NAME=deepseek_v4_flash_pretrain_config,CASE_NAME=dsv4_adam_bf16 slurm_pretrain.sh
#   Muon BF16 (default):
#     sbatch --job-name=dsv4-muon-bf16 --export=ALL,RECIPE_NAME=deepseek_v4_flash_pretrain_muon_config,CASE_NAME=dsv4_muon_bf16 slurm_pretrain.sh
#   Adam MXFP8:
#     sbatch --job-name=dsv4-adam-mxfp8 --export=ALL,RECIPE_NAME=deepseek_v4_flash_pretrain_mxfp8_config,CASE_NAME=dsv4_adam_mxfp8 slurm_pretrain.sh
#
# Fast smoke override:
#   sbatch --export=ALL,SEQ_LENGTH=128,TRAIN_ITERS=60,EVAL_INTERVAL=20,EVAL_ITERS=5 slurm_pretrain.sh
#
# Checkpointed Muon BF16 split run:
#   ./submit_muon_bf16_two_stage.sh
# or submit the stages manually with SAVE_CHECKPOINTS=true, SAVE_INTERVAL=<split>,
# then LOAD_CHECKPOINT=<stage1-outdir>/checkpoints on an afterok-dependent job.
# ==============================================================================

#SBATCH --job-name=dsv4-pretrain
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch_long
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/%u/logs/dsv4_pretrain/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/%u/logs/dsv4_pretrain/%x_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# Cluster-local paths
# ==============================================================================

# CW and OCI use similar path shapes, but they are separate Lustre filesystems.
# Verify these paths on the target cluster before submitting.
export WKDIR="${WKDIR:-/lustre/fsw/portfolios/coreai/users/${USER}}"
export WORKSPACE="${WORKSPACE:-${WKDIR}/nemo_workspace}"

WORKDIR="${WORKDIR:-${WORKSPACE}/Megatron-Bridge-dsv4-train-fused-csa}"
MCORE_DIR="${MCORE_DIR:-${WORKSPACE}/Megatron-LM-dsv4-pr4894}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${WKDIR}/sqsh/nemo_26.04.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre}"

HF_CONFIG="${HF_CONFIG:-${WKDIR}/models/deepseek-ai/DeepSeek-V4-Flash}"
HF_HOME="${HF_HOME:-${WKDIR}/hf_home}"
TRANSFORMERS_SITE="${TRANSFORMERS_SITE:-${WKDIR}/training/dsv4_transformers_5_8_1_nodeps}"
EMERGING_OPTIMIZERS_SITE="${EMERGING_OPTIMIZERS_SITE:-${WKDIR}/training/dsv4_emerging_optimizers_0_2_0}"
PY_DEPS="${PY_DEPS:-${WKDIR}/training/dsv4_python_deps}"

# Fused CSA dependencies. The unfused backend does not require these, but keeping
# them on PYTHONPATH makes it easy to compare unfused and cudnn_dsa jobs.
FHT_SITE="${FHT_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/fast_hadamard_site}"
CUDNN_FE_SITE="${CUDNN_FE_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/cudnn_fe_site}"
FLASH_MLA_SITE="${FLASH_MLA_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/flash_mla_site}"

LOGDIR="${LOGDIR:-${WKDIR}/logs/dsv4_pretrain}"
OUTBASE="${OUTBASE:-${WKDIR}/training/dsv4_pretrain}"

# Set DATASET_TYPE=llm-pretrain and DATASET_NAME=dclm to use preprocessed DCLM.
DATASET_NAME="${DATASET_NAME:-dclm}"
DCLM_DATA_DIR="${DCLM_DATA_DIR:-${WKDIR}/data/dclm/preprocessed}"
DCLM_CACHE="${DCLM_CACHE:-${WKDIR}/.cache}"

# W&B is initialized by Megatron Bridge on the last global rank when this is set.
# The run URL normally appears in the Slurm .err/.out stream and under WANDB_SAVE_DIR.
WANDB_PROJECT="${WANDB_PROJECT:-megatron-bridge-dsv4}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia-nemo-fw-public}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-}"
WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-${OUTBASE}/wandb}"
WANDB_MODE="${WANDB_MODE:-online}"

# ==============================================================================
# Recipe and training configuration
# ==============================================================================

MODEL_NAME="${MODEL_NAME:-deepseek_v4_flash}"
RECIPE_NAME="${RECIPE_NAME:-deepseek_v4_flash_pretrain_muon_config}"
DATASET_TYPE="${DATASET_TYPE:-llm-pretrain}"
STEP_FUNC="${STEP_FUNC:-gpt_step}"
DSA_KERNEL_FUSION="${DSA_KERNEL_FUSION:-true}"
MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"

SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-50}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"
FULLY_PARALLEL_SAVE="${FULLY_PARALLEL_SAVE:-}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"

TP="${TP:-1}"
PP="${PP:-4}"
CP="${CP:-1}"
EP="${EP:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29571}"

NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
RECOMPUTE_OVERRIDES="${RECOMPUTE_OVERRIDES:-model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

case "$RECIPE_NAME" in
    deepseek_v4_flash_pretrain_config|deepseek_v4_flash_pretrain_muon_config|deepseek_v4_flash_pretrain_mxfp8_config)
        ;;
    *)
        echo "ERROR: slurm_pretrain.sh release recipes are deepseek_v4_flash_pretrain_config, deepseek_v4_flash_pretrain_muon_config, and deepseek_v4_flash_pretrain_mxfp8_config."
        echo "       Use a custom script or explicit recipe test harness for experimental recipes."
        exit 1
        ;;
esac

CASE_NAME="${CASE_NAME:-${RECIPE_NAME}_pr4894_fused_dsa}"
OUTDIR="${OUTDIR:-${OUTBASE}/${CASE_NAME}_${SLURM_JOB_ID}}"

# ==============================================================================
# Environment setup and validation
# ==============================================================================

if [ -z "$WANDB_EXP_NAME" ]; then
    WANDB_EXP_NAME="${CASE_NAME}_tp${TP}_pp${PP}_ep${EP}_cp${CP}_seq${SEQ_LENGTH}_gbs${GLOBAL_BATCH_SIZE}_${SLURM_JOB_ID}"
fi

mkdir -p "$LOGDIR" "$OUTDIR" "$PY_DEPS" "$EMERGING_OPTIMIZERS_SITE" "$WANDB_SAVE_DIR" "$DCLM_CACHE"

for required_path in "$WORKDIR" "$MCORE_DIR" "$CONTAINER_IMAGE" "$HF_CONFIG"; do
    if [ ! -e "$required_path" ]; then
        echo "ERROR: required path does not exist: $required_path"
        exit 1
    fi
done

if [ -n "$LOAD_CHECKPOINT" ] && [ ! -e "$LOAD_CHECKPOINT" ]; then
    echo "ERROR: LOAD_CHECKPOINT does not exist: $LOAD_CHECKPOINT"
    exit 1
fi

if [ "$DSA_KERNEL_FUSION" = "true" ]; then
    for required_path in "$FHT_SITE" "$CUDNN_FE_SITE" "$FLASH_MLA_SITE"; do
        if [ ! -e "$required_path" ]; then
            echo "ERROR: DSA_KERNEL_FUSION=true requires path: $required_path"
            exit 1
        fi
    done
fi

MASTER_ADDR=$(python3 - <<'PY'
import os
import re

s = os.environ.get("SLURM_NODELIST", "")
m = re.match(r"([\w-]+)\[(\d+)", s)
print(m.group(1) + m.group(2) if m else s.split(",")[0])
PY
)

DATASET_OVERRIDES=""
if [ "$DATASET_TYPE" = "llm-pretrain" ] && [ "$DATASET_NAME" = "dclm" ]; then
    BLEND_PATHS=""
    for i in $(seq 1 10); do
        pad=$(printf "%02d" "$i")
        prefix="${DCLM_DATA_DIR}/dclm_01_${pad}_text_document"
        if [ -f "${prefix}.bin" ]; then
            BLEND_PATHS="${BLEND_PATHS}\"${prefix}\","
        fi
    done
    BLEND_PATHS="${BLEND_PATHS%,}"
    if [ -z "$BLEND_PATHS" ]; then
        echo "ERROR: no *_text_document.bin files found under $DCLM_DATA_DIR"
        exit 1
    fi
    DATASET_OVERRIDES="dataset.blend=[[${BLEND_PATHS}],null] dataset.split='\"9999,8,2\"' dataset.path_to_cache=${DCLM_CACHE}"
elif [ "$DATASET_TYPE" = "llm-pretrain" ]; then
    echo "WARNING: DATASET_TYPE=llm-pretrain without DATASET_NAME=dclm; provide dataset.blend through EXTRA_OVERRIDES."
fi

CHECKPOINT_OVERRIDES="checkpoint.save=null checkpoint.save_interval=0"
if [ "$SAVE_CHECKPOINTS" = "true" ]; then
    CHECKPOINT_OVERRIDES="checkpoint.save=${OUTDIR}/checkpoints checkpoint.save_interval=${SAVE_INTERVAL}"
fi
if [ -n "$LOAD_CHECKPOINT" ]; then
    CHECKPOINT_OVERRIDES="$CHECKPOINT_OVERRIDES checkpoint.load=${LOAD_CHECKPOINT}"
fi
if [ -n "$FULLY_PARALLEL_SAVE" ]; then
    CHECKPOINT_OVERRIDES="$CHECKPOINT_OVERRIDES checkpoint.fully_parallel_save=${FULLY_PARALLEL_SAVE}"
fi

echo "======================================"
echo "DeepSeek-V4-Flash Pretraining"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Recipe: ${RECIPE_NAME}"
echo "DSA kernel fusion: ${DSA_KERNEL_FUSION}"
echo "Parallelism: TP=${TP} PP=${PP} CP=${CP} EP=${EP}"
echo "Sequence length: ${SEQ_LENGTH}"
echo "Train iters: ${TRAIN_ITERS}"
echo "LR warmup iters: ${LR_WARMUP_ITERS}"
echo "LR decay iters: ${LR_DECAY_ITERS}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Dataset type/name: ${DATASET_TYPE}/${DATASET_NAME}"
echo "Workdir: ${WORKDIR}"
echo "MCore dir: ${MCORE_DIR}"
echo "HF config: ${HF_CONFIG}"
echo "Transformers site: ${TRANSFORMERS_SITE}"
echo "Emerging optimizers site: ${EMERGING_OPTIMIZERS_SITE}"
echo "Output dir: ${OUTDIR}"
echo "Log dir: ${LOGDIR}"
echo "W&B project/entity/name: ${WANDB_PROJECT}/${WANDB_ENTITY}/${WANDB_EXP_NAME}"
echo "W&B save dir: ${WANDB_SAVE_DIR}"
echo "Recompute overrides: ${RECOMPUTE_OVERRIDES}"
echo "Checkpoint overrides: ${CHECKPOINT_OVERRIDES}"
echo "Fully parallel save: ${FULLY_PARALLEL_SAVE:-<recipe default>}"
echo "Extra overrides: ${EXTRA_OVERRIDES:-<none>}"
echo "Load checkpoint: ${LOAD_CHECKPOINT:-<none>}"
echo "======================================"

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNTS --no-container-mount-home"

CLI_OVERRIDES=" \
    train.train_iters=$TRAIN_ITERS \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    dataset.sequence_length=$SEQ_LENGTH \
    model.seq_length=$SEQ_LENGTH \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.context_parallel_size=$CP \
    model.expert_model_parallel_size=$EP \
    model.apply_dsa_kernel_fusion=$DSA_KERNEL_FUSION \
    model.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER_TYPE \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    scheduler.lr_decay_iters=$LR_DECAY_ITERS \
    $CHECKPOINT_OVERRIDES \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_entity=$WANDB_ENTITY \
    logger.wandb_exp_name=$WANDB_EXP_NAME \
    logger.wandb_save_dir=$WANDB_SAVE_DIR \
    logger.log_interval=$LOG_INTERVAL \
    $RECOMPUTE_OVERRIDES \
    $DATASET_OVERRIDES \
    $EXTRA_OVERRIDES"
CMD="uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/training/run_recipe.py \
    --recipe $RECIPE_NAME \
    --dataset $DATASET_TYPE \
    --step_func $STEP_FUNC \
    --hf_path $HF_CONFIG \
    $CLI_OVERRIDES"

$SRUN_CMD bash -lc "
    set -euo pipefail
    export HF_HOME=$HF_HOME
    export HF_HUB_OFFLINE=1
    export UV_CACHE_DIR=/tmp/uv_cache
    export WANDB_MODE=$WANDB_MODE
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export NCCL_PXN_DISABLE=$NCCL_PXN_DISABLE
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_TIMEOUT=1800000
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    cd $WORKDIR
    rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.pth \
          /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.py
    export PYTHONPATH=$PY_DEPS:$TRANSFORMERS_SITE:$EMERGING_OPTIMIZERS_SITE:$FLASH_MLA_SITE:$FHT_SITE:$CUDNN_FE_SITE:$WORKDIR/src:$MCORE_DIR:\${PYTHONPATH:-}
    source $WORKDIR/examples/models/deepseek_v4/hsg_runtime.sh
    ensure_transformers_site $TRANSFORMERS_SITE
    if [[ $RECIPE_NAME == *muon* ]]; then
        ensure_emerging_optimizers_site $EMERGING_OPTIMIZERS_SITE
    fi
    echo \"$CMD\"
    $CMD
"

echo "======================================"
echo "DeepSeek-V4 pretraining job completed"
echo "======================================"
