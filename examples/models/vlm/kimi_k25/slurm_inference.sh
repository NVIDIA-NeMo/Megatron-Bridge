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
# Kimi K2.5 VL — Megatron Inference (text + optional vision)
#
# Runs generation from either:
#   (a) a pre-converted Megatron checkpoint  (set MEGATRON_CHECKPOINT), or
#   (b) the HuggingFace model directly       (leave MEGATRON_CHECKPOINT empty)
#
# Default: TP=2 EP=128, 32 nodes (256 GPUs)
#
# NOTE: The model has 384 routed experts × 60 MoE layers = ~1T expert params.
#       In bf16 this is ~1890 GB. EP=128 gives ~15 GB/GPU for experts. TP=2
#       halves attention/shared params. Total ~27 GB/GPU leaves ample room
#       for TE workspace buffers on 80 GB GPUs.
#
# Usage:
#   sbatch examples/models/vlm/kimi_k25/slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=kimi-k25-vl-infer
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/yuya/logs/kimi_k25_infer_%j.log
#SBATCH --exclusive

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/users/yuya/containers/mbridge-260128.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/coreai/users/yuya/Megatron-Hub:/opt/Megatron-Bridge,/lustre/fsw/portfolios/coreai/users/yuya/root/data:/opt/data"

# ── Paths ────────────────────────────────────────────────────────────────
WORKDIR="/opt/Megatron-Bridge"
HF_MODEL_ORIG="/lustre/fsw/portfolios/coreai/users/yuya/kimi-k25-real"
HF_MODEL_PATH="/lustre/fsw/portfolios/coreai/users/yuya/kimi-k25-real-patched"

# Option: Load from pre-converted Megatron checkpoint (faster, skip on-the-fly conversion)
# On-the-fly conversion from HF (no pre-converted checkpoint needed)
MEGATRON_CHECKPOINT=""

# ── Inference config ─────────────────────────────────────────────────────
PROMPT="What is the meaning of life? Please answer in detail."
MAX_NEW_TOKENS=200

# ── Parallelism ──────────────────────────────────────────────────────────
TP=2
EP=128

# ── Tokens / Caches ──────────────────────────────────────────────────────
export GH_TOKEN="${GH_TOKEN:?Set GH_TOKEN}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/yuya/HF_HOME
export UV_CACHE_DIR="/lustre/fsw/portfolios/coreai/users/yuya/uv_cache_main"

# ── NCCL ─────────────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================"
echo "Kimi K2.5 VL — Megatron Inference"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "TP=$TP EP=$EP"
echo "HF model (orig): $HF_MODEL_ORIG"
echo "HF model (patched): $HF_MODEL_PATH"
if [ -n "$MEGATRON_CHECKPOINT" ]; then
    echo "Megatron ckpt: $MEGATRON_CHECKPOINT"
fi
echo "======================================"

MEGATRON_ARG=""
if [ -n "$MEGATRON_CHECKPOINT" ]; then
    MEGATRON_ARG="--megatron_model_path $MEGATRON_CHECKPOINT"
fi

CMD="cd $WORKDIR && "
CMD="${CMD}if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync --locked 2>&1 | tail -5; else sleep 30; fi && "
CMD="${CMD}if [ \"\$SLURM_PROCID\" -eq 0 ]; then "
CMD="${CMD}uv run --no-sync python examples/models/vlm/kimi_k25/patch_kimi_k25_compat.py "
CMD="${CMD}--source $HF_MODEL_ORIG --output $HF_MODEL_PATH; fi && "
CMD="${CMD}sleep 10 && "
CMD="${CMD}rm -rf nemo_experiments && "
CMD="${CMD}uv run --no-sync python examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path $HF_MODEL_PATH \
    $MEGATRON_ARG \
    --prompt \"$PROMPT\" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --trust_remote_code \
    --tp $TP \
    --ep $EP"

srun --mpi=pmix \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --no-container-mount-home \
    bash -c "$CMD"

echo "======================================"
echo "Inference complete"
echo "======================================"
