#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# GLM-5 Multi-Node Inference (Slurm)
#
# GLM-5 (MoE + MLA + DSA: 256 routed experts, top-8, ~800B+ params, BF16)
# Full model requires 8 nodes (64 GPUs) minimum.
# TP does NOT reduce expert memory — increase EP instead.
# Recommended: TP=2, EP=32, PP=1 (64 GPUs, 8 nodes).
#
# Loads the HF checkpoint, converts to Megatron in-memory, and runs greedy
# text generation via examples/conversion/hf_to_megatron_generate_text.py.
#
# Requirements: transformers >= 5.2.0
#
# Usage:
#   sbatch examples/models/glm5/slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=glm5-infer
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=0:30:00
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --partition=batch
#SBATCH --output=logs/glm5_inference_%j.log
#SBATCH --exclusive

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to your .sqsh container path}"
CONTAINER_MOUNTS="/lustre:/lustre"
WORKDIR="/opt/Megatron-Bridge"
BRIDGE_PATH="${BRIDGE_PATH:?Set BRIDGE_PATH to your Megatron-Bridge checkout on shared storage}"

# ── Tokens / Caches ──────────────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN for gated model access}"
export HF_HOME="${HF_HOME:?Set HF_HOME to your HuggingFace cache directory}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"

# ── Model / Parallelism ──────────────────────────────────────────────────
# Use the direct local snapshot path to avoid 64 processes calling
# snapshot_download simultaneously (causes Lustre race conditions).
HF_MODEL_PATH="${HF_HOME}/hub/models--zai-org--GLM-5/snapshots/$(ls ${HF_HOME}/hub/models--zai-org--GLM-5/snapshots/ | head -1)"
TP=2
EP=32
PP=1

PROMPT="What is artificial intelligence?"
MAX_NEW_TOKENS=100

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "GLM-5 Inference"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "TP=$TP PP=$PP EP=$EP (Total GPUs: $((SLURM_JOB_NUM_NODES * 8)))"
echo "======================================"

mkdir -p logs

srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="${BRIDGE_PATH}:${WORKDIR},${CONTAINER_MOUNTS}" \
  --no-container-mount-home \
  bash -c "
    export HF_TOKEN='$HF_TOKEN'
    export HF_HOME='$HF_HOME'
    export UV_CACHE_DIR='$UV_CACHE_DIR'
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export HF_HUB_OFFLINE=1
    export NCCL_TIMEOUT=1800000
    MASTER_ADDR=\$(python3 -c \"
import re, os
s = os.environ.get('SLURM_NODELIST', '')
m = re.match(r'([\w-]+)\[(\d+)', s)
print(m.group(1) + m.group(2) if m else s.split(',')[0])
\")
    cd $WORKDIR
    export PYTHONPATH=$WORKDIR/.venv/lib/python3.12/site-packages:\${PYTHONPATH:-}
    uv run --no-sync python -m torch.distributed.run \
      --nproc_per_node=8 \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=\$SLURM_PROCID \
      --master_addr=\$MASTER_ADDR \
      --master_port=29500 \
      examples/conversion/hf_to_megatron_generate_text.py \
      --hf_model_path $HF_MODEL_PATH \
      --prompt '$PROMPT' \
      --max_new_tokens $MAX_NEW_TOKENS \
      --tp $TP --ep $EP --pp $PP
  "

echo "======================================"
echo "Inference completed (exit $?)"
echo "======================================"
