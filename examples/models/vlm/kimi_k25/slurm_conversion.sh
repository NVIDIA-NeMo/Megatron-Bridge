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
# Kimi K2.5 VL — HF → Megatron Distributed Conversion
#
# Converts the full Kimi K2.5 VL model (61 layers, 384 experts, INT4/FP8
# quantized) from HuggingFace format to a distributed Megatron checkpoint.
#
# Default: TP=2 EP=64, 16 nodes (128 GPUs)
# The bridge auto-dequantizes INT4 routed experts and FP8 weights during import.
#
# NOTE: The model has 384 routed experts × 60 MoE layers = ~1T expert params.
#       In bf16 this is ~1890 GB. EP=64 gives ~30 GB/GPU for experts. TP=2
#       halves attention/shared params (~12 GB vs 24 GB), leaving headroom
#       for TE workspace and activations on 80 GB GPUs.
#
# Usage:
#   sbatch examples/models/vlm/kimi_k25/slurm_conversion.sh
# ==============================================================================

#SBATCH --job-name=kimi-k25-vl-convert
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/yuya/logs/kimi_k25_convert_%j.log
#SBATCH --exclusive

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/users/yuya/containers/mbridge-260128.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/coreai/users/yuya/Megatron-Hub:/opt/Megatron-Bridge,/lustre/fsw/portfolios/coreai/users/yuya/root/data:/opt/data"

# ── Paths ────────────────────────────────────────────────────────────────
WORKDIR="/opt/Megatron-Bridge"
HF_MODEL_ORIG="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/aot/kimi/kimi-k2.5-test-weights_vv1"
HF_MODEL_PATH="/lustre/fsw/portfolios/coreai/users/yuya/kimi-k25-vl-patched"
MEGATRON_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/yuya/checkpoints/kimi-k25-vl-megatron-tp2ep64"

# ── Parallelism ──────────────────────────────────────────────────────────
TP=2
EP=64

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
echo "Kimi K2.5 VL — HF → Megatron Conversion"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "TP=$TP EP=$EP"
echo "HF model (orig): $HF_MODEL_ORIG"
echo "HF model (patched): $HF_MODEL_PATH"
echo "Save to:  $MEGATRON_PATH"
echo "======================================"

CMD="cd $WORKDIR && "
CMD="${CMD}if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync --locked 2>&1 | tail -5; else sleep 30; fi && "
CMD="${CMD}if [ \"\$SLURM_PROCID\" -eq 0 ]; then "
CMD="${CMD}uv run --no-sync python examples/models/vlm/kimi_k25/patch_kimi_k25_compat.py "
CMD="${CMD}--source $HF_MODEL_ORIG --output $HF_MODEL_PATH; fi && "
CMD="${CMD}sleep 10 && "
CMD="${CMD}rm -rf nemo_experiments && "
CMD="${CMD}uv run --no-sync python examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model $HF_MODEL_PATH \
    --megatron-path $MEGATRON_PATH \
    --trust-remote-code \
    --tp $TP \
    --ep $EP"

srun --mpi=pmix \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --no-container-mount-home \
    bash -c "$CMD"

echo "======================================"
echo "Conversion complete"
echo "======================================"
