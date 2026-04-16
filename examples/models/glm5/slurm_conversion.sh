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
# GLM-5 Conversion Round-Trip Verification (Multi-Node via Slurm)
#
# GLM-5 (MoE + MLA + DSA: 256 routed experts, top-8, ~800B+ params, BF16)
# The full model requires multi-node — minimum 8 nodes (64 GPUs) with
# EP >= 32.  TP does NOT reduce expert memory — increase EP instead.
#
# Runs HF -> Megatron -> HF round-trip conversion and verifies weight fidelity.
# Saves the exported HF checkpoint to OUTPUT_DIR.
#
# Requirements: transformers >= 5.2.0
#
# Usage:
#   1. Set CONTAINER_IMAGE, CONTAINER_MOUNTS, and token exports (or use defaults)
#   2. Adjust PARALLELISM_CONFIGS if needed
#   3. Submit: sbatch examples/models/glm5/slurm_conversion.sh
# ==============================================================================

#SBATCH --job-name=glm5-roundtrip
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --partition=batch
#SBATCH --output=logs/glm5_roundtrip_%j.log
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

# ── Parallelism configs: "TP,PP,EP" per entry ────────────────────────────
# TP*PP*EP must equal total GPUs (NODES * GPUS_PER_NODE = 64).
# EP must divide 256 (number of routed experts).
PARALLELISM_CONFIGS=("2,1,32")

# ── Model ─────────────────────────────────────────────────────────────────
# Use the direct local snapshot path to avoid 64 processes calling
# snapshot_download simultaneously (causes Lustre race conditions).
HF_MODEL_PATH="${HF_HOME}/hub/models--zai-org--GLM-5/snapshots/$(ls ${HF_HOME}/hub/models--zai-org--GLM-5/snapshots/ | head -1)"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/glm5_converted}"

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "GLM-5 Round-Trip Conversion Sweep"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "Parallelism configs: ${PARALLELISM_CONFIGS[*]}"
echo "======================================"

mkdir -p logs

CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP <<< "$CONFIG"
    CONFIG_INDEX=$((CONFIG_INDEX + 1))

    echo ""
    echo "======================================"
    echo "Config $CONFIG_INDEX/${#PARALLELISM_CONFIGS[@]}: TP=$TP, PP=$PP, EP=$EP"
    echo "======================================"

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
          examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
          --hf-model-id $HF_MODEL_PATH \
          --output-dir $OUTPUT_DIR \
          --tp $TP --pp $PP --ep $EP
      "
    RUN_EXIT=$?
    if [ $RUN_EXIT -ne 0 ]; then
        echo "ERROR: Config TP=$TP, PP=$PP, EP=$EP failed (exit $RUN_EXIT)"
        exit $RUN_EXIT
    fi
    echo "[OK] Config $CONFIG_INDEX: TP=$TP, PP=$PP, EP=$EP passed"
done

echo ""
echo "======================================"
echo "All ${#PARALLELISM_CONFIGS[@]} configs passed"
echo "======================================"
