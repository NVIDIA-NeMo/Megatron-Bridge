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
#
# DeepSeek-V4-Flash Megatron-Bridge forward pass on 1 node (4 B200 192GB GPUs).
# DSv4 requires TP=1; we scale via EP=4. PP=1 (single node).
# Per the PR README the model does not fit on A100 80 GB at TP=1 -- this script
# targets a B200 cluster (e.g. OCI hsg).

#SBATCH --job-name=coreai_dlalgo_llm-mbridge:dsv4-parity-mb
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/<EDIT-ME>/users/<EDIT-ME>/logs/dsv4_parity_mb_%j.log
#SBATCH --exclusive

# --- Paths to edit before submitting ----------------------------------------
CONTAINER_IMAGE=/lustre/<EDIT-ME>/users/<EDIT-ME>/sqsh/nemo_26.04.sqsh
USER_LUSTRE=/lustre/<EDIT-ME>/users/<EDIT-ME>
BRIDGE_HOST=${USER_LUSTRE}/megatron-bridge-3                # host path to this repo
MCORE_HOST=${USER_LUSTRE}/Megatron-LM                        # host path to fork with DSv4 MCore commits
HF_SNAPSHOT=${USER_LUSTRE}/models/deepseek-ai/DeepSeek-V4-Flash
# ----------------------------------------------------------------------------

WORKDIR=/chcui/megatron-bridge-3
MCORE_DIR=/chcui/Megatron-LM
OUTPUT=/chcui/parity/dsv4/logits_mb.pt
PROMPTS_FILE=examples/conversion/deepseek_v4_parity/prompts.json

TP=1
PP=1
EP=4
ETP=1

mkdir -p ${USER_LUSTRE}/logs ${USER_LUSTRE}/parity/dsv4

echo "======================================"
echo "DeepSeek-V4-Flash Megatron-Bridge parity"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES | TP=$TP PP=$PP EP=$EP"
echo "HF snapshot: $HF_SNAPSHOT"
echo "Output:      $OUTPUT"
echo "======================================"

srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="${USER_LUSTRE}:/chcui" \
  --no-container-mount-home \
  bash -c "
    export HF_TOKEN='${HF_TOKEN:-}'
    export HF_HOME=/chcui/hf_home
    export HF_HUB_OFFLINE=1
    export UV_CACHE_DIR=/tmp/uv_cache
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_TIMEOUT=1800000
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    MASTER_ADDR=\$(python3 -c \"
import re, os
s = os.environ.get('SLURM_NODELIST', '')
m = re.match(r'([\w-]+)\[(\d+)', s)
print(m.group(1) + m.group(2) if m else s.split(',')[0])
\")
    cd $WORKDIR
    # Remove the bridge editable install hook so our PYTHONPATH override resolves to the
    # local checkout (the container ships /opt/Megatron-Bridge as a stale snapshot).
    rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron_bridge*.pth \
          /opt/venv/lib/python3.12/site-packages/__editable__*megatron_bridge*.py
    # DSv4 MCore prerequisites are not yet on a tagged release; point at the fork that
    # carries PR #3430, #4458, #4481, #4518 (see examples/models/deepseek_v4/README.md).
    export PYTHONPATH=$WORKDIR/src:$MCORE_DIR:\${PYTHONPATH:-}
    uv run --no-sync python -m torch.distributed.run \
      --nproc_per_node=4 \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=\$SLURM_PROCID \
      --master_addr=\$MASTER_ADDR \
      --master_port=29500 \
      examples/conversion/deepseek_v4_parity/run_mbridge.py \
      --hf_model_path '$HF_SNAPSHOT' \
      --prompts_file '$PROMPTS_FILE' \
      --output '$OUTPUT' \
      --tp $TP --pp $PP --ep $EP --etp $ETP \
      --pad_to_multiple_of 8 \
      --sequence_parallel false
  "

echo "Exit $?"
