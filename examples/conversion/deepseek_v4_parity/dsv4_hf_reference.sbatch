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
# DeepSeek-V4-Flash HuggingFace reference forward pass on 1 node (4 B200 GPUs).
# Uses native ``DeepseekV4ForCausalLM`` (no trust_remote_code) with
# device_map="auto" to shard across 4 B200 GPUs.

#SBATCH --job-name=coreai_dlalgo_llm-mbridge:dsv4-parity-hf
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/<EDIT-ME>/users/<EDIT-ME>/logs/dsv4_parity_hf_%j.log
#SBATCH --exclusive

# --- Paths to edit before submitting ----------------------------------------
CONTAINER_IMAGE=/lustre/<EDIT-ME>/users/<EDIT-ME>/sqsh/nemo_26.04.sqsh
USER_LUSTRE=/lustre/<EDIT-ME>/users/<EDIT-ME>
HF_SNAPSHOT=${USER_LUSTRE}/models/deepseek-ai/DeepSeek-V4-Flash
# ----------------------------------------------------------------------------

WORKDIR=/chcui/megatron-bridge-3
OUTPUT=/chcui/parity/dsv4/logits_hf.pt
PROMPTS_FILE=examples/conversion/deepseek_v4_parity/prompts.json

mkdir -p ${USER_LUSTRE}/logs ${USER_LUSTRE}/parity/dsv4

echo "======================================"
echo "DeepSeek-V4-Flash HF reference parity"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd $WORKDIR
    # We do not need Megatron-Bridge edits on the HF side, but removing the editable hook
    # keeps the import path consistent with the MB-side job for easier debugging.
    rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron_bridge*.pth \
          /opt/venv/lib/python3.12/site-packages/__editable__*megatron_bridge*.py
    uv run --no-sync python examples/conversion/deepseek_v4_parity/run_hf_reference.py \
      --hf_model_path '$HF_SNAPSHOT' \
      --prompts_file '$PROMPTS_FILE' \
      --output '$OUTPUT' \
      --device_map auto \
      --pad_to_multiple_of 8 \
      --pad_token_id 0
  "

echo "Exit $?"
