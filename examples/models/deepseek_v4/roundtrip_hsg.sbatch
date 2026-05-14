#!/bin/bash
# Export an existing DeepSeek-V4-Flash-Base Megatron checkpoint back to HF on HSG.
# This validates the DSv4 bridge with native transformers DeepseekV4 support.

#SBATCH --job-name=coreai_dlalgo_llm-mbridge:dsv4-roundtrip
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/chcui/logs/dsv4_roundtrip_%j.log

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/weijiac/sqsh/nemo_26.04.sqsh}
WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-Bridge-dsv4-train}
MCORE_DIR=${MCORE_DIR:-$WORKDIR/3rdparty/Megatron-LM}
FHT_SITE=${FHT_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/fast_hadamard_site}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/roundtrip/dsv4_transformers_5_8_1_nodeps}
HF_SNAPSHOT=${HF_SNAPSHOT:-/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash-Base}
MEGATRON_DIR=${MEGATRON_DIR:-/lustre/fsw/portfolios/coreai/users/chcui/mbridge_home/models/deepseek-ai/DeepSeek-V4-Flash-Base}
OUT_ROOT=${OUT_ROOT:-/lustre/fsw/portfolios/coreai/users/chcui/roundtrip/dsv4_native_tf581}

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-4}
ETP=${ETP:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29578}

mkdir -p /lustre/fsw/portfolios/coreai/users/chcui/logs "$OUT_ROOT"

HF_EXPORT_DIR=${HF_EXPORT_DIR:-$OUT_ROOT/DeepSeek-V4-Flash-Base-hf-${SLURM_JOB_ID}}

MASTER_ADDR=$(python3 - <<'PY'
import os
import re

s = os.environ.get("SLURM_NODELIST", "")
m = re.match(r"([\w-]+)\[(\d+)", s)
print(m.group(1) + m.group(2) if m else s.split(",")[0])
PY
)

echo "DeepSeek-V4 roundtrip export | job=$SLURM_JOB_ID"
echo "workdir=$WORKDIR"
echo "mcore=$MCORE_DIR"
echo "hf_snapshot=$HF_SNAPSHOT"
echo "megatron_dir=$MEGATRON_DIR"
echo "hf_export_dir=$HF_EXPORT_DIR"
echo "transformers_site=$TRANSFORMERS_SITE"
echo "parallelism TP=$TP PP=$PP EP=$EP ETP=$ETP"

srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  bash -lc "
    set -euo pipefail
    export HF_HOME=/lustre/fsw/portfolios/coreai/users/chcui/hf_home
    export HF_HUB_OFFLINE=1
    export UV_CACHE_DIR=/tmp/uv_cache
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_TIMEOUT=1800000
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    export PYTHONPATH=$TRANSFORMERS_SITE:$FHT_SITE:$WORKDIR/src:$MCORE_DIR:\${PYTHONPATH:-}
    source $WORKDIR/examples/models/deepseek_v4/hsg_runtime.sh
    ensure_transformers_site $TRANSFORMERS_SITE

    cd $WORKDIR
    rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.pth \
          /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.py

    /opt/venv/bin/python - <<'PY'
from transformers import AutoConfig, DeepseekV4Config, DeepseekV4ForCausalLM

from megatron.bridge import AutoBridge

cfg = AutoConfig.from_pretrained('$HF_SNAPSHOT')
assert isinstance(cfg, DeepseekV4Config), type(cfg)
assert DeepseekV4ForCausalLM.__name__ == 'DeepseekV4ForCausalLM'
provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
print('transformers_native_dsv4=OK', cfg.model_type, provider.experimental_attention_variant)
PY

    /opt/venv/bin/python -m torch.distributed.run \
      --nproc_per_node=$NPROC_PER_NODE \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      examples/conversion/convert_checkpoints_multi_gpu.py export \
      --hf-model $HF_SNAPSHOT \
      --megatron-path $MEGATRON_DIR \
      --hf-path $HF_EXPORT_DIR \
      --tp $TP --pp $PP --ep $EP --etp $ETP \
      --torch-dtype bfloat16 \
      --trust-remote-code \
      --not-strict \
      --distributed-save \
      --use-cpu-init
  "

echo "Exit $?"
