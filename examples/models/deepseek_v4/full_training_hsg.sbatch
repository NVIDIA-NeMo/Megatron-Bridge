#!/bin/bash
# One-iteration DeepSeek-V4-Flash full-config training smoke on HSG.
# This uses the real HF config, random initialization, Adam, and mock GPT data.

#SBATCH --job-name=coreai_dlalgo_llm-mbridge:dsv4-full-adam
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/chcui/logs/dsv4_full_adam_%j.log
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/weijiac/sqsh/nemo_26.04.sqsh}
WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-Bridge-dsv4-train}
MCORE_DIR=${MCORE_DIR:-/lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-LM-pr4518}
FHT_SITE=${FHT_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/fast_hadamard_site}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/training/dsv4_transformers_5_8_1_nodeps}
HF_CONFIG=${HF_CONFIG:-/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash}
OUTDIR=${OUTDIR:-/lustre/fsw/portfolios/coreai/users/chcui/training/dsv4_full_adam}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}

SEQ_LENGTH=${SEQ_LENGTH:-128}
TRAIN_ITERS=${TRAIN_ITERS:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
TP=${TP:-1}
PP=${PP:-4}
CP=${CP:-1}
EP=${EP:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29571}

mkdir -p /lustre/fsw/portfolios/coreai/users/chcui/logs "$OUTDIR"

MASTER_ADDR=$(python3 - <<'PY'
import os
import re

s = os.environ.get("SLURM_NODELIST", "")
m = re.match(r"([\w-]+)\[(\d+)", s)
print(m.group(1) + m.group(2) if m else s.split(",")[0])
PY
)

echo "DeepSeek-V4 full Adam training smoke | job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NUM_NODES nproc_per_node=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"
echo "parallelism TP=$TP PP=$PP CP=$CP EP=$EP"
echo "seq_length=$SEQ_LENGTH train_iters=$TRAIN_ITERS gbs=$GLOBAL_BATCH_SIZE mbs=$MICRO_BATCH_SIZE"
echo "workdir=$WORKDIR"
echo "hf_config=$HF_CONFIG"
echo "outdir=$OUTDIR"
echo "transformers_site=$TRANSFORMERS_SITE"
echo "moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER_TYPE"
echo "nccl_pxn_disable=$NCCL_PXN_DISABLE"

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
    export NCCL_PXN_DISABLE=$NCCL_PXN_DISABLE
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_TIMEOUT=1800000
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    cd $WORKDIR
    rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.pth \
          /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.py
    mkdir -p $TRANSFORMERS_SITE
    export PYTHONPATH=$TRANSFORMERS_SITE:$FHT_SITE:$WORKDIR/src:$MCORE_DIR:\${PYTHONPATH:-}
    source $WORKDIR/examples/models/deepseek_v4/hsg_runtime.sh
    ensure_transformers_site $TRANSFORMERS_SITE
    uv run --no-sync python -m torch.distributed.run \
      --nproc_per_node=$NPROC_PER_NODE \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=\$SLURM_PROCID \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      examples/models/deepseek_v4/toy_training.py \
      --model-size full \
      --case adam \
      --hf-config-path $HF_CONFIG \
      --output-dir $OUTDIR \
      --seq-length $SEQ_LENGTH \
      --train-iters $TRAIN_ITERS \
      --micro-batch-size $MICRO_BATCH_SIZE \
      --global-batch-size $GLOBAL_BATCH_SIZE \
      --tensor-parallel-size $TP \
      --pipeline-parallel-size $PP \
      --context-parallel-size $CP \
      --expert-parallel-size $EP \
      --moe-token-dispatcher-type $MOE_TOKEN_DISPATCHER_TYPE \
      --csa-backend unfused
  "

echo "Exit $?"
