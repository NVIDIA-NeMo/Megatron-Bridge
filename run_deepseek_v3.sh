#!/bin/bash
# Usage:
#   Normal run: ./run_deepseek_v3.sh
#   Deterministic mode: DETERMINISTIC=true ./run_deepseek_v3.sh
#   Deterministic with Flash Attention: DETERMINISTIC=true BACKEND=flash ./run_deepseek_v3.sh
#   Run on GB200: GPU=gb200 ./run_deepseek_v3.sh
set -euo pipefail
source ../../secrets.sh

GPU=${GPU:-"h100"}

# Use cuDNN 9.18.0.45 for deterministic MLA support and GQA fixes
export CUDNN_HOME="/lustre/fsw/coreai_dlalgo_llm/zhiyul/deterministics/Megatron-Bridge/cudnn_lib/9.18.0.45/cudnn"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDNN_HOME/include:${CPATH:-}"

if [ "$GPU" = "h100" ]; then
    CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
    ACCOUNT="coreai_dlalgo_nemorl"
    PARTITION="batch_short"
    NUM_GPUS=1024
    GPUS_PER_NODE=8
elif [ "$GPU" = "gb200" ]; then
    CONTAINER="/lustre/fsw/coreai_dlalgo_llm/zhiyul/containers/nemo-25.11.sqsh"
    ACCOUNT="coreai_dlalgo_llm"
    PARTITION="batch"
    NUM_GPUS=256
    GPUS_PER_NODE=4
else
    echo "Invalid GPU: $GPU"
    exit 1
fi
# Get current directory to mount
WORKDIR=$(pwd)

# Base commit for Megatron-LM changes
BASE_COMMIT="0d8e0714cd29c01e164fe6de9f532182bdffa942"
MEGATRON_DIR="3rdparty/Megatron-LM"

# Dynamically construct mounts for changed files in Megatron-LM
CUSTOM_MOUNTS=""
if [ -d "$MEGATRON_DIR" ]; then
    CHANGED_FILES=$(git -C "$MEGATRON_DIR" diff --name-only --diff-filter=AM "$BASE_COMMIT" HEAD)
    for f in $CHANGED_FILES; do
        CUSTOM_MOUNTS="${CUSTOM_MOUNTS},$WORKDIR/$MEGATRON_DIR/$f:/opt/megatron-lm/$f"
    done
fi

export DETERMINISTIC=${DETERMINISTIC:-false}
export BACKEND=${BACKEND:-fused}  # Allow Flash Attention in deterministic mode
export RECOMPUTE_ARGS=""

export NVTE_DEBUG=1   # disables/enables debugging
export NVTE_DEBUG_LEVEL=2

if [ "$BACKEND" = "flash" ]; then
    export NVTE_FUSED_ATTN=0
    export NVTE_UNFUSED_ATTN=0
    export NVTE_FLASH_ATTN=1
    export additional_args="model.attention_backend=flash"
elif [ "$BACKEND" = "fused" ]; then
    export NVTE_FUSED_ATTN=1
    export NVTE_UNFUSED_ATTN=0
    export NVTE_FLASH_ATTN=0
    export additional_args="model.attention_backend=fused"
elif [ "$BACKEND" = "local" ]; then
    export NVTE_FUSED_ATTN=0
    export NVTE_UNFUSED_ATTN=0
    export NVTE_FLASH_ATTN=0
    export additional_args="model.attention_backend=local"
elif [ "$BACKEND" = "unfused" ]; then
    export NVTE_FUSED_ATTN=0
    export NVTE_UNFUSED_ATTN=1
    export NVTE_FLASH_ATTN=0
    export additional_args="model.attention_backend=unfused"
else
    echo "Invalid backend: $BACKEND"
    exit 1
fi

# AssertionError: Modules must not have hooks registered at the time they are passed. However, registering hooks on modules after passing them through make_graphed_callables is allowed.
# export additional_args="${additional_args} model.cuda_graph_impl=none"
# These env vars might help if the hardware detection isn't working
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1

if [ "$DETERMINISTIC" = true ]; then
    # Deterministic mode environment variables (all required)
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    # Disable CUDA graphs in deterministic mode - hooks conflict with make_graphed_callables
    export additional_args="${additional_args} model.deterministic_mode=true model.cross_entropy_loss_fusion=false comm_overlap.tp_comm_overlap=false"
    export EXP_NAME="deterministic-${BACKEND}-${GPU}"
else
    export EXP_NAME="non-deterministic-${BACKEND}-${GPU}"
fi

python scripts/performance/setup_experiment.py \
    --account $ACCOUNT \
    --partition $PARTITION \
    --gpu $GPU \
    --time_limit "01:00:00" \
    -m deepseek \
    -s v3 \
    -ng $NUM_GPUS \
    -gn $GPUS_PER_NODE \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge,$CUDNN_HOME:/opt/cudnn$CUSTOM_MOUNTS" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "deepseek-v3-nemo-25.11-${EXP_NAME}" \
    --task pretrain \
    logger.tensorboard_dir=/nemo_run/tensorboard \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_memory_to_tensorboard=true \
    logger.throughput_window_size=1 \
    logger.tensorboard_log_interval=1 \
    $additional_args
