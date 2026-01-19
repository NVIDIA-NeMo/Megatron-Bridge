#!/bin/bash
# Usage:
#   Normal run: ./run_llama3_405b.sh
#   Deterministic mode: DETERMINISTIC=true ./run_llama3_405b.sh
#   Deterministic with Flash Attention: DETERMINISTIC=true BACKEND=flash ./run_llama3_405b.sh
set -euo pipefail
source ../../secrets.sh

GPU=${GPU:-"h100"}
if [ "$GPU" = "h100" ]; then
    CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
    ACCOUNT="coreai_dlalgo_nemorl"
    PARTITION="batch"
    NUM_GPUS=512
    GPUS_PER_NODE=8
elif [ "$GPU" = "gb200" ] || [ "$GPU" = "b200" ]; then
    CONTAINER="/lustre/fsw/coreai_dlalgo_llm/zhiyul/containers/nemo-25.11.sqsh"
    ACCOUNT="coreai_dlalgo_llm"
    PARTITION="batch"
    NUM_GPUS=128
    GPUS_PER_NODE=4
    if [ "$GPU" = "gb200" ]; then
        # These env vars help if the hardware detection isn't working on GB200
        export NVLINK_DOMAIN_SIZE=72
        export USE_MNNVL=1
    fi
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
    export RECOMPUTE_ARGS="+model.recompute_granularity=full +model.recompute_method=block +model.recompute_num_layers=1"
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
else
    echo "Invalid backend: $BACKEND"
    exit 1
fi

if { [ "$GPU" = "gb200" ] || [ "$GPU" = "b200" ]; } && [ "$BACKEND" = "fused" ]; then
    # use cudnn 9.18.0.76 for deterministic fused attention support
    CONTAINER="/lustre/fsw/coreai_dlalgo_llm/zhiyul/containers/nemo-25.11-cudnn9.18.0.76.sqsh"
    export CUDNN_HOME=/lustre/fsw/coreai_dlalgo_llm/zhiyul/deterministics/Megatron-Bridge/cudnn_lib/9.18.0.76/cudnn/
    export LD_LIBRARY_PATH='$CUDNN_HOME/lib64:$LD_LIBRARY_PATH'
fi


if [ "$DETERMINISTIC" = true ]; then
    # Deterministic mode environment variables (all required)
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export additional_args="${additional_args} model.deterministic_mode=true model.cross_entropy_loss_fusion=false comm_overlap.tp_comm_overlap=false"
    export EXP_NAME="deterministic-${BACKEND}-${GPU}"
else
    export EXP_NAME="non-deterministic-${BACKEND}-${GPU}"
fi

python scripts/performance/setup_experiment.py \
    --account $ACCOUNT \
    --partition $PARTITION \
    --gpu $GPU \
    -m llama31 \
    -s 405b \
    -ng $NUM_GPUS \
    -gn $GPUS_PER_NODE \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "llama31-405b-nemo-25.11-${EXP_NAME}" \
    --task pretrain \
    logger.tensorboard_dir=/nemo_run/tensorboard \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_memory_to_tensorboard=true \
    logger.throughput_window_size=1 \
    logger.tensorboard_log_interval=1 \
    $RECOMPUTE_ARGS \
    $additional_args
