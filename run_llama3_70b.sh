#!/bin/bash
# Usage:
#   Normal run: ./run_llama3_70b.sh
#   Deterministic mode: DETERMINISTIC=true ./run_llama3_70b.sh
#   Deterministic with Flash Attention: DETERMINISTIC=true BACKEND=flash ./run_llama3_70b.sh
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh

CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
# Get current directory to mount
WORKDIR=$(pwd)


export DETERMINISTIC=${DETERMINISTIC:-false}
export BACKEND=${BACKEND:-fused}  # Allow Flash Attention in deterministic mode
export RECOMPUTE_ARGS=""

export NVTE_DEBUG=1   # disables/enables debugging
export NVTE_DEBUG_LEVEL=2

if [ "$BACKEND" = "flash" ]; then
    # export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export additional_args="model.attention_backend=flash"
    # Memory optimization flags
    export NVTE_FUSED_ATTN=0
    export NVTE_UNFUSED_ATTN=0
    export NVTE_FLASH_ATTN=1
    # export RECOMPUTE_ARGS="+model.recompute_granularity=full +model.recompute_method=block +model.recompute_num_layers=1"
    # export BACKEND="flash-recompute"
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


if [ "$DETERMINISTIC" = true ]; then
    # Deterministic mode environment variables (all required)
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export additional_args="${additional_args} model.deterministic_mode=true model.cross_entropy_loss_fusion=false comm_overlap.tp_comm_overlap=false"
    export EXP_NAME="deterministic-${BACKEND}"
else
    export EXP_NAME="non-deterministic-${BACKEND}"
fi

python scripts/performance/setup_experiment.py \
    --account $ACCOUNT \
    --partition $PARTITION \
    --gpu h100 \
    -m llama3 \
    -s 70b \
    -ng 64 \
    -gn 8 \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge,$WORKDIR/3rdparty/Megatron-LM:/opt/megatron-lm" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "llama3-70b-nemo-25.11-${EXP_NAME}" \
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
