#!/bin/bash
# Usage:
#   Normal run: ./run_llama3_405b_fp8.sh
#   Deterministic mode: DETERMINISTIC=true ./run_llama3_405b_fp8.sh
#   Deterministic with Flash Attention: DETERMINISTIC=true BACKEND=flash ./run_llama3_405b_fp8.sh
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh

# CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11-cuda13.1-cudnn9.18.0.76.sqsh"
export CUDNN_HOME=/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/Megatron-Bridge/cudnn_lib/9.18.0.76/cudnn/
export LD_LIBRARY_PATH='$CUDNN_HOME/lib64:$LD_LIBRARY_PATH'

ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
# Get current directory to mount
WORKDIR=$(pwd)


export DETERMINISTIC=${DETERMINISTIC:-false}
export BACKEND=${BACKEND:-fused}  # Allow Flash Attention in deterministic mode
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


# FP8 memory optimization
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Force Python to not use bytecode cache (.pyc files) to ensure code changes are picked up
export PYTHONDONTWRITEBYTECODE=1

# Fix TorchInductor/Triton cache race condition
# Use persistent cache on Lustre instead of /tmp to avoid multi-rank compilation conflicts
# export TORCHINDUCTOR_CACHE_DIR=/lustre/fs1/portfolios/coreai/users/zhiyul/.cache/torchinductor
# export TRITON_CACHE_DIR=/lustre/fs1/portfolios/coreai/users/zhiyul/.cache/triton

# Determinism is broken with recompute
# export RECOMPUTE_ARGS="+model.recompute_granularity=full +model.recompute_method=block +model.recompute_num_layers=1"
RECOMPUTE_ARGS=""

python scripts/performance/setup_experiment.py \
    --account $ACCOUNT \
    --partition $PARTITION \
    --gpu h100 \
    -m llama31 \
    -s 405b \
    -ng 512 \
    -gn 8 \
    -c fp8_cs \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge,$WORKDIR/3rdparty/Megatron-LM:/opt/megatron-lm" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "llama31-405b-fp8-nemo-25.11-${EXP_NAME}" \
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


