#!/bin/bash
# Usage:
#   Normal run: ./run_llama3_405b_fp8.sh
#   Deterministic mode: DETERMINISTIC=true ./run_llama3_405b_fp8.sh
#   Deterministic with Fused Attention: DETERMINISTIC=true DETERMINISTIC_FUSED=true bash run_llama3_405b_fp8_fused.sh
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh

CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
# Get current directory to mount
WORKDIR=$(pwd)

export NVTE_DEBUG=1   # disables/enables debugging
export NVTE_DEBUG_LEVEL=2

export DETERMINISTIC=${DETERMINISTIC:-false}
export DETERMINISTIC_FUSED=${DETERMINISTIC_FUSED:-false}  # Allow fused attention in deterministic mode
if [ "$DETERMINISTIC" = true ]; then
    # Deterministic mode environment variables (all required)
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    if [ "$DETERMINISTIC_FUSED" = true ]; then
        export additional_args="model.deterministic_mode=true model.cross_entropy_loss_fusion=false model.attention_backend=fused comm_overlap.tp_comm_overlap=false"
        export DETERMINISTIC_FLAG="deterministic-fused"
    else
        export additional_args="model.deterministic_mode=true model.cross_entropy_loss_fusion=false model.attention_backend=local comm_overlap.tp_comm_overlap=false"
        export DETERMINISTIC_FLAG="deterministic-local"
    fi
else
    export additional_args=""
    export DETERMINISTIC_FLAG="non-deterministic"
fi

# FP8 memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Determinism is broken with recompute
# export RECOMPUTE_ARGS="+model.recompute_granularity=full +model.recompute_method=block +model.recompute_num_layers=1"
RECOMPUTE_ARGS=""

python scripts/performance/setup_experiment.py \
    --account $ACCOUNT \
    --partition $PARTITION \
    --gpu h100 \
    --time_limit "04:00:00" \
    -m llama31 \
    -s 405b \
    -ng 512 \
    -gn 8 \
    -c fp8_cs \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "llama31-405b-fp8-nemo-25.11-${DETERMINISTIC_FLAG}" \
    --task pretrain \
    train.train_iters=200 \
    logger.tensorboard_dir=/nemo_run/tensorboard \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_memory_to_tensorboard=true \
    logger.throughput_window_size=1 \
    logger.tensorboard_log_interval=1 \
    $RECOMPUTE_ARGS \
    $additional_args


