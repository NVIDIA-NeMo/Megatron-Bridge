#!/bin/bash
# Usage:
#   Normal run: ./run_llama3_70b.sh
#   Deterministic mode: DETERMINISTIC=true ./run_llama3_70b.sh
#   Deterministic with Flash Attention: DETERMINISTIC=true DETERMINISTIC_FLASH=true ./run_llama3_70b.sh
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh

CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
# Get current directory to mount
WORKDIR=$(pwd)

export DETERMINISTIC=${DETERMINISTIC:-false}
export DETERMINISTIC_FLASH=${DETERMINISTIC_FLASH:-false}  # Allow Flash Attention in deterministic mode
if [ "$DETERMINISTIC" = true ]; then
    # Deterministic mode environment variables (all required)
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    if [ "$DETERMINISTIC_FLASH" = true ]; then
        export additional_args="model.deterministic_mode=true model.cross_entropy_loss_fusion=false model.attention_backend=flash comm_overlap.tp_comm_overlap=false"
        export DETERMINISTIC_FLAG="deterministic-flash"
    else
        export additional_args="model.deterministic_mode=true model.cross_entropy_loss_fusion=false model.attention_backend=local comm_overlap.tp_comm_overlap=false"
        export DETERMINISTIC_FLAG="deterministic"
    fi
else
    export additional_args=""
    export DETERMINISTIC_FLAG="non-deterministic"
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
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "llama3-70b-nemo-25.11-${DETERMINISTIC_FLAG}" \
    --task pretrain \
    logger.tensorboard_dir=/nemo_run/tensorboard \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_memory_to_tensorboard=true \
    logger.throughput_window_size=1 \
    logger.tensorboard_log_interval=1 \
    $additional_args
