#!/bin/bash
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh

CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch_short"
# Get current directory to mount
WORKDIR=$(pwd)

export DETERMINISTIC=${DETERMINISTIC:-false}
if [ "$DETERMINISTIC" = true ]; then
    # Deterministic mode environment variables (all required)
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export additional_args="model.deterministic_mode=true model.cross_entropy_loss_fusion=false model.attention_backend=local"
    export DETERMINISTIC_FLAG="deterministic"
else
    export additional_args=""
    export DETERMINISTIC_FLAG="non-deterministic"
fi

python scripts/performance/setup_experiment.py \
    --account $ACCOUNT \
    --partition $PARTITION \
    --gpu h100 \
    -m nemotronh \
    -s nano_30b_a3b \
    -ng 32 \
    -gn 8 \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "nemotron3-nano-30b-a3b-nemo-25.11-${DETERMINISTIC_FLAG}" \
    --task pretrain \
    logger.tensorboard_dir=/nemo_run/tensorboard \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_memory_to_tensorboard=true \
    logger.throughput_window_size=1 \
    logger.tensorboard_log_interval=1 \
    $additional_args

