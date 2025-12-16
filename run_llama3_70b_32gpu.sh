#!/bin/bash
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh

CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11.sqsh"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch_short"
# Get current directory to mount
WORKDIR=$(pwd)

# Deterministic mode environment variables (all required)
export NCCL_ALGO="Ring"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/performance/setup_experiment.py \
    --account $ACCOUNT \
    --partition $PARTITION \
    --gpu h100 \
    -m llama3 \
    -s 70b \
    -ng 32 \
    -gn 8 \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/workdir" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "llama3-70b-pretrain-32gpu-deterministic-fused-log-fixed" \
    --task pretrain \
    model.tensor_model_parallel_size=4 \
    model.pipeline_model_parallel_size=4 \
    model.context_parallel_size=2 \
    model.virtual_pipeline_model_parallel_size=5 \
    comm_overlap.tp_comm_overlap=false \
    model.deterministic_mode=true \
    model.cross_entropy_loss_fusion=false \
    model.attention_backend=local \
    model.cross_entropy_loss_fusion=false \
    logger.tensorboard_dir=/nemo_run/tensorboard \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_memory_to_tensorboard=true \
    logger.throughput_window_size=1 \
    logger.tensorboard_log_interval=1

