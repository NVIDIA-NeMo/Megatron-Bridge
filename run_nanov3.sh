#!/bin/bash
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh

CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-25.11-nano-v3-nsys.sqsh"
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

    # mamba deterministic mode environment variables (all required)
    export MAMBA_DETERMINISTIC=1
    export CAUSAL_CONV1D_DETERMINISTIC=1
    export TRITON_ENABLE_AUTOTUNE=0
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
    -ng 16 \
    -gn 8 \
    --container_image $CONTAINER \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge,$WORKDIR/3rdparty/Megatron-LM:/opt/megatron-lm" \
    -hf $HF_TOKEN \
    -wdk $WANDB_API_KEY \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "nemotron3-nano-30b-a3b-nemo-25.11-${DETERMINISTIC_FLAG}" \
    --task pretrain \
    train.global_batch_size=128 \
    model.tensor_model_parallel_size=2 \
    model.sequence_parallel=true \
    model.expert_model_parallel_size=8 \
    model.pipeline_model_parallel_size=1 \
    model.context_parallel_size=1 \
    model.moe_token_dispatcher_type=alltoall \
    logger.tensorboard_dir=/nemo_run/tensorboard \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_memory_to_tensorboard=true \
    logger.throughput_window_size=1 \
    logger.tensorboard_log_interval=1 \
    $additional_args

