#!/bin/bash
set -euxo pipefail

# Path to Megatron-MoE-Scripts
export WORKSPACE=$(dirname "$(readlink -f "$0")")

# Benchmarking configurations (must be set)
export MODEL=${MODEL:-"your_own_model"}
export MBRIDGE_PATH=${MBRIDGE_PATH:-"your_own_megatron_bridge_path"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_own_wandb_api_key"}
export MBRIDGE_RELEASE_VERSION=${MBRIDGE_RELEASE_VERSION:-"your_megatron_bridge_version"}

# Load common configurations
source "${WORKSPACE}/runtime_configs/common.conf"
# Load model-specific configurations
source "${WORKSPACE}/runtime_configs/runtime.conf"

# Initialize training parameters
TRAINING_PARAMS=${TRAINING_PARAMS:-""}

# Process training parameters
if [[ -f ${TRAINING_PARAMS_PATH} ]]; then
    envsubst < ${TRAINING_PARAMS_PATH} > ${TRAINING_PARAMS_PATH}.tmp
    TRAINING_PARAMS_PATH=${TRAINING_PARAMS_PATH}.tmp
else
    echo "Error: TRAINING_PARAMS_PATH does not exist: ${TRAINING_PARAMS_PATH}."
    exit 1
fi

# Append any command line arguments to TRAINING_PARAMS
if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} $@"
fi

# Extract environment variables to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' ${TRAINING_PARAMS_PATH})
while IFS='=' read -r KEY VALUE; do
    if [[ -n ${KEY} ]]; then
        export "${KEY}"="${VALUE}"
    fi
done < <(echo "${ENV_VARS}" | tr ' ' '\n')

OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-0}
# FP8 arguments
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} mixed_precision.fp8_recipe=blockwise mixed_precision.fp8=e4m3"
    if [[ ${OPTIMIZER_OFFLOAD} == 0 ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} mixed_precision.fp8_param_gather=true" # Optimizer CPU offload does not support fp8 param gather now.
    fi
    TRAINING_PARAMS="${TRAINING_PARAMS} optimizer.use_precision_aware_optimizer=true optimizer.main_grads_dtype=torch.float32 optimizer.main_params_dtype=torch.float32 optimizer.exp_avg_dtype=torch.bfloat16 optimizer.exp_avg_sq_dtype=torch.bfloat16"
    TRAINING_PARAMS="${TRAINING_PARAMS} model.moe_router_padding_for_fp8=true"
elif [[ ${PR} == "mxfp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} mixed_precision.fp8_recipe=mxfp8 mixed_precision.fp8=e4m3"
    if [[ ${OPTIMIZER_OFFLOAD} == 0 ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} mixed_precision.fp8_param_gather=true mixed_precision.reuse_grad_buf_for_mxfp8_param_ag=true" # Optimizer CPU offload does not support fp8 param gather now.
    fi
    TRAINING_PARAMS="${TRAINING_PARAMS} optimizer.use_precision_aware_optimizer=true optimizer.main_grads_dtype=torch.float32 optimizer.main_params_dtype=torch.float32 optimizer.exp_avg_dtype=torch.bfloat16 optimizer.exp_avg_sq_dtype=torch.bfloat16"
    TRAINING_PARAMS="${TRAINING_PARAMS} model.moe_router_padding_for_fp8=true"
fi


# 1F1B overlapping arguments and environment variables
A2A_OVERLAP=${A2A_OVERLAP:-0}
if [[ ${A2A_OVERLAP} == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NVTE_FWD_LAYERNORM_SM_MARGIN=20
    export NVTE_BWD_LAYERNORM_SM_MARGIN=20
    TRAINING_PARAMS="${TRAINING_PARAMS} comm_overlap.delay_wgrad_compute=true comm_overlap.overlap_moe_expert_parallel_comm=true"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
    TRAINING_PARAMS="${TRAINING_PARAMS} comm_overlap.overlap_grad_reduce=true comm_overlap.overlap_param_gather=true"
fi

# Long context arguments
if [[ ${SEQ_LEN} -gt 4096 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} model.max_position_embeddings=${SEQ_LEN}"
fi

# Profile command
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-memory-usage true \
        -f true -x true \
        -o ${NSYS_PATH}/${MODEL}-benchmarking-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 50 --profile-step-end 55 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

# Start training
cd ${MBRIDGE_PATH} || {
    echo "Error: Failed to change directory to megatron-bridge path ${MBRIDGE_PATH}"
    exit 1
}

# Dry run check if set
if [[ ${DRY_RUN:-0} -eq 1 ]]; then
    echo "=== DRY RUN - Training Command ==="
    echo "${PROFILE_CMD} python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"
    echo "=== End of DRY RUN ==="
else
    ${PROFILE_CMD} python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}
fi
