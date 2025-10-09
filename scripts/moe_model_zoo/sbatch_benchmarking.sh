#!/bin/bash
set -euxo pipefail

# Path to Megatron-MoE-Scripts
export WORKSPACE=$(dirname "$(readlink -f "$0")")

# Benchmarking configurations (must be set)
export MODEL=${MODEL:-"your_own_model"}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"your_own_container_image"}
export MBRIDGE_PATH=${MBRIDGE_PATH:-"your_own_megatron_bridge_path"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_own_wandb_api_key"}
export MBRIDGE_RELEASE_VERSION=${MBRIDGE_RELEASE_VERSION:-"your_megatron_bridge_version"}

# Load common configurations
source "${WORKSPACE}/runtime_configs/benchmarking/common.conf"
# Load model-specific configurations
source "${WORKSPACE}/runtime_configs/benchmarking/runtime.conf"

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
        echo "${KEY}=${VALUE}"
    fi
done < <(echo "${ENV_VARS}" | tr ' ' '\n')


# FP8 arguments
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-recipe blockwise --fp8-format e4m3"
    if [[ ${OPTIMIZER_OFFLOAD} == 0 ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-param-gather" # Optimizer CPU offload does not support fp8 param gather now.
    fi
    TRAINING_PARAMS="${TRAINING_PARAMS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-router-padding-for-fp8"
fi

if [[ ${PR} == "mxfp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-recipe mxfp8 --fp8-format e4m3"
    if [[ ${OPTIMIZER_OFFLOAD} == 0 ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-param-gather --reuse-grad-buf-for-mxfp8-param-ag" # Optimizer CPU offload does not support fp8 param gather now.
    fi
    TRAINING_PARAMS="${TRAINING_PARAMS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-router-padding-for-fp8"
fi

# 1F1B overlapping arguments and environment variables
A2A_OVERLAP=${A2A_OVERLAP:-0}
if [[ ${A2A_OVERLAP} == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NVTE_FWD_LAYERNORM_SM_MARGIN=20
    export NVTE_BWD_LAYERNORM_SM_MARGIN=20
    TRAINING_PARAMS="${TRAINING_PARAMS} config_container.comm_overlap.delay_wgrad_compute=true config_container.comm_overlap.overlap_moe_expert_parallel_comm=true"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
    TRAINING_PARAMS="${TRAINING_PARAMS} config_container.comm_overlap.overlap_grad_reduce=true config_container.comm_overlap.overlap_param_gather=true"
fi

# Long context arguments
if [[ ${SEQ_LEN} -gt 4096 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} config_container.model.max_position_embeddings=${SEQ_LEN}"
fi

# Profile command
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        -f true -x true \
        -o ${NSYS_PATH}/${MODEL}-benchmarking-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 20 --profile-step-end 22 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

# Export training command
export TRAINING_CMD="${PROFILE_CMD} python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Generate timestamp for job name
TIMESTAMP=$(date +'%y%m%d_%H%M%S')

N_TASKS_PER_NODE=${N_TASKS_PER_NODE:-8}

# Submit SLURM job
set +e
if [[ ${DRY_RUN:-0} -eq 1 ]]; then
    echo "=== DRY RUN - SLURM Job Script ==="
    cat <<EOF
#!/bin/bash

#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks-per-node=${N_TASKS_PER_NODE}
#SBATCH --time=${RUN_TIME}
#SBATCH --job-name=${ACCOUNT}-moe-${RUN_NAME}-${TIMESTAMP}
#SBATCH --dependency=singleton
#SBATCH --output=${WORKSPACE}/slurm.log
#SBATCH --exclusive

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_\${SLURM_NODEID}"}

srun \
    --mpi=pmix -l \
    --no-container-mount-home \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --container-workdir=${MBRIDGE_PATH} \
    bash -c \\\${TRAINING_CMD} 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log
EOF
    echo "=== End of DRY RUN ==="
    echo "=== Full Training Command ==="
    echo "${TRAINING_CMD}"
    echo "=== End of Full Training Command ==="
else
    sbatch ${SBATCH_ARG} <<EOF
#!/bin/bash

#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks-per-node=${N_TASKS_PER_NODE}
#SBATCH --time=${RUN_TIME}
#SBATCH --job-name=${ACCOUNT}-moe-${RUN_NAME}-${TIMESTAMP}
#SBATCH --output=${WORKSPACE}/slurm.log
#SBATCH --exclusive

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_\${SLURM_NODEID}"}

srun \
    --mpi=pmix -l \
    --no-container-mount-home \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --container-workdir=${MBRIDGE_PATH} \
    bash -c \\\${TRAINING_CMD} 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log
EOF
fi
set -e
