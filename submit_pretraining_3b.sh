#!/bin/bash
set -x

# Parse command-line arguments
DIRECT=false
N_JOBS=1
INITIAL_JOB_ID=""
TRAIN_ITERS=""
CUSTOM_JOB_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --direct) DIRECT=true; shift ;;
        --train-iters) TRAIN_ITERS=$2; shift 2 ;;
        --job-name) CUSTOM_JOB_NAME=$2; shift 2 ;;
        *) 
            if [[ -z "$N_JOBS_SET" ]]; then
                N_JOBS=$1; N_JOBS_SET=1
            elif [[ -z "$INITIAL_JOB_ID_SET" ]]; then
                INITIAL_JOB_ID=$1; INITIAL_JOB_ID_SET=1
            fi
            shift ;;
    esac
done

# Validate arguments
if ! [[ "$N_JOBS" =~ ^[0-9]+$ ]] || [ "$N_JOBS" -lt 1 ]; then
    echo "Error: n must be a positive integer"
    echo "Usage: bash submit_pretraining_3b.sh [--direct] [--train-iters N] [--job-name NAME] [n] [initial_job_id]"
    exit 1
fi

if [ -n "$INITIAL_JOB_ID" ] && ! [[ "$INITIAL_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: initial_job_id must be a numeric job ID"
    echo "Usage: bash submit_pretraining_3b.sh [--direct] [--train-iters N] [--job-name NAME] [n] [initial_job_id]"
    exit 1
fi

# ============================================
# Job Configuration (cluster-specific)
# ============================================
JOB_FAMILY="ministral_3b"
JOB_NAME="v3"
JOB_NAME="${JOB_FAMILY}_${JOB_NAME}"
if [ -n "$CUSTOM_JOB_NAME" ]; then
    JOB_NAME="$CUSTOM_JOB_NAME"
fi
DURATION=4  # hours

# Cluster-specific configurations
SLURM_NUM_NODES=16
GPUS_PER_NODE=8
PARTITION="batch"
ACCOUNT="coreai_dlalgo_llm"
DATA_ARGS_PATH=examples/diffusion/recipes/nemotron_diffusion/conf/climb_nm5.5_phase3_mistral_nemo.sh
if [ "$DIRECT" = "true" ]; then
    DATA_ARGS_PATH=examples/diffusion/recipes/nemotron_diffusion/conf/climb_nm5.5_phase3_mistral_nemo_debug.sh
fi
hf_path="/lustre/fsw/portfolios/nvr/users/snorouzi/models/Ministral-3-3B-Base-2512_converted"
pretrained_checkpoint="/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/hf_to_mb_3b"
ar_teacher_model_path="/lustre/fsw/portfolios/nvr/users/snorouzi/models/Ministral-3-8B-Base-2512_1t_ft"
dlm_teacher_model_path="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/abhgarg/megatron_exp/ministral_8b_sbd64_llada_combine_before_weighting_1e_5_rerun_32/iter_0012500_hf/Ministral-3-8B-Base-2512_1t_ft"

DISTILL=${DISTILL:-false}
DISTILL_DLM=${DISTILL_DLM:-false}
TEACHER_ARG=""

TENSOR_MODEL_PARALLEL_SIZE=1
if [ "$DISTILL_DLM" = "true" ]; then
    JOB_NAME="${JOB_FAMILY}_dlm_to_dlm_distill"
    TEACHER_ARG="--teacher-model-path=${dlm_teacher_model_path}"
    TENSOR_MODEL_PARALLEL_SIZE=2
elif [ "$DISTILL" = "true" ]; then
    JOB_NAME="${JOB_FAMILY}_distill_8b_teacher"
    TEACHER_ARG="--teacher-model-path=${ar_teacher_model_path}"
fi

USER_PATH=/lustre/fsw/portfolios/coreai/users
USER=snorouzi
export SQSH_CACHE_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi
MB_DIR=${MB_DIR:-~/code/Megatron-Bridge}

# Container image
CONTAINER_IMAGE="nvcr.io/nvidian/nemo:26.04.rc4"
# Container mounts
MOUNTS="${MB_DIR}:/opt/Megatron-Bridge,${MB_DIR}/3rdparty/Megatron-LM:/opt/megatron-lm"

# ============================================
# Submit jobs in a chain
# ============================================
PREVIOUS_JOB_ID=${INITIAL_JOB_ID}

for ((i=1; i<=N_JOBS; i++)); do
    if [ "$N_JOBS" -gt 1 ]; then
        UNIQUE_JOB_NAME="${JOB_NAME}_${i}"
    else
        UNIQUE_JOB_NAME="${JOB_NAME}"
    fi

    # finetune=true only for the first job (load pretrained weights, skip optimizer state)
    if [ "$i" -eq 1 ]; then
        FINETUNE=true
    else
        FINETUNE=false
    fi

    TRAIN_CMD="torchrun \
  --nproc_per_node \$SUBMIT_GPUS \
  --master_addr \$MASTER_ADDR \
  --master_port \$MASTER_PORT \
  --nnodes \$NUM_NODES \
  --node_rank \$NODE_RANK \
  /opt/Megatron-Bridge/examples/diffusion/recipes/nemotron_diffusion/ar_to_dlm.py \
  --hf-path=${hf_path} \
  --data-args-path /opt/Megatron-Bridge/${DATA_ARGS_PATH} \
  ${TEACHER_ARG} \
  checkpoint.finetune=${FINETUNE} \
  checkpoint.pretrained_checkpoint=${pretrained_checkpoint} \
  model.calculate_per_token_loss=true \
  model.different_seed_per_dp=true \
  model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
  logger.wandb_project=megatron \
  logger.wandb_exp_name=${UNIQUE_JOB_NAME} \
  logger.wandb_save_dir=${USER_PATH}/${USER}/megatron_exp/${UNIQUE_JOB_NAME}/wandb \
  dataset.path_to_cache=${USER_PATH}/${USER}/megatron_exp/data_cache \
  checkpoint.save=${USER_PATH}/${USER}/megatron_exp/${JOB_NAME} \
  checkpoint.load=${USER_PATH}/${USER}/megatron_exp/${JOB_NAME} \
  --model-size 3b"

    if [ -n "$TRAIN_ITERS" ]; then
        TRAIN_CMD="${TRAIN_CMD} train.train_iters=${TRAIN_ITERS}"
    fi

    FULL_CMD="export CUDA_DEVICE_MAX_CONNECTIONS=1; export PYTORCH_ALLOC_CONF=expandable_segments:True; export TRANSFORMERS_OFFLINE=1; export HF_DATASETS_OFFLINE=1; \
  export USER=snorouzi; \
  export ENABLE_FUSED_ADAM=True; \
  export WANDB_API_KEY=4194b2dcd12956a7ae960885c6244df12699849f; \
  export HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_cache; \
  export PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM:\$PYTHONPATH; \
  ${TRAIN_CMD}"

    if [ "$DIRECT" = "true" ]; then
        # ============================================
        # Run directly on this machine
        # ============================================
        export SUBMIT_GPUS=${GPUS_PER_NODE}
        export MASTER_ADDR=localhost
        export MASTER_PORT=29501
        export NUM_NODES=1
        export NODE_RANK=0
        eval "${FULL_CMD} train.global_batch_size=8 train.micro_batch_size=1"
    else
        # ============================================
        # Submit via Slurm
        # ============================================
        # SUBMIT_GPUS, MASTER_ADDR, MASTER_PORT, NUM_NODES, NODE_RANK are set by submit_job

        echo "Submitting job ${i}/${N_JOBS}"

        DEP_ARG=""
        if [ -n "$PREVIOUS_JOB_ID" ]; then
            DEP_ARG="--dependency afterany:${PREVIOUS_JOB_ID}"
        fi

        JOB_OUTPUT=$(submit_job \
            --account ${ACCOUNT} \
            --logroot ${USER_PATH}/${USER}/logs \
            --gpu ${GPUS_PER_NODE} \
            --nodes ${SLURM_NUM_NODES} \
            --partition ${PARTITION} \
            --duration ${DURATION} \
            -n ${UNIQUE_JOB_NAME} \
            --autoresume_before_timelimit 15 \
            ${DEP_ARG} \
            --image ${CONTAINER_IMAGE} \
            --mounts ${MOUNTS} \
            --command "${FULL_CMD}" 2>&1)

        NEW_JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '(Submitted batch job|Job Id is) [0-9]+' | grep -oE '[0-9]+' | head -1)

        if [ -z "$NEW_JOB_ID" ]; then
            echo "Warning: Could not extract job ID from output. Output was:"
            echo "$JOB_OUTPUT"
            if [ "$N_JOBS" -gt 1 ]; then
                echo "Stopping job submission chain."
                exit 1
            fi
        else
            echo "Job ${i} submitted with ID: ${NEW_JOB_ID}"
            PREVIOUS_JOB_ID=$NEW_JOB_ID
        fi
    fi
done

if [ "$N_JOBS" -gt 1 ]; then
    echo "Successfully submitted ${N_JOBS} jobs in a chain."
fi
