#!/bin/bash
# Megatron-native evaluation launcher for dLLM and AR models.
#
# Usage:
#   bash eval_megatron.sh --parallel-tasks --expts BL1A,BL1C --modes dllm,ar
#   bash eval_megatron.sh --parallel-tasks --expts BL1A --modes ar --eval-tasks mbpp_plus --seeds 0,1234
#   bash eval_megatron.sh --parallel-tasks   # (uses all defaults)
#   bash eval_megatron.sh --direct           # run directly on GPU node
#   bash eval_megatron.sh --parallel-models  # one Slurm job per model
#
# Options:
#   --expts       Comma-separated experiments (default: BL1A,BL1C,BL1D,BL2A,BL3A,BL4A)
#   --modes       Comma-separated eval modes (default: dllm,ar)
#   --eval-tasks  Comma-separated tasks (default: gsm8k_cot,humaneval,mbpp,humaneval_plus,mbpp_plus)
#   --seeds       Comma-separated seeds (default: 42,0,1234,3407,1337,2024,5678,8765)
#   --gpus N          GPUs per node (auto-detected by cluster)
#   --checkpoint PATH Override checkpoint path (use with --exp-name)
#   --exp-name NAME   Experiment name for output paths
#   --hf-model-id ID  Override HF model/config path
#   --limit N         Limit number of eval samples (for testing)
#
# GPU configurations:
#   TP=1, GPUS_PER_NODE=1  → single GPU, plain python
#   TP=1, GPUS_PER_NODE=8  → 8 GPUs, DP=8, accelerate launch
#   TP=2, GPUS_PER_NODE=8  → 8 GPUs, DP=4, torchrun

set -euo pipefail

# --- Job Configuration ---
JOB_NAME="megatron_dllm_eval"
ACCOUNT="coreai_dlalgo_llm"
# ACCOUNT="nvr_lpr_llm"
# ACCOUNT="coreai_dlalgo_genai"

hostname=$(hostname)

GPUS_PER_NODE=8
if [[ "$hostname" == *"nrt"* ]]; then
    PARTITION="backfill,batch_block1"
elif [[ "$hostname" == *"ord"* ]]; then
    PARTITION="polar,polar3,polar4,grizzly"
elif [[ "$hostname" == *"hsg"* ]]; then
    PARTITION="batch"
    GPUS_PER_NODE=4
elif [[ "$hostname" == *"cw-dfw"* ]]; then
    PARTITION="batch"
else
    PARTITION="backfill,batch"
fi

USER_ROOT_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users"
if [[ "$hostname" == *"eos"* ]]; then
    USER_ROOT_DIR="/lustre/fsw/nvr_lpr_llm"
fi

NODES=1
DURATION_HOURS=4
USER_NAME="snorouzi"
USER_PATH="/lustre/fsw/portfolios/coreai/users/${USER_NAME}"
LOG_ROOT="${USER_PATH}/logs"

# --- Paths ---
CODE_ROOT="/root/code"
MEGATRON_BRIDGE="${CODE_ROOT}/Megatron-Bridge"
EVAL_SCRIPT="${MEGATRON_BRIDGE}/examples/diffusion/recipes/nemotron_diffusion/eval/eval_megatron.py"
EVAL_SCRIPT_CONTAINER="/opt/Megatron-Bridge/examples/diffusion/recipes/nemotron_diffusion/eval/eval_megatron.py"
IMAGE_PATH="nvcr.io/nvidian/nemo:26.04.rc4"
MB_DIR=${MB_DIR:-/home/snorouzi/code/Megatron-Bridge}
MOUNTS="${MB_DIR}:/opt/Megatron-Bridge,${MB_DIR}/3rdparty/Megatron-LM:/opt/megatron-lm"
export SQSH_CACHE_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/users/${USER_NAME}
# --- Control Flow ---
EXECUTE_DIRECTLY=true
PARALLEL_TASKS=false
PARALLEL_MODELS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --direct)
      EXECUTE_DIRECTLY=true; PARALLEL_TASKS=false; PARALLEL_MODELS=false
      echo "Executing inner command directly."
      shift ;;
    --parallel-tasks)
      EXECUTE_DIRECTLY=false; PARALLEL_TASKS=true; PARALLEL_MODELS=false
      echo "Running tasks in parallel for each model."
      shift ;;
    --parallel-models)
      EXECUTE_DIRECTLY=false; PARALLEL_TASKS=false; PARALLEL_MODELS=true
      echo "Running one job per model (all tasks sequential within each job)."
      shift ;;
    --gpus)
      GPUS_PER_NODE="$2"; echo "GPUS_PER_NODE=$GPUS_PER_NODE"; shift 2 ;;
    --expts)
      EXPT_LIST_STR="$2"; shift 2 ;;
    --modes)
      MODES_STR="$2"; shift 2 ;;
    --eval-tasks)
      EVAL_TASKS_STR="$2"; shift 2 ;;
    --seeds)
      SEEDS_STR="$2"; shift 2 ;;
    --checkpoint)
      CUSTOM_CHECKPOINT="$2"; shift 2 ;;
    --exp-name)
      CUSTOM_EXP_NAME="$2"; shift 2 ;;
    --hf-model-id)
      CUSTOM_HF_MODEL_ID="$2"; shift 2 ;;
    --steps-per-block)
      STEPS_PER_BLOCK="$2"; shift 2 ;;
    --denoising-threshold)
      DENOISING_THRESHOLD="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--direct|--parallel-tasks|--parallel-models] [--expts E1,E2] [--modes dllm,ar] [--eval-tasks T1,T2] [--seeds S1,S2] [--gpus N]"
      exit 1 ;;
  esac
done

# --- Model config ---
HF_MODEL_ID="/lustre/fsw/portfolios/nvr/users/snorouzi/models/Ministral-3-3B-Base-2512_converted"
HF_MODEL_ID="${CUSTOM_HF_MODEL_ID:-${HF_MODEL_ID}}"
TOKENIZER="${HF_MODEL_ID}"
MASK_TOKEN_ID=100
MEGATRON_EXP_ROOT="${USER_PATH}/megatron_exp"

# --- Parse comma-separated CLI args into arrays (with defaults) ---
IFS=',' read -ra EXPT_LIST <<< "${EXPT_LIST_STR:-3b}"
IFS=',' read -ra MODES <<< "${MODES_STR:-dllm,ar}"
IFS=',' read -ra SEEDS <<< "${SEEDS_STR:-42}"

echo "Experiments: ${EXPT_LIST[*]}"
echo "Modes:       ${MODES[*]}"
echo "Seeds:       ${SEEDS[*]}"

# --- Experiment configs (built from EXPT_LIST x MODES) ---
exp_names=()
eval_modes=()
block_lengths=()
shift_logits=()
checkpoint_paths=()
tp_sizes=()

if [ -n "${CUSTOM_CHECKPOINT:-}" ]; then
    _name="${CUSTOM_EXP_NAME:-custom_checkpoint}"
    for _mode in "${MODES[@]}"; do
        exp_names+=("${_name}_${_mode}")
        eval_modes+=("${_mode}")
        if [ "${_mode}" = "dllm" ]; then
            block_lengths+=(32)
            shift_logits+=(False)
        else
            block_lengths+=(1)
            shift_logits+=(True)
        fi
        checkpoint_paths+=("${CUSTOM_CHECKPOINT}")
        tp_sizes+=(1)
    done
else
    for _expt in "${EXPT_LIST[@]}"; do
        for _mode in "${MODES[@]}"; do
            exp_names+=("ministral_${_expt}_${_mode}")
            eval_modes+=("${_mode}")
            if [ "${_mode}" = "dllm" ]; then
                block_lengths+=(32)
                shift_logits+=(False)
            else
                block_lengths+=(1)
                shift_logits+=(True)
            fi
            checkpoint_paths+=("${MEGATRON_EXP_ROOT}/ministral_${_expt}_/iter_0012500")
            tp_sizes+=(1)
        done
    done
fi

# Scalar or per-experiment array — scalar applies to all experiments.
denoising_thresholds=None
if [ -n "${DENOISING_THRESHOLD:-}" ]; then denoising_thresholds="$DENOISING_THRESHOLD"; fi
neg_entropy=True
temperature=0.0

# --- Task configs (all available) ---
ALL_TASKS=("gsm8k_cot" "humaneval" "mbpp" "humaneval_plus" "mbpp_plus" "minerva_math")
ALL_NSHOTS=(          8          0     3              0          3           4)
ALL_MAX_NEW_TOKENS=(256        512   512            512        512         512)

# Filter tasks if --eval-tasks was provided
if [ -n "${EVAL_TASKS_STR:-}" ]; then
    IFS=',' read -ra TASK_FILTER <<< "$EVAL_TASKS_STR"
    tasks=()
    nshots=()
    max_new_tokens=()
    for ft in "${TASK_FILTER[@]}"; do
        for i in "${!ALL_TASKS[@]}"; do
            if [ "${ALL_TASKS[$i]}" = "$ft" ]; then
                tasks+=("${ALL_TASKS[$i]}")
                nshots+=("${ALL_NSHOTS[$i]}")
                max_new_tokens+=("${ALL_MAX_NEW_TOKENS[$i]}")
            fi
        done
    done
else
    tasks=("${ALL_TASKS[@]}")
    nshots=("${ALL_NSHOTS[@]}")
    max_new_tokens=("${ALL_MAX_NEW_TOKENS[@]}")
fi

echo "Tasks:       ${tasks[*]}"

# --- Other settings ---
BATCH_SIZE=1
LIMIT="${LIMIT:-}"  # Set to a number for quick testing, empty string for full eval

# --- Pip installs needed inside container ---
PIP_INSTALLS="\
  python -m pip uninstall nvidia-lm-eval -y -q 2>/dev/null || true; \
  python ${EVAL_SCRIPT%/*}/patch_minerva_deps.py; \
  python -m pip install --upgrade huggingface_hub evaluate -q --no-cache-dir"

# --- Helper: get value from scalar-or-array ---
get_param() {
    local var_name=$1
    local idx=$2
    if [[ "$(declare -p "$var_name" 2>/dev/null)" =~ "declare -a" ]]; then
        local -n arr=$var_name
        echo "${arr[$idx]}"
    else
        echo "${!var_name}"
    fi
}

# --- Helper: build launch command ---
build_launch_cmd() {
    local tp=$1
    local total_gpus=${GPUS_PER_NODE}

    if [ "${tp}" -gt 1 ]; then
        echo "torchrun --nproc_per_node=${total_gpus}"
    elif [ "${total_gpus}" -gt 1 ]; then
        echo "accelerate launch --num_processes ${total_gpus}"
    else
        echo "python"
    fi
}

# --- Helper: build eval command ---
build_eval_command() {
    local model_idx=$1
    local task_idx=$2
    local seed=$3

    local exp_name="${exp_names[$model_idx]}"
    local eval_mode="${eval_modes[$model_idx]}"
    local block_length="${block_lengths[$model_idx]}"
    local shift_log="${shift_logits[$model_idx]}"
    local ckpt="${checkpoint_paths[$model_idx]}"
    local tp="${tp_sizes[$model_idx]}"

    local task="${tasks[$task_idx]}"
    local nshot="${nshots[$task_idx]}"
    local max_new_tok="${max_new_tokens[$task_idx]}"

    local cur_temp
    cur_temp=$(get_param temperature "$model_idx")
    local cur_threshold
    cur_threshold=$(get_param denoising_thresholds "$model_idx")
    local cur_neg_entropy
    cur_neg_entropy=$(get_param neg_entropy "$model_idx")

    local output_path="${USER_PATH}/megatron_eval_results/${exp_name}/seed_${seed}/${task}-ns${nshot}"
    local nfe_log="${output_path}__nfes.json"

    local MODEL_ARGS="megatron_load_path=${ckpt}"
    MODEL_ARGS="${MODEL_ARGS},hf_model_id=${HF_MODEL_ID}"
    MODEL_ARGS="${MODEL_ARGS},tokenizer=${TOKENIZER}"
    MODEL_ARGS="${MODEL_ARGS},mask_token_id=${MASK_TOKEN_ID}"
    MODEL_ARGS="${MODEL_ARGS},eval_mode=${eval_mode}"
    MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${max_new_tok}"
    MODEL_ARGS="${MODEL_ARGS},max_sequence_length=4096"
    MODEL_ARGS="${MODEL_ARGS},steps_per_block=${STEPS_PER_BLOCK:-32}"
    MODEL_ARGS="${MODEL_ARGS},temperature=${cur_temp}"
    MODEL_ARGS="${MODEL_ARGS},block_length=${block_length}"
    MODEL_ARGS="${MODEL_ARGS},shift_logits=${shift_log}"
    MODEL_ARGS="${MODEL_ARGS},neg_entropy=${cur_neg_entropy}"
    MODEL_ARGS="${MODEL_ARGS},denoising_threshold=${cur_threshold}"
    MODEL_ARGS="${MODEL_ARGS},tp=${tp},pp=1"
    MODEL_ARGS="${MODEL_ARGS},nfe_log_path=${nfe_log}"
    MODEL_ARGS="${MODEL_ARGS},load_hf_weights=False"

    local LIMIT_ARG=""
    if [ -n "${LIMIT}" ]; then
        LIMIT_ARG="--limit ${LIMIT}"
    fi

    local LAUNCH_CMD
    LAUNCH_CMD=$(build_launch_cmd "${tp}")

    echo "${LAUNCH_CMD} ${EVAL_SCRIPT} \
--model megatron_dllm \
--model_args \"${MODEL_ARGS}\" \
--tasks ${task} \
--batch_size ${BATCH_SIZE} \
--output_path ${output_path} \
--num_fewshot ${nshot} \
--log_samples \
--confirm_run_unsafe_code \
--seed ${seed} \
${LIMIT_ARG}"
}

# ================================================================
# Direct execution
# ================================================================
if [ "$EXECUTE_DIRECTLY" = true ]; then
    echo "Running all models and tasks sequentially"
    export PYTHONPATH="${MEGATRON_BRIDGE}/src:${MEGATRON_BRIDGE}/examples:${MEGATRON_BRIDGE}:${PYTHONPATH:-}"
    export HF_ALLOW_CODE_EVAL=1
    export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
    if [ -f ~/hf_token.txt ]; then export HF_TOKEN=$(cat ~/hf_token.txt); fi


    for model_idx in "${!exp_names[@]}"; do
        echo "=========================================="
        echo "Model $((model_idx + 1))/${#exp_names[@]}: ${exp_names[$model_idx]}"
        echo "  eval_mode=${eval_modes[$model_idx]}, tp=${tp_sizes[$model_idx]}"
        echo "=========================================="

        for task_idx in "${!tasks[@]}"; do
            for seed in "${SEEDS[@]}"; do
                EVAL_CMD=$(build_eval_command "$model_idx" "$task_idx" "$seed")
                echo ">>> ${tasks[$task_idx]} (${nshots[$task_idx]}-shot) seed=${seed}"
                eval "${EVAL_CMD}"
                echo "Completed: ${exp_names[$model_idx]} / ${tasks[$task_idx]} / seed=${seed}"
                echo ""
            done
        done
    done
    echo "All evaluations completed!"

# ================================================================
# Slurm: one job per task
# ================================================================
elif [ "$PARALLEL_TASKS" = true ]; then
    EVAL_SCRIPT="${EVAL_SCRIPT_CONTAINER}"
    echo "Submitting separate Slurm jobs for each model/task pair"
    job_counter=0

    for model_idx in "${!exp_names[@]}"; do
        for task_idx in "${!tasks[@]}"; do
          for seed in "${SEEDS[@]}"; do
            exp_name="${exp_names[$model_idx]}"
            task="${tasks[$task_idx]}"
            nshot="${nshots[$task_idx]}"

            CURRENT_JOB_NAME="${JOB_NAME}_${exp_name}_${task}_ns${nshot}_s${seed}_${job_counter}"
            TASK_CMD=$(build_eval_command "$model_idx" "$task_idx" "$seed")
            INNER_COMMAND="python /opt/Megatron-Bridge/examples/diffusion/recipes/nemotron_diffusion/eval/patch_minerva_deps.py; export PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/examples:/opt/Megatron-Bridge:/opt/megatron-lm:\${PYTHONPATH:-}; export HF_ALLOW_CODE_EVAL=1; export HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_cache; export HF_TOKEN=$(cat ~/hf_token.txt 2>/dev/null || true); ${TASK_CMD}"
            INNER_COMMAND_ONELINE=$(echo "${INNER_COMMAND}" | tr -s ' ' | sed 's/ \\ / /g' | tr -d '\n')

            SUBMIT_CMD="submit_job --account ${ACCOUNT} \
                --logroot ${LOG_ROOT} \
                --gpu ${GPUS_PER_NODE} \
                --nodes ${NODES} \
                --partition ${PARTITION} \
                --duration ${DURATION_HOURS} \
                -n ${CURRENT_JOB_NAME} \
                --image ${IMAGE_PATH} \
                --mounts ${MOUNTS} \
                --command '${INNER_COMMAND_ONELINE}'"

            echo "Submitting job ${job_counter}: ${exp_name} - ${task} (${nshot}-shot) seed=${seed}"
            eval "${SUBMIT_CMD}"
            sleep 1
            job_counter=$((job_counter + 1))
          done
        done
    done
    echo "Submitted ${job_counter} jobs total!"

# ================================================================
# Slurm: one job per model (all tasks sequential)
# ================================================================
elif [ "$PARALLEL_MODELS" = true ]; then
    EVAL_SCRIPT="${EVAL_SCRIPT_CONTAINER}"
    echo "Submitting one Slurm job per model"
    job_counter=0

    for model_idx in "${!exp_names[@]}"; do
        exp_name="${exp_names[$model_idx]}"
        CURRENT_JOB_NAME="${JOB_NAME}_${exp_name}_${job_counter}"

        INNER_COMMAND="python /opt/Megatron-Bridge/examples/diffusion/recipes/nemotron_diffusion/eval/patch_minerva_deps.py; export PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/examples:/opt/Megatron-Bridge:/opt/megatron-lm:\${PYTHONPATH:-}; export HF_ALLOW_CODE_EVAL=1; export HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_cache; export HF_TOKEN=$(cat ~/hf_token.txt 2>/dev/null || true)"

        for task_idx in "${!tasks[@]}"; do
            for seed in "${SEEDS[@]}"; do
                TASK_CMD=$(build_eval_command "$model_idx" "$task_idx" "$seed")
                INNER_COMMAND="${INNER_COMMAND} ; ${TASK_CMD}"
            done
        done

        INNER_COMMAND_ONELINE=$(echo "${INNER_COMMAND}" | tr -s ' ' | sed 's/ \\ / /g' | tr -d '\n')

        SUBMIT_CMD="submit_job --account ${ACCOUNT} \
            --logroot ${LOG_ROOT} \
            --gpu ${GPUS_PER_NODE} \
            --nodes ${NODES} \
            --partition ${PARTITION} \
            --duration ${DURATION_HOURS} \
            -n ${CURRENT_JOB_NAME} \
            --image ${IMAGE_PATH} \
            --mounts ${MOUNTS} \
            --command '${INNER_COMMAND_ONELINE}'"

        echo "Submitting job ${job_counter}: ${exp_name}"
        eval "${SUBMIT_CMD}"
        sleep 1
        job_counter=$((job_counter + 1))
    done
    echo "Submitted ${job_counter} jobs total!"

else
    echo "Error: specify --direct, --parallel-tasks, or --parallel-models"
    exit 1
fi
