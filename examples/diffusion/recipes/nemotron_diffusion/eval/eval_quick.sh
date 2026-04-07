#!/bin/bash
# Quick eval script for nemotron_diffusion models.
#
# Usage:
#   bash eval_quick.sh --model-size 3b
#   bash eval_quick.sh --model-size 8b --tasks humaneval,mbpp --limit 16
#   bash eval_quick.sh --model-size cascade --parallel-tasks   # submit one job per task
#   bash eval_quick.sh --model-size 14b --parallel-models      # submit one job (all tasks)
#   bash eval_quick.sh --model-size cascade --cascade-schedule "ckpt|n|hf|..."
set -euo pipefail

# --- Paths ---
MEGATRON_BRIDGE="${HOME}/code/Megatron-Bridge"
EVAL_SCRIPT="${MEGATRON_BRIDGE}/examples/diffusion/recipes/nemotron_diffusion/eval_megatron.py"
TOKENIZER="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/yongganf/miscs/models/Nemotron-H-8B-Base-8K"

HF_MODEL_ID_3B="/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/ministral_3b/iter_0012500_hf/Ministral-3-3B-Base-2512_converted"
HF_MODEL_ID_8B="/lustre/fsw/portfolios/nvr/users/abhgarg/miscs/models/Ministral-3-8B-Base-2512_1t_ft"
HF_MODEL_ID_14B="mistralai/Ministral-3-14B-Base-2512"
CHECKPOINT_3B="/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/ministral_3b/iter_0012500"
CHECKPOINT_8B="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/abhgarg/megatron_exp/ministral_8b_sbd64_llada_combine_before_weighting_1e_5_rerun_32/iter_0012500"
CHECKPOINT_14B="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/abhgarg/megatron_exp/ministral_14b_BL1A/iter_0025000"

# --- submit_job config ---
ACCOUNT="coreai_dlalgo_llm"
NODES=1
DURATION_HOURS=4
USER_PATH="/lustre/fsw/portfolios/coreai/users/snorouzi"
LOG_ROOT="${USER_PATH}/logs"
IMAGE_PATH="nvcr.io/nvidia/nemo:25.09"
SQSH_CACHE_DIR="${USER_PATH}/.sqsh_cache"

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

# --- Task registry: task -> "nshots max_new_tokens temperature" ---
declare -A task_registry
task_registry=(
    [gsm8k_cot]="8 256 0.0"
    [humaneval]="0 512 0.0"
    [mbpp]="3 512 0.0"
    [humaneval_plus]="0 512 0.0"
    [mbpp_plus]="3 512 0.0"
    [minerva_math]="4 512 0.0"
)
all_tasks=(gsm8k_cot humaneval mbpp humaneval_plus mbpp_plus minerva_math)

# --- Defaults ---
MODEL_SIZE=""
LIMIT=""
SEED=42
CASCADE_SCHEDULE=""
DIFFUSION_STEPS=""
STEPS_PER_BLOCK=""
THRESHOLD="None"
EXECUTE_DIRECTLY=true
PARALLEL_TASKS=false
PARALLEL_MODELS=false
cli_tasks=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-size)       MODEL_SIZE="$2";       shift 2 ;;
    --limit)            LIMIT="$2";            shift 2 ;;
    --tasks)            cli_tasks="$2";        shift 2 ;;
    --seed)             SEED="$2";             shift 2 ;;
    --cascade-schedule)  CASCADE_SCHEDULE="$2";  shift 2 ;;
    --diffusion-steps)   DIFFUSION_STEPS="$2";   shift 2 ;;
    --steps-per-block)   STEPS_PER_BLOCK="$2";  shift 2 ;;
    --threshold)         THRESHOLD="$2";         shift 2 ;;
    --direct)           EXECUTE_DIRECTLY=true;  PARALLEL_TASKS=false; PARALLEL_MODELS=false; shift ;;
    --parallel-tasks)   EXECUTE_DIRECTLY=false; PARALLEL_TASKS=true;  PARALLEL_MODELS=false; shift ;;
    --parallel-models)  EXECUTE_DIRECTLY=false; PARALLEL_TASKS=false; PARALLEL_MODELS=true;  shift ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --model-size 3b|8b|14b|cascade [--tasks T1,T2] [--limit N] [--seed S]"
      echo "          [--cascade-schedule SCHEDULE] [--direct|--parallel-tasks|--parallel-models]"
      exit 1 ;;
  esac
done

if [[ -z "$MODEL_SIZE" ]]; then
  echo "Error: --model-size is required (3b, 8b, 14b, or cascade)"
  exit 1
fi

# Resolve model checkpoint/hf
case "$MODEL_SIZE" in
  3b)      CHECKPOINT="$CHECKPOINT_3B"; HF_MODEL_ID="$HF_MODEL_ID_3B" ;;
  8b)      CHECKPOINT="$CHECKPOINT_8B"; HF_MODEL_ID="$HF_MODEL_ID_8B" ;;
  14b)     CHECKPOINT="$CHECKPOINT_14B"; HF_MODEL_ID="$HF_MODEL_ID_14B" ;;
  cascade)
    CHECKPOINT="$CHECKPOINT_3B"
    HF_MODEL_ID="$HF_MODEL_ID_3B"
    STEPS_PER_BLOCK="${STEPS_PER_BLOCK:-6}"
    if [[ -z "$CASCADE_SCHEDULE" ]]; then
        local_steps=$(( STEPS_PER_BLOCK / 2 ))
        CASCADE_SCHEDULE="${CHECKPOINT_8B}|${local_steps}|${HF_MODEL_ID_8B}|${CHECKPOINT_3B}|${local_steps}|${HF_MODEL_ID_3B}"
    fi
    ;;
  *)
    echo "Error: unknown --model-size '${MODEL_SIZE}'. Choose from: 3b, 8b, 14b, cascade"
    exit 1 ;;
esac

# Resolve tasks
if [[ -n "$cli_tasks" ]]; then
    IFS=',' read -ra tasks <<< "$cli_tasks"
    for t in "${tasks[@]}"; do
        if [[ -z "${task_registry[$t]+x}" ]]; then
            echo "Error: Unknown task '$t'. Available: ${!task_registry[*]}"
            exit 1
        fi
    done
else
    tasks=("${all_tasks[@]}")
fi

# Expand task registry into parallel arrays
nshots=(); max_new_tokens_arr=(); temperatures=()
for t in "${tasks[@]}"; do
    read -r ns mnt temp <<< "${task_registry[$t]}"
    nshots+=("$ns"); max_new_tokens_arr+=("$mnt"); temperatures+=("$temp")
done

echo "=== nemotron_diffusion eval: model=${MODEL_SIZE}, tasks=${tasks[*]}, limit=${LIMIT:-full}, seed=${SEED} ==="
echo "    checkpoint: ${CHECKPOINT}"
[[ -n "$CASCADE_SCHEDULE" ]] && echo "    cascade_schedule: ${CASCADE_SCHEDULE}"

NEMO_DFM="${HOME}/code/nemo-dfm"
PIP_INSTALLS_BASE="python -m pip uninstall nvidia-lm-eval -y -q 2>/dev/null || true; python -m pip install lm-eval==0.4.10 -q --no-cache-dir; python -m pip install --upgrade huggingface_hub evaluate -q --no-cache-dir; python -m pip install antlr4-python3-runtime==4.9.3 -q --no-cache-dir"
PIP_INSTALLS_MATH="python -m pip uninstall nvidia-lm-eval -y -q 2>/dev/null || true; python -m pip install lm-eval==0.4.10 -q --no-cache-dir; python -m pip install 'lm-eval[math]==0.4.10' -q --no-cache-dir; python -m pip install --upgrade huggingface_hub evaluate -q --no-cache-dir; python -m pip install antlr4-python3-runtime==4.9.3 -q --no-cache-dir; python ${MEGATRON_BRIDGE}/examples/diffusion/recipes/nemotron_diffusion/patch_minerva_deps.py"
PIP_INSTALLS="$PIP_INSTALLS_BASE"

# Build eval command for a single task
build_task_cmd() {
    local task_idx=$1
    local task="${tasks[$task_idx]}"
    local nshot="${nshots[$task_idx]}"
    local max_new_tokens="${max_new_tokens_arr[$task_idx]}"
    local temperature="${temperatures[$task_idx]}"
    local output_path="${USER_PATH}/megatron_eval_results/ministral_${MODEL_SIZE}/seed_${SEED}/${task}-ns${nshot}"

    local MODEL_ARGS="megatron_load_path=${CHECKPOINT}"
    MODEL_ARGS="${MODEL_ARGS},hf_model_id=${HF_MODEL_ID}"
    MODEL_ARGS="${MODEL_ARGS},tokenizer=${TOKENIZER}"
    MODEL_ARGS="${MODEL_ARGS},mask_token_id=100"
    MODEL_ARGS="${MODEL_ARGS},eval_mode=dllm"
    MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${max_new_tokens}"
    MODEL_ARGS="${MODEL_ARGS},max_sequence_length=4096"
    local _diffusion_steps
    if [[ -n "${STEPS_PER_BLOCK}" ]]; then
        _diffusion_steps=$(( STEPS_PER_BLOCK * max_new_tokens / 32 ))
    else
        _diffusion_steps="${DIFFUSION_STEPS:-${max_new_tokens}}"
    fi
    MODEL_ARGS="${MODEL_ARGS},diffusion_steps=${_diffusion_steps}"
    MODEL_ARGS="${MODEL_ARGS},temperature=${temperature}"
    MODEL_ARGS="${MODEL_ARGS},block_length=32"
    MODEL_ARGS="${MODEL_ARGS},shift_logits=False"
    MODEL_ARGS="${MODEL_ARGS},neg_entropy=True"
    MODEL_ARGS="${MODEL_ARGS},denoising_threshold=${THRESHOLD}"
    MODEL_ARGS="${MODEL_ARGS},tp=1,pp=1"
    MODEL_ARGS="${MODEL_ARGS},load_hf_weights=False"
    [[ -n "$CASCADE_SCHEDULE" ]] && MODEL_ARGS="${MODEL_ARGS},cascade_schedule=${CASCADE_SCHEDULE}"

    local LIMIT_ARG=""
    [[ -n "$LIMIT" ]] && LIMIT_ARG="--limit ${LIMIT}"

    local eval_script="${MEGATRON_BRIDGE}/examples/diffusion/recipes/nemotron_diffusion/eval_megatron.py"
    echo "accelerate launch --num_processes ${GPUS_PER_NODE} ${eval_script} \
  --model megatron_dllm \
  --model_args \"${MODEL_ARGS}\" \
  --tasks ${task} \
  --batch_size 1 \
  --output_path ${output_path} \
  --num_fewshot ${nshot} \
  --log_samples \
  --confirm_run_unsafe_code \
  --seed ${SEED} \
  ${LIMIT_ARG}"
}

# ================================================================
# Direct execution
# ================================================================
if [[ "$EXECUTE_DIRECTLY" == "true" ]]; then
    source "${NEMO_DFM}/prepare.sh"
    export PYTHONPATH="${MEGATRON_BRIDGE}/src:${MEGATRON_BRIDGE}/examples:${MEGATRON_BRIDGE}:${PYTHONPATH:-}"
    export HF_ALLOW_CODE_EVAL=1
    export HF_HOME="${USER_PATH}/hf_cache"
    echo "Installing dependencies..."
    eval "${PIP_INSTALLS}"

    for task_idx in "${!tasks[@]}"; do
        task="${tasks[$task_idx]}"
        echo "--- Running task: ${task} ---"
        if [[ "$task" == "minerva_math" ]]; then
            eval "${PIP_INSTALLS_MATH}"
        else
            eval "${PIP_INSTALLS_BASE}"
        fi
        eval "$(build_task_cmd $task_idx)"
    done

# ================================================================
# Slurm: one job per task
# ================================================================
elif [[ "$PARALLEL_TASKS" == "true" ]]; then
    job_counter=0
    for task_idx in "${!tasks[@]}"; do
        task="${tasks[$task_idx]}"
        nshot="${nshots[$task_idx]}"
        CURRENT_JOB_NAME="ministral_eval_${MODEL_SIZE}_${task}_ns${nshot}_s${SEED}"
        TASK_CMD=$(build_task_cmd $task_idx)
        if [[ "$task" == "minerva_math" ]]; then TASK_PIP="${PIP_INSTALLS_MATH}"; else TASK_PIP="${PIP_INSTALLS_BASE}"; fi
        INNER_COMMAND="source ${NEMO_DFM}/prepare.sh; export PYTHONPATH=${MEGATRON_BRIDGE}/src:${MEGATRON_BRIDGE}/examples:${MEGATRON_BRIDGE}:\${PYTHONPATH:-}; export HF_ALLOW_CODE_EVAL=1; export HF_HOME=${USER_PATH}/hf_cache; ${TASK_PIP}; ${TASK_CMD}"
        INNER_COMMAND_ONELINE=$(echo "${INNER_COMMAND}" | tr -s ' ' | sed 's/ \\ / /g' | tr -d '\n')

        SUBMIT_CMD="submit_job --account ${ACCOUNT} \
            --logroot ${LOG_ROOT} \
            --gpu ${GPUS_PER_NODE} \
            --nodes ${NODES} \
            --partition ${PARTITION} \
            --duration ${DURATION_HOURS} \
            -n ${CURRENT_JOB_NAME} \
            --image ${IMAGE_PATH} \
            --command '${INNER_COMMAND_ONELINE}'"

        export SQSH_CACHE_DIR
    echo "Submitting job ${job_counter}: ${task} (${nshot}-shot)"
        eval "${SUBMIT_CMD}"
        sleep 1
        job_counter=$((job_counter + 1))
    done
    echo "Submitted ${job_counter} jobs total."

# ================================================================
# Slurm: one job (all tasks sequential)
# ================================================================
elif [[ "$PARALLEL_MODELS" == "true" ]]; then
    CURRENT_JOB_NAME="ministral_eval_${MODEL_SIZE}_s${SEED}"
    INNER_COMMAND="source ${NEMO_DFM}/prepare.sh; export PYTHONPATH=${MEGATRON_BRIDGE}/src:${MEGATRON_BRIDGE}/examples:${MEGATRON_BRIDGE}:\${PYTHONPATH:-}; export HF_ALLOW_CODE_EVAL=1; export HF_HOME=${USER_PATH}/hf_cache; ${PIP_INSTALLS}"
    for task_idx in "${!tasks[@]}"; do
        TASK_CMD=$(build_task_cmd $task_idx)
        INNER_COMMAND="${INNER_COMMAND} ; ${TASK_CMD}"
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
        --command '${INNER_COMMAND_ONELINE}'"

    export SQSH_CACHE_DIR
    echo "Submitting job: ${CURRENT_JOB_NAME}"
    eval "${SUBMIT_CMD}"
fi
