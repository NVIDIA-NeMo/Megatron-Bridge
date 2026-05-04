#!/bin/bash
# Run heterogeneous MIMO LLaVA E2E test with various parallelism configurations
# Usage: ./run_hetero_llava_parallelism_tests.sh [--gpus N] [--config CONFIG_NAME]
#
# Examples:
#   ./run_hetero_llava_parallelism_tests.sh                    # Run all configs with 8 GPUs
#   ./run_hetero_llava_parallelism_tests.sh --gpus 4           # Run all configs with 4 GPUs
#   ./run_hetero_llava_parallelism_tests.sh --config tp2_dp2   # Run only tp2_dp2 config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/megatron_mimo_training_llava.py"

# Default values
NUM_GPUS=${NUM_GPUS:-8}
SINGLE_CONFIG=""

# Training defaults (can be overridden via env vars)
# MBS is set per-config (must be divisible by every module's DP size).
# GBS must be divisible by MBS.  num_microbatches = GBS / MBS.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-96}
TRAIN_ITERS=${TRAIN_ITERS:-100}
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-2.0e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-60}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
CLIP_GRAD=${CLIP_GRAD:-1.0}
LOG_INTERVAL=${LOG_INTERVAL:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"Megatron-Bridge-MIMO"}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-"/tmp/wandb"}
DATASET_ROOT=${DATASET_ROOT:-"/path/to/llava/pretrain/dataset"}
UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv_cache/}

# HuggingFace source models for checkpoint conversion
HF_VISION_MODEL=${HF_VISION_MODEL:-"openai/clip-vit-large-patch14-336"}
HF_LLM_MODEL=${HF_LLM_MODEL:-"lmsys/vicuna-7b-v1.5"}
MEGATRON_VOCAB_SIZE=${MEGATRON_VOCAB_SIZE:-32256}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/workspace/megatron_mimo_checkpoints}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            SINGLE_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Hetero MIMO LLaVA Parallelism E2E Tests"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="

# Define configurations as: "name|llm_tp|llm_pp|llm_dp|llm_offset|vision_tp|vision_pp|vision_dp|vision_offset|mbs"
# Note: Vision encoder (CLIPViT) does not support PP > 1, only LLM can use PP
# Heterogeneous: LLM and vision occupy non-overlapping GPU sets
# MBS must be divisible by every module's DP size (enforced by build_megatron_mimo_data_loaders)

declare -a CONFIGS_8GPU=(
    # 4+4 split
    "tp4_both|4|1|1|0|4|1|1|4|4"
    "tp2_dp2_both|2|1|2|0|2|1|2|4|4"
    "tp2_pp2_llm_tp4_vision|2|2|1|0|4|1|1|4|4"
    "tp1_dp4_both|1|1|4|0|1|1|4|4|4"
    "pp4_llm_tp4_vision|1|4|1|0|4|1|1|4|4"
    "pp4_llm_dp4_vision|1|4|1|0|1|1|4|4|4"
    "pp2_dp2_llm_tp2_dp2|1|2|2|0|2|1|2|4|4"
    "tp4_llm_dp4_vision|4|1|1|0|1|1|4|4|4"
    "tp2_pp2_llm_tp2_dp2_vision|2|2|1|0|2|1|2|4|4"
    "tp2_dp2_llm_dp4_vision|2|1|2|0|1|1|4|4|4"
    # Asymmetric configs
    "asymmetric_2_6_pp2|1|2|1|0|2|1|3|2|3"
    "asymmetric_2_6_tp2|2|1|1|0|2|1|3|2|3"
)

declare -a CONFIGS_4GPU=(
    # 2+2 split
    "tp2_both|2|1|1|0|2|1|1|2|4"
    "tp1_dp2_both|1|1|2|0|1|1|2|2|4"
)

declare -a CONFIGS_2GPU=(
    "tp1_both|1|1|1|0|1|1|1|1|4"
)

# Select configs based on GPU count
if [[ $NUM_GPUS -ge 8 ]]; then
    CONFIGS=("${CONFIGS_8GPU[@]}")
elif [[ $NUM_GPUS -ge 4 ]]; then
    CONFIGS=("${CONFIGS_4GPU[@]}")
else
    CONFIGS=("${CONFIGS_2GPU[@]}")
fi

# Track results
declare -a RESULTS=()
declare -a FAILED_CONFIGS=()
TOTAL=0
PASSED=0

convert_checkpoints() {
    local vision_tp="$1"
    local llm_tp="$2"

    local clip_ckpt_dir="${CHECKPOINT_BASE_DIR}/clip_tp${vision_tp}"
    local llm_ckpt_dir="${CHECKPOINT_BASE_DIR}/llm_tp${llm_tp}"

    # Convert CLIP checkpoint if not already cached for this TP size
    if [[ ! -d "${clip_ckpt_dir}/tp_rank_00" ]]; then
        echo "  Converting CLIP checkpoint (TP=${vision_tp})..."
        uv run python "${SCRIPT_DIR}/convert_hf_clip_to_megatron.py" \
            --hf-model "${HF_VISION_MODEL}" \
            --output "${clip_ckpt_dir}" \
            --tensor-parallel-size "${vision_tp}" \
            --use-te
    else
        echo "  Using cached CLIP checkpoint: ${clip_ckpt_dir}"
    fi

    # Convert LLM checkpoint if not already cached for this TP size
    if [[ ! -d "${llm_ckpt_dir}/tp_rank_00" ]]; then
        echo "  Converting LLM checkpoint (TP=${llm_tp})..."
        uv run python "${SCRIPT_DIR}/convert_hf_llama_to_megatron.py" \
            --hf-model "${HF_LLM_MODEL}" \
            --output "${llm_ckpt_dir}" \
            --tensor-parallel-size "${llm_tp}" \
            --use-te \
            --megatron-vocab-size "${MEGATRON_VOCAB_SIZE}"
    else
        echo "  Using cached LLM checkpoint: ${llm_ckpt_dir}"
    fi

    # Return paths via global variables
    CONVERTED_CLIP_CKPT="${clip_ckpt_dir}"
    CONVERTED_LLM_CKPT="${llm_ckpt_dir}"
}

build_wandb_exp_name() {
    local name="$1"
    local llm_tp="$2" llm_pp="$3" llm_dp="$4"
    local vision_tp="$5" vision_pp="$6" vision_dp="$7"
    local mbs="$8"

    echo "hetero-llava-${name}-${NUM_GPUS}gpu-llm_tp${llm_tp}_pp${llm_pp}_dp${llm_dp}-vis_tp${vision_tp}_pp${vision_pp}_dp${vision_dp}-mbs${mbs}"
}

run_config() {
    local config="$1"
    local name llm_tp llm_pp llm_dp llm_offset vision_tp vision_pp vision_dp vision_offset mbs

    IFS='|' read -r name llm_tp llm_pp llm_dp llm_offset vision_tp vision_pp vision_dp vision_offset mbs <<< "$config"

    local exp_name
    exp_name=$(build_wandb_exp_name "${name}" "${llm_tp}" "${llm_pp}" "${llm_dp}" "${vision_tp}" "${vision_pp}" "${vision_dp}" "${mbs}")

    echo ""
    echo "----------------------------------------"
    echo "Running: ${name}"
    echo "  LLM:    TP=${llm_tp}, PP=${llm_pp}, DP=${llm_dp}, offset=${llm_offset}"
    echo "  Vision: TP=${vision_tp}, PP=${vision_pp}, DP=${vision_dp}, offset=${vision_offset}"
    echo "  MBS:    ${mbs}"
    echo "  W&B:    ${exp_name}"
    echo "----------------------------------------"

    TOTAL=$((TOTAL + 1))

    # Convert checkpoints for this config's TP sizes
    convert_checkpoints "${vision_tp}" "${llm_tp}"

    local start_time=$(date +%s)

    if MIMO_LLM_TP="${llm_tp}" \
       MIMO_LLM_PP="${llm_pp}" \
       MIMO_LLM_DP="${llm_dp}" \
       MIMO_LLM_OFFSET="${llm_offset}" \
       MIMO_VISION_TP="${vision_tp}" \
       MIMO_VISION_PP="${vision_pp}" \
       MIMO_VISION_DP="${vision_dp}" \
       MIMO_VISION_OFFSET="${vision_offset}" \
       UV_CACHE_DIR="${UV_CACHE_DIR}" \
       uv run torchrun \
           --nproc_per_node "${NUM_GPUS}" \
           --nnodes 1 \
           "${TEST_FILE}" \
           --micro-batch-size "${mbs}" \
           --global-batch-size "${GLOBAL_BATCH_SIZE}" \
           --train-iters "${TRAIN_ITERS}" \
           --adam-beta1 "${ADAM_BETA1}" \
           --adam-beta2 "${ADAM_BETA2}" \
           --clip-grad "${CLIP_GRAD}" \
           --log-interval "${LOG_INTERVAL}" \
           --lr "${LR}" \
           --lr-warmup-iters "${LR_WARMUP_ITERS}" \
           --min-lr "${MIN_LR}" \
           --weight-decay "${WEIGHT_DECAY}" \
           --wandb-project "${WANDB_PROJECT}" \
           --wandb-exp-name "${exp_name}" \
           --wandb-save-dir "${WANDB_SAVE_DIR}" \
           --dataset-root "${DATASET_ROOT}" \
           --vision-encoder-checkpoint "${CONVERTED_CLIP_CKPT}" \
           --language-model-checkpoint "${CONVERTED_LLM_CKPT}" \
           2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        RESULTS+=("PASS|${name}|${duration}s")
        PASSED=$((PASSED + 1))
        echo "[PASS] ${name} (${duration}s)"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        RESULTS+=("FAIL|${name}|${duration}s")
        FAILED_CONFIGS+=("${name}")
        echo "[FAIL] ${name} (${duration}s)"
        return 1
    fi
    return 0
}

# Run tests
if [[ -n "${SINGLE_CONFIG}" ]]; then
    # Run single config
    found=false
    for config in "${CONFIGS[@]}"; do
        name="${config%%|*}"
        if [[ "${name}" == "${SINGLE_CONFIG}" ]]; then
            run_config "${config}"
            found=true
            break
        fi
    done
    if [[ "${found}" == "false" ]]; then
        echo "Error: Config '${SINGLE_CONFIG}' not found. Available configs:"
        for config in "${CONFIGS[@]}"; do
            echo "  - ${config%%|*}"
        done
        exit 1
    fi
else
    # Run all configs - abort on any failure
    for config in "${CONFIGS[@]}"; do
        if ! run_config "${config}"; then
            name="${config%%|*}"
            echo ""
            echo "=========================================="
            echo "FATAL: Config '${name}' failed. Aborting."
            echo "=========================================="
            exit 1
        fi
    done
fi

# Print summary
echo ""
echo "=========================================="
echo "SUMMARY: ${PASSED}/${TOTAL} passed"
echo "=========================================="
printf "%-6s | %-35s | %s\n" "Status" "Configuration" "Time"
echo "-------|-------------------------------------|-------"
for result in "${RESULTS[@]}"; do
    IFS='|' read -r status name duration <<< "$result"
    if [[ "${status}" == "PASS" ]]; then
        printf "\033[32m%-6s\033[0m | %-35s | %s\n" "${status}" "${name}" "${duration}"
    else
        printf "\033[31m%-6s\033[0m | %-35s | %s\n" "${status}" "${name}" "${duration}"
    fi
done
echo "=========================================="

if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed configurations:"
    for cfg in "${FAILED_CONFIGS[@]}"; do
        echo "  - ${cfg}"
    done
    exit 1
fi

echo ""
echo "All tests passed!"
exit 0
