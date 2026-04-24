#!/bin/bash
# Run heterogeneous MIMO LLaVA+audio E2E test with various parallelism configurations
# Usage: ./run_hetero_llava_audio_parallelism_tests.sh [--gpus N] [--config CONFIG_NAME]
#
# Examples:
#   ./run_hetero_llava_audio_parallelism_tests.sh                              # Run all configs with 8 GPUs
#   ./run_hetero_llava_audio_parallelism_tests.sh --config tp4_llm_tp2_vis_tp2_aud  # Run a single config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/megatron_mimo_training_llava_audio.py"

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
CLIP_GRAD=${CLIP_GRAD:-0.0}
LOG_INTERVAL=${LOG_INTERVAL:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"Megatron-Bridge-MIMO"}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-"/tmp/wandb"}
DATASET_ROOT=${DATASET_ROOT:-"/path/to/LLaVA-Pretrain-Audio-Augmented"}
UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv_cache/}

# HuggingFace source models for checkpoint conversion
HF_VISION_MODEL=${HF_VISION_MODEL:-"openai/clip-vit-large-patch14-336"}
HF_LLM_MODEL=${HF_LLM_MODEL:-"lmsys/vicuna-7b-v1.5"}
HF_AUDIO_MODEL=${HF_AUDIO_MODEL:-"openai/whisper-base"}
MEGATRON_VOCAB_SIZE=${MEGATRON_VOCAB_SIZE:-32256}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/workspace/megatron_mimo_checkpoints}

# Audio-augmented dataset (set by prepare_llava_pretrain_audio.sh). The audio
# encoder is only exercised when AUDIO_COLUMN is non-empty.
HF_DATA_FILES=${HF_DATA_FILES:-blip_laion_cc_sbu_558k_with_audio.json}
AUDIO_COLUMN=${AUDIO_COLUMN:-audio}

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

# Define configurations as:
#   "name|llm_tp|llm_pp|llm_dp|llm_offset|vision_tp|vision_pp|vision_dp|vision_offset|audio_tp|audio_pp|audio_dp|audio_offset|mbs"
# Notes:
#   - Vision encoder (CLIPViT) and audio encoder (Whisper) do not support PP > 1
#   - Modules occupy non-overlapping GPU sets; offsets are the first rank of each module
#   - MBS must be divisible by every module's DP size (enforced by build_megatron_mimo_data_loaders)
#   - Encoder DP must be >= LLM DP (required for embedding alignment across batches)
#   - Total GPUs = llm_tp*llm_pp*llm_dp + vision_tp*vision_pp*vision_dp + audio_tp*audio_pp*audio_dp

declare -a CONFIGS_8GPU=(
    # LLM on 4 GPUs + vision/audio 2+2 split on remaining 4 (offsets: LLM=0, vision=4, audio=6)
    "tp4_llm_tp2_vis_tp2_aud|4|1|1|0|2|1|1|4|2|1|1|6|4"
    "tp4_llm_dp2_vis_dp2_aud|4|1|1|0|1|1|2|4|1|1|2|6|2"
    "tp2_dp2_llm_dp2_vis_dp2_aud|2|1|2|0|1|1|2|4|1|1|2|6|2"
    "tp2_pp2_llm_tp2_vis_tp2_aud|2|2|1|0|2|1|1|4|2|1|1|6|4"
    "pp4_llm_tp2_vis_tp2_aud|1|4|1|0|2|1|1|4|2|1|1|6|4"
    "pp4_llm_dp2_vis_dp2_aud|1|4|1|0|1|1|2|4|1|1|2|6|2"
    "pp2_dp2_llm_dp2_vis_dp2_aud|1|2|2|0|1|1|2|4|1|1|2|6|2"
    # Asymmetric: LLM on 2 GPUs + vision/audio share remaining 6 (LLM=0, vision=2, audio=2+vision_size)
    # LLM tp2
    "asym_tp2_llm_tp4_vis_tp2_aud|2|1|1|0|4|1|1|2|2|1|1|6|4"
    "asym_tp2_llm_tp2_vis_tp4_aud|2|1|1|0|2|1|1|2|4|1|1|4|4"
    "asym_tp2_llm_dp3_vis_dp3_aud|2|1|1|0|1|1|3|2|1|1|3|5|3"
    "asym_tp2_llm_dp4_vis_tp2_aud|2|1|1|0|1|1|4|2|2|1|1|6|4"
    "asym_tp2_llm_dp2_vis_dp4_aud|2|1|1|0|1|1|2|2|1|1|4|4|4"
    # LLM pp2
    "asym_pp2_llm_tp4_vis_tp2_aud|1|2|1|0|4|1|1|2|2|1|1|6|4"
    "asym_pp2_llm_tp2_vis_tp4_aud|1|2|1|0|2|1|1|2|4|1|1|4|4"
    "asym_pp2_llm_dp3_vis_dp3_aud|1|2|1|0|1|1|3|2|1|1|3|5|3"
    "asym_pp2_llm_dp4_vis_tp2_aud|1|2|1|0|1|1|4|2|2|1|1|6|4"
)

CONFIGS=("${CONFIGS_8GPU[@]}")

# Track results
declare -a RESULTS=()
declare -a FAILED_CONFIGS=()
TOTAL=0
PASSED=0

convert_checkpoints() {
    local vision_tp="$1"
    local llm_tp="$2"
    local audio_tp="$3"

    local clip_ckpt_dir="${CHECKPOINT_BASE_DIR}/clip_tp${vision_tp}"
    local llm_ckpt_dir="${CHECKPOINT_BASE_DIR}/llm_tp${llm_tp}"
    local whisper_ckpt_dir="${CHECKPOINT_BASE_DIR}/whisper_tp${audio_tp}"

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

    # Convert Whisper checkpoint if not already cached for this TP size
    if [[ ! -d "${whisper_ckpt_dir}/tp_rank_00" ]]; then
        echo "  Converting Whisper checkpoint (TP=${audio_tp})..."
        uv run python "${SCRIPT_DIR}/whisper/convert_hf_whisper_to_megatron.py" \
            --hf-model "${HF_AUDIO_MODEL}" \
            --output "${whisper_ckpt_dir}" \
            --tensor-parallel-size "${audio_tp}" \
            --use-te
    else
        echo "  Using cached Whisper checkpoint: ${whisper_ckpt_dir}"
    fi

    # Return paths via global variables
    CONVERTED_CLIP_CKPT="${clip_ckpt_dir}"
    CONVERTED_LLM_CKPT="${llm_ckpt_dir}"
    CONVERTED_WHISPER_CKPT="${whisper_ckpt_dir}"
}

build_wandb_exp_name() {
    local name="$1"
    local llm_tp="$2" llm_pp="$3" llm_dp="$4"
    local vision_tp="$5" vision_pp="$6" vision_dp="$7"
    local audio_tp="$8" audio_pp="$9" audio_dp="${10}"
    local mbs="${11}"

    echo "hetero-llava-${name}-${NUM_GPUS}gpu-llm_tp${llm_tp}_pp${llm_pp}_dp${llm_dp}-vis_tp${vision_tp}_pp${vision_pp}_dp${vision_dp}-aud_tp${audio_tp}_pp${audio_pp}_dp${audio_dp}-mbs${mbs}"
}

run_config() {
    local config="$1"
    local name llm_tp llm_pp llm_dp llm_offset
    local vision_tp vision_pp vision_dp vision_offset
    local audio_tp audio_pp audio_dp audio_offset mbs

    IFS='|' read -r name \
        llm_tp llm_pp llm_dp llm_offset \
        vision_tp vision_pp vision_dp vision_offset \
        audio_tp audio_pp audio_dp audio_offset \
        mbs <<< "$config"

    local exp_name
    exp_name=$(build_wandb_exp_name "${name}" \
        "${llm_tp}" "${llm_pp}" "${llm_dp}" \
        "${vision_tp}" "${vision_pp}" "${vision_dp}" \
        "${audio_tp}" "${audio_pp}" "${audio_dp}" \
        "${mbs}")

    echo ""
    echo "----------------------------------------"
    echo "Running: ${name}"
    echo "  LLM:    TP=${llm_tp}, PP=${llm_pp}, DP=${llm_dp}, offset=${llm_offset}"
    echo "  Vision: TP=${vision_tp}, PP=${vision_pp}, DP=${vision_dp}, offset=${vision_offset}"
    echo "  Audio:  TP=${audio_tp}, PP=${audio_pp}, DP=${audio_dp}, offset=${audio_offset}"
    echo "  MBS:    ${mbs}"
    echo "  W&B:    ${exp_name}"
    echo "----------------------------------------"

    TOTAL=$((TOTAL + 1))

    # Convert checkpoints for this config's TP sizes
    convert_checkpoints "${vision_tp}" "${llm_tp}" "${audio_tp}"

    local start_time=$(date +%s)

    if MIMO_LLM_TP="${llm_tp}" \
       MIMO_LLM_PP="${llm_pp}" \
       MIMO_LLM_DP="${llm_dp}" \
       MIMO_LLM_OFFSET="${llm_offset}" \
       MIMO_VISION_TP="${vision_tp}" \
       MIMO_VISION_PP="${vision_pp}" \
       MIMO_VISION_DP="${vision_dp}" \
       MIMO_VISION_OFFSET="${vision_offset}" \
       MIMO_AUDIO_TP="${audio_tp}" \
       MIMO_AUDIO_PP="${audio_pp}" \
       MIMO_AUDIO_DP="${audio_dp}" \
       MIMO_AUDIO_OFFSET="${audio_offset}" \
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
           --hf-data-files "${HF_DATA_FILES}" \
           --audio-column "${AUDIO_COLUMN}" \
           --vision-encoder-checkpoint "${CONVERTED_CLIP_CKPT}" \
           --language-model-checkpoint "${CONVERTED_LLM_CKPT}" \
           --audio-encoder-checkpoint "${CONVERTED_WHISPER_CKPT}" \
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
