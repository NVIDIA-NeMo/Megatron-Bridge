#!/bin/bash
# Heterogeneous MIMO LLaVA smoke test against the mini audio-augmented dataset.
# LLM on ranks 0-3 (TP=4), CLIP on ranks 4-5 (TP=2), Whisper on ranks 6-7 (TP=2).
#
# Assumes ./prepare_llava_pretrain_audio.sh has been run, or will auto-build via prepare_dataset().

set -euo pipefail

GPUS_PER_NODE=8
NUM_NODES=1

# Resolve this script's directory so we can locate the HF->Megatron converters
# (convert_hf_clip_to_megatron.py / convert_hf_llama_to_megatron.py /
# whisper/convert_hf_whisper_to_megatron.py) and the training entrypoint
# regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- MIMO parallelism config -------------------------------------------------
# Consumed by megatron_mimo_training_llava_audio.py's _build_parallelism_config()
# via these env vars. The LLM, vision encoder, and audio encoder occupy
# non-overlapping GPU sets (heterogeneous layout). Defaults reproduce the
# 4+2+2 split: LLM on ranks 0-3 (TP=4), CLIP on ranks 4-5 (TP=2), Whisper on
# ranks 6-7 (TP=2). For each module TP*PP*DP must equal its GPU count, the three
# modules must not overlap, and together they must cover all GPUS_PER_NODE ranks.
# rank_offset is the first global rank of each module. Note: the vision encoder
# (CLIPViT) and audio encoder (Whisper) do not support PP > 1.
MIMO_LLM_TP=${MIMO_LLM_TP:-4}
MIMO_LLM_PP=${MIMO_LLM_PP:-1}
MIMO_LLM_DP=${MIMO_LLM_DP:-1}
MIMO_LLM_OFFSET=${MIMO_LLM_OFFSET:-0}
MIMO_VISION_TP=${MIMO_VISION_TP:-2}
MIMO_VISION_PP=${MIMO_VISION_PP:-1}
MIMO_VISION_DP=${MIMO_VISION_DP:-1}
MIMO_VISION_OFFSET=${MIMO_VISION_OFFSET:-4}
MIMO_AUDIO_TP=${MIMO_AUDIO_TP:-2}
MIMO_AUDIO_PP=${MIMO_AUDIO_PP:-1}
MIMO_AUDIO_DP=${MIMO_AUDIO_DP:-1}
MIMO_AUDIO_OFFSET=${MIMO_AUDIO_OFFSET:-6}
export MIMO_LLM_TP MIMO_LLM_PP MIMO_LLM_DP MIMO_LLM_OFFSET
export MIMO_VISION_TP MIMO_VISION_PP MIMO_VISION_DP MIMO_VISION_OFFSET
export MIMO_AUDIO_TP MIMO_AUDIO_PP MIMO_AUDIO_DP MIMO_AUDIO_OFFSET

# Set DETERMINISTIC=1 to export deterministic NCCL/CUBLAS/cuDNN/TE env vars
# AND pass --deterministic to the training script (FP32, unfused attention, etc.).
# Also disables gradient clipping (clip-grad=0.0), which is non-associative under
# distributed reductions and introduces run-to-run variance.
DETERMINISTIC=${DETERMINISTIC:-0}
# Set UNFREEZE_LLM=1 to train the language model
UNFREEZE_LLM=${UNFREEZE_LLM:-0}
FREEZE_LLM=$([[ "${UNFREEZE_LLM}" == "1" ]] && echo "false" || echo "true")
LLM_TAG=$([[ "${UNFREEZE_LLM}" == "1" ]] && echo "unfrozen-llm" || echo "frozen-llm")
LR=$([[ "${UNFREEZE_LLM}" == "1" ]] && echo "1.0e-4" || echo "1e-3")
MIN_LR=$([[ "${UNFREEZE_LLM}" == "1" ]] && echo "1.0e-5" || echo "2.0e-5")
DETERMINISTIC_FLAG=""
EXP_SUFFIX=""
CLIP_GRAD=1.0
if [[ "${DETERMINISTIC}" == "1" ]]; then
    DETERMINISTIC_FLAG="--deterministic"
    EXP_SUFFIX="-fp32"
    CLIP_GRAD=0.0
    # Pin Ring algorithm for deterministic reduction order.
    # Tree is faster for some message sizes but NCCL 2.28 Tree doesn't support
    # AllGather with Int8 (used by torch.distributed.all_gather_object), and
    # letting NCCL choose per-operation (^NVLS) still leaves Tree/Ring selection
    # non-deterministic.  Ring supports all collective ops.
    export NCCL_ALGO=Ring
    export NCCL_PROTO=Simple
    # Disable NCCL's topology-aware optimizations that can change paths between runs
    export NCCL_TUNER_PLUGIN=""
    # For full CUDA-level determinism
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    # Force deterministic cuDNN attention (disable non-deterministic workspace)
    export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
    # Required by Transformer Engine when deterministic_mode=True
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
fi

# Audio-augmented dataset. Empty DATASET_ROOT triggers a build via
# prepare_llava_pretrain_audio.sh into AUDIO_DATASET_DIR (TTS synthesis over
# LLaVA-Pretrain; resume-safe, skipped if already built). Set DATASET_ROOT to an
# existing audio-augmented tree to skip the build. The audio encoder is
# exercised only when AUDIO_COLUMN is non-empty.
DATASET_ROOT=${DATASET_ROOT:-""}
AUDIO_DATASET_DIR=${AUDIO_DATASET_DIR:-/workspace/llava_pretrain_audio_augmented}
HF_DATA_FILES=${HF_DATA_FILES:-blip_laion_cc_sbu_558k_with_audio.json}
AUDIO_COLUMN=${AUDIO_COLUMN:-audio}

# Training sizing (overridable), used both in the torchrun launch below and to
# size the audio build: only train_iters * global_batch_size samples are consumed
# over the whole run, so synthesize audio for just that many records via LIMIT
# instead of all 558k. Override LIMIT to force a specific count.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-96}
TRAIN_ITERS=${TRAIN_ITERS:-100}
LIMIT=${LIMIT:-$((TRAIN_ITERS * GLOBAL_BATCH_SIZE))}

# Ensure the audio-augmented dataset is available. When DATASET_ROOT is empty,
# run prepare_llava_pretrain_audio.sh (which auto-downloads LLaVA-Pretrain as
# needed, synthesizes per-sample audio, and merges ${HF_DATA_FILES}) into
# AUDIO_DATASET_DIR, then point DATASET_ROOT at it. Skipped when the merged JSON
# is already present. Pass DATASET_DOWNLOAD_DIR/LIMIT/NUM_SHARDS/TTS_* through the environment to
# tune the build (see prepare_llava_pretrain_audio.sh).
prepare_dataset() {
    if [[ -n "${DATASET_ROOT}" ]]; then
        echo "Using DATASET_ROOT: ${DATASET_ROOT}"
        return
    fi

    DATASET_ROOT="${AUDIO_DATASET_DIR}"
    if [[ -f "${DATASET_ROOT}/${HF_DATA_FILES}" ]]; then
        echo "Using cached audio-augmented dataset at ${DATASET_ROOT}"
        return
    fi

    echo "DATASET_ROOT not set; building audio-augmented dataset (LIMIT=${LIMIT}) under ${DATASET_ROOT}"
    AUGMENTED_DATASET_DIR="${DATASET_ROOT}" LIMIT="${LIMIT}" "${SCRIPT_DIR}/prepare_llava_pretrain_audio.sh"
    echo "  Audio-augmented dataset ready at ${DATASET_ROOT}"
}

# --- Checkpoint conversion config --------------------------------------------
# HuggingFace source models converted to per-TP-rank Megatron checkpoints. The
# converted checkpoints are cached under CHECKPOINT_BASE_DIR keyed by TP size
# and loaded into the LLM / vision encoder / audio encoder (see
# convert_checkpoints below). The correct checkpoint is picked from
# MIMO_LLM_TP / MIMO_VISION_TP / MIMO_AUDIO_TP.
HF_VISION_MODEL=${HF_VISION_MODEL:-"openai/clip-vit-large-patch14-336"}
HF_LLM_MODEL=${HF_LLM_MODEL:-"lmsys/vicuna-7b-v1.5"}
HF_AUDIO_MODEL=${HF_AUDIO_MODEL:-"openai/whisper-base"}
MEGATRON_VOCAB_SIZE=${MEGATRON_VOCAB_SIZE:-32256}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/workspace/megatron_mimo_checkpoints}

# Convert the HF source models into per-TP-rank Megatron checkpoints matching
# the configured LLM, vision, and audio TP sizes. Results are cached per TP size
# (reused if already present). PP > 1 needs no separate conversion: the training
# script remaps globally-numbered layer keys to each PP stage's local indices on
# load. Sets CONVERTED_{CLIP,LLM,WHISPER}_CKPT for the launch below.
convert_checkpoints() {
    local vision_tp="$1"
    local llm_tp="$2"
    local audio_tp="$3"

    local clip_ckpt_dir="${CHECKPOINT_BASE_DIR}/clip_tp${vision_tp}"
    local llm_ckpt_dir="${CHECKPOINT_BASE_DIR}/llm_tp${llm_tp}"
    local whisper_ckpt_dir="${CHECKPOINT_BASE_DIR}/whisper_tp${audio_tp}"

    if [[ ! -d "${clip_ckpt_dir}/tp_rank_00" ]]; then
        echo "Converting CLIP checkpoint (TP=${vision_tp})..."
        uv run python "${SCRIPT_DIR}/convert_hf_clip_to_megatron.py" \
            --hf-model "${HF_VISION_MODEL}" \
            --output "${clip_ckpt_dir}" \
            --tensor-parallel-size "${vision_tp}" \
            --use-te
    else
        echo "Using cached CLIP checkpoint: ${clip_ckpt_dir}"
    fi

    if [[ ! -d "${llm_ckpt_dir}/tp_rank_00" ]]; then
        echo "Converting LLM checkpoint (TP=${llm_tp})..."
        uv run python "${SCRIPT_DIR}/convert_hf_llama_to_megatron.py" \
            --hf-model "${HF_LLM_MODEL}" \
            --output "${llm_ckpt_dir}" \
            --tensor-parallel-size "${llm_tp}" \
            --use-te \
            --megatron-vocab-size "${MEGATRON_VOCAB_SIZE}"
    else
        echo "Using cached LLM checkpoint: ${llm_ckpt_dir}"
    fi

    if [[ ! -d "${whisper_ckpt_dir}/tp_rank_00" ]]; then
        echo "Converting Whisper checkpoint (TP=${audio_tp})..."
        uv run python "${SCRIPT_DIR}/whisper/convert_hf_whisper_to_megatron.py" \
            --hf-model "${HF_AUDIO_MODEL}" \
            --output "${whisper_ckpt_dir}" \
            --tensor-parallel-size "${audio_tp}" \
            --use-te
    else
        echo "Using cached Whisper checkpoint: ${whisper_ckpt_dir}"
    fi

    CONVERTED_CLIP_CKPT="${clip_ckpt_dir}"
    CONVERTED_LLM_CKPT="${llm_ckpt_dir}"
    CONVERTED_WHISPER_CKPT="${whisper_ckpt_dir}"
}

# Ensure dataset is available (builds the audio-augmented tree when DATASET_ROOT is empty)
prepare_dataset

# Convert (or reuse cached) checkpoints for the configured LLM/vision/audio TP sizes
convert_checkpoints "${MIMO_VISION_TP}" "${MIMO_LLM_TP}" "${MIMO_AUDIO_TP}"

uv run torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    "${SCRIPT_DIR}/megatron_mimo_training_llava_audio.py" \
    --micro-batch-size 4 \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters "${TRAIN_ITERS}" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad ${CLIP_GRAD} \
    --log-interval 1 \
    --lr ${LR} \
    --lr-warmup-iters 60 \
    --min-lr ${MIN_LR} \
    --weight-decay 0.0 \
    --wandb-project "Megatron-Bridge-MIMO" \
    --wandb-exp-name "mimo-llava-audio-hetero-e2e-${LLM_TAG}-test${EXP_SUFFIX}" \
    --wandb-save-dir "/tmp/wandb" \
    --dataset-root "${DATASET_ROOT}" \
    --hf-data-files "${HF_DATA_FILES}" \
    --audio-column "${AUDIO_COLUMN}" \
    --freeze-llm ${FREEZE_LLM} \
    ${DETERMINISTIC_FLAG} \
    --vision-encoder-checkpoint "${CONVERTED_CLIP_CKPT}" \
    --language-model-checkpoint "${CONVERTED_LLM_CKPT}" \
    --audio-encoder-checkpoint "${CONVERTED_WHISPER_CKPT}"
