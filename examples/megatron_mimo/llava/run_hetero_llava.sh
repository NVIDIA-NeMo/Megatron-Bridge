#!/bin/bash
# Heterogeneous MIMO LLaVA training — LLM on ranks 0-3, CLIP on ranks 4-7.
set -euo pipefail
GPUS_PER_NODE=8
NUM_NODES=1

# Resolve this script's directory so we can locate the HF->Megatron converters
# (convert_hf_clip_to_megatron.py / convert_hf_llama_to_megatron.py) and the
# training entrypoint regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- MIMO parallelism config -------------------------------------------------
# Consumed by megatron_mimo_training_llava.py's _build_parallelism_config() via
# these env vars. The LLM and vision encoder occupy non-overlapping GPU sets
# (heterogeneous layout). Defaults reproduce the 4+4 split: LLM on ranks 0-3,
# CLIP on ranks 4-7. For each module TP*PP*DP must equal its GPU count, the two
# modules must not overlap, and together they must cover all GPUS_PER_NODE ranks.
# rank_offset is the first global rank of each module. Note: the vision encoder
# (CLIPViT) does not support PP > 1.
MIMO_LLM_TP=${MIMO_LLM_TP:-4}
MIMO_LLM_PP=${MIMO_LLM_PP:-1}
MIMO_LLM_DP=${MIMO_LLM_DP:-1}
MIMO_LLM_OFFSET=${MIMO_LLM_OFFSET:-0}
MIMO_VISION_TP=${MIMO_VISION_TP:-4}
MIMO_VISION_PP=${MIMO_VISION_PP:-1}
MIMO_VISION_DP=${MIMO_VISION_DP:-1}
MIMO_VISION_OFFSET=${MIMO_VISION_OFFSET:-4}
export MIMO_LLM_TP MIMO_LLM_PP MIMO_LLM_DP MIMO_LLM_OFFSET
export MIMO_VISION_TP MIMO_VISION_PP MIMO_VISION_DP MIMO_VISION_OFFSET

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

# LLaVA-Pretrain dataset. Empty DATASET_ROOT triggers an auto-download + extract
# into DATASET_DOWNLOAD_DIR (skipped if already present).
DATASET_ROOT=${DATASET_ROOT:-""}
DATASET_DOWNLOAD_DIR=${DATASET_DOWNLOAD_DIR:-/workspace/llava_pretrain}
LLAVA_PRETRAIN_REPO=${LLAVA_PRETRAIN_REPO:-liuhaotian/LLaVA-Pretrain}

# --- Checkpoint conversion config --------------------------------------------
# HuggingFace source models converted to per-TP-rank Megatron checkpoints. The
# converted checkpoints are cached under CHECKPOINT_BASE_DIR keyed by TP size
# and loaded into the LLM / vision encoder (see convert_checkpoints below). The
# correct checkpoint is picked from MIMO_LLM_TP / MIMO_VISION_TP.
HF_VISION_MODEL=${HF_VISION_MODEL:-"openai/clip-vit-large-patch14-336"}
HF_LLM_MODEL=${HF_LLM_MODEL:-"lmsys/vicuna-7b-v1.5"}
MEGATRON_VOCAB_SIZE=${MEGATRON_VOCAB_SIZE:-32256}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/workspace/megatron_mimo_checkpoints}

# Ensure the LLaVA-Pretrain dataset is available. When DATASET_ROOT is empty,
# download liuhaotian/LLaVA-Pretrain (captions JSON + images.zip) into
# DATASET_DOWNLOAD_DIR and extract images.zip there. The JSON's image paths
# (e.g. "00453/004531425.jpg") resolve against DATASET_ROOT, so images.zip is
# extracted directly into it. Both steps are skipped if already present.
prepare_dataset() {
    if [[ -n "${DATASET_ROOT}" ]]; then
        echo "Using DATASET_ROOT: ${DATASET_ROOT}"
        return
    fi

    DATASET_ROOT="${DATASET_DOWNLOAD_DIR}"
    local json_file="${DATASET_ROOT}/blip_laion_cc_sbu_558k.json"
    local images_zip="${DATASET_ROOT}/images.zip"

    # Already prepared: captions JSON + extracted image shards present. This
    # holds even if images.zip was deleted post-extraction, so we skip both the
    # download and the extraction.
    if [[ -f "${json_file}" && -d "${DATASET_ROOT}/00000" ]]; then
        echo "Using cached LLaVA-Pretrain dataset at ${DATASET_ROOT}"
        return
    fi

    echo "DATASET_ROOT not set; preparing ${LLAVA_PRETRAIN_REPO} under ${DATASET_ROOT}"
    mkdir -p "${DATASET_ROOT}"

    if [[ -f "${json_file}" && -f "${images_zip}" ]]; then
        echo "  Using cached download in ${DATASET_ROOT}"
    else
        echo "  Downloading ${LLAVA_PRETRAIN_REPO} (this can take a while)..."
        uv run python - "${LLAVA_PRETRAIN_REPO}" "${DATASET_ROOT}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
PY
    fi

    # images.zip extracts to 5-digit shard dirs (00000, 00001, ...) at the root.
    if [[ -d "${DATASET_ROOT}/00000" ]]; then
        echo "  Images already extracted."
    else
        echo "  Extracting images.zip..."
        unzip -q -o "${images_zip}" -d "${DATASET_ROOT}"
    fi

    echo "  Dataset ready at ${DATASET_ROOT}"
}

# Convert the HF source models into per-TP-rank Megatron checkpoints matching
# the configured LLM and vision TP sizes. Results are cached per TP size (reused
# if already present). PP > 1 needs no separate conversion: the training script
# remaps globally-numbered layer keys to each PP stage's local indices on load.
# Sets CONVERTED_CLIP_CKPT / CONVERTED_LLM_CKPT for the launch below.
convert_checkpoints() {
    local vision_tp="$1"
    local llm_tp="$2"

    local clip_ckpt_dir="${CHECKPOINT_BASE_DIR}/clip_tp${vision_tp}"
    local llm_ckpt_dir="${CHECKPOINT_BASE_DIR}/llm_tp${llm_tp}"

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

    CONVERTED_CLIP_CKPT="${clip_ckpt_dir}"
    CONVERTED_LLM_CKPT="${llm_ckpt_dir}"
}

# Ensure dataset is available (downloads + extracts when DATASET_ROOT is empty)
prepare_dataset

# Convert (or reuse cached) checkpoints for the configured LLM/vision TP sizes
convert_checkpoints "${MIMO_VISION_TP}" "${MIMO_LLM_TP}"

uv run torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    "${SCRIPT_DIR}/megatron_mimo_training_llava.py" \
    --micro-batch-size 4 \
    --global-batch-size 96 \
    --train-iters 100 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad ${CLIP_GRAD} \
    --log-interval 1 \
    --lr ${LR} \
    --lr-warmup-iters 60 \
    --min-lr ${MIN_LR} \
    --weight-decay 0.0 \
    --wandb-project "Megatron-Bridge-MIMO" \
    --wandb-exp-name "mimo-llava-hetero-e2e-${LLM_TAG}-test${EXP_SUFFIX}" \
    --wandb-save-dir "/tmp/wandb" \
    --dataset-root "${DATASET_ROOT}" \
    --vision-encoder-checkpoint "${CONVERTED_CLIP_CKPT}" \
    --language-model-checkpoint "${CONVERTED_LLM_CKPT}" \
    ${DETERMINISTIC_FLAG} \
    --freeze-llm ${FREEZE_LLM} \
