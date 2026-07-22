#!/bin/bash
# =============================================================================
# Gemma-4 E4B Full Pipeline: HF → Convert → Parity Check → Training
#
# Usage (from Megatron-Bridge root):
#   NVIDIA_VISIBLE_DEVICES=0,1 bash examples/models/gemma/gemma4/slurm_pretrain.sh
#
# Key overrides:
#   HF_MODEL_DIR        : path to downloaded HF model  (default: ~/models/gemma-4-E4B-it)
#   MEGATRON_CKPT       : base path for converted checkpoints
#                         → text checkpoint: ${MEGATRON_CKPT}-text
#                         → vl/audio checkpoint: ${MEGATRON_CKPT}-vl
#   TRAIN_DATA_PATH     : data prefix for training  (required for real training)
#   SAVE_DIR            : where to save training checkpoints
#   SKIP_CONVERT        : set to 1 to skip BOTH conversions
#   SKIP_TEXT_CONVERT   : set to 1 to skip only the text conversion
#   SKIP_VL_CONVERT     : set to 1 to skip only the vl/audio conversion
#   SKIP_PARITY         : set to 1 to skip all parity checks
#   TRAIN_ITERS         : number of training iterations (default: 1000)
#   SEQ_LENGTH          : sequence length (default: 4096)
#
# Parity checks run for all three modalities automatically:
#   text  → TEXT_CKPT: text tokens, compares GPTModel vs HF CausalLM
#   vl    → VL_CKPT:  image tokens + patch tensor, compares full image forward
#   audio → VL_CKPT:  audio tokens + mel-spectrogram, compares full audio forward
#
# Example:
#   HF_MODEL_DIR=/path/to/gemma-4-E4B-it \
#   MEGATRON_CKPT=/path/to/gemma4-e4b-megatron \
#   NVIDIA_VISIBLE_DEVICES=0,1 bash examples/models/gemma/gemma4/slurm_pretrain.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
MEGATRON_LM_ROOT=${MEGATRON_LM_ROOT:-$(cd "$BRIDGE_ROOT/../Megatron-LM" 2>/dev/null && pwd)}

if [ ! -f "$MEGATRON_LM_ROOT/pretrain_gpt.py" ]; then
    echo "Error: Megatron-LM root not found: $MEGATRON_LM_ROOT"
    echo "Set MEGATRON_LM_ROOT=/path/to/Megatron-LM"
    exit 1
fi

export MEGATRON_LM_ROOT
export PYTHONPATH="$BRIDGE_ROOT/src:$MEGATRON_LM_ROOT:${PYTHONPATH:-}"
cd "$MEGATRON_LM_ROOT"

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
HF_MODEL_DIR=${HF_MODEL_DIR:-$HOME/models/gemma-4-E4B-it}
MEGATRON_CKPT=${MEGATRON_CKPT:-$HOME/checkpoints/gemma4-e4b-megatron}
SAVE_DIR=${SAVE_DIR:-$HOME/checkpoints/gemma4-e4b-finetune}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-}

# Derived checkpoint paths (text-only for training, vl for multi-modal parity)
TEXT_CKPT="${MEGATRON_CKPT}-text"
VL_CKPT="${MEGATRON_CKPT}-vl"

# Pipeline control
SKIP_CONVERT=${SKIP_CONVERT:-1}
SKIP_TEXT_CONVERT=${SKIP_TEXT_CONVERT:-${SKIP_CONVERT}}
SKIP_VL_CONVERT=${SKIP_VL_CONVERT:-${SKIP_CONVERT}}
SKIP_PARITY=${SKIP_PARITY:-0}

# Hardware
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
TP_SIZE=2
PP_SIZE=1
MASTER_PORT=${MASTER_PORT:-6200}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Training hyperparameters
TRAIN_ITERS=${TRAIN_ITERS:-1000}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
LR=${LR:-2e-5}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "Error: HF model not found at $HF_MODEL_DIR"
    echo "  Download with: hf download google/gemma-4-E4B-it --local-dir $HF_MODEL_DIR"
    exit 1
fi

TORCHRUN_BIN=${TORCHRUN_BIN:-"uv run python -m torch.distributed.run"}

echo ""
echo "========================================"
echo "  Gemma-4 E4B Pipeline"
echo "  bridge      : $BRIDGE_ROOT"
echo "  mcore       : $MEGATRON_LM_ROOT"
echo "  hf_model    : $HF_MODEL_DIR"
echo "  text_ckpt   : $TEXT_CKPT"
echo "  vl_ckpt     : $VL_CKPT"
echo "  save_dir    : $SAVE_DIR"
echo "  gpus        : $GPUS_PER_NODE  TP=$TP_SIZE  PP=$PP_SIZE"
echo "  train_iters : $TRAIN_ITERS  seq=$SEQ_LENGTH"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: run one conversion
# ---------------------------------------------------------------------------
_convert() {
    local mode="$1"
    local ckpt_path="$2"
    local port="$3"
    echo "  Converting in mode='${mode}' → ${ckpt_path}"
    mkdir -p "$ckpt_path"
    GEMMA4_CONVERSION_MODE="$mode" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 $TORCHRUN_BIN \
        --nproc_per_node $TP_SIZE \
        --nnodes 1 --node_rank 0 \
        --master_addr localhost \
        --master_port "$port" \
        "$BRIDGE_ROOT/scripts/conversion/run_conversion.py" import \
        --device gpu \
        --hf-model "$HF_MODEL_DIR" \
        --megatron-path "$ckpt_path" \
        --tp $TP_SIZE \
        --pp $PP_SIZE \
        --torch-dtype bfloat16 \
        --distributed-timeout-minutes 30
    echo "  Conversion done → $ckpt_path"
}

# ---------------------------------------------------------------------------
# Helper: run one parity check
# ---------------------------------------------------------------------------
_parity() {
    local mode="$1"
    local ckpt_path="$2"
    local port="$3"
    local log_dir="${GEMMA4_LOG_ROOT:?'Error: set GEMMA4_LOG_ROOT to a writable log directory'}/gemma4_e4b_parity_${mode}"
    # VL image parity runs through a much longer bf16 path (280 image tokens),
    # so it uses a wider tolerance than text/audio.
    local atol=3.0
    [ "$mode" = "vl" ] && atol=6.0
    echo ""
    echo "  ── Parity [${mode^^}] against $ckpt_path (atol=${atol}) ──"
    $TORCHRUN_BIN \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes 1 --node_rank 0 \
        --master_addr localhost \
        --master_port "$port" \
        --log_dir "$log_dir" \
        --redirects 3 --tee 3 \
        "$SCRIPT_DIR/parity_check_e4b.py" \
        --hf-dir "$HF_MODEL_DIR" \
        --megatron-ckpt "$ckpt_path" \
        --tp $TP_SIZE --bf16 \
        --mode "$mode" \
        --atol "$atol"
    echo "  Parity [${mode^^}] PASSED"
}

# ---------------------------------------------------------------------------
# STEP 1a: Convert HF → Megatron (text-only, used for training)
# ---------------------------------------------------------------------------
echo "========================================"
echo "  Step 1a: Convert HF → Megatron (text mode, TP=$TP_SIZE)"
echo "========================================"

if [ "${SKIP_TEXT_CONVERT}" = "1" ] && \
   [ -f "${TEXT_CKPT}/latest_checkpointed_iteration.txt" ]; then
    echo "  Skipping: text checkpoint already exists at $TEXT_CKPT"
else
    _convert "text" "$TEXT_CKPT" $((MASTER_PORT + 10))
fi

# ---------------------------------------------------------------------------
# STEP 1b: Convert HF → Megatron (vl/audio mode, used for multi-modal parity)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Step 1b: Convert HF → Megatron (audio mode, TP=$TP_SIZE)"
echo "========================================"

if [ "${SKIP_VL_CONVERT}" = "1" ] && \
   [ -f "${VL_CKPT}/latest_checkpointed_iteration.txt" ]; then
    echo "  Skipping: vl checkpoint already exists at $VL_CKPT"
else
    _convert "audio" "$VL_CKPT" $((MASTER_PORT + 12))
fi

# ---------------------------------------------------------------------------
# STEP 2: Parity checks — all three modalities
#
# Modality-specific inputs:
#   text  : text tokens [0, 1, …, SEQ-1]
#   vl    : [image_token_id]*280 + 4 text tokens, patch tensor [1, 2520, 768]
#   audio : [audio_token_id]*12 + text tokens, mel-spectrogram [1, 48, 128]
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Step 2: Parity Checks (all 3 modalities)"
echo "========================================"

if [ "${SKIP_PARITY}" = "1" ]; then
    echo "  Skipping all parity checks."
else
    _parity "text"  "$TEXT_CKPT" $((MASTER_PORT + 1))
    _parity "vl"    "$VL_CKPT"   $((MASTER_PORT + 3))
    _parity "audio" "$VL_CKPT"   $((MASTER_PORT + 5))
    echo ""
    echo "  All parity checks PASSED"
fi

# ---------------------------------------------------------------------------
# STEP 3: Fine-tuning via run_recipe.py + gemma4_e4b_pretrain_config
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Step 3: Training ($TRAIN_ITERS iters)"
echo "========================================"

mkdir -p "$SAVE_DIR"
TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-${GEMMA4_LOG_ROOT:?'Error: set GEMMA4_LOG_ROOT to a writable log directory'}/gemma4_e4b_train_logs}
rm -rf "$TRAIN_LOG_DIR" && mkdir -p "$TRAIN_LOG_DIR"

if [ -n "$TRAIN_DATA_PATH" ]; then
    DATASET_TYPE="megatron-indexed"
    DATA_OVERRIDES=(
        "dataset.blend=[[$TRAIN_DATA_PATH],null]"
        "tokenizer.tokenizer_type=HuggingFaceTokenizer"
        "tokenizer.tokenizer_model=$HF_MODEL_DIR"
    )
else
    echo "  WARNING: TRAIN_DATA_PATH not set, using mock data."
    DATASET_TYPE="mock"
    DATA_OVERRIDES=()
fi

$TORCHRUN_BIN \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 --node_rank 0 \
    --master_addr localhost \
    --master_port $MASTER_PORT \
    --log_dir "$TRAIN_LOG_DIR" \
    --redirects 3 --tee 3 \
    "$BRIDGE_ROOT/scripts/training/run_recipe.py" \
    --recipe gemma4_e4b_pretrain_config \
    --dataset "$DATASET_TYPE" \
    "checkpoint.pretrained_checkpoint=$TEXT_CKPT" \
    "checkpoint.save=$SAVE_DIR" \
    "train.train_iters=$TRAIN_ITERS" \
    "model.seq_length=$SEQ_LENGTH" \
    "dataset.seq_length=$SEQ_LENGTH" \
    "${DATA_OVERRIDES[@]}"

echo ""
echo "========================================"
echo "  Training complete."
echo "  Checkpoints saved to: $SAVE_DIR"
echo "========================================"
