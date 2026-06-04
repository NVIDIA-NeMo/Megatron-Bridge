#!/bin/bash
# =============================================================================
# Gemma-4 E4B Full Pipeline: HF → Convert → Parity Check → Training
#
# Usage (from Megatron-Bridge root):
#   NVIDIA_VISIBLE_DEVICES=0,1 bash examples/models/gemma/gemma4/train_gemma4_e4b_pipeline.sh
#
# Key overrides:
#   HF_MODEL_DIR     : path to downloaded HF model  (default: ~/models/gemma-4-E4B-it)
#   MEGATRON_CKPT    : where to save the converted checkpoint
#   TRAIN_DATA_PATH  : data prefix for training     (required for real training)
#   SAVE_DIR         : where to save training checkpoints
#   SKIP_CONVERT     : set to 1 to skip conversion if checkpoint already exists
#   SKIP_PARITY      : set to 1 to skip parity check
#   TRAIN_ITERS      : number of training iterations (default: 1000)
#   SEQ_LENGTH       : sequence length (default: 4096)
#
# Example:
#   HF_MODEL_DIR=/path/to/gemma-4-E4B-it \
#   MEGATRON_CKPT=/path/to/gemma4-e4b-megatron \
#   TRAIN_DATA_PATH=/mnt/nvme0/data/train \
#   SAVE_DIR=/path/to/gemma4-e4b-finetune \
#   NVIDIA_VISIBLE_DEVICES=0,1 bash examples/models/gemma/gemma4/train_gemma4_e4b_pipeline.sh
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
export PYTHONPATH="$BRIDGE_ROOT/src:$SCRIPT_DIR:$MEGATRON_LM_ROOT:$MEGATRON_LM_ROOT/tools/checkpoint:${PYTHONPATH:-}"
cd "$MEGATRON_LM_ROOT"

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
HF_MODEL_DIR=${HF_MODEL_DIR:-$HOME/models/gemma-4-E4B-it}
MEGATRON_CKPT=${MEGATRON_CKPT:-$HOME/checkpoints/gemma4-e4b-megatron}
SAVE_DIR=${SAVE_DIR:-$HOME/checkpoints/gemma4-e4b-finetune}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-}  # e.g. /mnt/data/train_text_document

# Pipeline control
SKIP_CONVERT=${SKIP_CONVERT:-0}
SKIP_PARITY=${SKIP_PARITY:-0}

# Hardware
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
TP_SIZE=2
PP_SIZE=1
MASTER_PORT=${MASTER_PORT:-6200}

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
    echo "  Download with: huggingface-cli download google/gemma-4-E4B-it --local-dir $HF_MODEL_DIR"
    exit 1
fi

TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}

echo ""
echo "========================================"
echo "  Gemma-4 E4B Pipeline"
echo "  bridge      : $BRIDGE_ROOT"
echo "  mcore       : $MEGATRON_LM_ROOT"
echo "  hf_model    : $HF_MODEL_DIR"
echo "  megatron_ck : $MEGATRON_CKPT"
echo "  save_dir    : $SAVE_DIR"
echo "  gpus        : $GPUS_PER_NODE  TP=$TP_SIZE  PP=$PP_SIZE"
echo "  train_iters : $TRAIN_ITERS  seq=$SEQ_LENGTH"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# STEP 1: Convert HF checkpoint → Megatron format
# ---------------------------------------------------------------------------
echo "========================================"
echo "  Step 1: Convert HF → Megatron (TP=$TP_SIZE)"
echo "========================================"

if [ "${SKIP_CONVERT}" = "1" ] && [ -f "$MEGATRON_CKPT/latest_checkpointed_iteration.txt" ]; then
    echo "  Skipping: checkpoint already exists at $MEGATRON_CKPT"
else
    mkdir -p "$MEGATRON_CKPT"
    CUDA_DEVICE_MAX_CONNECTIONS=1 python "$MEGATRON_LM_ROOT/tools/checkpoint/convert.py" \
        --model-type GPT \
        --loader gemma4_hf \
        --saver core \
        --load-dir "$HF_MODEL_DIR" \
        --save-dir "$MEGATRON_CKPT" \
        --model-size gemma4-e4b \
        --tokenizer-model "$HF_MODEL_DIR" \
        --bf16 \
        --target-tensor-parallel-size $TP_SIZE \
        --target-pipeline-parallel-size $PP_SIZE \
        --no-checking

    echo "  Conversion done → $MEGATRON_CKPT"
fi

# ---------------------------------------------------------------------------
# STEP 2: Parity check (verify conversion correctness)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Step 2: Parity Check (HF vs Megatron)"
echo "========================================"

if [ "${SKIP_PARITY}" = "1" ]; then
    echo "  Skipping parity check."
else
    PARITY_LOG=/tmp/gemma4_e4b_parity_logs
    $TORCHRUN_BIN \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes 1 --node_rank 0 \
        --master_addr localhost \
        --master_port $((MASTER_PORT + 1)) \
        --log_dir "$PARITY_LOG" \
        --redirects 3 --tee 3 \
        "$SCRIPT_DIR/parity_check_e4b.py" \
        --hf-dir "$HF_MODEL_DIR" \
        --megatron-ckpt "$MEGATRON_CKPT" \
        --tp $TP_SIZE --bf16 \
        --atol 3.0  # bf16 + 42 layers: expected max diff ~3.0

    echo "  Parity check PASSED"
fi

# ---------------------------------------------------------------------------
# STEP 3: Fine-tuning
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Step 3: Training ($TRAIN_ITERS iters)"
echo "========================================"

mkdir -p "$SAVE_DIR"
TRAIN_LOG_DIR=/tmp/gemma4_e4b_train_logs
rm -rf "$TRAIN_LOG_DIR" && mkdir -p "$TRAIN_LOG_DIR"

# Model architecture (Gemma-4 E4B)
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 42
    --hidden-size 2560
    --ffn-hidden-size 10240
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --global-kv-channels 512
    --num-global-query-groups 2

    --seq-length $SEQ_LENGTH
    --max-position-embeddings 131072

    --position-embedding-type rope
    --rotary-percent 1.0
    --sliding-window-rope-base 10000
    --full-attention-rope-base 1000000
    --full-attention-rope-partial-factor 0.25

    --window-size "511,0"
    --window-attn-skip-freq 6
    --num-kv-shared-layers 18

    --geglu-tanh
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear

    --vocab-size 262143
    --make-vocab-size-divisible-by 128
    --scale-embeddings-by-hidden-size

    --per-layer-embed-vocab-size 262144
    --per-layer-embed-dim 256

    --spec megatron.bridge.models.gemma.gemma4_layer_specs gemma4_layer_spec
    --transformer-impl local
    --attention-backend auto
    --init-method-std 0.02
)

# Training settings
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --lr-warmup-iters 100
    --lr $LR
    --min-lr 2e-6
    --lr-decay-style cosine
    --lr-decay-iters $TRAIN_ITERS
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.99
    --clip-grad 1.0
    --bf16
    --calculate-per-token-loss
    --no-masked-softmax-fusion
    --no-rope-fusion
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --use-distributed-optimizer
    --load "$MEGATRON_CKPT"
    --save "$SAVE_DIR"
    --save-interval 200
    --finetune
    --no-load-optim
    --no-load-rng
)

# Parallelism
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size 1
)

# Data
if [ -n "$TRAIN_DATA_PATH" ]; then
    DATA_ARGS=(
        --data-path "$TRAIN_DATA_PATH"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$HF_MODEL_DIR"
        --split "98,1,1"
        --no-mmap-bin-files
        --num-workers 4
    )
else
    echo "  WARNING: TRAIN_DATA_PATH not set, using mock data."
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --split "99,1,0"
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
fi

# Logging / eval
LOGGING_ARGS=(
    --log-interval 10
    --eval-iters 10
    --eval-interval 200
    --tensorboard-dir "$SAVE_DIR/tensorboard"
    --no-save-optim
    --no-save-rng
    --distributed-timeout-minutes 30
)

export CUDA_DEVICE_MAX_CONNECTIONS=1

$TORCHRUN_BIN \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 --node_rank 0 \
    --master_addr localhost \
    --master_port $MASTER_PORT \
    --log_dir "$TRAIN_LOG_DIR" \
    --redirects 3 --tee 3 \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"

echo ""
echo "========================================"
echo "  Training complete."
echo "  Checkpoints saved to: $SAVE_DIR"
echo "========================================"
