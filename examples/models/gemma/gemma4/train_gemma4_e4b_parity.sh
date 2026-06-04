#!/bin/bash
# Logit parity check: converted Megatron Gemma-4 E4B vs HF Gemma-4 E4B.
#
# Loads the converted Megatron checkpoint (TP=2) and the original HF model,
# runs the same token sequence through both, and checks that max |logit diff|
# is within --atol. Expected to pass with atol ~1.0 for bf16.
#
# Usage (from Megatron-Bridge root):
#   NVIDIA_VISIBLE_DEVICES=0,1 bash examples/models/gemma/gemma4/train_gemma4_e4b_parity.sh
#
# Overrides:
#   MEGATRON_LM_ROOT=...  GEMMA4_HF_DIR=...  GEMMA4_CKPT=...
#   TP_SIZE=...  ATOL=...  BF16=...  bash ...

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
MEGATRON_LM_ROOT=${MEGATRON_LM_ROOT:-$(cd "$BRIDGE_ROOT/../Megatron-LM" 2>/dev/null && pwd)}

if [ ! -f "$MEGATRON_LM_ROOT/pretrain_gpt.py" ]; then
    echo "Error: Megatron-LM root not found: $MEGATRON_LM_ROOT"
    echo "Set MEGATRON_LM_ROOT=/path/to/Megatron-LM"
    exit 1
fi

GEMMA4_HF_DIR=${GEMMA4_HF_DIR:-$HOME/models/gemma-4-E4B-it}
GEMMA4_CKPT=${GEMMA4_CKPT:-$HOME/checkpoints/gemma4-e4b-megatron}
ATOL=${ATOL:-3.0}
BF16=${BF16:-1}

if [ ! -d "$GEMMA4_HF_DIR" ]; then
    echo "Error: HF model dir not found: $GEMMA4_HF_DIR"
    echo "Set GEMMA4_HF_DIR=/path/to/gemma-4-E4B-it"
    exit 1
fi
if [ ! -f "$GEMMA4_CKPT/latest_checkpointed_iteration.txt" ]; then
    echo "Error: Megatron checkpoint not found at $GEMMA4_CKPT"
    echo "Set GEMMA4_CKPT=/path/to/gemma4-e4b-megatron"
    exit 1
fi

TP_SIZE=${TP_SIZE:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-$TP_SIZE}
MASTER_PORT=${MASTER_PORT:-6101}
TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR:-/tmp/gemma4_e4b_parity_logs}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON_LM_ROOT
export PYTHONPATH="$BRIDGE_ROOT/src:$SCRIPT_DIR:$MEGATRON_LM_ROOT:$MEGATRON_LM_ROOT/tools/checkpoint:${PYTHONPATH:-}"

rm -rf "$TORCHRUN_LOG_DIR"
mkdir -p "$TORCHRUN_LOG_DIR"

echo "========================================"
echo "  Gemma-4 E4B parity check (TP=$TP_SIZE)"
echo "  bridge : $BRIDGE_ROOT"
echo "  mcore  : $MEGATRON_LM_ROOT"
echo "  hf_dir : $GEMMA4_HF_DIR"
echo "  ckpt   : $GEMMA4_CKPT"
echo "  gpus   : $GPUS_PER_NODE"
echo "  atol   : $ATOL"
echo "  bf16   : $BF16"
echo "========================================"

DTYPE_ARGS=()
if [ "$BF16" = "1" ]; then
    DTYPE_ARGS+=(--bf16)
fi

cd "$MEGATRON_LM_ROOT"

torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes 1 --node_rank 0 \
    --master_addr localhost \
    --master_port "$MASTER_PORT" \
    --log_dir "$TORCHRUN_LOG_DIR" \
    --redirects 3 --tee 3 \
    "$SCRIPT_DIR/parity_check_e4b.py" \
    --hf-dir "$GEMMA4_HF_DIR" \
    --megatron-ckpt "$GEMMA4_CKPT" \
    --tp "$TP_SIZE" \
    --atol "$ATOL" \
    "${DTYPE_ARGS[@]}"

echo "========================================"
echo "  Parity check PASSED"
echo "========================================"
