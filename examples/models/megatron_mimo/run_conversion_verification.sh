#!/bin/bash
# Verify CLIP and LLM (Vicuna-7B) HF→Megatron weight conversion across TP=1, 2, 4.
#
# Converts weights at each TP size, then runs the Megatron model and compares
# outputs against HuggingFace.  Exits non-zero on the first failure.
#
# Requirements:
#   - GPUs >= max TP size tested (default 4)
#   - HF model weights accessible (downloads on first run)
#
# Usage:
#   bash run_conversion_verification.sh                       # default paths
#   bash run_conversion_verification.sh --ckpt-root /scratch  # custom output dir
#   bash run_conversion_verification.sh --tp-sizes "1 2"      # subset of TP sizes
#   bash run_conversion_verification.sh --models llm --tp-sizes "2 4"  # LLM only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults (override via flags)
# ---------------------------------------------------------------------------
CKPT_ROOT="${CKPT_ROOT:-/tmp/conversion_verify}"
HF_CLIP_MODEL="${HF_CLIP_MODEL:-openai/clip-vit-large-patch14-336}"
HF_LLM_MODEL="${HF_LLM_MODEL:-lmsys/vicuna-7b-v1.5}"
MEGATRON_VOCAB_SIZE=32256
DTYPE="${DTYPE:-fp32}"
TP_SIZES="${TP_SIZES:-1 2 4}"
MODELS="${MODELS:-clip llm}"  # "clip", "llm", or "clip llm"

# ---------------------------------------------------------------------------
# Parse optional flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-root)  CKPT_ROOT="$2";  shift 2 ;;
        --dtype)      DTYPE="$2";       shift 2 ;;
        --tp-sizes)   TP_SIZES="$2";    shift 2 ;;
        --hf-clip)    HF_CLIP_MODEL="$2"; shift 2 ;;
        --hf-llm)     HF_LLM_MODEL="$2";  shift 2 ;;
        --models)     MODELS="$2";       shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
RESULTS=()

run_step() {
    local label="$1"; shift
    echo ""
    echo "================================================================"
    echo "  $label"
    echo "================================================================"
    if "$@"; then
        echo "  >> $label: PASSED"
        RESULTS+=("PASS  $label")
        PASS=$((PASS + 1))
    else
        echo "  >> $label: FAILED"
        RESULTS+=("FAIL  $label")
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# Run conversions and verifications
# ---------------------------------------------------------------------------
for TP in $TP_SIZES; do
    # --- CLIP ---
    if [[ " $MODELS " == *" clip "* ]]; then
        CLIP_CKPT="${CKPT_ROOT}/clip_tp${TP}"
        run_step "CLIP convert TP=${TP}" \
            python "${SCRIPT_DIR}/convert_hf_clip_to_megatron.py" \
                --hf-model "$HF_CLIP_MODEL" \
                --output "$CLIP_CKPT" \
                --tensor-parallel-size "$TP" \
                --use-te

        run_step "CLIP verify  TP=${TP}" \
            torchrun --nproc_per_node="$TP" \
                "${SCRIPT_DIR}/verify_clip_conversion.py" \
                --checkpoint-dir "$CLIP_CKPT" \
                --hf-model "$HF_CLIP_MODEL" \
                --dtype "$DTYPE" \
                --tensor-parallel-size "$TP"
    fi

    # --- LLM ---
    if [[ " $MODELS " == *" llm "* ]]; then
        LLM_CKPT="${CKPT_ROOT}/llm_tp${TP}"
        run_step "LLM  convert TP=${TP}" \
            python "${SCRIPT_DIR}/convert_hf_llama_to_megatron.py" \
                --hf-model "$HF_LLM_MODEL" \
                --output "$LLM_CKPT" \
                --tensor-parallel-size "$TP" \
                --use-te \
                --megatron-vocab-size "$MEGATRON_VOCAB_SIZE"

        run_step "LLM  verify  TP=${TP}" \
            torchrun --nproc_per_node="$TP" \
                "${SCRIPT_DIR}/verify_llama_conversion.py" \
                --checkpoint-dir "$LLM_CKPT" \
                --hf-model "$HF_LLM_MODEL" \
                --dtype "$DTYPE" \
                --tensor-parallel-size "$TP" \
                --megatron-vocab-size "$MEGATRON_VOCAB_SIZE"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  CONVERSION VERIFICATION SUMMARY"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Total: $((PASS + FAIL))  Passed: ${PASS}  Failed: ${FAIL}"
echo "================================================================"

if [[ $FAIL -gt 0 ]]; then
    echo "SOME VERIFICATIONS FAILED"
    exit 1
fi

echo "ALL VERIFICATIONS PASSED"
exit 0
