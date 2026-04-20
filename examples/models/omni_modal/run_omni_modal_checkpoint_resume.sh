#!/bin/bash
# MIMO checkpoint save→resume round-trip e2e test.
#
# Runs the test in two phases (separate torchrun invocations) for each
# parallelism configuration:
#   Phase 1 (save):   Train for 5 steps, save checkpoint.
#   Phase 2 (resume): Resume from checkpoint, train to 10 steps, verify continuity.
#
# Usage:
#   ./run_omni_modal_checkpoint_resume.sh                         # 8 GPUs, all configs
#   ./run_omni_modal_checkpoint_resume.sh --gpus 8                # explicit GPU count
#   ./run_omni_modal_checkpoint_resume.sh --config tp4_both       # single config only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/omni_modal_checkpoint_resume_e2e.py"

NUM_GPUS=${NUM_GPUS:-8}
SINGLE_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --config) SINGLE_CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "MIMO Checkpoint Resume E2E Tests"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="

# Config format: "name|llm_tp|llm_pp|llm_dp|llm_offset|vision_tp|vision_pp|vision_dp|vision_offset"
# Note: CLIPViT does not support PP > 1; only LLM can use PP.
declare -a CONFIGS_8GPU=(
    "dp4_both|1|1|4|0|1|1|4|4"
    "tp4_both|4|1|1|0|4|1|1|4"
    "tp2_dp2_both|2|1|2|0|2|1|2|4"
    "pp2_llm_dp4_vision|1|2|2|0|1|1|4|4"
)

declare -a CONFIGS_4GPU=(
    "dp2_both|1|1|2|0|1|1|2|2"
    "tp2_both|2|1|1|0|2|1|1|2"
)

declare -a CONFIGS_2GPU=(
    "dp1_both|1|1|1|0|1|1|1|1"
)

if [[ $NUM_GPUS -ge 8 ]]; then
    CONFIGS=("${CONFIGS_8GPU[@]}")
elif [[ $NUM_GPUS -ge 4 ]]; then
    CONFIGS=("${CONFIGS_4GPU[@]}")
else
    CONFIGS=("${CONFIGS_2GPU[@]}")
fi

declare -a RESULTS=()
declare -a FAILED_CONFIGS=()
TOTAL=0
PASSED=0

run_config() {
    local config="$1"
    local name llm_tp llm_pp llm_dp llm_offset vision_tp vision_pp vision_dp vision_offset
    IFS='|' read -r name llm_tp llm_pp llm_dp llm_offset vision_tp vision_pp vision_dp vision_offset <<< "$config"

    echo ""
    echo "----------------------------------------"
    echo "Config: ${name}"
    echo "  LLM:    TP=${llm_tp}, PP=${llm_pp}, DP=${llm_dp}, offset=${llm_offset}"
    echo "  Vision: TP=${vision_tp}, PP=${vision_pp}, DP=${vision_dp}, offset=${vision_offset}"
    echo "----------------------------------------"

    TOTAL=$((TOTAL + 1))
    local start_time=$(date +%s)

    CKPT_DIR=$(mktemp -d -t "omni_modal_ckpt_${name}_XXXXXX")

    local env_prefix="MIMO_LLM_TP=${llm_tp} MIMO_LLM_PP=${llm_pp} MIMO_LLM_DP=${llm_dp} MIMO_LLM_OFFSET=${llm_offset}"
    env_prefix="${env_prefix} MIMO_VISION_TP=${vision_tp} MIMO_VISION_PP=${vision_pp} MIMO_VISION_DP=${vision_dp} MIMO_VISION_OFFSET=${vision_offset}"

    local ok=true

    echo "  Phase 1: SAVE"
    if ! env ${env_prefix} \
        python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" \
        "${TEST_FILE}" --phase save --ckpt-dir "${CKPT_DIR}" 2>&1; then
        ok=false
    fi

    if $ok; then
        echo "  Phase 2: RESUME"
        if ! env ${env_prefix} \
            python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" \
            "${TEST_FILE}" --phase resume --ckpt-dir "${CKPT_DIR}" 2>&1; then
            ok=false
        fi
    fi

    rm -rf "${CKPT_DIR}"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if $ok; then
        RESULTS+=("PASS|${name}|${duration}s")
        PASSED=$((PASSED + 1))
        echo "  [PASS] ${name} (${duration}s)"
    else
        RESULTS+=("FAIL|${name}|${duration}s")
        FAILED_CONFIGS+=("${name}")
        echo "  [FAIL] ${name} (${duration}s)"
        return 1
    fi
    return 0
}

if [[ -n "${SINGLE_CONFIG}" ]]; then
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
        echo "Error: Config '${SINGLE_CONFIG}' not found. Available:"
        for config in "${CONFIGS[@]}"; do echo "  - ${config%%|*}"; done
        exit 1
    fi
else
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

echo ""
echo "=========================================="
echo "SUMMARY: ${PASSED}/${TOTAL} passed"
echo "=========================================="
printf "%-6s | %-25s | %s\n" "Status" "Configuration" "Time"
echo "-------|---------------------------|-------"
for result in "${RESULTS[@]}"; do
    IFS='|' read -r status name duration <<< "$result"
    if [[ "${status}" == "PASS" ]]; then
        printf "\033[32m%-6s\033[0m | %-25s | %s\n" "${status}" "${name}" "${duration}"
    else
        printf "\033[31m%-6s\033[0m | %-25s | %s\n" "${status}" "${name}" "${duration}"
    fi
done
echo "=========================================="

if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed configurations:"
    for cfg in "${FAILED_CONFIGS[@]}"; do echo "  - ${cfg}"; done
    exit 1
fi

echo ""
echo "All checkpoint resume tests passed!"
exit 0
