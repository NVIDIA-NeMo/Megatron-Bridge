#!/bin/bash
# Run MIMO E2E test with various parallelism configurations
# Usage: ./run_mimo_parallelism_tests.sh [--gpus N] [--config CONFIG_NAME]
#
# Examples:
#   ./run_mimo_parallelism_tests.sh                    # Run all configs with 8 GPUs
#   ./run_mimo_parallelism_tests.sh --gpus 4           # Run all configs with 4 GPUs
#   ./run_mimo_parallelism_tests.sh --config tp2_both  # Run only tp2_both config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/test_mimo_training_e2e.py"

# Default values
NUM_GPUS=${NUM_GPUS:-8}
SINGLE_CONFIG=""

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
echo "MIMO Parallelism E2E Tests"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="

# Define configurations as: "name|llm_tp|llm_pp|llm_dp|llm_offset|vision_tp|vision_pp|vision_dp|vision_offset"
# Note: Vision encoder (CLIPViT) does not support PP > 1, only LLM can use PP
declare -a CONFIGS_8GPU=(
    "baseline_dp_only|1|1|4|0|1|1|4|4"
    # "tp2_both|2|1|2|0|2|1|2|4"
    # "tp2_llm_dp_vision|2|1|2|0|1|1|4|4"
    # "pp2_llm_only|1|2|2|0|1|1|4|4"
    # "tp4_both|4|1|1|0|4|1|1|4"
    # "tp4_llm_tp2_vision|4|1|1|0|2|1|2|4"
    # "3d_llm_dp_vision|2|2|1|0|1|1|4|4"
    # "asymmetric_6_2_pp|2|3|1|0|2|1|1|6"
)

# Note: PP > 1 not included for 4 GPU configs (would need at least 2 ranks for PP)
declare -a CONFIGS_4GPU=(
    "baseline_dp_only|1|1|2|0|1|1|2|2"
    "tp2_both|2|1|1|0|2|1|1|2"
)

declare -a CONFIGS_2GPU=(
    "baseline_dp_only|1|1|1|0|1|1|1|1"
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

run_config() {
    local config="$1"
    local name llm_tp llm_pp llm_dp llm_offset vision_tp vision_pp vision_dp vision_offset
    
    IFS='|' read -r name llm_tp llm_pp llm_dp llm_offset vision_tp vision_pp vision_dp vision_offset <<< "$config"
    
    echo ""
    echo "----------------------------------------"
    echo "Running: ${name}"
    echo "  LLM:    TP=${llm_tp}, PP=${llm_pp}, DP=${llm_dp}, offset=${llm_offset}"
    echo "  Vision: TP=${vision_tp}, PP=${vision_pp}, DP=${vision_dp}, offset=${vision_offset}"
    echo "----------------------------------------"
    
    TOTAL=$((TOTAL + 1))
    
    # Set environment variables and run
    local start_time=$(date +%s)
    
    if MIMO_LLM_TP="${llm_tp}" \
       MIMO_LLM_PP="${llm_pp}" \
       MIMO_LLM_DP="${llm_dp}" \
       MIMO_LLM_OFFSET="${llm_offset}" \
       MIMO_VISION_TP="${vision_tp}" \
       MIMO_VISION_PP="${vision_pp}" \
       MIMO_VISION_DP="${vision_dp}" \
       MIMO_VISION_OFFSET="${vision_offset}" \
       python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" "${TEST_FILE}" 2>&1; then
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
    for cfg in "${FAILED_CONFIGS[@]}"; do
        echo "  - ${cfg}"
    done
    exit 1
fi

echo ""
echo "All tests passed!"
exit 0
