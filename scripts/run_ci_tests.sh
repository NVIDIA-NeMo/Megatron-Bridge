#!/usr/bin/env bash
# run_ci_tests.sh — CI-like test runner for interactive environments.
# Reproduces the GitHub CI pipeline locally or inside Docker. Pipeline stages:
# - uv sync (resolve/install all dependency groups)
# - Lint: pre-commit 3.6.0
# - Unit tests: pytest with coverage
# - Functional tests: L0/L1/L2 tier scripts under tests/functional_tests/
# - Coverage: combine and report
#
# Test tiers (cumulative — each tier includes lower tiers):
# - L0: core smoke tests; runs on every PR
# - L1: broader model/recipe coverage; runs on daily / merge-to-main
# - L2: VL models, checkpoints, heavy quantization; runs nightly / weekly
#
# Modes:
# - local  (default): runs L{tier}_Launch_*.sh scripts directly
# - docker: builds docker/Dockerfile.ci and runs inside a GPU-enabled container
#
# Requirements:
# - local: Python 3.10+; GPUs + CUDA for functional tests
# - docker: Docker with GPU runtime (nvidia-container-toolkit)
#
# Environment variables:
# - HF_HOME: Hugging Face cache directory (default: <repo>/.hf_home)
# - CUDA_VISIBLE_DEVICES: GPU ids to use (default: 0,1)
# - GH_TOKEN: GitHub token used by tests/tools requiring GitHub API (required)
# - NO_UV: if set to 1, behaves as --no-uv
#
# Examples:
#   bash scripts/run_ci_tests.sh                        # L0 only (PR mode)
#   bash scripts/run_ci_tests.sh --tier L1              # L0 + L1
#   bash scripts/run_ci_tests.sh --tier L2              # all tiers
#   bash scripts/run_ci_tests.sh --mode docker --tier L1
#   bash scripts/run_ci_tests.sh --gpus 0 --skip-functional
#   bash scripts/run_ci_tests.sh --skip-lint --skip-unit
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_ci_tests_${TIMESTAMP}.log"
# Mirror all output to console and file
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] Writing output to ${LOG_FILE}"

MODE="local"           # local | docker
TIER="L0"              # L0 | L1 | L2
SKIP_LINT="false"
SKIP_UNIT="false"
SKIP_FUNCTIONAL="false"
USE_UV="true"
CUDA_DEVICES_DEFAULT="0,1"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES_DEFAULT}}
HF_HOME=${HF_HOME:-"${REPO_ROOT}/.hf_home"}

# Track functional test failures while allowing subsequent scripts to continue
FUNC_FAIL=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode [local|docker]     Run tests locally or inside Docker (default: local)
  --tier [L0|L1|L2]         Test tier to run; each tier includes lower tiers (default: L0)
                              L0 — core smoke tests (PR gate)
                              L1 — L0 + broader model/recipe coverage (daily)
                              L2 — all tests including VL, ckpts, heavy quant (nightly)
  --no-uv                   Do not use uv; use system python/pip instead
  --skip-lint               Skip lint/pre-commit step
  --skip-unit               Skip unit tests
  --skip-functional         Skip functional tests
  --gpus <ids>              Set CUDA_VISIBLE_DEVICES (default: ${CUDA_DEVICES_DEFAULT})
  --hf-home <path>          Set HF_HOME cache directory (default: ${REPO_ROOT}/.hf_home)
  -h, --help                Show this help

Examples:
  $(basename "$0")                          # PR: L0 tests only
  $(basename "$0") --tier L1               # daily: L0 + L1 tests
  $(basename "$0") --tier L2               # nightly: all tests
  $(basename "$0") --mode docker --tier L1
  $(basename "$0") --gpus 0 --skip-functional
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --tier)
      TIER="${2:-}"
      shift 2
      ;;
    --no-uv)
      USE_UV="false"
      shift 1
      ;;
    --skip-lint)
      SKIP_LINT="true"
      shift 1
      ;;
    --skip-unit)
      SKIP_UNIT="true"
      shift 1
      ;;
    --skip-functional)
      SKIP_FUNCTIONAL="true"
      shift 1
      ;;
    --gpus)
      CUDA_VISIBLE_DEVICES="${2:-}"
      shift 2
      ;;
    --hf-home)
      HF_HOME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "${TIER}" in
  L0|L1|L2) ;;
  *) echo "Unknown tier: ${TIER} (expected L0, L1, or L2)" >&2; usage; exit 2 ;;
esac

export HF_HOME
export CUDA_VISIBLE_DEVICES

# Require GH_TOKEN to be set for operations that need GitHub API access.
if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "[env] GH_TOKEN is not set. Please export GH_TOKEN before running this script." >&2
  exit 1
fi

# Select tooling based on USE_UV or NO_UV
if [[ "${USE_UV}" == "true" && -z "${NO_UV:-}" ]]; then
  PYTHON="uv run python"
  COVERAGE="uv run coverage"
  PIP="uv pip"
  PRECOMMIT="uv run pre-commit"
  SYNC_CMD="uv sync --all-groups"
else
  PYTHON="python"
  COVERAGE="python -m coverage"
  PIP="pip"
  PRECOMMIT="pre-commit"
  SYNC_CMD="true"
fi

# Return a list of L{n}_Launch_*.sh glob patterns to run for a given max tier.
# Tiers are cumulative: L1 includes L0, L2 includes L0+L1.
tier_patterns() {
  local max_tier="$1"
  case "${max_tier}" in
    L0) echo "L0" ;;
    L1) echo "L0 L1" ;;
    L2) echo "L0 L1 L2" ;;
  esac
}

run_lint_local() {
  if [[ "${SKIP_LINT}" == "true" ]]; then
    echo "[lint] Skipped"
    return 0
  fi
  ${PRECOMMIT} run --all-files --show-diff-on-failure --color=always
}

run_unit_local() {
  if [[ "${SKIP_UNIT}" == "true" ]]; then
    echo "[unit] Skipped"
    return 0
  fi
  echo "[unit] Running unit tests with coverage"
  ${COVERAGE} erase || true
  ${COVERAGE} run -a -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/unit_tests -m "not pleasefixme"
}

run_functional_local() {
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then
    echo "[functional] Skipped"
    return 0
  fi

  local patterns
  patterns=$(tier_patterns "${TIER}")
  echo "[functional] Running tier=${TIER} (includes: ${patterns})"

  set +e
  FUNC_FAIL=0
  for tier in ${patterns}; do
    for script in "${REPO_ROOT}"/tests/functional_tests/${tier}_Launch_*.sh; do
      [[ -e "${script}" ]] || continue
      echo "[functional] Running $(basename "${script}")"
      bash "${script}"
      if [[ $? -ne 0 ]]; then
        echo "[functional] FAILED: $(basename "${script}")"
        FUNC_FAIL=1
      fi
    done
  done
  set -e
  return 0
}

run_local() {
  echo "[env] tier=${TIER} HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  ${SYNC_CMD}
  ${PIP} install -U pygithub
  rm -rf "${REPO_ROOT}/nemo_experiments" "${REPO_ROOT}/NeMo_experiments" || true
  run_lint_local
  run_unit_local
  run_functional_local
  echo "[coverage] Combine & report"
  ${COVERAGE} combine -q || true
  ${COVERAGE} report -i
  # Fail overall if any functional script failed, but only after coverage is reported
  if [[ "${FUNC_FAIL}" -ne 0 ]]; then
    echo "[functional] One or more functional test scripts failed"
    exit 1
  fi
}

run_docker() {
  if [[ "${USE_UV}" == "true" && -z "${NO_UV:-}" ]]; then
    DOCKER_SETUP_PREFIX="uv sync --all-groups && uv pip install -U pygithub && rm -rf nemo_experiments NeMo_experiments || true"
    DOCKER_LINT_PREFIX="uv pip install -U pre-commit==3.6.0 coverage[toml] && uv run pre-commit install && uv run pre-commit run --all-files --show-diff-on-failure --color=always"
    DOCKER_COVERAGE_REPORT="uv run coverage report -i"
  else
    DOCKER_SETUP_PREFIX="pip install -U pygithub && rm -rf nemo_experiments NeMo_experiments || true"
    DOCKER_LINT_PREFIX="pip install -U pre-commit==3.6.0 'coverage[toml]' && pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always"
    DOCKER_COVERAGE_REPORT="python -m coverage report -i"
  fi

  if [[ "${SKIP_LINT}" == "true" ]]; then LINT_CMD="true"; else LINT_CMD="${DOCKER_LINT_PREFIX}"; fi
  if [[ "${SKIP_UNIT}" == "true" ]]; then UNIT_CMD="true"; else UNIT_CMD="bash tests/unit_tests/Launch_Unit_Tests.sh"; fi
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then
    FUNC_CMD="true"
  else
    # Build the tier patterns string for the container shell
    local patterns
    patterns=$(tier_patterns "${TIER}")
    # Run all L{n}_Launch_*.sh scripts for the selected tiers; continue on failure
    FUNC_CMD="shopt -s nullglob; rc=0; for tier in ${patterns}; do for f in tests/functional_tests/\${tier}_Launch_*.sh; do echo \"[functional] Running \$(basename \"\$f\")\"; bash \"\$f\" || { echo \"[functional] FAILED: \$(basename \"\$f\")\"; rc=1; }; done; done; exit \$rc"
  fi

  echo "[docker] Building image from docker/Dockerfile.ci"
  docker build -f "${REPO_ROOT}/docker/Dockerfile.ci" -t megatron-bridge "${REPO_ROOT}"

  HOST_HF_HOME="${HF_HOME}"
  CONTAINER_HF_HOME="/home/TestData/HF_HOME"
  mkdir -p "${HOST_HF_HOME}"

  echo "[docker] Running tier=${TIER} tests in container (HF_HOME=${CONTAINER_HF_HOME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  docker run --rm -it --gpus all \
    -e HF_HOME="${CONTAINER_HF_HOME}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -e GH_TOKEN="${GH_TOKEN}" \
    -v "${REPO_ROOT}":/workspace \
    -v "${HOST_HF_HOME}":"${CONTAINER_HF_HOME}" \
    -w /workspace \
    megatron-bridge bash -lc "${DOCKER_SETUP_PREFIX} && ${LINT_CMD} && ${UNIT_CMD} && ( ${FUNC_CMD} ); FUNC_STATUS=\$?; ${DOCKER_COVERAGE_REPORT}; exit \${FUNC_STATUS}"
}

case "${MODE}" in
  local)
    run_local
    ;;
  docker)
    run_docker
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 2
    ;;
esac

echo "[done]"
