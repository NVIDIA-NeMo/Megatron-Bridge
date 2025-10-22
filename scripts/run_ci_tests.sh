#!/usr/bin/env bash
# run_ci_tests.sh — CI-like test runner for interactive environments.
# Reproduces the GitHub CI pipeline locally or inside Docker. Pipeline stages:
# - uv sync (resolve/install all dependency groups)
# - Lint: pre-commit 3.6.0
# - Unit tests: pytest with coverage
# - Functional tests: training (DDP; optional inprocess restart via ft_launcher), converter, models, recipes
# - Coverage: combine and report
#
# Behavior updates:
# - Installs PyGithub (uv pip install -U pygithub) for tests that call GitHub APIs
# - Cleans any leftover NeMo experiment dirs to avoid interference (removes nemo_experiments/NeMo_experiments)
# - Local mode runs all functional groups; skips only in-process restart in the first training sweep
# - Docker mode runs functional tests directly with pytest and skips heavy suites that historically
#   required pre-mounted test data (gemma/gemma2/gemma3/glm45) to keep CI runtime stable
#
# Modes:
# - local  (default): uses system Python environment
# - docker: builds docker/Dockerfile.ci and runs tests inside the GPU-enabled container
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
#   bash scripts/run_ci_tests.sh
#   bash scripts/run_ci_tests.sh --mode docker
#   bash scripts/run_ci_tests.sh --gpus 0 --skip-functional
#   bash scripts/run_ci_tests.sh --skip-lint --skip-unit
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

MODE="local"           # local | docker
SKIP_LINT="false"
SKIP_UNIT="false"
SKIP_FUNCTIONAL="false"
USE_UV="true"
CUDA_DEVICES_DEFAULT="0,1"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES_DEFAULT}}
HF_HOME=${HF_HOME:-"${REPO_ROOT}/.hf_home"}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode [local|docker]     Run tests locally (python) or inside Docker (default: local)
  --no-uv                   Do not use uv; use system python/pip instead
  --skip-lint               Skip lint/pre-commit step
  --skip-unit               Skip unit tests
  --skip-functional         Skip functional tests
  --gpus <ids>              Set CUDA_VISIBLE_DEVICES (default: ${CUDA_DEVICES_DEFAULT})
  --hf-home <path>          Set HF_HOME cache directory (default: "+${REPO_ROOT}/.hf_home+")
  -h, --help                Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --mode docker
  $(basename "$0") --gpus 0 --skip-functional
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
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

  echo "[functional] Training group (excluding inprocess restart)"
  ${PYTHON} -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run -a -m pytest \
    -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/training -k "not test_inprocess_restart and not load_model"

  if command -v ft_launcher >/dev/null 2>&1; then
    echo "[functional] Inprocess restart with ft_launcher"
    export TORCH_CPP_LOG_LEVEL="error"
    ft_launcher \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
      --nnodes=1 --nproc-per-node=2 \
      --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
      --ft-param-rank_out_of_section_timeout=300 \
      --monitor-interval=5 --max-restarts=3 \
      --ft-restart-policy=min-healthy \
      -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
      tests/functional_tests/training/test_inprocess_restart.py
  else
    echo "[functional] ft_launcher not found; skipping inprocess restart test"
  fi

  echo "[functional] Converter group"
  ${COVERAGE} run -a -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/converter

  echo "[functional] Models group"
  ${COVERAGE} run -a -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/models

  echo "[functional] Recipes group (2 GPUs)"
  ${PYTHON} -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run -a -m pytest \
    -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/recipes
}

run_local() {
  echo "[env] Using HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  ${SYNC_CMD}
  ${PIP} install -U pygithub
  rm -rf "${REPO_ROOT}/nemo_experiments" "${REPO_ROOT}/NeMo_experiments" || true
  run_lint_local
  run_unit_local
  run_functional_local
  echo "[coverage] Combine & report"
  ${COVERAGE} combine -q || true
  ${COVERAGE} report -i
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
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then FUNC_CMD="true"; else FUNC_CMD="python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m 'not pleasefixme' --tb=short -rA tests/functional_tests/training -k 'not test_inprocess_restart' && coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m 'not pleasefixme' --tb=short -rA tests/functional_tests/converter -k 'not gemma and not gemma2 and not gemma3 and not glm45' && coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m 'not pleasefixme' --tb=short -rA tests/functional_tests/models -k 'not gemma and not gemma2 and not gemma3 and not glm45' && python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m 'not pleasefixme' --tb=short -rA tests/functional_tests/recipes -k 'not gemma and not gemma2 and not gemma3 and not glm45'"; fi

  echo "[docker] Building image from docker/Dockerfile.ci"
  docker build -f "${REPO_ROOT}/docker/Dockerfile.ci" -t megatron-bridge "${REPO_ROOT}"

  HOST_HF_HOME="${HF_HOME}"
  CONTAINER_HF_HOME="/home/TestData/HF_HOME"
  mkdir -p "${HOST_HF_HOME}"

  echo "[docker] Running tests in container (HF_HOME=${CONTAINER_HF_HOME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  docker run --rm -it --gpus all \
    -e HF_HOME="${CONTAINER_HF_HOME}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -e GH_TOKEN="${GH_TOKEN}" \
    -v "${REPO_ROOT}":/workspace \
    -v "${HOST_HF_HOME}":"${CONTAINER_HF_HOME}" \
    -w /workspace \
    megatron-bridge bash -lc "${DOCKER_SETUP_PREFIX} && ${LINT_CMD} && ${UNIT_CMD} && ${FUNC_CMD} && ${DOCKER_COVERAGE_REPORT}"
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


