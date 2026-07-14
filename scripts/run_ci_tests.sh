#!/usr/bin/env bash
# run_ci_tests.sh — CI-like test runner for interactive environments.
# Reproduces the GitHub CI pipeline locally, inside Docker, or through Slurm/Pyxis.
#
# Pipeline stages:
# - uv sync / setup, unless --skip-setup is set
# - Lint: pre-commit
# - Unit tests: pytest with coverage
# - Functional tests: L0/L1/L2 launch scripts under tests/functional_tests/launch_scripts/
# - Coverage: combine and report
#
# Test tiers are cumulative: L1 includes L0, and L2 includes L0+L1.
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

MODE="local"                  # local | docker | srun
TIER="L0"                     # L0 | L1 | L2
HARDWARE="h100"               # h100 | gb200
SCRIPT_DIR_OVERRIDE=""        # relative to tests/functional_tests/launch_scripts
FUNCTIONAL_SCRIPT=""          # script basename, path, or basename without .sh
SKIP_LINT="false"
SKIP_UNIT="false"
SKIP_FUNCTIONAL="false"
SKIP_SETUP="false"
USE_UV="true"
DRY_RUN="false"
LIST_FUNCTIONAL="false"

CUDA_DEVICES_DEFAULT="0,1"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES_DEFAULT}}
HF_HOME=${HF_HOME:-"${REPO_ROOT}/.hf_home"}

CONTAINER_IMAGE="${MB_CONTAINER_IMAGE:-}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/opt/Megatron-Bridge}"
CONTAINER_MOUNTS="${MB_CONTAINER_MOUNTS:-}"
SRUN_ACCOUNT="${SRUN_ACCOUNT:-${SLURM_ACCOUNT:-}}"
SRUN_PARTITION="${SRUN_PARTITION:-batch}"
SRUN_NODES="${SRUN_NODES:-1}"
SRUN_GPUS_PER_NODE="${SRUN_GPUS_PER_NODE:-}"
SRUN_JOB_NAME="${SRUN_JOB_NAME:-}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"
SRUN_ARGS=()

FUNC_FAIL=0
FUNCTIONAL_SCRIPTS=()
LOG_FILE=""

start_logging() {
  local timestamp
  local log_dir
  timestamp=$(date +%Y%m%d_%H%M%S)
  log_dir="${REPO_ROOT}/logs"
  mkdir -p "${log_dir}"
  LOG_FILE="${log_dir}/run_ci_tests_${timestamp}.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "[log] Writing output to ${LOG_FILE}"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode [local|docker|srun]       Run locally, inside Docker, or via Slurm/Pyxis (default: local)
  --tier [L0|L1|L2]                Test tier to run; each tier includes lower tiers (default: L0)
  --hardware [h100|gb200]          Functional-test hardware directory (default: h100)
  --script-dir <path>              Functional script dir under launch_scripts/ (default: <hardware>/active)
  --script <name-or-path>          Run one functional launch script
  --list-functional                Print selected functional scripts and exit
  --dry-run                        Print commands/scripts without executing them
  --no-uv                          Do not use uv for runner-managed commands
  --skip-setup                     Skip uv sync and the runner-managed pygithub install
  --skip-lint                      Skip lint/pre-commit step
  --skip-unit                      Skip unit tests
  --skip-functional                Skip functional tests
  --gpus <ids>                     Set CUDA_VISIBLE_DEVICES (default: ${CUDA_DEVICES_DEFAULT})
  --hf-home <path>                 Set HF_HOME cache directory (default: ${REPO_ROOT}/.hf_home)

srun/Pyxis options:
  --container-image <path>         Container image; defaults to MB_CONTAINER_IMAGE
  --container-mounts <mounts>      Pyxis mount string; defaults to repo mount plus /lustre when applicable
  --container-workdir <path>       Repo path inside container (default: /opt/Megatron-Bridge)
  --srun-account <account>         Slurm account; defaults to SRUN_ACCOUNT or SLURM_ACCOUNT
  --srun-partition <partition>     Slurm partition (default: batch)
  --srun-nodes <n>                 Slurm node count (default: 1)
  --srun-gpus-per-node <n|none>    GPUs per node for clusters that use this Slurm flag (default: unset)
  --srun-arg <arg>                 Extra srun argument; repeat for multiple argv entries
  --srun-extra-args <arg>          Extra single srun argument; prefer repeatable --srun-arg

Examples:
  $(basename "$0") --list-functional
  $(basename "$0") --tier L1 --hardware h100
  $(basename "$0") --script L0_Launch_converter --skip-lint --skip-unit
  MB_CONTAINER_IMAGE=/path/to/mbridge.sqsh $(basename "$0") --mode srun --script L0_Launch_converter --skip-lint --skip-unit
EOF
}

quote_args() {
  local quoted=()
  local arg
  local q
  for arg in "$@"; do
    printf -v q "%q" "${arg}"
    quoted+=("${q}")
  done
  printf "%s " "${quoted[@]}"
}

container_path_for_host_path() {
  local host_path="$1"
  if [[ "${host_path}" == "${REPO_ROOT}" ]]; then
    printf "%s" "${CONTAINER_WORKDIR}"
  elif [[ "${host_path}" == "${REPO_ROOT}/"* ]]; then
    printf "%s/%s" "${CONTAINER_WORKDIR}" "${host_path#"${REPO_ROOT}/"}"
  else
    printf "%s" "${host_path}"
  fi
}

default_container_mounts() {
  local mounts="${REPO_ROOT}:${CONTAINER_WORKDIR}"
  if [[ "${REPO_ROOT}" == *,* || "${CONTAINER_WORKDIR}" == *,* ]]; then
    echo "[srun] Default Pyxis mounts cannot contain commas: ${REPO_ROOT}:${CONTAINER_WORKDIR}" >&2
    exit 2
  fi
  if [[ "${REPO_ROOT}" == /lustre/* || "${HF_HOME}" == /lustre/* || "${UV_CACHE_DIR:-}" == /lustre/* || "${NEMO_HOME:-}" == /lustre/* ]]; then
    mounts="/lustre:/lustre,${mounts}"
  fi
  printf "%s" "${mounts}"
}

functional_script_dir() {
  if [[ -n "${SCRIPT_DIR_OVERRIDE}" ]]; then
    printf "%s" "${SCRIPT_DIR_OVERRIDE%/}"
  else
    printf "%s/active" "${HARDWARE}"
  fi
}

tier_patterns() {
  local max_tier="$1"
  case "${max_tier}" in
    L0) echo "L0" ;;
    L1) echo "L0 L1" ;;
    L2) echo "L0 L1 L2" ;;
  esac
}

resolve_functional_script() {
  local root="$1"
  local spec="$2"
  local candidate=""
  local name="${spec}"

  if [[ "${spec}" == *.sh ]]; then
    name="${spec}"
  else
    name="${spec}.sh"
  fi

  if [[ "${spec}" == /* || "${spec}" == ./* || "${spec}" == tests/* ]]; then
    candidate="${spec}"
    [[ "${candidate}" == *.sh ]] || candidate="${candidate}.sh"
  elif [[ -f "${root}/${name}" ]]; then
    candidate="${root}/${name}"
  else
    mapfile -t matches < <(find "${REPO_ROOT}/tests/functional_tests/launch_scripts" -type f -name "${name}" | sort)
    if [[ "${#matches[@]}" -eq 1 ]]; then
      candidate="${matches[0]}"
    elif [[ "${#matches[@]}" -gt 1 ]]; then
      echo "[functional] Ambiguous script '${spec}'. Matches:" >&2
      printf "  %s\n" "${matches[@]}" >&2
      exit 2
    fi
  fi

  if [[ -z "${candidate}" || ! -f "${candidate}" ]]; then
    echo "[functional] Script not found: ${spec}" >&2
    echo "[functional] Looked under: ${root}" >&2
    exit 2
  fi

  printf "%s" "${candidate}"
}

collect_functional_scripts() {
  FUNCTIONAL_SCRIPTS=()
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then
    return 0
  fi

  local dir
  local root
  dir=$(functional_script_dir)
  root="${REPO_ROOT}/tests/functional_tests/launch_scripts/${dir}"

  if [[ ! -d "${root}" ]]; then
    echo "[functional] Launch script directory does not exist: ${root}" >&2
    exit 2
  fi

  if [[ -n "${FUNCTIONAL_SCRIPT}" ]]; then
    FUNCTIONAL_SCRIPTS+=("$(resolve_functional_script "${root}" "${FUNCTIONAL_SCRIPT}")")
    return 0
  fi

  local patterns
  local tier
  local script
  patterns=$(tier_patterns "${TIER}")
  for tier in ${patterns}; do
    for script in "${root}/${tier}"_*.sh; do
      [[ -e "${script}" ]] || continue
      FUNCTIONAL_SCRIPTS+=("${script}")
    done
  done

  if [[ "${#FUNCTIONAL_SCRIPTS[@]}" -eq 0 ]]; then
    echo "[functional] No launch scripts matched tier=${TIER} in ${root}" >&2
    exit 2
  fi
}

print_functional_scripts() {
  collect_functional_scripts
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then
    echo "[functional] Skipped"
    return 0
  fi

  local script
  for script in "${FUNCTIONAL_SCRIPTS[@]}"; do
    printf "%s\n" "${script#${REPO_ROOT}/}"
  done
}

inner_functional_script_arg() {
  if [[ -z "${FUNCTIONAL_SCRIPT}" ]]; then
    return 0
  fi

  if [[ "${FUNCTIONAL_SCRIPT}" == "${REPO_ROOT}/"* ]]; then
    printf "%s" "${FUNCTIONAL_SCRIPT#${REPO_ROOT}/}"
  elif [[ "${FUNCTIONAL_SCRIPT}" == /* ]]; then
    echo "[functional] Absolute --script paths outside the repo cannot be used inside containers: ${FUNCTIONAL_SCRIPT}" >&2
    return 2
  else
    printf "%s" "${FUNCTIONAL_SCRIPT}"
  fi
}

get_ci_timeout_minutes() {
  local script="$1"
  local timeout_minutes
  timeout_minutes=$(grep -m1 '^# CI_TIMEOUT=' "${script}" | cut -d= -f2)
  printf "%s" "${timeout_minutes:-30}"
}

ensure_functional_workdir() {
  local script
  if [[ "${REPO_ROOT}" == "/opt/Megatron-Bridge" ]]; then
    return 0
  fi

  for script in "${FUNCTIONAL_SCRIPTS[@]}"; do
    if grep -q '/opt/Megatron-Bridge' "${script}"; then
      echo "[functional] ${script#${REPO_ROOT}/} assumes /opt/Megatron-Bridge paths." >&2
      echo "[functional] Run functional tests with --mode docker or --mode srun, or mount the repo at /opt/Megatron-Bridge." >&2
      exit 2
    fi
  done
}

warn_functional_workdir_for_dry_run() {
  local script
  if [[ "${REPO_ROOT}" == "/opt/Megatron-Bridge" ]]; then
    return 0
  fi

  for script in "${FUNCTIONAL_SCRIPTS[@]}"; do
    if grep -q '/opt/Megatron-Bridge' "${script}"; then
      echo "[dry-run] NOTE: actual local functional execution would fail outside /opt/Megatron-Bridge; use --mode docker or --mode srun."
      return 0
    fi
  done
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
    --hardware)
      HARDWARE="${2:-}"
      shift 2
      ;;
    --script-dir)
      SCRIPT_DIR_OVERRIDE="${2:-}"
      shift 2
      ;;
    --script)
      FUNCTIONAL_SCRIPT="${2:-}"
      shift 2
      ;;
    --list-functional)
      LIST_FUNCTIONAL="true"
      shift 1
      ;;
    --dry-run)
      DRY_RUN="true"
      shift 1
      ;;
    --no-uv)
      USE_UV="false"
      shift 1
      ;;
    --skip-setup)
      SKIP_SETUP="true"
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
    --container-image)
      CONTAINER_IMAGE="${2:-}"
      shift 2
      ;;
    --container-mounts)
      CONTAINER_MOUNTS="${2:-}"
      shift 2
      ;;
    --container-workdir)
      CONTAINER_WORKDIR="${2:-}"
      shift 2
      ;;
    --srun-account)
      SRUN_ACCOUNT="${2:-}"
      shift 2
      ;;
    --srun-partition)
      SRUN_PARTITION="${2:-}"
      shift 2
      ;;
    --srun-nodes)
      SRUN_NODES="${2:-}"
      shift 2
      ;;
    --srun-gpus-per-node)
      SRUN_GPUS_PER_NODE="${2:-}"
      shift 2
      ;;
    --srun-arg)
      SRUN_ARGS+=("${2:-}")
      shift 2
      ;;
    --srun-extra-args)
      SRUN_EXTRA_ARGS="${2:-}"
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

case "${MODE}" in
  local|docker|srun) ;;
  *) echo "Unknown mode: ${MODE} (expected local, docker, or srun)" >&2; usage; exit 2 ;;
esac

case "${TIER}" in
  L0|L1|L2) ;;
  *) echo "Unknown tier: ${TIER} (expected L0, L1, or L2)" >&2; usage; exit 2 ;;
esac

case "${HARDWARE}" in
  h100|gb200) ;;
  *) echo "Unknown hardware: ${HARDWARE} (expected h100 or gb200)" >&2; usage; exit 2 ;;
esac

export HF_HOME
export CUDA_VISIBLE_DEVICES
if [[ -n "${GH_TOKEN:-}" ]]; then
  export GH_TOKEN
fi

if [[ "${LIST_FUNCTIONAL}" == "true" ]]; then
  print_functional_scripts
  exit 0
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  start_logging
fi

if [[ "${DRY_RUN}" != "true" && "${SKIP_FUNCTIONAL}" != "true" && -z "${GH_TOKEN:-}" ]]; then
  echo "[env] GH_TOKEN is not set. Export GH_TOKEN before running functional tests." >&2
  exit 1
fi

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

run_setup_local() {
  if [[ "${SKIP_SETUP}" == "true" ]]; then
    echo "[setup] Skipped"
    return 0
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] ${SYNC_CMD}"
    echo "[dry-run] ${PIP} install -U pygithub"
    return 0
  fi

  ${SYNC_CMD}
  ${PIP} install -U pygithub
}

run_lint_local() {
  if [[ "${SKIP_LINT}" == "true" ]]; then
    echo "[lint] Skipped"
    return 0
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] ${PRECOMMIT} run --all-files --show-diff-on-failure --color=always"
    return 0
  fi
  ${PRECOMMIT} run --all-files --show-diff-on-failure --color=always
}

run_unit_local() {
  if [[ "${SKIP_UNIT}" == "true" ]]; then
    echo "[unit] Skipped"
    return 0
  fi
  local unit_scripts=(
    "tests/unit_tests/Launch_Unit_Tests_Core.sh"
    "tests/unit_tests/Launch_Unit_Tests_Diffusion.sh"
  )
  if [[ "${DRY_RUN}" == "true" ]]; then
    for script in "${unit_scripts[@]}"; do
      echo "[dry-run] bash ${script}"
    done
    return 0
  fi

  if [[ "${REPO_ROOT}" == "/opt/Megatron-Bridge" ]]; then
    echo "[unit] Running CI unit launch scripts"
    local script
    for script in "${unit_scripts[@]}"; do
      bash "${script}"
    done
  else
    echo "[unit] Running direct pytest fallback outside /opt/Megatron-Bridge"
    ${COVERAGE} erase || true
    ${COVERAGE} run -a -m pytest \
      -o log_cli=true \
      -o log_cli_level=INFO \
      --disable-warnings \
      -vs tests/unit_tests -m "not pleasefixme"
  fi
}

run_functional_local() {
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then
    echo "[functional] Skipped"
    return 0
  fi

  collect_functional_scripts
  echo "[functional] Running tier=${TIER} hardware=${HARDWARE} dir=$(functional_script_dir) count=${#FUNCTIONAL_SCRIPTS[@]}"

  local script
  if [[ "${DRY_RUN}" == "true" ]]; then
    warn_functional_workdir_for_dry_run
    for script in "${FUNCTIONAL_SCRIPTS[@]}"; do
      echo "[dry-run] bash ${script#${REPO_ROOT}/}"
    done
    return 0
  fi
  ensure_functional_workdir

  set +e
  FUNC_FAIL=0
  for script in "${FUNCTIONAL_SCRIPTS[@]}"; do
    local timeout_minutes
    local timeout_seconds
    timeout_minutes=$(get_ci_timeout_minutes "${script}")
    timeout_seconds=$((timeout_minutes * 60))
    echo "[functional] Running ${script#${REPO_ROOT}/}"
    if command -v timeout >/dev/null 2>&1; then
      timeout "${timeout_seconds}s" bash "${script}"
    else
      echo "[functional] timeout command not found; running without CI timeout"
      bash "${script}"
    fi
    local status=$?
    if [[ "${status}" -eq 124 ]]; then
      echo "[functional] TIMEOUT after ${timeout_minutes} minutes: ${script#${REPO_ROOT}/}"
      FUNC_FAIL=1
    elif [[ "${status}" -ne 0 ]]; then
      echo "[functional] FAILED: ${script#${REPO_ROOT}/}"
      FUNC_FAIL=1
    fi
  done
  set -e
  return 0
}

run_coverage_report_local() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] ${COVERAGE} combine -q || true"
    echo "[dry-run] ${COVERAGE} report -i"
    return 0
  fi
  echo "[coverage] Combine & report"
  ${COVERAGE} combine -q || true
  ${COVERAGE} report -i || true
}

run_local() {
  cd "${REPO_ROOT}"
  echo "[env] mode=local tier=${TIER} hardware=${HARDWARE} HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  run_setup_local
  if [[ "${DRY_RUN}" != "true" ]]; then
    rm -rf "${REPO_ROOT}/nemo_experiments" "${REPO_ROOT}/NeMo_experiments" || true
  fi
  run_lint_local
  run_unit_local
  run_functional_local
  run_coverage_report_local

  if [[ "${FUNC_FAIL}" -ne 0 ]]; then
    echo "[functional] One or more functional test scripts failed"
    exit 1
  fi
}

inner_runner_args() {
  local inner_hf_home="$1"
  local propagate_dry_run="${2:-false}"
  local inner_script=""
  local -a args=(
    "--mode" "local"
    "--tier" "${TIER}"
    "--hardware" "${HARDWARE}"
    "--gpus" "${CUDA_VISIBLE_DEVICES}"
    "--hf-home" "${inner_hf_home}"
  )

  [[ -n "${SCRIPT_DIR_OVERRIDE}" ]] && args+=("--script-dir" "${SCRIPT_DIR_OVERRIDE}")
  if [[ -n "${FUNCTIONAL_SCRIPT}" ]]; then
    inner_script=$(inner_functional_script_arg) || return 2
    args+=("--script" "${inner_script}")
  fi
  [[ "${SKIP_SETUP}" == "true" ]] && args+=("--skip-setup")
  [[ "${SKIP_LINT}" == "true" ]] && args+=("--skip-lint")
  [[ "${SKIP_UNIT}" == "true" ]] && args+=("--skip-unit")
  [[ "${SKIP_FUNCTIONAL}" == "true" ]] && args+=("--skip-functional")
  [[ "${USE_UV}" != "true" ]] && args+=("--no-uv")
  [[ "${propagate_dry_run}" == "true" && "${DRY_RUN}" == "true" ]] && args+=("--dry-run")

  quote_args "${args[@]}"
}

run_docker() {
  echo "[docker] Building image from docker/Dockerfile.ci"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] docker build -f ${REPO_ROOT}/docker/Dockerfile.ci -t megatron-bridge ${REPO_ROOT}"
  else
    docker build -f "${REPO_ROOT}/docker/Dockerfile.ci" -t megatron-bridge "${REPO_ROOT}"
  fi

  local host_hf_home="${HF_HOME}"
  local container_hf_home="/home/TestData/HF_HOME"

  local inner_args
  inner_args=$(inner_runner_args "${container_hf_home}") || exit 2
  local quoted_workdir
  printf -v quoted_workdir "%q" "${CONTAINER_WORKDIR}"
  local inner_cmd
  inner_cmd="cd ${quoted_workdir} && bash scripts/run_ci_tests.sh ${inner_args}"

  echo "[docker] Running tier=${TIER} hardware=${HARDWARE} in container"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] docker run --rm -it --gpus all -e HF_HOME=${container_hf_home} -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -e GH_TOKEN=<redacted> -v ${REPO_ROOT}:${CONTAINER_WORKDIR} -v ${host_hf_home}:${container_hf_home} -w ${CONTAINER_WORKDIR} megatron-bridge bash -lc $(printf %q "${inner_cmd}")"
    return 0
  fi
  mkdir -p "${host_hf_home}"

  docker run --rm -it --gpus all \
    -e HF_HOME="${container_hf_home}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -e GH_TOKEN="${GH_TOKEN:-}" \
    -v "${REPO_ROOT}":"${CONTAINER_WORKDIR}" \
    -v "${host_hf_home}":"${container_hf_home}" \
    -w "${CONTAINER_WORKDIR}" \
    megatron-bridge bash -lc "${inner_cmd}"
}

run_srun() {
  if [[ -z "${CONTAINER_IMAGE}" ]]; then
    echo "[srun] No container image set. Use --container-image or export MB_CONTAINER_IMAGE." >&2
    exit 2
  fi
  if [[ "${DRY_RUN}" != "true" ]] && ! command -v srun >/dev/null 2>&1; then
    echo "[srun] srun not found in PATH" >&2
    exit 2
  fi

  [[ -n "${CONTAINER_MOUNTS}" ]] || CONTAINER_MOUNTS=$(default_container_mounts)
  if [[ -z "${SRUN_JOB_NAME}" ]]; then
    if [[ -n "${SRUN_ACCOUNT}" ]]; then
      SRUN_JOB_NAME="${SRUN_ACCOUNT}-mbridge.ci-tests"
    else
      SRUN_JOB_NAME="mbridge.ci-tests"
    fi
  fi

  local container_hf_home
  container_hf_home=$(container_path_for_host_path "${HF_HOME}")

  local inner_args
  inner_args=$(inner_runner_args "${container_hf_home}") || exit 2
  local quoted_workdir
  printf -v quoted_workdir "%q" "${CONTAINER_WORKDIR}"
  local inner_cmd
  inner_cmd="cd ${quoted_workdir} && export UV_NO_SYNC=\${UV_NO_SYNC:-1} && bash scripts/run_ci_tests.sh ${inner_args}"

  local -a srun_cmd=(
    srun
    --mpi=pmix
    --no-kill
    -N "${SRUN_NODES}"
    --ntasks=1
    --container-image "${CONTAINER_IMAGE}"
    --container-mounts "${CONTAINER_MOUNTS}"
    --no-container-mount-home
    -J "${SRUN_JOB_NAME}"
  )

  if [[ -n "${SRUN_GPUS_PER_NODE}" && "${SRUN_GPUS_PER_NODE}" != "none" ]]; then
    srun_cmd+=("--gpus-per-node=${SRUN_GPUS_PER_NODE}")
  fi

  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ -n "${SRUN_ACCOUNT}" ]] && srun_cmd+=(--account "${SRUN_ACCOUNT}")
    [[ -n "${SRUN_PARTITION}" ]] && srun_cmd+=(-p "${SRUN_PARTITION}")
  fi

  if [[ -n "${SRUN_EXTRA_ARGS}" ]]; then
    if [[ "${SRUN_EXTRA_ARGS}" =~ [[:space:]] ]]; then
      echo "[srun] --srun-extra-args accepts one argv entry. Use repeated --srun-arg for values with spaces." >&2
      exit 2
    fi
    srun_cmd+=("${SRUN_EXTRA_ARGS}")
  fi
  if [[ "${#SRUN_ARGS[@]}" -gt 0 ]]; then
    srun_cmd+=("${SRUN_ARGS[@]}")
  fi

  srun_cmd+=(bash -lc "${inner_cmd}")

  echo "[srun] image=${CONTAINER_IMAGE}"
  echo "[srun] mounts=${CONTAINER_MOUNTS}"
  echo "[srun] workdir=${CONTAINER_WORKDIR} HF_HOME=${container_hf_home}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo -n "[dry-run] "
    printf "%q " "${srun_cmd[@]}"
    echo
    return 0
  fi

  "${srun_cmd[@]}"
}

case "${MODE}" in
  local)
    run_local
    ;;
  docker)
    run_docker
    ;;
  srun)
    run_srun
    ;;
esac

echo "[done]"
