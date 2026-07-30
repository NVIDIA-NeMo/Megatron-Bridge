#!/usr/bin/env bash
# Stage 1/3 of the determinism-trace e2e pipeline.
#
#   dettrace_submit.sh  ->  dettrace_wait.sh  ->  dettrace_diff.sh
#
# Submits BOTH trace arms (A + B) of the Nemotron 3 Ultra gb300 MXFP8 recipe with the
# cross-process tracer enabled (DET_TRACE_*), and records each arm's Slurm job id and
# stream dir to a state file the wait/diff stages read. The two arms are identical; they
# (likely) land on different node sets, which is what exercises cross-process
# non-determinism. See docs/determinism-debug-tool.md.
#
# Required env: HF_TOKEN WANDB_API_KEY PARTITION CONTAINER_IMAGE REPO_ROOT HF_CACHE STREAMS_ROOT
# Optional:
#   ACCOUNT (=nemotron_sw_pre)  NGPUS (=64, multiple of 64)  MAX_STEPS (=5)
#   DET_TRACE_ITERS (=1-3)  DET_TRACE_OPS (=1)  STATE_FILE (=./dettrace-state.env)
#   GRES (=gpu:4)  HF_HUB_OFFLINE (=0)  WANDB_PROJECT (=mbridge-dev)  PYTHON (=python)
#   ADDITIONAL_SLURM_PARAMS
#
# Run from the repo root.

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${PARTITION:?set PARTITION}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE (gb300 NeMo image validated for MXFP8)}"
: "${REPO_ROOT:?set REPO_ROOT (absolute path to this checkout)}"
: "${HF_CACHE:?set HF_CACHE (shared HF cache dir)}"
: "${STREAMS_ROOT:?set STREAMS_ROOT (shared-FS dir; arms write \$STREAMS_ROOT/A and /B)}"

ACCOUNT="${ACCOUNT:-nemotron_sw_pre}"
WANDB_PROJECT="${WANDB_PROJECT:-mbridge-dev}"
PYTHON="${PYTHON:-python}"
NGPUS="${NGPUS:-64}"
MAX_STEPS="${MAX_STEPS:-5}"
DET_TRACE_ITERS="${DET_TRACE_ITERS:-1-3}"
DET_TRACE_OPS="${DET_TRACE_OPS:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
GRES="${GRES:-gpu:4}"
STATE_FILE="${STATE_FILE:-./dettrace-state.env}"
ADDITIONAL_SLURM_PARAMS="${ADDITIONAL_SLURM_PARAMS:-}"

[[ "$NGPUS" =~ ^[0-9]+$ ]] && [ "$((10#$NGPUS))" -ge 64 ] && [ "$(((10#$NGPUS) % 64))" -eq 0 ] || {
  echo "ERROR: NGPUS must be a positive multiple of 64 (NVLink domain); got '$NGPUS'" >&2
  exit 2
}
NGPUS=$((10#$NGPUS))

# Guard the submodule-sync landmine before spending any queue time.
if [ ! -f "${REPO_ROOT}/3rdparty/Megatron-LM/megatron/training/models/gpt.py" ]; then
  echo "ERROR: 3rdparty/Megatron-LM is not synced (missing megatron/training/models/gpt.py)." >&2
  echo "       Fix: git -C '${REPO_ROOT}' submodule update --init --recursive" >&2
  exit 3
fi

GRES_ARG=()
[ -n "$GRES" ] && GRES_ARG=(--gres "$GRES")
SLURM_EXTRA_ARG=()
[ -n "$ADDITIONAL_SLURM_PARAMS" ] && SLURM_EXTRA_ARG=(--additional_slurm_params "$ADDITIONAL_SLURM_PARAMS")
OPS_ARG=()
[ "$DET_TRACE_OPS" = "1" ] && OPS_ARG=(-E DET_TRACE_OPS=1)
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge"

# Submit one arm; echo ONLY its Slurm job id on stdout (submit log goes to stderr).
submit_arm() {
  local arm="$1" streams="${STREAMS_ROOT}/${arm}" logf jid
  mkdir -p "$streams"
  logf=$(mktemp)
  if ! "${PYTHON}" scripts/performance/setup_experiment.py \
      --account "${ACCOUNT}" --partition "${PARTITION}" --gpu gb300 --time_limit 01:00:00 \
      -m nemotronh -mr nemotron_3_ultra -c fp8_mx -cv v1 \
      -ng "${NGPUS}" -gn 4 "${GRES_ARG[@]}" "${SLURM_EXTRA_ARG[@]}" \
      --container_image "${CONTAINER_IMAGE}" --custom_mounts "${MOUNTS}" \
      --hf_token "${HF_TOKEN}" -wdk "${WANDB_API_KEY}" -wdp "${WANDB_PROJECT}" \
      -wdj "nemotron-3-ultra-mxfp8-dettrace-${arm}" \
      --task pretrain --max_steps "${MAX_STEPS}" \
      --moe_flex_dispatcher_backend hybridep \
      -E NCCL_CUMEM_ENABLE=1 \
      -E HF_HOME="${HF_CACHE}" -E HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
      -E TRANSFORMERS_CACHE="${HF_CACHE}" \
      -E HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" -E TRANSFORMERS_OFFLINE="${HF_HUB_OFFLINE}" \
      -E DET_TRACE_OUT_DIR="${streams}" -E DET_TRACE_ITERS="${DET_TRACE_ITERS}" \
      "${OPS_ARG[@]}" >"$logf" 2>&1; then
    echo "ERROR: submit failed for arm ${arm}:" >&2; cat "$logf" >&2; rm -f "$logf"; return 1
  fi
  cat "$logf" >&2
  jid=$(grep -oE "Job id: [0-9]+" "$logf" | grep -oE "[0-9]+" | tail -1)
  rm -f "$logf"
  [ -n "$jid" ] || { echo "ERROR: no 'Job id' in submit output for arm ${arm}" >&2; return 1; }
  echo "$jid"
}

echo ">>> submitting arm A" >&2; JOB_A=$(submit_arm A)
echo ">>> submitting arm B" >&2; JOB_B=$(submit_arm B)

cat > "${STATE_FILE}" <<EOF
JOBID_A=${JOB_A}
JOBID_B=${JOB_B}
STREAMS_A=${STREAMS_ROOT}/A
STREAMS_B=${STREAMS_ROOT}/B
EOF

echo
echo "Submitted determinism-trace arms: A=${JOB_A}  B=${JOB_B}  (${NGPUS} GPU, gb300 MXFP8, HybridEP)."
echo "State written to ${STATE_FILE}."
echo "Next: scripts/determinism/dettrace_wait.sh   then   scripts/determinism/dettrace_diff.sh"
