#!/usr/bin/env bash
# Two-arm cross-process DETERMINISM TRACE of the Nemotron 3 Ultra gb300 MXFP8 recipe.
#
# Submits two identical jobs (arm A and arm B) with the determinism tracer enabled
# (DET_TRACE_*), so their per-rank fingerprint streams can be diffed to the FIRST
# divergent op — the empirical first source of cross-process non-determinism. See
# docs/determinism-debug-tool.md (Reproduction section) for the full method.
#
# Recipe: nemotron_3_ultra @ gb300 / MXFP8 (-c fp8_mx), Megatron-FSDP + flex/HybridEP,
# CuteDSL fused grouped MLP, selective moe_act recompute. Default NGPUS=64 (16 nodes,
# 1 NVLink domain) so it schedules quickly; scale up in multiples of 64.
#
# Three landmines this script defends against (each costs a full multi-hour queue
# round if missed):
#   1. Submodule must be synced or the run dies at import
#      (ModuleNotFoundError: megatron.training.models.gpt). Checked below.
#   2. HybridEP must be kept: setup_experiment forces alltoall unless
#      --moe_flex_dispatcher_backend hybridep is passed (done below).
#   3. The HybridEP combine is a tracer blind spot, so DET_TRACE_OPS=1 is required
#      (default on) — otherwise the MoE path produces no trace records.
#
# Required env:
#   HF_TOKEN WANDB_API_KEY PARTITION CONTAINER_IMAGE REPO_ROOT HF_CACHE STREAMS_ROOT
#     CONTAINER_IMAGE  gb300 NeMo squashfs (26.08.rc2 validated for the MXFP8 path)
#     REPO_ROOT        absolute path to this checkout (mounted into the container)
#     HF_CACHE         shared HF cache dir (HF online is fine at <= 128 GPU)
#     STREAMS_ROOT     shared-FS dir for the .fp streams; arms write $STREAMS_ROOT/{A,B}
# Optional:
#   ACCOUNT (=nemotron_sw_pre)  NGPUS (=64, multiple of 64)  MAX_STEPS (=5)
#   DET_TRACE_ITERS (=1-3)  DET_TRACE_OPS (=1)  ADDITIONAL_SLURM_PARAMS  GRES (=gpu:4)
#   HF_HUB_OFFLINE (=0)  WANDB_PROJECT (=mbridge-dev)  PYTHON (=python)
#
# Run from the repo root. The two arms land on (likely) different node sets, which is
# what exercises cross-process / topology-dependent non-determinism.

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${PARTITION:?set PARTITION}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE (gb300 NeMo 26.08.rc2 squashfs)}"
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
ADDITIONAL_SLURM_PARAMS="${ADDITIONAL_SLURM_PARAMS:-}"

# NGPUS must be a positive multiple of 64 (the gb300 NVLink domain / EP=64 group).
[[ "$NGPUS" =~ ^[0-9]+$ ]] && [ "$((10#$NGPUS))" -ge 64 ] && [ "$(((10#$NGPUS) % 64))" -eq 0 ] || {
  echo "ERROR: NGPUS must be a positive multiple of 64 (NVLink domain); got '$NGPUS'" >&2
  exit 2
}
NGPUS=$((10#$NGPUS))

# Landmine #1 guard: the mounted submodule must contain the training GPT builder.
if [ ! -f "${REPO_ROOT}/3rdparty/Megatron-LM/megatron/training/models/gpt.py" ]; then
  echo "ERROR: 3rdparty/Megatron-LM is not synced to the branch pin" >&2
  echo "       (missing megatron/training/models/gpt.py -> import crash before iter 1)." >&2
  echo "       Fix: git -C '${REPO_ROOT}' submodule update --init --recursive" >&2
  exit 3
fi

GRES_ARG=()
[ -n "$GRES" ] && GRES_ARG=(--gres "$GRES")
SLURM_EXTRA_ARG=()
[ -n "$ADDITIONAL_SLURM_PARAMS" ] && SLURM_EXTRA_ARG=(--additional_slurm_params "$ADDITIONAL_SLURM_PARAMS")
OPS_ARG=()
[ "$DET_TRACE_OPS" = "1" ] && OPS_ARG=(-E DET_TRACE_OPS=1)

# Mount the repo on top of the container's copy so this exact checkout (incl. the
# synced submodule and any local tracer edits) is what runs.
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge"

submit_arm() {
  local arm="$1"
  local streams="${STREAMS_ROOT}/${arm}"
  mkdir -p "$streams"
  echo ">>> submitting determinism-trace arm ${arm}  (NGPUS=${NGPUS}, streams=${streams})"
  "${PYTHON}" scripts/performance/setup_experiment.py \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --gpu gb300 \
    --time_limit 01:00:00 \
    -m nemotronh -mr nemotron_3_ultra -c fp8_mx -cv v1 \
    -ng "${NGPUS}" -gn 4 \
    "${GRES_ARG[@]}" \
    "${SLURM_EXTRA_ARG[@]}" \
    --container_image "${CONTAINER_IMAGE}" \
    --custom_mounts "${MOUNTS}" \
    --hf_token "${HF_TOKEN}" \
    -wdk "${WANDB_API_KEY}" \
    -wdp "${WANDB_PROJECT}" \
    -wdj "nemotron-3-ultra-mxfp8-dettrace-${arm}" \
    --task pretrain \
    --max_steps "${MAX_STEPS}" \
    --moe_flex_dispatcher_backend hybridep \
    -E NCCL_CUMEM_ENABLE=1 \
    -E HF_HOME="${HF_CACHE}" \
    -E HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
    -E TRANSFORMERS_CACHE="${HF_CACHE}" \
    -E HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    -E TRANSFORMERS_OFFLINE="${HF_HUB_OFFLINE}" \
    -E DET_TRACE_OUT_DIR="${streams}" \
    -E DET_TRACE_ITERS="${DET_TRACE_ITERS}" \
    "${OPS_ARG[@]}"
}

submit_arm A
submit_arm B

cat <<EOF

Both determinism-trace arms submitted (${NGPUS} GPU, gb300 MXFP8, HybridEP,
op-trace iters ${DET_TRACE_ITERS}). When both jobs finish and have written streams,
diff them to the first divergence:

  ${PYTHON} src/megatron/bridge/training/utils/determinism/diff_streams.py \\
      ${STREAMS_ROOT}/A  ${STREAMS_ROOT}/B
EOF
