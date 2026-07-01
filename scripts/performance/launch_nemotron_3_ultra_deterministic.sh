#!/usr/bin/env bash
# Launch a deterministic Nemotron 3 Ultra (550B-A55B) pretrain run on GB200 (24 nodes / 96 GPUs).
#
# This is the recipe verified bit-exact across 2026-06-09 jobs 2074557 / 2074641 / 2074651 /
# 2076499 / 2076503 (vanilla deterministic) and reproduced 2026-06-12 jobs 2102770 / 2103151
# (deterministic + DDP overlap).
#
# Dispatcher note: this recipe runs with ``moe_token_dispatcher_type=alltoall``
# and leaves ``moe_flex_dispatcher_backend=null`` (the alltoall default). HybridEP
# is intentionally NOT selected: its buffer fails to allocate on NVL16-block
# hardware at EP=32 (CUDA fabric handle import requires a single NVL72-style
# NVLink domain). ``moe_flex_dispatcher_backend`` is only read when
# ``moe_token_dispatcher_type=flex``, so with ``alltoall`` the backend value is moot.
#
# Required env vars before running:
#   HF_TOKEN          Hugging Face token (for the Nemotron tokenizer)
#   WANDB_API_KEY     Weights & Biases API key
#   ACCOUNT           Slurm account (e.g. coreai_dlalgo_llm)
#   PARTITION         Slurm partition (e.g. gb200)
#   CONTAINER_IMAGE   Path to enroot squashfs (e.g. .../nvcr.io#nvidia/nemo:26.04.01.squashfs)
#   REPO_ROOT         Absolute path to this checkout
#   HF_CACHE          Absolute path to a shared HF cache directory
#
# Optional:
#   WANDB_PROJECT     Defaults to "mbridge-dev"
#   WANDB_JOB_NAME    Defaults to "nemotron-3-ultra-deterministic-bf16"

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
ACCOUNT="${ACCOUNT:-nemotron_sw_pre}"
: "${PARTITION:?set PARTITION}"
# Absolute path to a local enroot squashfs (enroot/pyxis won't resolve a relative path).
# For many-rank runs, stripe it across all OSTs (`lfs setstripe -c -1 <dir>` then copy the
# image in) so image reads at startup don't bottleneck a few OSTs.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/images/nemo-26.04.01.squashfs}"
: "${REPO_ROOT:?set REPO_ROOT (absolute path to this checkout)}"
: "${HF_CACHE:?set HF_CACHE (shared HF cache dir)}"

WANDB_PROJECT="${WANDB_PROJECT:-mbridge-dev}"
WANDB_JOB_NAME="${WANDB_JOB_NAME:-nemotron-3-ultra-deterministic-bf16}"
# Interpreter that has nemo_run (override if `python` on PATH lacks it).
PYTHON="${PYTHON:-python}"
# HF Hub offline: default TRUE (offline). At scale, online makes every rank call
# the HF API during tokenizer load -> 429 rate-limit -> failure. Offline reads only
# the local (pre-staged) cache. Set HF_HUB_OFFLINE=0 to force online. Drives
# TRANSFORMERS_OFFLINE too. Requires the cache to be pre-staged.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# --- Cluster-aware Slurm GPU request -----------------------------------------
# Some partitions (e.g. a generic ``batch`` partition) don't auto-allocate GPUs
# and reject jobs that don't request them; others (gb200 partitions) do. Auto-
# detect by Slurm ClusterName; override with GRES=... (GRES="" forces no --gres).
if [ -z "${GRES+x}" ]; then
  # `|| true`: scontrol can return non-zero transiently; without it, set -o pipefail
  # would make this assignment fail and `set -e` would silently kill the launcher.
  _cluster=$(scontrol show config 2>/dev/null | awk -F= '/^[[:space:]]*ClusterName/{gsub(/[[:space:]]/,"",$2);print $2}' || true)
  case "$_cluster" in
    oci-hsg-cs-001*) GRES="gpu:4" ;;  # NVL72 batch partition needs an explicit GPU request
    *)               GRES="" ;;        # default: partition auto-allocates GPUs
  esac
fi
GRES_ARG=()
[ -n "$GRES" ] && GRES_ARG=(--gres "$GRES")

# GPU count (default 24 nodes / 96 GPUs). Override NGPUS to scale; global_batch_size
# auto-scales in set_post_overrides. Keep NGPUS a multiple of 96 so dense DP
# (NGPUS/6) and expert EDP (NGPUS/96) stay integer: 96, 192, ... 3072 (=768 nodes).
NGPUS="${NGPUS:-96}"
GN="${GN:-4}"
[[ "$NGPUS" =~ ^[0-9]+$ ]] && [ "$((10#$NGPUS))" -ge 1 ] || { echo "ERROR: NGPUS must be a positive integer (got '$NGPUS')" >&2; exit 2; }
NGPUS=$((10#$NGPUS))

# Optional extra Slurm params (semicolon-separated key=value) -> setup_experiment
# --additional_slurm_params. For large-scale reserved runs, e.g.:
#   ADDITIONAL_SLURM_PARAMS="reservation=<your_reservation>;qos=<your_qos>"
ADDITIONAL_SLURM_PARAMS="${ADDITIONAL_SLURM_PARAMS:-}"
SLURM_EXTRA_ARG=()
[ -n "$ADDITIONAL_SLURM_PARAMS" ] && SLURM_EXTRA_ARG=(--additional_slurm_params "$ADDITIONAL_SLURM_PARAMS")

# Mount the repo on top of the container's /opt/Megatron-Bridge so local edits
# (e.g. submodule pin) take effect inside the run.
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge"

"${PYTHON}" scripts/performance/setup_experiment.py \
  --account "${ACCOUNT}" \
  --partition "${PARTITION}" \
  --gpu gb200 \
  --time_limit 00:30:00 \
  -m nemotronh -mr nemotron_3_ultra -c bf16 -cv v1 \
  -ng "${NGPUS}" -gn "${GN}" \
  "${GRES_ARG[@]}" \
  "${SLURM_EXTRA_ARG[@]}" \
  --container_image "${CONTAINER_IMAGE}" \
  --custom_mounts "${MOUNTS}" \
  -hf "${HF_TOKEN}" \
  -wdk "${WANDB_API_KEY}" \
  -wdp "${WANDB_PROJECT}" \
  -wdj "${WANDB_JOB_NAME}" \
  --task pretrain \
  -E NCCL_ALGO=Ring \
  -E NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
  -E CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -E MAMBA_DETERMINISTIC=1 \
  -E TRITON_CACHE_AUTOTUNING=1 \
  -E HF_HOME="${HF_CACHE}" \
  -E HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
  -E TRANSFORMERS_CACHE="${HF_CACHE}" \
  -E HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  -E TRANSFORMERS_OFFLINE="${HF_HUB_OFFLINE}" \
  model.attention_backend=fused \
  model.deterministic_mode=true \
  model.cross_entropy_loss_fusion=false \
  model.moe_router_fusion=true \
  model.moe_token_dispatcher_type=alltoall \
  model.moe_flex_dispatcher_backend=null \
  ddp.overlap_grad_reduce=false \
  ddp.overlap_param_gather=false \
  logger.tensorboard_dir=/nemo_run/tensorboard \
  logger.log_interval=1 \
  logger.log_throughput=true \
  logger.log_throughput_to_tensorboard=true \
  logger.log_memory_to_tensorboard=true \
  logger.throughput_window_size=1 \
  logger.tensorboard_log_interval=1 \
  ddp.overlap_grad_reduce=true \
  ddp.overlap_param_gather=true \
  train.manual_gc=true \
  train.manual_gc_interval=100 \
  train.fill_uninitialized_memory=false
