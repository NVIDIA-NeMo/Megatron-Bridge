#!/usr/bin/env bash
# Launch a deterministic Nemotron 3 Ultra (550B-A55B) pretrain run on GB200 (24 nodes / 96 GPUs).
#
# This is the recipe verified bit-exact across 2026-06-09 jobs 2074557 / 2074641 / 2074651 /
# 2076499 / 2076503 (vanilla deterministic) and reproduced 2026-06-12 jobs 2102770 / 2103151
# (deterministic + DDP overlap).
#
# Dispatcher note: this recipe runs with ``moe_token_dispatcher_type=alltoall``.
# ``--moe_flex_dispatcher_backend hybridep`` is intentionally NOT set. The HybridEP
# buffer fails to allocate on NVL16-block hardware at EP=32 (CUDA fabric handle
# import requires a single NVL72-style NVLink domain). The two positional
# ``model.moe_flex_dispatcher_backend=*`` overrides below are stored but unread
# because ``moe_token_dispatcher_type`` stays ``alltoall``.
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
: "${ACCOUNT:?set ACCOUNT}"
: "${PARTITION:?set PARTITION}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE}"
: "${REPO_ROOT:?set REPO_ROOT (absolute path to this checkout)}"
: "${HF_CACHE:?set HF_CACHE (shared HF cache dir)}"

WANDB_PROJECT="${WANDB_PROJECT:-mbridge-dev}"
WANDB_JOB_NAME="${WANDB_JOB_NAME:-nemotron-3-ultra-deterministic-bf16}"

# --- Cluster-aware Slurm GPU request -----------------------------------------
# Some partitions (e.g. a generic ``batch`` partition) don't auto-allocate GPUs
# and reject jobs that don't request them; others (gb200 partitions) do. Auto-
# detect by Slurm ClusterName; override with GRES=... (GRES="" forces no --gres).
if [ -z "${GRES+x}" ]; then
  _cluster=$(scontrol show config 2>/dev/null | awk -F= '/^[[:space:]]*ClusterName/{gsub(/[[:space:]]/,"",$2);print $2}')
  case "$_cluster" in
    oci-hsg-cs-001*) GRES="gpu:4" ;;  # NVL72 batch partition needs an explicit GPU request
    *)               GRES="" ;;        # default: partition auto-allocates GPUs
  esac
fi
GRES_ARG=()
[ -n "$GRES" ] && GRES_ARG=(--gres "$GRES")

# Mount the repo on top of the container's /opt/Megatron-Bridge so local edits
# (e.g. submodule pin) take effect inside the run.
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge"

python scripts/performance/setup_experiment.py \
  --account "${ACCOUNT}" \
  --partition "${PARTITION}" \
  --gpu gb200 \
  --time_limit 00:30:00 \
  -m nemotronh -mr nemotron_3_ultra -c bf16 -cv v1 \
  -ng 96 -gn 4 \
  "${GRES_ARG[@]}" \
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
  model.attention_backend=fused \
  model.deterministic_mode=true \
  model.cross_entropy_loss_fusion=false \
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
  model.moe_flex_dispatcher_backend=hybridep \
  ddp.overlap_grad_reduce=true \
  ddp.overlap_param_gather=true \
  train.manual_gc=true \
  train.manual_gc_interval=100
