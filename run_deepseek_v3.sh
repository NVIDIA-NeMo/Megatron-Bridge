#!/bin/bash
# Usage (mirrors origin/zhiyul/deterministics_gb200_dsv3_debug_forward_backward):
#   ./run_deepseek_v3.sh
#   DETERMINISTIC=true BACKEND=fused GPU=gb200 ./run_deepseek_v3.sh
#   DETERMINISTIC=true BACKEND=fused GPU=gb200 DET_DEBUG=true ./run_deepseek_v3.sh
set -euo pipefail
source ../../secrets.sh

GPU=${GPU:-"gb200"}
NUM_GPUS=${NUM_GPUS:-256}

# Optional small-scale overrides (default: keep PerfConfig production values).
# Use these to run a smaller-scale reproducer (e.g. 2 nodes / 8 GPUs).
#   NUM_LAYERS=8 PP_SIZE=2 VP_SIZE=2 EP_SIZE=4 TP_SIZE=1 NUM_GPUS=8 \
#   NUM_EXPERTS=32 FLEX_BACKEND=deepep \
#   RACE_NOISE=1 DETERMINISTIC=true BACKEND=fused GPU=gb200 ./run_deepseek_v3.sh
PP_SIZE=${PP_SIZE:-}
VP_SIZE=${VP_SIZE:-}
EP_SIZE=${EP_SIZE:-}
TP_SIZE=${TP_SIZE:-}
NUM_LAYERS=${NUM_LAYERS:-}
# NUM_EXPERTS: override model.num_moe_experts. DSv3 default (from HF config)
# is 256; with EP < 64 this gives too many local_experts per rank for the
# HybridEPBuffer to size correctly (→ -N dim / illegal address crash).
# Moonlight 16B uses 64 experts with EP=8 → 8 local/rank ratio. Pick
# NUM_EXPERTS = (4..8) * EP_SIZE to keep the ratio production-comparable.
NUM_EXPERTS=${NUM_EXPERTS:-}
# FLEX_BACKEND: model.moe_flex_dispatcher_backend override. Options:
#   hybridep  — GB200/GB300 NVL72 path (production); has the EP<64 init bug
#   deepep    — Hopper/B200/B300 path; falls back to alltoall on GB200
#   alltoall  — pure pytorch all_to_all_single (manually disables flex)
FLEX_BACKEND=${FLEX_BACKEND:-}
# RACE_NOISE=1 activates the in-train_step RacingStreams wrap added in
# 3rdparty/Megatron-LM/megatron/training/training.py (bind-mounted via the
# CUSTOM_MOUNTS logic below).
RACE_NOISE=${RACE_NOISE:-0}
export RACE_NOISE

if [ "$GPU" = "h100" ]; then
    CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-26.04.sqsh"
    ACCOUNT="coreai_dlalgo_nemorl"
    PARTITION="${PARTITION:-batch_short}"
    GPUS_PER_NODE=8
elif [ "$GPU" = "gb200" ] || [ "$GPU" = "b200" ]; then
    CONTAINER="/lustre/fsw/coreai_dlalgo_llm/zhiyul/containers/nemo-26.04.sqsh"
    ACCOUNT="coreai_dlalgo_llm"
    PARTITION="${PARTITION:-gb200}"
    GPUS_PER_NODE=4
    # Debug-branch note (kept verbatim): "AssertionError: Modules must not have
    # hooks registered at the time they are passed. However, registering hooks
    # on modules after passing them through make_graphed_callables is allowed."
    # → only flip cuda_graph_impl=none when the determinism plugin is active.
    export NVLINK_DOMAIN_SIZE=72
    export USE_MNNVL=1
else
    echo "Invalid GPU: $GPU"; exit 1
fi

WORKDIR=$(pwd)

# v0.4.1 submodule pin; anything above is "ours" (per-manager HybridEP fix etc.)
BASE_COMMIT="d7288711ba278d160d2a5a22c099915c9fe1395c"
MEGATRON_DIR="3rdparty/Megatron-LM"

# Mount changed MLM files (committed + working-tree + staged + untracked .py).
# 26.04 editable install path: /opt/Megatron-Bridge/3rdparty/Megatron-LM/
CUSTOM_MOUNTS=""
if [ -d "$MEGATRON_DIR" ]; then
    CHANGED_COMMITTED=$(git -C "$MEGATRON_DIR" diff --name-only --diff-filter=AM "$BASE_COMMIT" HEAD 2>/dev/null || true)
    CHANGED_WT=$(git -C "$MEGATRON_DIR" diff --name-only --diff-filter=AM 2>/dev/null || true)
    CHANGED_STAGED=$(git -C "$MEGATRON_DIR" diff --name-only --diff-filter=AM --cached 2>/dev/null || true)
    CHANGED_UNTRACKED=$(git -C "$MEGATRON_DIR" ls-files --others --exclude-standard -- '*.py' 2>/dev/null || true)
    CHANGED_FILES=$(printf '%s\n' $CHANGED_COMMITTED $CHANGED_WT $CHANGED_STAGED $CHANGED_UNTRACKED | sort -u)
    for f in $CHANGED_FILES; do
        [ -z "$f" ] && continue
        CUSTOM_MOUNTS="${CUSTOM_MOUNTS},$WORKDIR/$MEGATRON_DIR/$f:/opt/Megatron-Bridge/$MEGATRON_DIR/$f"
    done
fi

echo "--- MLM bind-mounts (target: /opt/Megatron-Bridge/$MEGATRON_DIR/...): ---"
echo "$CUSTOM_MOUNTS" | tr ',' '\n' | sed -n '/Megatron-LM/p' | head -40
echo "------------------------------------------------------------------------"
if ! echo "$CUSTOM_MOUNTS" | grep -q "fused_a2a.py"; then
    echo "WARN: no MLM file edits detected (stock v0.4.1 MLM). Continuing." >&2
fi

export DETERMINISTIC=${DETERMINISTIC:-false}
export BACKEND=${BACKEND:-fused}
export DET_DEBUG=${DET_DEBUG:-false}

export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

# MINIMAL=true: strict-minimum-override mode for testing whether the stock
# v0.4.x recipe + `model.deterministic_mode=true` alone is enough. Drops all
# other Hydra overrides (attention_backend, cross_entropy_loss_fusion,
# tp_comm_overlap) and only keeps determinism-required env vars.
MINIMAL=${MINIMAL:-false}

if [ "$BACKEND" = "flash" ]; then
    export NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0 NVTE_FLASH_ATTN=1
elif [ "$BACKEND" = "fused" ]; then
    export NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0 NVTE_FLASH_ATTN=0
else
    echo "Invalid BACKEND=$BACKEND"; exit 1
fi

if [ "$MINIMAL" = "true" ]; then
    additional_args=""
else
    additional_args="model.attention_backend=${BACKEND}"
fi

if [ "$DETERMINISTIC" = "true" ]; then
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    if [ "$MINIMAL" = "true" ]; then
        # Per _validate_and_apply_deterministic_mode in src/megatron/bridge/training/config.py:
        #   - deterministic_mode=true is the toggle (auto-calls torch.use_deterministic_algorithms)
        #   - cross_entropy_loss_fusion=false is asserted (REQUIRED)
        # tp_comm_overlap=false is defensive: the DSv3 recipe already sets it False, but
        # asserting here makes the determinism contract explicit and resilient to recipe drift.
        additional_args="model.deterministic_mode=true model.cross_entropy_loss_fusion=false comm_overlap.tp_comm_overlap=false"
        EXP_NAME="deterministic-minimal-${GPU}"
    else
        additional_args="${additional_args} \
            model.deterministic_mode=true \
            model.cross_entropy_loss_fusion=false \
            comm_overlap.tp_comm_overlap=false"
        EXP_NAME="deterministic-${BACKEND}-${GPU}"
    fi
else
    EXP_NAME="non-deterministic-${BACKEND}-${GPU}"
fi

# Aggressive recompute knob (opt-in) — full-granularity uniform recompute.
# Bypasses the per-module recompute_modules path (which wraps MoE in
# custom_forward and trips DeepEP at this version). Full granularity
# recomputes every transformer layer's forward in the backward pass.
if [ "${AGG_RECOMPUTE:-false}" = "true" ]; then
    # HybridEP-safe non-MoE recompute: mla_up_proj (PerfConfig default) +
    # layernorm. Both use output-discarding checkpointing, neither replays
    # forward through HybridEP's dispatch_with_permute (which would trigger -4).
    # Adds ~5-10 GiB activation memory savings vs r1/r2 (mla_up_proj only) —
    # enough headroom for the +450 MiB step-4 graph workspace.
    additional_args="${additional_args} \
        model.recompute_granularity=selective \
        model.recompute_modules=[mla_up_proj,layernorm]"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    EXP_NAME="${EXP_NAME}-aggrec"
fi

# DeterminismDebugPlugin: hooks conflict with make_graphed_callables → must
# turn cuda_graph off, AND it costs memory, so this is opt-in.
if [ "$DET_DEBUG" = "true" ]; then
    DET_INTERVAL=${DET_INTERVAL:-10}
    DET_SAMPLE=${DET_SAMPLE:-0.05}
    additional_args="${additional_args} \
        model.cuda_graph_impl=none \
        model.determinism_debug_enabled=true \
        model.determinism_check_interval=${DET_INTERVAL} \
        model.determinism_layer_sample_rate=${DET_SAMPLE} \
        model.determinism_num_repeats=2"
    EXP_NAME="${EXP_NAME}-detprobe"
fi

# Optional pinned nodelist for two-rerun pairing.
SLURM_EXTRA=""
if [ -n "${NODELIST:-}" ]; then
    SLURM_EXTRA="--additional_slurm_params nodelist=${NODELIST}"
fi

# Optional global batch size override (-gb flag in argument_parser.py).
GBS_ARG=""
if [ -n "${GBS:-}" ]; then
    GBS_ARG="-gb ${GBS}"
fi

# Small-scale model overrides.
# Pass NUM_LAYERS via the proper --num_layers argparse flag (not as a Hydra
# override).  set_user_overrides() in overrides.py handles --num_layers and
# automatically resets moe_layer_freq to match (3 dense + N-3 MoE for DSv3).
# Using the Hydra path (model.num_layers=N) bypasses that fix and causes:
#   AssertionError: Invalid length of moe_layer_pattern: 61, expected N
# Also null pipeline_model_parallel_layout: PerfConfig pins a 61-layer layout
# that won't match a smaller model.
NUM_LAYERS_ARG=""
if [ -n "$NUM_LAYERS" ]; then
    NUM_LAYERS_ARG="--num_layers ${NUM_LAYERS}"
    # Small-scale mode: disable CUDA graphs.  Production runs with
    # cuda_graph_impl=transformer_engine but at small scale (PP=2, VP=2,
    # NUM_LAYERS=8) the graph capture exposes an AccumulateGrad
    # stream-mismatch that triggers cudaErrorIllegalAddress on the PP
    # watchdog during the first backward pass.  Memory budget at small
    # scale is generous, so disabling graphs is safe.
    additional_args="${additional_args} \
        model.pipeline_model_parallel_layout=null \
        model.cuda_graph_impl=none"
fi

# Parallelism CLI overrides (defaults: PerfConfig values for the recipe).
PARALLELISM_ARGS=""
[ -n "$PP_SIZE" ] && PARALLELISM_ARGS="${PARALLELISM_ARGS} -pp ${PP_SIZE}"
[ -n "$VP_SIZE" ] && PARALLELISM_ARGS="${PARALLELISM_ARGS} -vp ${VP_SIZE}"
[ -n "$EP_SIZE" ] && PARALLELISM_ARGS="${PARALLELISM_ARGS} -ep ${EP_SIZE}"
[ -n "$TP_SIZE" ] && PARALLELISM_ARGS="${PARALLELISM_ARGS} -tp ${TP_SIZE}"

# MoE-shape overrides (Hydra args). Keep num_local_experts = num_experts/EP
# at a buffer-friendly value to avoid the HybridEP -N dim crash that fires
# when num_local_experts vastly exceeds what production sees (256/64=4 prod;
# 256/4=64 here → buffer overflows).
if [ -n "$NUM_EXPERTS" ]; then
    additional_args="${additional_args} model.num_moe_experts=${NUM_EXPERTS}"
fi
if [ -n "$FLEX_BACKEND" ]; then
    if [ "$FLEX_BACKEND" = "alltoall" ]; then
        # Manually disable flex; force the plain alltoall dispatcher.
        additional_args="${additional_args} \
            model.moe_token_dispatcher_type=alltoall \
            model.moe_flex_dispatcher_backend=null"
    else
        additional_args="${additional_args} model.moe_flex_dispatcher_backend=${FLEX_BACKEND}"
    fi
fi

# Use the 2602-side venv for nemo_run (with upgraded version on /tmp PYTHONPATH).
PYTHON="${PYTHON:-/lustre/fsw/coreai_dlalgo_llm/zhiyul/deterministics/Megatron-Bridge-2602/venv/bin/python}"

env PYTHONPATH=/tmp/nemo_run_2604 $PYTHON scripts/performance/setup_experiment.py \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --gpu "$GPU" \
    --time_limit "01:00:00" \
    -m deepseek \
    -mr deepseek_v3 \
    -cv v1 \
    -ng "$NUM_GPUS" \
    -gn "$GPUS_PER_NODE" \
    --container_image "$CONTAINER" \
    --custom_mounts "/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge$CUSTOM_MOUNTS" \
    --custom_env_vars "RACE_NOISE=${RACE_NOISE},CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0},HYBRIDEP_SYNC=${HYBRIDEP_SYNC:-1}" \
    -hf "$HF_TOKEN" \
    -wdk "$WANDB_API_KEY" \
    -wdp "mbridge-dev-zhiyul" \
    -wdj "${WANDB_JOB_NAME:-deepseek-v3-nemo-26.04-${EXP_NAME}}" \
    --task pretrain \
    $SLURM_EXTRA \
    $PARALLELISM_ARGS \
    $NUM_LAYERS_ARG \
    $GBS_ARG \
    $additional_args
