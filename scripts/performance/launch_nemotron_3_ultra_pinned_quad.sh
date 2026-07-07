#!/usr/bin/env bash
# Pinned-node 2x2 quad for Nemotron 3 Ultra determinism perf @ 24 nodes (96 GPUs).
#
# WHY THIS EXISTS
#   The det-vs-nondet step-time signal (~0.5%) is swamped by between-allocation
#   node-placement variance (~2%), so comparing runs from *different* allocations
#   cannot resolve it. And nsys instrumentation adds a persistent ~2% out-of-window
#   tax (CUPTI/injection stays resident on the profiled ranks the whole run, gating
#   the collective step time), which is likely det-heavier and thus inflates the
#   measured determinism gap. This script removes BOTH confounds by running all four
#   arms on the SAME physical nodes:
#
#     {det, nondet} x {nsys, no-nsys}   (4 jobs, serialized on one pinned node-set)
#
#   Yields:
#     det-nonsys vs nondet-nonsys  -> CLEAN determinism step-time delta (no nsys tax)
#     det-nsys   vs nondet-nsys    -> the section-2 NVTX decomposition, same nodes
#     det-nsys   vs det-nonsys     -> nsys out-of-window tax on identical hardware
#
#   Node pinning: arm 1 (det-nonsys) is submitted with NO nodelist; once it is
#   allocated we capture its node-set and pin the remaining three arms to it with an
#   afterany dependency chain, so they run one-at-a-time on the identical nodes.
#
# Positional Hydra overrides mirror launch_nemotron_3_ultra_nsys_compare.sh exactly
# (bit-exact-proven recipe) minus the nsys toggle, which is now per-arm.
#
# Required env: HF_TOKEN, WANDB_API_KEY
# Optional env (defaults recovered from the prior 24n router-fusion runs):
#   ACCOUNT (nemotron_sw_post), PARTITION (batch), CONTAINER_IMAGE, REPO_ROOT,
#   HF_CACHE, WANDB_PROJECT (mbridge-dev), OUT_DIR (./nsys-pinned-quad-24node),
#   NGPUS (96), GN (4), NSYS_START/STOP (15/18), HF_HUB_OFFLINE (1),
#   ALLOC_TIMEOUT_SEC (3600 -- how long to wait for arm-1 to grab nodes),
#   WAIT_TIMEOUT_SEC (14400 -- how long to wait for all 4 to finish),
#   RESERVATION / QOS (optional; appended to every arm's slurm params).

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
PARTITION="${PARTITION:-batch}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/images/nemo-26.04.01.squashfs}"
REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
HF_CACHE="${HF_CACHE:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/hf_cache}"
WANDB_PROJECT="${WANDB_PROJECT:-mbridge-dev}"
OUT_DIR="${OUT_DIR:-./nsys-pinned-quad-24node}"
NGPUS="${NGPUS:-96}"
GN="${GN:-4}"
NSYS_START="${NSYS_START:-15}"
NSYS_STOP="${NSYS_STOP:-18}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
ALLOC_TIMEOUT_SEC="${ALLOC_TIMEOUT_SEC:-3600}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-14400}"
PYTHON="${PYTHON:-python}"
RESERVATION="${RESERVATION:-}"
QOS="${QOS:-}"

mkdir -p "$OUT_DIR"
OUT_DIR=$(realpath "$OUT_DIR")
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge"
TS=$(date +%s)
PROFILE_RANKS_CSV="0,$((NGPUS / 2)),$((NGPUS - 1))"
PROFILE_RANKS_HYDRA="[${PROFILE_RANKS_CSV}]"
GRES_ARG=(--gres "gpu:${GN}")

# Global slurm params common to every arm (reservation / qos, if set).
_common_slurm() {
    local s=""
    [ -n "$RESERVATION" ] && s="reservation=${RESERVATION}"
    [ -n "$QOS" ] && s="${s:+$s;}qos=${QOS}"
    echo "$s"
}

# submit_arm <det|nondet> <true|false use_nsys> <nodelist|""> <dep_jobid|"">
# Writes jobid-<tag>.txt / wdj-<tag>.txt / submit-<tag>.log under OUT_DIR.
submit_arm() {
    local MODE="$1" USE_NSYS="$2" NODELIST="$3" DEP="$4"
    local tag="${MODE}-$([ "$USE_NSYS" = true ] && echo nsys || echo nonsys)"
    local WDJ="nemotron-3-ultra-pinnedquad-${tag}-${TS}"

    local DET_ENVS=() DET_MODE="false" CE_FUSION="true"
    if [ "$MODE" = det ]; then
        DET_ENVS=(-E NCCL_ALGO=Ring -E NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 -E CUBLAS_WORKSPACE_CONFIG=:4096:8 -E MAMBA_DETERMINISTIC=1)
        DET_MODE="true"; CE_FUSION="false"
    fi

    local NSYS_CLI=() NSYS_HY=()
    if [ "$USE_NSYS" = true ]; then
        NSYS_CLI=(--enable_nsys --profiling_start_step "$NSYS_START" --profiling_stop_step "$NSYS_STOP"
                  --profiling_ranks "$PROFILE_RANKS_CSV" --nsys_trace cuda-sw,nvtx --export_nsys_sqlite)
        NSYS_HY=(profiling.use_nsys_profiler=true profiling.profile_step_start="$NSYS_START"
                 profiling.profile_step_end="$NSYS_STOP" "profiling.profile_ranks=$PROFILE_RANKS_HYDRA"
                 profiling.nvtx_ranges=true profiling.record_shapes=false)
    fi

    local extra; extra="$(_common_slurm)"
    [ -n "$NODELIST" ] && extra="${extra:+$extra;}nodelist=${NODELIST}"
    [ -n "$DEP" ]      && extra="${extra:+$extra;}dependency=afterany:${DEP}"
    local SLURM_EXTRA=()
    [ -n "$extra" ] && SLURM_EXTRA=(--additional_slurm_params "$extra")

    echo ">>> submitting $tag  (nodelist='${NODELIST:-<scheduler>}'  dep='${DEP:-none}')" >&2
    "$PYTHON" scripts/performance/setup_experiment.py \
        --account "$ACCOUNT" --partition "$PARTITION" --gpu gb200 --time_limit 00:30:00 \
        -m nemotronh -mr nemotron_3_ultra -c bf16 -cv v1 -ng "$NGPUS" -gn "$GN" \
        "${GRES_ARG[@]}" "${SLURM_EXTRA[@]}" \
        --container_image "$CONTAINER_IMAGE" --custom_mounts "$MOUNTS" \
        -hf "$HF_TOKEN" -wdk "$WANDB_API_KEY" -wdp "$WANDB_PROJECT" -wdj "$WDJ" --task pretrain \
        "${NSYS_CLI[@]}" "${DET_ENVS[@]}" \
        -E TRITON_CACHE_AUTOTUNING=1 -E HF_HOME="$HF_CACHE" -E HF_DATASETS_CACHE="$HF_CACHE/datasets" \
        -E TRANSFORMERS_CACHE="$HF_CACHE" -E HF_HUB_OFFLINE="$HF_HUB_OFFLINE" -E TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE" \
        model.attention_backend=fused \
        model.deterministic_mode="$DET_MODE" \
        model.cross_entropy_loss_fusion="$CE_FUSION" \
        model.moe_router_fusion=true \
        model.moe_token_dispatcher_type=alltoall \
        model.moe_flex_dispatcher_backend=null \
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
        train.fill_uninitialized_memory=false \
        "${NSYS_HY[@]}" > "$OUT_DIR/submit-${tag}.log" 2>&1

    local jid
    jid=$(grep -oE "Job id: [0-9]+" "$OUT_DIR/submit-${tag}.log" | head -1 | awk '{print $3}' || true)
    if [ -z "$jid" ]; then
        echo "ERROR: no Job id parsed for $tag; see $OUT_DIR/submit-${tag}.log" >&2
        exit 1
    fi
    echo "$jid" > "$OUT_DIR/jobid-${tag}.txt"
    echo "$WDJ" > "$OUT_DIR/wdj-${tag}.txt"
    echo "    $tag -> job $jid  (wandb=$WDJ)" >&2
}

# Poll until <jobid> has an allocated node-set; echo the compressed nodelist.
wait_for_nodes() {
    local jid="$1" deadline
    deadline=$(( $(date +%s) + ALLOC_TIMEOUT_SEC ))
    while :; do
        local st n
        st=$(squeue -j "$jid" -h -o "%T" 2>/dev/null | tr -d ' ' || true)
        n=$(squeue -j "$jid" -h -o "%N" 2>/dev/null | tr -d ' ' || true)
        if [ -n "$n" ] && [ "$n" != "(null)" ] && [ "$n" != "n/a" ]; then
            echo "$n"; return 0
        fi
        case "$st" in
            ""|CANCELLED|FAILED|TIMEOUT|COMPLETED|NODE_FAIL|BOOT_FAIL)
                echo "ERROR: arm-1 job $jid reached state '$st' before a node-set could be captured" >&2
                return 1 ;;
        esac
        [ "$(date +%s)" -gt "$deadline" ] && { echo "ERROR: timed out waiting for job $jid to allocate nodes" >&2; return 1; }
        echo "    $(date -Iseconds)  job $jid state=$st, no nodes yet..." >&2
        sleep 20
    done
}

# ---- submit the quad -------------------------------------------------------
# Arm 1 picks the node-set; arms 2-4 pin to it via an afterany chain.
submit_arm det false "" ""
J1=$(cat "$OUT_DIR/jobid-det-nonsys.txt")
NODES=$(wait_for_nodes "$J1")
echo "$NODES" > "$OUT_DIR/nodeset.txt"
echo ">>> pinned node-set: $NODES" >&2

submit_arm nondet false "$NODES" "$J1"; J2=$(cat "$OUT_DIR/jobid-nondet-nonsys.txt")
submit_arm det    true  "$NODES" "$J2"; J3=$(cat "$OUT_DIR/jobid-det-nsys.txt")
submit_arm nondet true  "$NODES" "$J3"; J4=$(cat "$OUT_DIR/jobid-nondet-nsys.txt")

echo ""
echo "Submitted quad on node-set $NODES:"
echo "  det-nonsys=$J1  nondet-nonsys=$J2  det-nsys=$J3  nondet-nsys=$J4"
echo "Job ids + node-set saved under $OUT_DIR/"
echo "Analyze after completion with: OUT_DIR=$OUT_DIR bash scripts/performance/analyze_pinned_quad.sh"
