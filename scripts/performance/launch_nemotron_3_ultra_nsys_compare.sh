#!/usr/bin/env bash
# End-to-end nsys comparison for Nemotron 3 Ultra: submit one det + one non-det
# run with nsys profiling, then produce the side-by-side leaderboard.
#
# Modelled on https://github.com/NVIDIA/Megatron-LM/pull/5041 's
# run_nsys_breakdown.sh but adapted for multi-node Slurm + nemo_run.
#
# The positional Hydra overrides below MIRROR ``launch_nemotron_3_ultra_deterministic.sh``
# (the bit-exact-proven recipe — jobs 2074557 / 2074641 / 2074651 / 2076499 / 2076503,
# reproduced 2102770 / 2103151). Only differences:
#   1. determinism toggle on the 4 env vars + 2 model flags (per $MODE)
#   2. nsys profiling flags added
#
# Output layout (OUT_DIR defaults to ./nsys-compare):
#   nsys-det.csv      -- NVTX nvtx_sum CSV for det run (rank 0)
#   nsys-nondet.csv   -- NVTX nvtx_sum CSV for non-det run (rank 0)
#   leaderboard.txt   -- side-by-side report (top-20 by |delta|)
#
# Required env vars (same as launch_nemotron_3_ultra_deterministic.sh):
#   HF_TOKEN, WANDB_API_KEY, ACCOUNT, PARTITION, CONTAINER_IMAGE,
#   REPO_ROOT, HF_CACHE
# Optional:
#   WANDB_PROJECT    (default "mbridge-dev")
#   OUT_DIR          (default "./nsys-compare")
#   NSYS_START/STOP  (default 15/18)
#   WAIT_TIMEOUT_SEC (default 3600)
#   PYTHON           (default "python" -- override if interpreter w/ nemo_run is elsewhere)

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${ACCOUNT:?set ACCOUNT}"
: "${PARTITION:?set PARTITION}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE}"
: "${REPO_ROOT:?set REPO_ROOT (absolute path to this checkout)}"
: "${HF_CACHE:?set HF_CACHE (shared HF cache dir)}"

WANDB_PROJECT="${WANDB_PROJECT:-mbridge-dev}"
OUT_DIR="${OUT_DIR:-./nsys-compare}"
NSYS_START="${NSYS_START:-15}"
NSYS_STOP="${NSYS_STOP:-18}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-3600}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_DIR"
OUT_DIR=$(realpath "$OUT_DIR")
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge"
TS=$(date +%s)

submit_run() {
    local MODE="$1"  # "det" | "nondet"
    local WDJ="nemotron-3-ultra-${MODE}-nsys${NSYS_START}-${NSYS_STOP}-${TS}"

    # Determinism delta vs launch_nemotron_3_ultra_deterministic.sh:
    #   det:    full 4 env vars + deterministic_mode=true  + cross_entropy_loss_fusion=false
    #   nondet: no det env vars  + deterministic_mode=false + cross_entropy_loss_fusion=true
    local DET_ENVS=()
    local DET_MODE="false"
    local CE_FUSION="true"
    if [ "$MODE" = "det" ]; then
        DET_ENVS=(
            -E NCCL_ALGO=Ring
            -E NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
            -E CUBLAS_WORKSPACE_CONFIG=:4096:8
            -E MAMBA_DETERMINISTIC=1
        )
        DET_MODE="true"
        CE_FUSION="false"
    fi

    # --- Positional override block: mirrors launch_nemotron_3_ultra_deterministic.sh ---
    # The two ``false → true`` last-wins DDP overlap toggles and the trailing
    # ``model.moe_flex_dispatcher_backend=hybridep`` field are kept verbatim so
    # this run is the bit-exact-proven recipe + nsys instrumentation.
    "$PYTHON" scripts/performance/setup_experiment.py \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --gpu gb200 \
        --time_limit 00:30:00 \
        -m nemotronh -mr nemotron_3_ultra -c bf16 -cv v1 \
        -ng 96 -gn 4 \
        --container_image "$CONTAINER_IMAGE" \
        --custom_mounts "$MOUNTS" \
        -hf "$HF_TOKEN" \
        -wdk "$WANDB_API_KEY" \
        -wdp "$WANDB_PROJECT" \
        -wdj "$WDJ" \
        --task pretrain \
        --enable_nsys --profiling_start_step "$NSYS_START" --profiling_stop_step "$NSYS_STOP" \
        --profiling_ranks 0 --nsys_trace cuda-sw,nvtx --export_nsys_sqlite \
        "${DET_ENVS[@]}" \
        -E TRITON_CACHE_AUTOTUNING=1 \
        -E HF_HOME="$HF_CACHE" \
        -E HF_DATASETS_CACHE="$HF_CACHE/datasets" \
        -E TRANSFORMERS_CACHE="$HF_CACHE" \
        model.attention_backend=fused \
        model.deterministic_mode="$DET_MODE" \
        model.cross_entropy_loss_fusion="$CE_FUSION" \
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
        train.manual_gc_interval=100 \
        profiling.use_nsys_profiler=true \
        profiling.profile_step_start="$NSYS_START" \
        profiling.profile_step_end="$NSYS_STOP" \
        profiling.profile_ranks=[0] \
        profiling.nvtx_ranges=true \
        profiling.record_shapes=false 2>&1 | tee "$OUT_DIR/submit-${MODE}.log"

    grep -oE "Job id: [0-9]+" "$OUT_DIR/submit-${MODE}.log" | head -1 | awk '{print $3}' > "$OUT_DIR/jobid-${MODE}.txt"
    echo "$MODE job: $(cat "$OUT_DIR/jobid-${MODE}.txt")  (wandb=$WDJ)"
    echo "$WDJ" > "$OUT_DIR/wdj-${MODE}.txt"
}

submit_run det
submit_run nondet
JOB_DET=$(cat "$OUT_DIR/jobid-det.txt")
JOB_NONDET=$(cat "$OUT_DIR/jobid-nondet.txt")

# Wait for both to complete.
deadline=$(($(date +%s) + WAIT_TIMEOUT_SEC))
while :; do
    pending=$(squeue -j "$JOB_DET,$JOB_NONDET" -h 2>/dev/null | wc -l)
    [ "$pending" -eq 0 ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "ERROR: timed out waiting for jobs $JOB_DET / $JOB_NONDET" >&2
        exit 124
    fi
    echo "$(date -Iseconds)  waiting for jobs: $pending still in queue"
    sleep 30
done

# Convert .nsys-rep / .sqlite to NVTX nvtx_sum CSV.
generate_csv() {
    local MODE="$1"
    local WDJ
    WDJ=$(cat "$OUT_DIR/wdj-${MODE}.txt")
    local REP SQLITE
    REP=$(find "$HOME/.nemo_run/experiments/$WDJ" -name "profile_*.nsys-rep" 2>/dev/null | head -1)
    SQLITE=$(find "$HOME/.nemo_run/experiments/$WDJ" -name "profile_*.sqlite" 2>/dev/null | head -1)
    local CSV="$OUT_DIR/nsys-${MODE}.csv"
    if command -v nsys >/dev/null && [ -n "$REP" ]; then
        nsys stats --force-export=true --report nvtx_sum --format csv "$REP" > "$CSV"
    elif [ -n "$SQLITE" ]; then
        "$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/extract_nsys_csv.py" "$SQLITE" "$CSV"
    else
        echo "ERROR: no nsys-rep or sqlite found for $MODE under $HOME/.nemo_run/experiments/$WDJ" >&2
        exit 1
    fi
    echo "wrote $CSV  ($(wc -l < "$CSV") rows)"
}

generate_csv det
generate_csv nondet

# Side-by-side leaderboard.
"$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/print_nsys_leaderboard.py" "$OUT_DIR" \
    | tee "$OUT_DIR/leaderboard.txt"

echo ""
echo "Report written to $OUT_DIR/leaderboard.txt"
echo "CSVs:           $OUT_DIR/nsys-det.csv  $OUT_DIR/nsys-nondet.csv"
echo "Job IDs:        det=$JOB_DET  nondet=$JOB_NONDET"
