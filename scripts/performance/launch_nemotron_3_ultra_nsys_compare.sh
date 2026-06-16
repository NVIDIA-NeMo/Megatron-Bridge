#!/usr/bin/env bash
# End-to-end nsys comparison for Nemotron 3 Ultra: submit one det + one non-det
# run with nsys profiling, then produce the side-by-side leaderboard.
#
# Submits THREE jobs and produces both reports:
#   1. det + nsys    (perf-comparison side A)
#   2. non-det + nsys (perf-comparison side B)
#   3. det + NO nsys (bit-wise determinism check)
# Modelled on https://github.com/NVIDIA/Megatron-LM/pull/5041 run_nsys_breakdown.sh
# run_nsys_breakdown.sh but adapted for multi-node Slurm + nemo_run.
#
# The positional Hydra overrides below MIRROR ``launch_nemotron_3_ultra_deterministic.sh``
# (bit-exact-proven at 24 nodes / 96 GPUs — jobs 2074557 / 2074641 / 2074651 /
# 2076499 / 2076503, reproduced 2102770 / 2103151 / 2103633 / 2103637).
# This script now runs at 48 nodes / 192 GPUs — TP·PP·CP=3 and EP·ETP=8 both
# still divide 192 (DP_attn=64, DP_expert=24, microbatches/step=48), so the
# recipe is valid at the new scale, but the bit-wise check below is a FRESH
# determinism validation at 48 nodes, not a reproduction of the 24-node proof.
# Only differences vs launch_nemotron_3_ultra_deterministic.sh:
#   1. determinism toggle on the 4 env vars + 2 model flags (per $MODE)
#   2. nsys profiling flags added (for det / nondet runs only)
#   3. job size: 96 GPUs → 192 GPUs (24 → 48 nodes)
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
#   NGPUS            (default 192 -- must be a multiple of TP*PP*EP and PP*GN must divide it)
#   GN               (default 4 -- GPUs per node, GB200 = 4)
#   PP_SIZE          (default 3 -- pipeline_model_parallel_size, used to auto-derive
#                     profiling_ranks = first/middle/last PP stage)

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

# --- Auto-derive profiling_ranks = first / middle / last PP stage ---
# NGPUS = world_size, GN = GPUs per node, PP_SIZE = recipe pipeline_model_parallel_size.
# We always profile the first rank of each PP stage, so each stage is represented:
#   PP=3, NGPUS=192 → ranks 0, 64, 128
#   PP=3, NGPUS=96  → ranks 0, 32, 64
#   PP=2, NGPUS=192 → ranks 0, 96
#   PP=1            → rank 0 only
NGPUS="${NGPUS:-192}"
GN="${GN:-4}"
PP_SIZE="${PP_SIZE:-3}"
RANKS_PER_PP=$((NGPUS / PP_SIZE))
case "$PP_SIZE" in
    1) PROFILE_RANKS_CSV="0" ;;
    2) PROFILE_RANKS_CSV="0,${RANKS_PER_PP}" ;;
    *) PROFILE_RANKS_CSV="0,${RANKS_PER_PP},$((RANKS_PER_PP * (PP_SIZE - 1)))" ;;
esac
PROFILE_RANKS_HYDRA="[${PROFILE_RANKS_CSV}]"
echo "Auto-selected profiling_ranks: ${PROFILE_RANKS_CSV} (NGPUS=${NGPUS}, PP=${PP_SIZE})"

submit_run() {
    local MODE="$1"  # "det" | "nondet" | "det-bitwise"
    local WDJ
    local ENABLE_NSYS=true
    case "$MODE" in
        det)         WDJ="nemotron-3-ultra-det-nsys${NSYS_START}-${NSYS_STOP}-${TS}" ;;
        nondet)      WDJ="nemotron-3-ultra-nondet-nsys${NSYS_START}-${NSYS_STOP}-${TS}" ;;
        det-bitwise) WDJ="nemotron-3-ultra-det-bitwise-check-${TS}" ; ENABLE_NSYS=false ;;
    esac

    # Determinism delta vs launch_nemotron_3_ultra_deterministic.sh:
    #   det / det-bitwise: full 4 env vars + deterministic_mode=true  + cross_entropy_loss_fusion=false
    #   nondet:            no det env vars + deterministic_mode=false + cross_entropy_loss_fusion=true
    local DET_ENVS=()
    local DET_MODE="false"
    local CE_FUSION="true"
    if [ "$MODE" != "nondet" ]; then
        DET_ENVS=(
            -E NCCL_ALGO=Ring
            -E NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
            -E CUBLAS_WORKSPACE_CONFIG=:4096:8
            -E MAMBA_DETERMINISTIC=1
        )
        DET_MODE="true"
        CE_FUSION="false"
    fi

    # nsys CLI flags are only added for the two nsys runs; the bit-wise check is a
    # plain run so its iter-50 lm loss can be diff'd against a non-instrumented baseline.
    local NSYS_CLI_FLAGS=()
    local NSYS_HYDRA_OVERRIDES=()
    if [ "$ENABLE_NSYS" = "true" ]; then
        NSYS_CLI_FLAGS=(
            --enable_nsys
            --profiling_start_step "$NSYS_START"
            --profiling_stop_step "$NSYS_STOP"
            --profiling_ranks "$PROFILE_RANKS_CSV"
            --nsys_trace cuda-sw,nvtx
            --export_nsys_sqlite
        )
        NSYS_HYDRA_OVERRIDES=(
            profiling.use_nsys_profiler=true
            profiling.profile_step_start="$NSYS_START"
            profiling.profile_step_end="$NSYS_STOP"
            "profiling.profile_ranks=$PROFILE_RANKS_HYDRA"
            profiling.nvtx_ranges=true
            profiling.record_shapes=false
        )
    fi

    # --- Positional override block: mirrors launch_nemotron_3_ultra_deterministic.sh ---
    # The two ``false → true`` last-wins DDP overlap toggles and the trailing
    # ``model.moe_flex_dispatcher_backend=hybridep`` field are kept verbatim so
    # this run is the bit-exact-proven recipe + optional nsys instrumentation.
    "$PYTHON" scripts/performance/setup_experiment.py \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --gpu gb200 \
        --time_limit 00:30:00 \
        -m nemotronh -mr nemotron_3_ultra -c bf16 -cv v1 \
        -ng "$NGPUS" -gn "$GN" \
        --container_image "$CONTAINER_IMAGE" \
        --custom_mounts "$MOUNTS" \
        -hf "$HF_TOKEN" \
        -wdk "$WANDB_API_KEY" \
        -wdp "$WANDB_PROJECT" \
        -wdj "$WDJ" \
        --task pretrain \
        "${NSYS_CLI_FLAGS[@]}" \
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
        "${NSYS_HYDRA_OVERRIDES[@]}" 2>&1 | tee "$OUT_DIR/submit-${MODE}.log"

    grep -oE "Job id: [0-9]+" "$OUT_DIR/submit-${MODE}.log" | head -1 | awk '{print $3}' > "$OUT_DIR/jobid-${MODE}.txt"
    echo "$MODE job: $(cat "$OUT_DIR/jobid-${MODE}.txt")  (wandb=$WDJ)"
    echo "$WDJ" > "$OUT_DIR/wdj-${MODE}.txt"
}

# Three jobs total:
#   det:         det + nsys                (perf-comparison side A)
#   nondet:      non-det + nsys            (perf-comparison side B)
#   det-bitwise: det + NO nsys             (bit-wise determinism check —
#                                            diffed against existing det baseline)
submit_run det
submit_run nondet
submit_run det-bitwise
JOB_DET=$(cat "$OUT_DIR/jobid-det.txt")
JOB_NONDET=$(cat "$OUT_DIR/jobid-nondet.txt")
JOB_BITWISE=$(cat "$OUT_DIR/jobid-det-bitwise.txt")

# Wait for all three to complete.
deadline=$(($(date +%s) + WAIT_TIMEOUT_SEC))
while :; do
    pending=$(squeue -j "$JOB_DET,$JOB_NONDET,$JOB_BITWISE" -h 2>/dev/null | wc -l)
    [ "$pending" -eq 0 ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "ERROR: timed out waiting for jobs $JOB_DET / $JOB_NONDET / $JOB_BITWISE" >&2
        exit 124
    fi
    echo "$(date -Iseconds)  waiting for jobs: $pending still in queue"
    sleep 30
done

# Convert .nsys-rep / .sqlite to NVTX nvtx_sum CSV — one CSV per profiled rank.
# Output: nsys-${MODE}-rank<N>.csv for every captured rank, plus a stable
# symlink nsys-${MODE}.csv → nsys-${MODE}-rank0.csv so print_nsys_leaderboard.py
# keeps working unchanged.
generate_csv() {
    local MODE="$1"
    local WDJ
    WDJ=$(cat "$OUT_DIR/wdj-${MODE}.txt")
    local exp_dir="$HOME/.nemo_run/experiments/$WDJ"
    local count=0
    while IFS= read -r SQLITE; do
        [ -z "$SQLITE" ] && continue
        local RANK
        RANK=$(echo "$SQLITE" | grep -oE 'rank[0-9]+' | head -1)
        [ -z "$RANK" ] && RANK="rankunknown"
        local CSV="$OUT_DIR/nsys-${MODE}-${RANK}.csv"
        local REP="${SQLITE%.sqlite}.nsys-rep"
        if command -v nsys >/dev/null && [ -f "$REP" ]; then
            nsys stats --force-export=true --report nvtx_sum --format csv "$REP" > "$CSV"
        else
            "$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/extract_nsys_csv.py" "$SQLITE" "$CSV"
        fi
        echo "wrote $CSV  ($(wc -l < "$CSV") rows)"
        count=$((count + 1))
    done < <(find "$exp_dir" -name "profile_*.sqlite" 2>/dev/null | sort)
    if [ "$count" -eq 0 ]; then
        echo "ERROR: no sqlite found for $MODE under $exp_dir" >&2
        exit 1
    fi
    # Compat: leaderboard expects nsys-${MODE}.csv → point it at rank 0 (first PP stage).
    if [ -f "$OUT_DIR/nsys-${MODE}-rank0.csv" ]; then
        ln -sf "nsys-${MODE}-rank0.csv" "$OUT_DIR/nsys-${MODE}.csv"
    else
        local fallback
        fallback=$(ls "$OUT_DIR"/nsys-${MODE}-rank*.csv 2>/dev/null | head -1)
        [ -n "$fallback" ] && ln -sf "$(basename "$fallback")" "$OUT_DIR/nsys-${MODE}.csv"
    fi
}

generate_csv det
generate_csv nondet

# Side-by-side leaderboard for the perf comparison (det+nsys vs non-det+nsys).
"$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/print_nsys_leaderboard.py" "$OUT_DIR" \
    | tee "$OUT_DIR/leaderboard.txt"

# --- Bit-wise determinism check ---
# Diff iter-50 lm-loss of det+nsys (2103633-style) vs det+no-nsys (2103637-style).
# If they match to the last digit, nsys instrumentation didn't perturb determinism
# AND the recipe is bit-reproducible across separate Slurm allocations.
echo ""
echo "=== Bit-wise determinism check ==="
DET_LOG=$(find "$HOME/.nemo_run/experiments/$(cat $OUT_DIR/wdj-det.txt)" -name "log-*${JOB_DET}*.out" -type f 2>/dev/null | head -1)
BIT_LOG=$(find "$HOME/.nemo_run/experiments/$(cat $OUT_DIR/wdj-det-bitwise.txt)" -name "log-*${JOB_BITWISE}*.out" -type f 2>/dev/null | head -1)
STRIP='s/^ \[[^]]+\] //; s/elapsed time per iteration \(ms\): [0-9.]+ \| //; s/throughput per GPU \(TFLOP\/s\/GPU\): [0-9.]+ \| //'
{
    echo "det+nsys log:        $DET_LOG"
    echo "det+no-nsys log:     $BIT_LOG"
    echo ""
    printf "%-6s %-22s %-22s %s\n" "iter" "det+nsys lm loss" "det+no-nsys lm loss" "match"
    printf -- "------ ---------------------- ---------------------- ------\n"
    match_all=1
    for it in 1 5 10 20 30 40 50; do
        l_det=$(grep -E "iteration\s+${it}/" "$DET_LOG" 2>/dev/null | head -1 | grep -oP 'lm loss: \S+' | sed 's/lm loss: //')
        l_bit=$(grep -E "iteration\s+${it}/" "$BIT_LOG" 2>/dev/null | head -1 | grep -oP 'lm loss: \S+' | sed 's/lm loss: //')
        m="✗" ; [ -n "$l_det" ] && [ "$l_det" = "$l_bit" ] && m="✓" || match_all=0
        printf "%-6s %-22s %-22s %s\n" "$it" "${l_det:-?}" "${l_bit:-?}" "$m"
    done
    echo ""
    [ $match_all -eq 1 ] && echo "BIT-WISE DETERMINISTIC ✓ — all iter losses match" || echo "MISMATCH ✗ — recipe is not bit-exact reproducible"
} | tee "$OUT_DIR/bitwise_check.txt"

echo ""
echo "Reports written to:"
echo "  perf leaderboard:    $OUT_DIR/leaderboard.txt"
echo "  bit-wise check:      $OUT_DIR/bitwise_check.txt"
echo "CSVs:           $OUT_DIR/nsys-det.csv  $OUT_DIR/nsys-nondet.csv"
echo "Job IDs:        det=$JOB_DET  nondet=$JOB_NONDET  det-bitwise=$JOB_BITWISE"
