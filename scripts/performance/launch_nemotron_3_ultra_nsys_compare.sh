#!/usr/bin/env bash
# End-to-end nsys comparison for Nemotron 3 Ultra: submit one det + one non-det
# run with nsys profiling, plus TWO det runs without nsys, then produce the
# side-by-side leaderboard and two paired bit-wise diffs.
#
# Submits FOUR jobs and produces both reports:
#   1. det + nsys      (perf-comparison side A)
#   2. non-det + nsys  (perf-comparison side B)
#   3. det + NO nsys   (bit-wise check #1 — paired vs job 1 for nsys-on/off effect,
#                        and vs job 4 for no-nsys cross-allocation reproducibility)
#   4. det + NO nsys   (bit-wise check #2 — second independent allocation, no nsys)
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
#   GRES             (Slurm GPU request, auto-detected per cluster: on oci-hsg-cs-001
#                    defaults to "gpu:<GN>" since its batch partition won't auto-allocate
#                    GPUs; empty elsewhere. Set GRES="gpu:N" to override, GRES="" to disable.)
#   NGPUS            (default 192 -- must be a multiple of TP*PP*EP and GN must divide it)
#   GN               (default 4 -- GPUs per node, GB200 = 4)
#                    profiling_ranks is auto-derived as {0, NGPUS/2, NGPUS-1} (start/middle/last
#                    world rank), so no PP knowledge is required in this launcher.

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${ACCOUNT:?set ACCOUNT}"
: "${PARTITION:?set PARTITION}"
# Path to a local enroot squashfs. For many-rank runs, stripe it across all OSTs
# (`lfs setstripe -c -1 <dir>` then copy the image in) so image reads at startup
# don't bottleneck a few OSTs.
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE (local enroot squashfs)}"
: "${REPO_ROOT:?set REPO_ROOT (absolute path to this checkout)}"
: "${HF_CACHE:?set HF_CACHE (shared HF cache dir)}"

WANDB_PROJECT="${WANDB_PROJECT:-mbridge-dev}"
OUT_DIR="${OUT_DIR:-./nsys-compare}"
NSYS_START="${NSYS_START:-15}"
NSYS_STOP="${NSYS_STOP:-18}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-3600}"
PYTHON="${PYTHON:-python}"
# HF Hub offline: default TRUE (offline). At scale, going online makes every rank
# call the HF API during tokenizer load (is_base_mistral -> model_info), blowing
# the 1000-req/5min quota -> 429 -> tokenizer load fails -> NCCL cascade. Offline
# reads only the local (pre-staged) cache. Set HF_HUB_OFFLINE=0 to force online.
# Drives TRANSFORMERS_OFFLINE too. Requires the cache to be pre-staged.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$OUT_DIR"
OUT_DIR=$(realpath "$OUT_DIR")
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge"
TS=$(date +%s)

# --- Auto-derive profiling_ranks = start / middle / last world rank ---
# Independent of TP/PP/EP — ranks within the same PP stage execute the same
# compute graph, so "first rank of stage N" and "any rank of stage N" produce
# the same forward-graph profile. World-rank spread happens to land in
# different PP stages anyway at PP ≥ 3, while staying simpler and PP-config-agnostic.
#   NGPUS=192 → ranks 0, 96, 191
#   NGPUS=96  → ranks 0, 48, 95
#   NGPUS=2   → ranks 0, 1     (dedup)
#   NGPUS=1   → rank 0         (dedup)
NGPUS="${NGPUS:-192}"
GN="${GN:-4}"
# Validate + normalize NGPUS (catches non-numeric, leading-zero / octal traps, NGPUS<1).
[[ "$NGPUS" =~ ^[0-9]+$ ]] || { echo "ERROR: NGPUS must be a non-negative integer, got '$NGPUS'" >&2; exit 2; }
NGPUS=$((10#$NGPUS))
[[ "$NGPUS" -ge 1 ]] || { echo "ERROR: NGPUS must be >=1, got '$NGPUS'" >&2; exit 2; }
# Previous commit (1111af36) briefly supported PP_SIZE; this commit dropped it.
# Warn loudly so anyone relying on the short-lived knob notices.
if [ -n "${PP_SIZE:-}" ]; then
    echo "WARNING: PP_SIZE is no longer used; profiling_ranks is now derived from NGPUS alone." >&2
fi
case "$NGPUS" in
    1) PROFILE_RANKS_CSV="0" ;;
    2) PROFILE_RANKS_CSV="0,1" ;;
    *) PROFILE_RANKS_CSV="0,$((NGPUS / 2)),$((NGPUS - 1))" ;;
esac
PROFILE_RANKS_HYDRA="[${PROFILE_RANKS_CSV}]"
echo "Auto-selected profiling_ranks: ${PROFILE_RANKS_CSV} (NGPUS=${NGPUS})"

# --- Cluster-aware Slurm GPU request -----------------------------------------
# Some Slurm partitions (e.g. a generic ``batch`` partition) don't auto-allocate
# GPUs and reject jobs that don't request them ("Cannot find GPU specification");
# others (gb200 partitions) allocate GPUs from the partition itself. Auto-detect
# by Slurm ClusterName; override with GRES=... (GRES="" forces no --gres).
if [ -z "${GRES+x}" ]; then
    # `|| true`: scontrol can return non-zero transiently; without it, set -o pipefail
    # would make this assignment fail and `set -e` would silently kill the launcher.
    _cluster=$(scontrol show config 2>/dev/null | awk -F= '/^[[:space:]]*ClusterName/{gsub(/[[:space:]]/,"",$2);print $2}' || true)
    case "$_cluster" in
        oci-hsg-cs-001*) GRES="gpu:${GN}" ;;  # NVL72 batch partition needs an explicit GPU request
        *)               GRES="" ;;            # default: partition auto-allocates GPUs
    esac
    echo "Cluster '${_cluster:-unknown}' -> GRES='${GRES}'"
fi
GRES_ARG=()
[ -n "$GRES" ] && GRES_ARG=(--gres "$GRES")

# Optional extra Slurm params (semicolon-separated key=value) -> setup_experiment
# --additional_slurm_params. For reserved large-scale runs, e.g.:
#   ADDITIONAL_SLURM_PARAMS="reservation=<your_reservation>;qos=<your_qos>"
ADDITIONAL_SLURM_PARAMS="${ADDITIONAL_SLURM_PARAMS:-}"
# SERIALIZE=1 chains the runs with Slurm `afterany` dependencies so only ONE runs
# at a time (a single NGPUS-sized allocation) instead of all in parallel — polite to
# others sharing the reservation. Each run still gets its own wandb run / experiment
# dir (distinct -wdj); only scheduling is chained. Built per-run in submit_run().
SERIALIZE="${SERIALIZE:-0}"
PREV_JOBID=""

submit_run() {
    local MODE="$1"  # "det" | "nondet" | "det-bitwise" | "det-bitwise2"
    local WDJ
    local ENABLE_NSYS=true
    case "$MODE" in
        det)          WDJ="nemotron-3-ultra-det-nsys${NSYS_START}-${NSYS_STOP}-${TS}" ;;
        nondet)       WDJ="nemotron-3-ultra-nondet-nsys${NSYS_START}-${NSYS_STOP}-${TS}" ;;
        det-bitwise)  WDJ="nemotron-3-ultra-det-bitwise-check-${TS}" ; ENABLE_NSYS=false ;;
        det-bitwise2) WDJ="nemotron-3-ultra-det-bitwise-check2-${TS}" ; ENABLE_NSYS=false ;;
    esac

    # Determinism delta vs launch_nemotron_3_ultra_deterministic.sh:
    #   det / det-bitwise / det-bitwise2: full 4 env vars + deterministic_mode=true  + cross_entropy_loss_fusion=false
    #   nondet:                            no det env vars + deterministic_mode=false + cross_entropy_loss_fusion=true
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

    # Slurm extra params for THIS run: reservation/qos (global) + an afterany
    # dependency on the previous run when SERIALIZE=1 (one allocation at a time).
    local _extra="$ADDITIONAL_SLURM_PARAMS"
    if [ "$SERIALIZE" = "1" ] && [ -n "$PREV_JOBID" ]; then
        _extra="${_extra:+$_extra;}dependency=afterany:$PREV_JOBID"
    fi
    local SLURM_EXTRA_ARG=()
    [ -n "$_extra" ] && SLURM_EXTRA_ARG=(--additional_slurm_params "$_extra")

    # --- Positional override block: mirrors launch_nemotron_3_ultra_deterministic.sh ---
    # The two ``false → true`` last-wins DDP overlap toggles are kept verbatim so
    # this run is the bit-exact-proven recipe + optional nsys instrumentation.
    # ``moe_flex_dispatcher_backend`` stays ``null`` (the alltoall default); HybridEP
    # is not selected because ``moe_token_dispatcher_type`` stays ``alltoall``.
    "$PYTHON" scripts/performance/setup_experiment.py \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --gpu gb200 \
        --time_limit 00:30:00 \
        -m nemotronh -mr nemotron_3_ultra -c bf16 -cv v1 \
        -ng "$NGPUS" -gn "$GN" \
        "${GRES_ARG[@]}" \
        "${SLURM_EXTRA_ARG[@]}" \
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
        -E HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
        -E TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE" \
        model.attention_backend=fused \
        model.deterministic_mode="$DET_MODE" \
        model.cross_entropy_loss_fusion="$CE_FUSION" \
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
        train.fill_uninitialized_memory=false \
        "${NSYS_HYDRA_OVERRIDES[@]}" 2>&1 | tee "$OUT_DIR/submit-${MODE}.log"

    # Capture the jobid once; reuse it for the file, the log line, and the next run's
    # afterany dependency (SERIALIZE=1) — no re-reading the file.
    PREV_JOBID=$(grep -oE "Job id: [0-9]+" "$OUT_DIR/submit-${MODE}.log" | head -1 | awk '{print $3}')
    echo "$PREV_JOBID" > "$OUT_DIR/jobid-${MODE}.txt"
    echo "$MODE job: $PREV_JOBID  (wandb=$WDJ)"
    echo "$WDJ" > "$OUT_DIR/wdj-${MODE}.txt"
}

# Four jobs total:
#   det:          det + nsys                (perf-comparison side A)
#   nondet:       non-det + nsys            (perf-comparison side B)
#   det-bitwise:  det + NO nsys             (no-nsys bit-wise check #1 —
#                                             paired against det for nsys-on/off comparison
#                                             AND against det-bitwise2 for no-nsys reproducibility)
#   det-bitwise2: det + NO nsys             (no-nsys bit-wise check #2 — second independent
#                                             allocation; diffed against det-bitwise to measure
#                                             scale-induced residual without nsys confound)
submit_run det
submit_run nondet
submit_run det-bitwise
# det-bitwise2 (2nd no-nsys run): ENABLED so two independent no-nsys deterministic
# allocations can be diffed bit-wise against each other — the determinism-trust check.
submit_run det-bitwise2
JOB_DET=$(cat "$OUT_DIR/jobid-det.txt")
JOB_NONDET=$(cat "$OUT_DIR/jobid-nondet.txt")
JOB_BITWISE=$(cat "$OUT_DIR/jobid-det-bitwise.txt")
JOB_BITWISE2=$(cat "$OUT_DIR/jobid-det-bitwise2.txt")

# Wait for all four to complete.
deadline=$(($(date +%s) + WAIT_TIMEOUT_SEC))
while :; do
    pending=$(squeue -j "$JOB_DET,$JOB_NONDET,$JOB_BITWISE,$JOB_BITWISE2" -h 2>/dev/null | wc -l)
    [ "$pending" -eq 0 ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "ERROR: timed out waiting for jobs $JOB_DET / $JOB_NONDET / $JOB_BITWISE / $JOB_BITWISE2" >&2
        exit 124
    fi
    echo "$(date -Iseconds)  waiting for jobs: $pending still in queue"
    sleep 30
done

# Convert .nsys-rep / .sqlite to NVTX nvtx_sum CSV — one CSV per profiled rank.
# Output: nsys-${MODE}-rank<N>.csv for every captured rank, plus a stable
# symlink nsys-${MODE}.csv → nsys-${MODE}-rank0.csv so print_nsys_leaderboard.py
# keeps working unchanged.
#
# Iterates over unique profile *stems* (strip .nsys-rep / .sqlite extensions),
# so the loop also covers runs that emitted only one of the two artifacts.
# Per-stem preference: .nsys-rep + `nsys` binary, else .sqlite via the Python helper.
generate_csv() {
    local MODE="$1"
    local WDJ
    WDJ=$(cat "$OUT_DIR/wdj-${MODE}.txt")
    local exp_dir="$HOME/.nemo_run/experiments/$WDJ"
    local count=0
    while IFS= read -r STEM; do
        [ -z "$STEM" ] && continue
        local REP="${STEM}.nsys-rep"
        local SQLITE="${STEM}.sqlite"
        # Extract rank from the file BASENAME (not the full path) so a parent
        # directory named e.g. "rank-test" can't poison the rank label.
        local BASE RANK
        BASE="${STEM##*/}"
        RANK=$(echo "$BASE" | grep -oE 'rank[0-9]+' | head -1)
        [ -z "$RANK" ] && RANK="rankunknown"
        local CSV="$OUT_DIR/nsys-${MODE}-${RANK}.csv"
        if command -v nsys >/dev/null && [ -f "$REP" ]; then
            nsys stats --force-export=true --report nvtx_sum --format csv "$REP" > "$CSV"
        elif [ -f "$SQLITE" ]; then
            "$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/extract_nsys_csv.py" "$SQLITE" "$CSV"
        else
            echo "WARNING: stem $STEM has neither .nsys-rep nor .sqlite, skipping" >&2
            continue
        fi
        echo "wrote $CSV  ($(wc -l < "$CSV") rows)"
        count=$((count + 1))
    done < <(find "$exp_dir" \( -name "profile_*.nsys-rep" -o -name "profile_*.sqlite" \) 2>/dev/null \
               | sed -E 's/\.(nsys-rep|sqlite)$//' \
               | sort -uV)
    if [ "$count" -eq 0 ]; then
        echo "ERROR: no profile artifacts found for $MODE under $exp_dir" >&2
        exit 1
    fi
    # Compat: leaderboard expects nsys-${MODE}.csv → point it at rank 0 (first PP stage).
    if [ -f "$OUT_DIR/nsys-${MODE}-rank0.csv" ]; then
        ln -sf "nsys-${MODE}-rank0.csv" "$OUT_DIR/nsys-${MODE}.csv"
    else
        local fallback
        fallback=$(ls "$OUT_DIR"/nsys-${MODE}-rank*.csv 2>/dev/null | sort -V | head -1)
        [ -n "$fallback" ] && ln -sf "$(basename "$fallback")" "$OUT_DIR/nsys-${MODE}.csv"
    fi
}

generate_csv det
generate_csv nondet

# Side-by-side leaderboard for the perf comparison (det+nsys vs non-det+nsys).
"$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/print_nsys_leaderboard.py" "$OUT_DIR" \
    | tee "$OUT_DIR/leaderboard.txt"

# --- Bit-wise determinism check ---
# nsys-on vs nsys-off (det vs det-bitwise) — measures the nsys instrumentation effect
# on the deterministic recipe at this scale.
echo ""
echo "=== Bit-wise determinism check ==="
DET_LOG=$(find "$HOME/.nemo_run/experiments/$(cat $OUT_DIR/wdj-det.txt)" -name "log-*${JOB_DET}*.out" -type f 2>/dev/null | head -1)
BIT_LOG=$(find "$HOME/.nemo_run/experiments/$(cat $OUT_DIR/wdj-det-bitwise.txt)" -name "log-*${JOB_BITWISE}*.out" -type f 2>/dev/null | head -1)
BIT2_LOG=$(find "$HOME/.nemo_run/experiments/$(cat $OUT_DIR/wdj-det-bitwise2.txt)" -name "log-*${JOB_BITWISE2}*.out" -type f 2>/dev/null | head -1)

extract_loss() {
    local LOG="$1" it="$2"
    grep -E "iteration\s+${it}/" "$LOG" 2>/dev/null | head -1 | grep -oP 'lm loss: \S+' | sed 's/lm loss: //'
}
diff_table() {
    local LABEL_L="$1" LABEL_R="$2" LOG_L="$3" LOG_R="$4"
    echo ""
    echo "--- $LABEL_L  vs  $LABEL_R ---"
    echo "  left  log: $LOG_L"
    echo "  right log: $LOG_R"
    echo ""
    printf "  %-6s %-22s %-22s %s\n" "iter" "$LABEL_L lm loss" "$LABEL_R lm loss" "match"
    printf -- "  ------ ---------------------- ---------------------- ------\n"
    local match_all=1
    for it in 1 2 3 5 10 20 30 40 50; do
        local l r m
        l=$(extract_loss "$LOG_L" "$it")
        r=$(extract_loss "$LOG_R" "$it")
        m="✗" ; [ -n "$l" ] && [ "$l" = "$r" ] && m="✓" || match_all=0
        printf "  %-6s %-22s %-22s %s\n" "$it" "${l:-?}" "${r:-?}" "$m"
    done
    echo ""
    [ "$match_all" -eq 1 ] && echo "  → all sampled iters match exactly" \
                            || echo "  → at least one sampled iter disagrees"
}
{
    # nsys-on vs nsys-off: measures the nsys instrumentation effect on the det recipe.
    diff_table "det+nsys"    "det+no-nsys"    "$DET_LOG"  "$BIT_LOG"
    # two independent no-nsys det allocations vs each other: the determinism-trust check
    # (if these disagree, the recipe is NOT bit-exact across allocations at this scale).
    diff_table "det+no-nsys" "det+no-nsys#2"  "$BIT_LOG"  "$BIT2_LOG"
} | tee "$OUT_DIR/bitwise_check.txt"

echo ""
echo "Reports written to:"
echo "  perf leaderboard:    $OUT_DIR/leaderboard.txt"
echo "  bit-wise check:      $OUT_DIR/bitwise_check.txt"
echo "CSVs:           $OUT_DIR/nsys-det.csv  $OUT_DIR/nsys-nondet.csv"
echo "Job IDs:        det=$JOB_DET  nondet=$JOB_NONDET  det-bitwise=$JOB_BITWISE  det-bitwise2=$JOB_BITWISE2"
