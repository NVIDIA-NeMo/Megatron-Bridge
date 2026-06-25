#!/usr/bin/env bash
# End-to-end nsys comparison + determinism check for DeepSeek-V3 (671B) on GB200.
#
# DSv3 analog of launch_nemotron_3_ultra_nsys_compare.sh. Submits THREE jobs and
# produces both a perf leaderboard and a bit-wise determinism check:
#   1. det + nsys      (perf-comparison side A — deterministic)
#   2. non-det + nsys  (perf-comparison side B — measures determinism's perf cost)
#   3. det + NO nsys   (bit-wise determinism check — paired vs job 1)
#
# KEY DSv3 DIFFERENCE vs the Nemotron-3-Ultra recipe:
#   DSv3 runs the **HybridEP** MoE token dispatcher (moe_flex_dispatcher_backend=hybridep),
#   NOT alltoall. HybridEP at 26.04 needs the two-change determinism fix in
#   3rdparty/Megatron-LM/.../fused_a2a.py (enable_custom_allgather=False + a
#   torch.cuda.synchronize() after each non-blocking dispatch/combine), exposed as
#   the env knobs HYBRIDEP_CUSTOM_ALLGATHER / HYBRIDEP_SYNC. This launcher bind-mounts
#   the patched fused_a2a.py (see run_deepseek_v3.sh) and keeps the fix ON for ALL
#   runs so the det-vs-nondet delta isolates determinism (not the fix) and every run
#   is crash-safe. The fix's synchronize() costs overlap; set HYBRIDEP_SYNC=0 to
#   measure max-overlap perf, but that may crash / be non-deterministic at scale.
#
# Required env vars:
#   HF_TOKEN, WANDB_API_KEY, ACCOUNT, PARTITION, CONTAINER_IMAGE, REPO_ROOT, HF_CACHE
# Optional:
#   WANDB_PROJECT    (default "mbridge-dev-zhiyul")
#   OUT_DIR          (default "./nsys-compare-dsv3")
#   NSYS_START/STOP  (default 15/18)
#   WAIT_TIMEOUT_SEC (default 5400 — DSv3 50-iter run is ~23 min once it lands)
#   PYTHON           (default "python" — override if interpreter w/ nemo_run is elsewhere)
#   GRES             (Slurm GPU request, auto-detected per cluster; GRES="" to disable)
#   NGPUS            (default & MINIMUM 256 — DSv3 HybridEP EP=64 production scale;
#                    smaller is rejected. Scale up in multiples of 256.)
#   GN               (default 4 — GPUs per node, GB200 = 4)
#   HYBRIDEP_SYNC            (default 1 = sync on / fix on)
#   HYBRIDEP_CUSTOM_ALLGATHER(default 0 = enable_custom_allgather forced False / fix on)
#   HF_HUB_OFFLINE   (default 1 = offline; reads pre-staged cache, avoids HF 429 at scale)

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${ACCOUNT:?set ACCOUNT}"
: "${PARTITION:?set PARTITION}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE (local enroot squashfs)}"
: "${REPO_ROOT:?set REPO_ROOT (absolute path to this checkout)}"
: "${HF_CACHE:?set HF_CACHE (shared HF cache dir)}"

WANDB_PROJECT="${WANDB_PROJECT:-mbridge-dev-zhiyul}"
OUT_DIR="${OUT_DIR:-./nsys-compare-dsv3}"
NSYS_START="${NSYS_START:-15}"
NSYS_STOP="${NSYS_STOP:-18}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-5400}"
PYTHON="${PYTHON:-python}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# HybridEP determinism fix knobs (default = fix fully ON). Held constant across all
# three runs so the det-vs-nondet perf delta isolates determinism, not the fix.
HYBRIDEP_SYNC="${HYBRIDEP_SYNC:-1}"
HYBRIDEP_CUSTOM_ALLGATHER="${HYBRIDEP_CUSTOM_ALLGATHER:-0}"

mkdir -p "$OUT_DIR"
OUT_DIR=$(realpath "$OUT_DIR")
TS=$(date +%s)

# --- NGPUS / profiling_ranks (start / middle / last world rank) ---------------
# 256 GPUs (64 nodes x GN=4) is the SMALLEST valid base scale for DSv3: the HybridEP
# path runs at EP=64 (production), and smaller scales crash (the HybridEP buffer needs
# a full NVL domain / EP=64; sub-256 reproducers hit -N dim / illegal-address). Scale
# UP from 256 — keep NGPUS a multiple of 256 so EP=64 stays valid.
NGPUS="${NGPUS:-256}"
GN="${GN:-4}"
[[ "$NGPUS" =~ ^[0-9]+$ ]] || { echo "ERROR: NGPUS must be a positive integer, got '$NGPUS'" >&2; exit 2; }
NGPUS=$((10#$NGPUS))
[[ "$NGPUS" -ge 256 ]] || { echo "ERROR: DSv3 requires NGPUS>=256 (HybridEP EP=64 production scale); got '$NGPUS'" >&2; exit 2; }
PROFILE_RANKS_CSV="0,$((NGPUS / 2)),$((NGPUS - 1))"
PROFILE_RANKS_HYDRA="[${PROFILE_RANKS_CSV}]"
echo "Auto-selected profiling_ranks: ${PROFILE_RANKS_CSV} (NGPUS=${NGPUS})"

# --- Cluster-aware Slurm GPU request ------------------------------------------
if [ -z "${GRES+x}" ]; then
    _cluster=$(scontrol show config 2>/dev/null | awk -F= '/^[[:space:]]*ClusterName/{gsub(/[[:space:]]/,"",$2);print $2}')
    case "$_cluster" in
        oci-hsg-cs-001*) GRES="gpu:${GN}" ;;  # NVL72 batch partition needs an explicit GPU request
        *)               GRES="" ;;            # default: partition auto-allocates GPUs
    esac
    echo "Cluster '${_cluster:-unknown}' -> GRES='${GRES}'"
fi
GRES_ARG=()
[ -n "$GRES" ] && GRES_ARG=(--gres "$GRES")

# Optional extra Slurm params (semicolon-separated key=value) -> --additional_slurm_params.
ADDITIONAL_SLURM_PARAMS="${ADDITIONAL_SLURM_PARAMS:-}"
SLURM_EXTRA_ARG=()
[ -n "$ADDITIONAL_SLURM_PARAMS" ] && SLURM_EXTRA_ARG=(--additional_slurm_params "$ADDITIONAL_SLURM_PARAMS")

# --- MLM bind-mounts: overlay changed 3rdparty/Megatron-LM/*.py (the HybridEP fix) ---
# Mirrors run_deepseek_v3.sh: mount the repo over /opt/Megatron-Bridge AND overlay each
# changed MLM .py at the container's editable-install path so the fix is the imported code.
BASE_COMMIT="c8288b6c978f2b6b6c460a21b5b42114f5c0be3e"  # ultra-perf MLM pin
MEGATRON_DIR="3rdparty/Megatron-LM"
CUSTOM_MOUNTS=""
if [ -d "$REPO_ROOT/$MEGATRON_DIR" ]; then
    pushd "$REPO_ROOT" >/dev/null
    CHANGED_COMMITTED=$(git -C "$MEGATRON_DIR" diff --name-only --diff-filter=AM "$BASE_COMMIT" HEAD 2>/dev/null | grep '\.py$' || true)
    CHANGED_WT=$(git -C "$MEGATRON_DIR" diff --name-only --diff-filter=AM 2>/dev/null | grep '\.py$' || true)
    CHANGED_STAGED=$(git -C "$MEGATRON_DIR" diff --name-only --diff-filter=AM --cached 2>/dev/null | grep '\.py$' || true)
    CHANGED_UNTRACKED=$(git -C "$MEGATRON_DIR" ls-files --others --exclude-standard -- '*.py' 2>/dev/null || true)
    CHANGED_FILES=$(printf '%s\n' $CHANGED_COMMITTED $CHANGED_WT $CHANGED_STAGED $CHANGED_UNTRACKED | sort -u)
    for f in $CHANGED_FILES; do
        [ -z "$f" ] && continue
        CUSTOM_MOUNTS="${CUSTOM_MOUNTS},$REPO_ROOT/$MEGATRON_DIR/$f:/opt/Megatron-Bridge/$MEGATRON_DIR/$f"
    done
    popd >/dev/null
fi
echo "--- MLM bind-mounts ---"; echo "$CUSTOM_MOUNTS" | tr ',' '\n' | sed -n '/Megatron-LM/p' | head -40
echo "-----------------------"
if ! echo "$CUSTOM_MOUNTS" | grep -q "fused_a2a.py"; then
    echo "WARN: fused_a2a.py not detected as changed — the HybridEP determinism fix may be INACTIVE." >&2
fi
MOUNTS="/lustre:/lustre,${REPO_ROOT}:/opt/Megatron-Bridge${CUSTOM_MOUNTS}"

submit_run() {
    local MODE="$1"  # "det" | "nondet" | "det-bitwise"
    local WDJ
    local ENABLE_NSYS=true
    case "$MODE" in
        det)         WDJ="deepseek-v3-det-nsys${NSYS_START}-${NSYS_STOP}-${TS}" ;;
        nondet)      WDJ="deepseek-v3-nondet-nsys${NSYS_START}-${NSYS_STOP}-${TS}" ;;
        det-bitwise) WDJ="deepseek-v3-det-bitwise-check-${TS}" ; ENABLE_NSYS=false ;;
    esac

    # Determinism toggle: det/det-bitwise get the 3 det env vars + deterministic_mode=true
    # + cross_entropy_loss_fusion=false; nondet drops them. (No MAMBA_DETERMINISTIC — DSv3
    # has no Mamba layers.) The HybridEP fix knobs are passed for ALL modes (crash-safety).
    local DET_ENVS=()
    local DET_MODE="false"
    local CE_FUSION="true"
    if [ "$MODE" != "nondet" ]; then
        DET_ENVS=(
            -E NCCL_ALGO=Ring
            -E NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
            -E CUBLAS_WORKSPACE_CONFIG=:4096:8
        )
        DET_MODE="true"
        CE_FUSION="false"
    fi

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

    # DSv3 recipe: keep the production HybridEP dispatcher (do NOT force alltoall).
    "$PYTHON" scripts/performance/setup_experiment.py \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --gpu gb200 \
        --time_limit 00:40:00 \
        -m deepseek -mr deepseek_v3 -cv v1 \
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
        -E HYBRIDEP_SYNC="$HYBRIDEP_SYNC" \
        -E HYBRIDEP_CUSTOM_ALLGATHER="$HYBRIDEP_CUSTOM_ALLGATHER" \
        -E RACE_NOISE=0 \
        -E HF_HOME="$HF_CACHE" \
        -E HF_DATASETS_CACHE="$HF_CACHE/datasets" \
        -E TRANSFORMERS_CACHE="$HF_CACHE" \
        -E HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
        -E TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE" \
        model.attention_backend=fused \
        model.deterministic_mode="$DET_MODE" \
        model.cross_entropy_loss_fusion="$CE_FUSION" \
        comm_overlap.tp_comm_overlap=false \
        logger.tensorboard_dir=/nemo_run/tensorboard \
        logger.log_interval=1 \
        logger.log_throughput=true \
        logger.log_throughput_to_tensorboard=true \
        logger.log_memory_to_tensorboard=true \
        logger.throughput_window_size=1 \
        logger.tensorboard_log_interval=1 \
        train.manual_gc=true \
        train.manual_gc_interval=100 \
        "${NSYS_HYDRA_OVERRIDES[@]}" 2>&1 | tee "$OUT_DIR/submit-${MODE}.log"

    grep -oE "Job id: [0-9]+" "$OUT_DIR/submit-${MODE}.log" | head -1 | awk '{print $3}' > "$OUT_DIR/jobid-${MODE}.txt"
    echo "$MODE job: $(cat "$OUT_DIR/jobid-${MODE}.txt")  (wandb=$WDJ)"
    echo "$WDJ" > "$OUT_DIR/wdj-${MODE}.txt"
}

submit_run det
submit_run nondet
submit_run det-bitwise
JOB_DET=$(cat "$OUT_DIR/jobid-det.txt")
JOB_NONDET=$(cat "$OUT_DIR/jobid-nondet.txt")
JOB_BITWISE=$(cat "$OUT_DIR/jobid-det-bitwise.txt")

# Wait for all three.
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

# --- nsys -> NVTX CSV (one per profiled rank; symlink nsys-${MODE}.csv -> rank0) ---
generate_csv() {
    local MODE="$1"
    local WDJ; WDJ=$(cat "$OUT_DIR/wdj-${MODE}.txt")
    local exp_dir="$HOME/.nemo_run/experiments/$WDJ"
    local count=0
    while IFS= read -r STEM; do
        [ -z "$STEM" ] && continue
        local REP="${STEM}.nsys-rep" SQLITE="${STEM}.sqlite" BASE RANK CSV
        BASE="${STEM##*/}"
        RANK=$(echo "$BASE" | grep -oE 'rank[0-9]+' | head -1); [ -z "$RANK" ] && RANK="rankunknown"
        CSV="$OUT_DIR/nsys-${MODE}-${RANK}.csv"
        if command -v nsys >/dev/null && [ -f "$REP" ]; then
            nsys stats --force-export=true --report nvtx_sum --format csv "$REP" > "$CSV"
        elif [ -f "$SQLITE" ]; then
            "$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/extract_nsys_csv.py" "$SQLITE" "$CSV"
        else
            echo "WARNING: stem $STEM has neither .nsys-rep nor .sqlite, skipping" >&2; continue
        fi
        echo "wrote $CSV  ($(wc -l < "$CSV") rows)"; count=$((count + 1))
    done < <(find "$exp_dir" \( -name "profile_*.nsys-rep" -o -name "profile_*.sqlite" \) 2>/dev/null \
               | sed -E 's/\.(nsys-rep|sqlite)$//' | sort -uV)
    if [ "$count" -eq 0 ]; then
        echo "ERROR: no profile artifacts found for $MODE under $exp_dir" >&2; exit 1
    fi
    if [ -f "$OUT_DIR/nsys-${MODE}-rank0.csv" ]; then
        ln -sf "nsys-${MODE}-rank0.csv" "$OUT_DIR/nsys-${MODE}.csv"
    else
        local fallback; fallback=$(ls "$OUT_DIR"/nsys-${MODE}-rank*.csv 2>/dev/null | sort -V | head -1)
        [ -n "$fallback" ] && ln -sf "$(basename "$fallback")" "$OUT_DIR/nsys-${MODE}.csv"
    fi
}

generate_csv det
generate_csv nondet

# Perf leaderboard: det+nsys vs non-det+nsys.
"$PYTHON" "$REPO_ROOT/scripts/performance/perf_leaderboard/print_nsys_leaderboard.py" "$OUT_DIR" \
    | tee "$OUT_DIR/leaderboard.txt"

# --- Bit-wise determinism check: det+nsys vs det+no-nsys ---
echo ""
echo "=== Bit-wise determinism check (DeepSeek-V3) ==="
DET_LOG=$(find "$HOME/.nemo_run/experiments/$(cat $OUT_DIR/wdj-det.txt)" -name "log-*${JOB_DET}*.out" -type f 2>/dev/null | head -1)
BIT_LOG=$(find "$HOME/.nemo_run/experiments/$(cat $OUT_DIR/wdj-det-bitwise.txt)" -name "log-*${JOB_BITWISE}*.out" -type f 2>/dev/null | head -1)

extract_loss() { local LOG="$1" it="$2"; grep -E "iteration\s+${it}/" "$LOG" 2>/dev/null | head -1 | grep -oP 'lm loss: \S+' | sed 's/lm loss: //'; }
diff_table() {
    local LABEL_L="$1" LABEL_R="$2" LOG_L="$3" LOG_R="$4"
    echo ""; echo "--- $LABEL_L  vs  $LABEL_R ---"; echo "  left  log: $LOG_L"; echo "  right log: $LOG_R"; echo ""
    printf "  %-6s %-22s %-22s %s\n" "iter" "$LABEL_L lm loss" "$LABEL_R lm loss" "match"
    printf -- "  ------ ---------------------- ---------------------- ------\n"
    local match_all=1
    for it in 1 2 3 5 10 20 30 40 50; do
        local l r m
        l=$(extract_loss "$LOG_L" "$it"); r=$(extract_loss "$LOG_R" "$it")
        m="✗"; [ -n "$l" ] && [ "$l" = "$r" ] && m="✓" || match_all=0
        printf "  %-6s %-22s %-22s %s\n" "$it" "${l:-?}" "${r:-?}" "$m"
    done
    echo ""
    [ "$match_all" -eq 1 ] && echo "  → all sampled iters match exactly" || echo "  → at least one sampled iter disagrees"
}
{ diff_table "det+nsys" "det+no-nsys" "$DET_LOG" "$BIT_LOG"; } | tee "$OUT_DIR/bitwise_check.txt"

echo ""
echo "Reports written to:"
echo "  perf leaderboard:  $OUT_DIR/leaderboard.txt"
echo "  bit-wise check:    $OUT_DIR/bitwise_check.txt"
echo "Job IDs:  det=$JOB_DET  nondet=$JOB_NONDET  det-bitwise=$JOB_BITWISE"
