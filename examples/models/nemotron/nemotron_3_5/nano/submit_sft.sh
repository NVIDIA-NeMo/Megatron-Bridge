#!/bin/bash
# One-time-use submitter that fans out Nemotron 3.5 Nano SFT jobs in parallel.
# Edit the rows in the `# === jobs to submit ===` block, then `bash submit_sft.sh`.

set -euo pipefail

LAUNCHER="examples/models/nemotron/nemotron_3_5/nano/slurm_sft.sh.no_share"

# Shared knobs across every job in this submission.
# GBS=16 is the minimum that divides every DP we use: TP=2/CP=1 ⇒ DP=8 (16 % 8 = 0); TP=1/CP=1 ⇒ DP=16 (16 % 16 = 0).
# Real rows override TRAIN_ITERS/LR_WARMUP_ITERS/SAVE_INTERVAL.
COMMON_EXPORT="ALL"
COMMON_EXPORT+=",MICRO_BATCH_SIZE=1"
COMMON_EXPORT+=",GLOBAL_BATCH_SIZE=16"
COMMON_EXPORT+=",TRAIN_ITERS=10"
COMMON_EXPORT+=",LR_WARMUP_ITERS=2"
COMMON_EXPORT+=",SEQ_LENGTH=2048"
# EVAL_ITERS=0 + huge EVAL_INTERVAL → skip in-loop evaluation entirely.
COMMON_EXPORT+=",EVAL_ITERS=0"
COMMON_EXPORT+=",EVAL_INTERVAL=100000"
COMMON_EXPORT+=",WANDB_PROJECT=liding_nano35_release"
# Megatron-format checkpoint produced by conversion item 3 (HF → Megatron import).
COMMON_EXPORT+=",PRETRAINED_CHECKPOINT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/liding/nemo/workspace/experiments/nm6_nano_35/models/nemotron35-nano-base-050126/iter_0000000"
# Pin WORKSPACE + NEMO_HOME explicitly. The launcher defaults to ".../nm6_nano35_release"
# which is not our workspace, and enroot's selective env passthrough means relying on
# ALL is fragile. NEMO_HOME holds the prepared SQuAD + OpenMath caches.
COMMON_EXPORT+=",WORKSPACE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/liding/nemo/workspace/experiments/nm6_nano_35"
COMMON_EXPORT+=",NEMO_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/liding/nemo/workspace/experiments/nm6_nano_35/cache/nemo"

submit_run() {
    local id="$1"
    local exp="$2"
    local nodes="$3"
    local time="$4"
    local extra="$5"
    local job
    job="$(sbatch --parsable \
        --nodes="$nodes" \
        --time="$time" \
        --job-name="nano35-$exp" \
        --output="logs/nano35_${exp}_%j.out" \
        --error="logs/nano35_${exp}_%j.err" \
        --export="${COMMON_EXPORT},EXP_NAME=${exp},${extra}" \
        "${LAUNCHER}")"
    job="${job%%;*}"
    printf "%-6s %-60s %-12s %s\n" "$id" "$exp" "$job" "${nodes}node/${time}"
}

mkdir -p logs

printf "%-6s %-60s %-12s %s\n" "ID" "Run" "Job ID" "Resources"
printf "%-6s %-60s %-12s %s\n" "--" "---" "------" "---------"

# === jobs to submit ===========================================================
# Real SFT runs on SQuAD — 1000 iters / GBS=16 / SEQ=2048 / lr_warmup=50 / save_interval=250
# (4 checkpoints at 250/500/750/1000). All 2 nodes / 16 GPUs. 04:00:00 wall.
#
# Parallelism sweep on 2 nodes:
#   S-TP2     (TP=2/PP=1/EP=8/CP=1) → DP=8,  expert_DP=2,  ~50 GB/rank — comfortable fit
#   S-PP2     (TP=1/PP=2/EP=8/CP=1) → DP=8,  expert_DP=1*, ~42 GB/rank — PP=2 splits layers so expert state is halved
#   S-TP2-CP2 (TP=2/PP=1/EP=8/CP=2) → DP=4,  expert_DP=1,  ~74 GB/rank — borderline, CP doesn't split layers so expert state stays full

# SQuAD rows commented out — all three finished on 2026-05-14 (iter_1000 in
# $WORKSPACE/results/sft-squad-*). Uncomment to re-run.
# submit_run "S-TP2"     "sft-squad-tp2-ep8-cp1-2n-1k"      2 "04:00:00" \
#     "DATASET_NAME=squad,TP=2,PP=1,EP=8,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
#
# submit_run "S-PP2"     "sft-squad-tp1-pp2-ep8-2n-1k"      2 "04:00:00" \
#     "DATASET_NAME=squad,TP=1,PP=2,EP=8,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
#
# submit_run "S-TP2-CP2" "sft-squad-tp2-cp2-ep8-2n-1k"      2 "04:00:00" \
#     "DATASET_NAME=squad,TP=2,PP=1,EP=8,CP=2,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"

# OpenMathInstruct-2 sweep — SEQ=4096 (recipe override) on 2 nodes / 16 GPUs, GBS=16, 1000 iters.
# No CP=2 row: the prepared packed cache uses pad_seq_to_mult=1; CP=2 requires
# pad_seq_to_mult=2*CP=4 which would need a separate pre-pack run. Defer.
#
#   S-MATH-TP2  (TP=2/PP=1/EP=8/CP=1) → DP=8, expert_DP=2 — mirrors S-TP2, SEQ doubles vs SQuAD so ~60 GB/rank (borderline)
#   S-MATH-PP2  (TP=1/PP=2/EP=8/CP=1) → DP=8, expert_DP=1* — mirrors S-PP2, safest fit

submit_run "S-MATH-TP2" "sft-openmath-tp2-ep8-cp1-2n-1k"   2 "04:00:00" \
    "DATASET_NAME=openmath,RECIPE_NAME=nemotron_3_5_nano_sft_openmathinstruct2_config,SEQ_LENGTH=4096,TP=2,PP=1,EP=8,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"

submit_run "S-MATH-PP2" "sft-openmath-tp1-pp2-ep8-2n-1k"   2 "04:00:00" \
    "DATASET_NAME=openmath,RECIPE_NAME=nemotron_3_5_nano_sft_openmathinstruct2_config,SEQ_LENGTH=4096,TP=1,PP=2,EP=8,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
