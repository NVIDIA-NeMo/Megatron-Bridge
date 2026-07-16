#!/bin/bash
# One-time-use submitter that fans out Nemotron 3.5 Nano LoRA/DoRA jobs in parallel.
# Edit the rows in the `# === jobs to submit ===` block, then `bash submit_peft.sh`.

set -euo pipefail

LAUNCHER="examples/models/nemotron/nemotron_3_5/nano/slurm_peft.sh.no_share"

# Shared knobs across every job in this submission.
# GBS=16 — minimum that divides any DP we use (TP=2/CP=1 ⇒ DP=4; TP=1/CP=1 ⇒ DP=8; etc.).
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
COMMON_EXPORT+=",PEFT_SCHEME=lora"
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
# Real LoRA runs on SQuAD — 1000 iters / GBS=16 / SEQ=2048 / lr_warmup=50 / save_interval=250 / 04:00:00 wall.
# PEFT freezes the base model: no full-model grads/opt state → memory is much smaller than SFT.
# All three configs use EP=4 (lowered from EP=8) so they fit on **1 node / 8 GPUs**.
# EP=4 on 1 node: each rank holds 128/4 = 32 experts, ~13.5 GB bf16 expert weights — fits with frozen base.
#
#   L-TP2     (TP=2/PP=1/EP=4/CP=1)  → DP=2,        expert_DP=2
#   L-PP2     (TP=1/PP=2/EP=4/CP=1)  → DP=4/stage,  expert_DP=1 (PP halves expert state per rank)
#   L-TP2-CP2 (TP=2/PP=1/EP=4/CP=2)  → DP=2,        expert_DP=1

# SQuAD rows commented out — all three finished on 2026-05-14/15 (iter_1000 in
# $WORKSPACE/results/lora-squad-*). Uncomment to re-run.
# submit_run "L-TP2"     "lora-squad-tp2-ep4-cp1-1n-1k"     1 "04:00:00" \
#     "DATASET_NAME=squad,TP=2,PP=1,EP=4,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
#
# submit_run "L-PP2"     "lora-squad-tp1-pp2-ep4-1n-1k"     1 "04:00:00" \
#     "DATASET_NAME=squad,TP=1,PP=2,EP=4,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
#
# submit_run "L-TP2-CP2" "lora-squad-tp2-cp2-ep4-1n-1k"     1 "04:00:00" \
#     "DATASET_NAME=squad,TP=2,PP=1,EP=4,CP=2,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"

# OpenMathInstruct-2 LoRA sweep — SEQ=4096 (recipe override) on 1 node / 8 GPUs, GBS=16, 1000 iters.
# No CP=2 row: the prepared packed cache uses pad_seq_to_mult=1; CP=2 requires
# pad_seq_to_mult=2*CP=4 which would need a separate pre-pack run. Defer.
#
#   L-MATH-TP2  (TP=2/PP=1/EP=4/CP=1) → DP=2, expert_DP=2 — mirrors L-TP2
#   L-MATH-PP2  (TP=1/PP=2/EP=4/CP=1) → DP=4/stage, expert_DP=1 — mirrors L-PP2

submit_run "L-MATH-TP2" "lora-openmath-tp2-ep4-cp1-1n-1k"  1 "04:00:00" \
    "DATASET_NAME=openmath,RECIPE_NAME=nemotron_3_5_nano_peft_openmathinstruct2_config,SEQ_LENGTH=4096,TP=2,PP=1,EP=4,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"

submit_run "L-MATH-PP2" "lora-openmath-tp1-pp2-ep4-1n-1k"  1 "04:00:00" \
    "DATASET_NAME=openmath,RECIPE_NAME=nemotron_3_5_nano_peft_openmathinstruct2_config,SEQ_LENGTH=4096,TP=1,PP=2,EP=4,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
