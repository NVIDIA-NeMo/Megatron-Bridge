#!/bin/bash
# One-time-use submitter that fans out Nemotron 3.5 Nano pretrain jobs in parallel.
# Edit the rows in the `# === jobs to submit ===` block, then `bash submit_pretrain.sh`.
#
# Each row produces one sbatch job with a distinct EXP_NAME so wandb runs and
# checkpoint dirs don't collide.

set -euo pipefail

LAUNCHER="examples/models/nemotron/nemotron_3_5/nano/slurm_pretrain.sh.no_share"

# Shared knobs across every job in this submission. Small GBS + few iters → smoke test.
# EVAL_ITERS=0 + huge EVAL_INTERVAL → skip in-loop evaluation entirely.
COMMON_EXPORT="ALL"
COMMON_EXPORT+=",MICRO_BATCH_SIZE=1"
COMMON_EXPORT+=",GLOBAL_BATCH_SIZE=128"
COMMON_EXPORT+=",TRAIN_ITERS=10"
COMMON_EXPORT+=",LR_WARMUP_ITERS=2"
COMMON_EXPORT+=",SEQ_LENGTH=4096"
COMMON_EXPORT+=",EVAL_ITERS=0"
COMMON_EXPORT+=",EVAL_INTERVAL=100000"
COMMON_EXPORT+=",WANDB_PROJECT=liding_nano35_release"

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
# Pretrain smoke matrix on 2 nodes / 16 GPUs (H100), DATASET_NAME=mock, 10 iters.
# Validates that each parallelism layout starts up, completes the forward path
# (incl. MoE all-to-all and MTP), and produces a finite, decreasing loss.
#
# Why these two:
#   TP=8/EP=8    — stress: TP > num_kv_heads (KV replicated) + EP shares the same ranks
#   TP=4/CP=2/EP=8 — context parallel layered on top of TP+EP; SP must be True
#
# TP=1 was dropped: under 16-GPU layout it OOMs because non-expert params are replicated
# per rank and TE-scoped CUDA Graphs hold ~6 GiB private pools.

# Mock-data smoke runs (already validated — uncomment to rerun).
# submit_run "P-TP8"   "pretrain-mock-tp8-ep8-2n"    2 "00:30:00" \
#     "DATASET_NAME=mock,TP=8,PP=1,EP=8,CP=1,SP=True"
# submit_run "P-CP2"  "pretrain-mock-tp4-cp2-ep8-2n" 2 "00:30:00" \
#     "DATASET_NAME=mock,TP=4,PP=1,EP=8,CP=2,SP=True"

# Additional smoke: TP=4 / EP=8 (in flight as 11774213 — leave commented to avoid resubmitting).
# submit_run "P-TP4"   "pretrain-mock-tp4-ep8-2n"    2 "00:30:00" \
#     "DATASET_NAME=mock,TP=4,PP=1,EP=8,CP=1,SP=True"

# Additional smoke: TP=2 / EP=8 (in flight as 11774343 — leave commented to avoid resubmitting).
# submit_run "P-TP2"   "pretrain-mock-tp2-ep8-2n"    2 "00:30:00" \
#     "DATASET_NAME=mock,TP=2,PP=1,EP=8,CP=1,SP=True"

# TP=1 / EP=8 on 2 nodes — OOMed previously (job 11767053): non-expert weights+grads
# are not sharded under TP=1, and CUDA Graphs hold ~6.38 GiB private pools. Skipping.
# submit_run "P-TP1"   "pretrain-mock-tp1-ep8-2n"    2 "00:30:00" \
#     "DATASET_NAME=mock,TP=1,PP=1,EP=8,CP=1,SP=True"

# DCLM 50-iter smoke (validated as 11774033 — kept for reference, leave commented).
# submit_run "P-DCLM" "pretrain-dclm-tp8-ep8-2n"    2 "01:00:00" \
#     "DATASET_NAME=dclm,TP=8,PP=1,EP=8,CP=1,SP=True,TRAIN_ITERS=50,LR_WARMUP_ITERS=5"

# Realistic DCLM training sweep — all 4 nodes / 32 GPUs / 1000 iters / GBS=128 / save_interval=250.
# Variants A and B already completed (P-DCLM-1K=11774386, P-DCLM-1K-CP2=11774429); commented to avoid resubmitting.
# submit_run "P-DCLM-1K"     "pretrain-dclm-tp2-ep8-4n-1k"     4 "04:00:00" \
#     "DATASET_NAME=dclm,TP=2,PP=1,EP=8,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
# submit_run "P-DCLM-1K-CP2" "pretrain-dclm-tp1-cp2-ep8-4n-1k" 4 "04:00:00" \
#     "DATASET_NAME=dclm,TP=1,PP=1,EP=8,CP=2,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"

# Variant C — TP=1 / PP=2 / EP=8 / CP=1 on 4 nodes / 32 GPUs.
# Mirrors the SFT S-PP2 layout for cross-family comparison. PP=2 splits 52 layers across
# 2 stages; each rank holds half the expert state. expert_DP = 32/(1*8*2*1) = 2.
submit_run "P-DCLM-1K-PP2" "pretrain-dclm-tp1-pp2-ep8-4n-1k" 4 "04:00:00" \
    "DATASET_NAME=dclm,TP=1,PP=2,EP=8,CP=1,SP=True,TRAIN_ITERS=1000,LR_WARMUP_ITERS=50,SAVE_INTERVAL=250"
