#!/usr/bin/env bash
# 3072-GPU (768-node) MXFP8 det-ON-vs-det-OFF run for Nemotron 3 Ultra.
#
# Thin wrapper over ``launch_nemotron_3_ultra_mxfp8_compare.sh`` that only bumps
# the scale to 3072 GPUs. It inherits everything from that script — crucially the
# MXFP8 fix (``model.recompute_modules`` drops ``mlp`` so the fp8 dense-MLP
# recompute path doesn't crash on the ``padding_mask`` kwarg; see that script's
# header) — so there is no duplicated recipe body to drift out of sync.
#
# This is the 3072-GPU companion to the bf16 determinism study in
# ``scripts/performance/perf_leaderboard/analysis_report_det_vs_nondet_3072gpu.md``
# (that run was ``NGPUS=3072`` on the same compare machinery: TP=2 PP=3 EP=32
# ETP=1, GBS auto-scales 128×32 = 4096, 768 nodes × GB200, partition=batch,
# account=nemotron_sw_pre). This launcher reproduces that scale under MXFP8.
#
# Scale sanity: 3072 = 768 nodes × 4 GPUs. 3072/6 (dense DP) = 512 and 3072/96
# (expert EDP) = 32 are both integer, and GN=4 divides 3072 — so the recipe is
# valid at this scale.
#
# 768-node runs are large and slow to schedule. Recommended:
#   * SERIALIZE=1                       run one arm at a time (one 3072-GPU alloc)
#   * ADDITIONAL_SLURM_PARAMS="reservation=<res>;qos=<qos>"   if the pool requires it
#   * WAIT_TIMEOUT_SEC large (default in the inner script is fine only for short
#     queues; override to e.g. 43200 for a busy 768-node queue)
#
# Required env vars are identical to launch_nemotron_3_ultra_mxfp8_compare.sh:
#   HF_TOKEN, WANDB_API_KEY, ACCOUNT, PARTITION, CONTAINER_IMAGE, REPO_ROOT, HF_CACHE
# Optional overrides (NGPUS, OUT_DIR, SERIALIZE, ADDITIONAL_SLURM_PARAMS, ...) pass
# straight through. Run from the repo root (the inner script uses repo-relative paths).

set -euo pipefail

# 3072 GPUs by default; still overridable (e.g. NGPUS=1536) for a scaling sweep.
export NGPUS="${NGPUS:-3072}"
export OUT_DIR="${OUT_DIR:-./mxfp8-compare-3072gpu}"
# The inner script defaults WAIT_TIMEOUT_SEC=3600 (1h) — too short for a 768-node
# queue, where scheduling alone can exceed an hour and the driver would give up
# while the jobs are still valid. Default to 12h here; still overridable.
export WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-43200}"

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${_here}/launch_nemotron_3_ultra_mxfp8_compare.sh" "$@"
