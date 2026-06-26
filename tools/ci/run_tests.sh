#!/usr/bin/env bash
# Run the CPU unit suite against two checkouts and fail only on regressions —
# tests that pass in BASE_DIR (main) but fail in HEAD_DIR (the PR). This keeps
# the public NeMo base image's pre-existing CPU-only failures from blocking PRs.
#
# Usage:  IMAGE=<image> tools/ci/run_tests.sh <base_dir> <head_dir>
# Local:  IMAGE=nvcr.io/nvidia/nemo:26.06 tools/ci/run_tests.sh ../main .
set -uo pipefail

IMAGE="${IMAGE:?set IMAGE to the test image}"
BASE_DIR="${1:?usage: run_tests.sh <base_dir> <head_dir>}"
HEAD_DIR="${2:?usage: run_tests.sh <base_dir> <head_dir>}"

# Run the unit suite inside the image against a checkout; all output -> log file.
# --continue-on-collection-errors: keep CPU-unimportable modules from aborting the
# run (exit 2); they become per-item ERRORs the regression diff handles normally.
run_suite() {  # $1 = source dir, $2 = log path
  docker run --rm -v "$1:/branch:ro" -w /branch \
    -e PYTHONPATH=/branch/src -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES="" \
    "${IMAGE}" \
    pytest tests/unit_tests --ignore=tests/unit_tests/diffusion -m "not pleasefixme" \
      --continue-on-collection-errors -p no:cacheprovider -q -rfE >"$2" 2>&1
}

# pytest exits 0 (all passed) or 1 (some failed); both mean the suite ran, so the
# results are trustworthy. Any other code is a crash (OOM 137, segfault 139,
# docker 125, ...) — abort rather than report a misleading diff.
require_clean_run() {  # $1 = exit code, $2 = source dir, $3 = log path
  if [ "$1" -ne 0 ] && [ "$1" -ne 1 ]; then
    echo "::error::test suite for '$2' exited $1 (crash/infra failure)" >&2
    cat "$3" >&2
    exit 2
  fi
}

# Extract failing pytest node IDs from a log, sorted and de-duplicated.
extract_failures() { grep -E '^(FAILED|ERROR) ' "$1" | awk '{print $2}' | sort -u; }

run_suite "${BASE_DIR}" base.log
require_clean_run $? "${BASE_DIR}" base.log

run_suite "${HEAD_DIR}" head.log
require_clean_run $? "${HEAD_DIR}" head.log

extract_failures base.log >base_failures.txt
extract_failures head.log >head_failures.txt

# Regressions = failing on head but not on main (comm -13 keeps lines only in the
# 2nd file, dropping main-only failures and failures shared by both).
comm -13 base_failures.txt head_failures.txt >regressions.txt

echo "main failures: $(wc -l <base_failures.txt) | head failures: $(wc -l <head_failures.txt)"
if [ -s regressions.txt ]; then
  echo "::error::PR introduces $(wc -l <regressions.txt) test regression(s) vs main:"
  cat regressions.txt
  exit 1
fi
echo "No regressions vs main (ignored $(wc -l <base_failures.txt) pre-existing base failures)."
