#!/usr/bin/env bash
# Stage 3/3 of the determinism-trace e2e pipeline: diff the two arms' fingerprint streams
# to the FIRST divergent op via diff_streams.py.
#
# Reads the stream dirs from the state file written by dettrace_submit.sh, or takes them
# as args. Fails clearly if a stream dir is empty (the usual cause: the run crashed before
# iteration 1's end-of-step flush, so nothing was written — fix the run first).
#
# Usage:
#   dettrace_diff.sh                          # read $STATE_FILE (default ./dettrace-state.env)
#   dettrace_diff.sh <STREAMS_A> <STREAMS_B>
# Optional env: STATE_FILE  REPO_ROOT (=.)  PYTHON (=python)

set -euo pipefail
STATE_FILE="${STATE_FILE:-./dettrace-state.env}"
PYTHON="${PYTHON:-python}"
REPO_ROOT="${REPO_ROOT:-.}"
DIFF_TOOL="src/megatron/bridge/training/utils/determinism/diff_streams.py"

if [ "$#" -ge 2 ]; then
  A="$1"; B="$2"
else
  [ -f "$STATE_FILE" ] || { echo "ERROR: no stream dirs given and $STATE_FILE not found" >&2; exit 2; }
  # shellcheck disable=SC1090
  . "$STATE_FILE"
  A="${STREAMS_A:?state file missing STREAMS_A}"
  B="${STREAMS_B:?state file missing STREAMS_B}"
fi

for d in "$A" "$B"; do
  if [ -z "$(find "$d" -type f -print -quit 2>/dev/null)" ]; then
    echo "ERROR: no stream files under '$d' — the run wrote nothing to diff." >&2
    echo "       Streams flush only at the step boundary, so a crash before iteration 1" >&2
    echo "       ends leaves them empty. Check the job log and rerun before diffing." >&2
    exit 4
  fi
done

exec "${PYTHON}" "${REPO_ROOT%/}/${DIFF_TOOL}" "$A" "$B"
