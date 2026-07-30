#!/usr/bin/env bash
# Stage 2/3 of the determinism-trace e2e pipeline: wait for both trace arms to reach a
# terminal Slurm state, then print their final sacct rows and stream-file counts.
#
# Reads job ids from the state file written by dettrace_submit.sh, or takes them as args.
# Uses scontrol JobState (robust: a transient empty squeue does not look like completion).
#
# Usage:
#   dettrace_wait.sh                    # read $STATE_FILE (default ./dettrace-state.env)
#   dettrace_wait.sh <JOBID_A> <JOBID_B>
# Optional env: STATE_FILE  POLL_SEC (=120)  MAX_POLLS (=720, ~24h)

set -euo pipefail
STATE_FILE="${STATE_FILE:-./dettrace-state.env}"
POLL_SEC="${POLL_SEC:-120}"
MAX_POLLS="${MAX_POLLS:-720}"

if [ "$#" -ge 2 ]; then
  A="$1"; B="$2"
else
  [ -f "$STATE_FILE" ] || { echo "ERROR: no job ids given and $STATE_FILE not found" >&2; exit 2; }
  # shellcheck disable=SC1090
  . "$STATE_FILE"
  A="${JOBID_A:?state file missing JOBID_A}"
  B="${JOBID_B:?state file missing JOBID_B}"
fi

TERMINAL="COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|DEADLINE|PREEMPTED"

# scontrol JobState, or PURGED once the controller no longer knows the (completed) job.
jstate() {
  local s
  s=$(scontrol show job "$1" 2>/dev/null | grep -oE "JobState=[A-Z_]+" | head -1 | cut -d= -f2)
  [ -z "$s" ] && s=PURGED
  echo "$s"
}

echo "Waiting on A=${A} B=${B} (poll ${POLL_SEC}s, cap $((MAX_POLLS * POLL_SEC / 3600))h)..."
prev=""
for _ in $(seq 1 "$MAX_POLLS"); do
  sa=$(jstate "$A"); sb=$(jstate "$B")
  cur="A=$sa B=$sb"
  [ "$cur" != "$prev" ] && { echo "[$(date +%H:%M)] $cur"; prev="$cur"; }
  if echo "$sa" | grep -qE "$TERMINAL|PURGED" && echo "$sb" | grep -qE "$TERMINAL|PURGED"; then
    echo "Both terminal."
    sacct -j "$A,$B" --format=JobID,State,Elapsed,ExitCode -X 2>/dev/null || true
    exit 0
  fi
  sleep "$POLL_SEC"
done

echo "ERROR: jobs still not terminal after the poll cap" >&2
exit 1
