#!/usr/bin/env bash
set -euo pipefail

workflow="${1:-.github/workflows/cicd-main.yml}"

awk '
  function check_job() {
    if (!in_queue_chain || !has_success) {
      return
    }

    checked++
    if (!allows_queue_bypass) {
      print job " uses success() but does not allow the intentional queue skip" > "/dev/stderr"
      failed = 1
    }
  }

  /^  [[:alnum:]_-]+:$/ {
    check_job()
    job = $1
    sub(/:$/, "", job)
    in_queue_chain = (in_queue_chain || job == "cicd-compute-build-matrix")
    if (job == "Nemo_CICD_Test") {
      in_queue_chain = 0
    }
    has_success = 0
    allows_queue_bypass = 0
    next
  }

  in_queue_chain && /success\(\)/ {
    has_success = 1
  }

  in_queue_chain && /needs\.configure\.outputs\.skip_queue == '\''true'\'' && !failure\(\)/ {
    allows_queue_bypass = 1
  }

  END {
    check_job()
    if (checked == 0) {
      print "No queue-dependent success() conditions found" > "/dev/stderr"
      exit 1
    }
    exit failed
  }
' "$workflow"
