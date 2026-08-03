#!/usr/bin/env bash
set -euo pipefail

compute_keys() {
  local base_ref=$1
  local pr_number=$2
  local mcore_ref=$3
  local mbridge_sha=$4
  local github_ref=$5
  local event_name=$6
  local ref_name=$7

  BASE_REF="${base_ref#refs/heads/}"
  CACHE_NAMESPACE=$(printf '%s' "$BASE_REF" | tr '/:@' '-' | tr -cd '[:alnum:]_.-')
  if [[ -z "$CACHE_NAMESPACE" ]]; then
    return 1
  fi

  BASELINE_KEY="${CACHE_NAMESPACE}-baseline"
  if [[ -n "$pr_number" ]]; then
    KEY="${CACHE_NAMESPACE}-${pr_number}"
  elif [[ -n "$mcore_ref" ]]; then
    KEY="${CACHE_NAMESPACE}-mcore-${mcore_ref:0:12}-${mbridge_sha:0:12}"
  elif [[ "$github_ref" == "refs/heads/$BASE_REF" || "$event_name" == "schedule" ]]; then
    KEY="$BASELINE_KEY"
  else
    BRANCH_SANITIZED=$(printf '%s' "$ref_name" | tr '/:@' '-' | tr -cd '[:alnum:]_.-')
    KEY="${CACHE_NAMESPACE}-${BRANCH_SANITIZED}"
  fi

  if [[ "${#KEY}" -gt 100 ]]; then
    KEY="${KEY:0:83}-$(printf '%s' "$KEY" | sha256sum | cut -c1-16)"
  fi
  if [[ "${#BASELINE_KEY}" -gt 100 ]]; then
    BASELINE_KEY="${BASELINE_KEY:0:83}-$(printf '%s' "$BASELINE_KEY" | sha256sum | cut -c1-16)"
  fi
}

assert_keys() {
  local expected_key=$1
  local expected_baseline=$2
  shift 2
  compute_keys "$@"
  test "$KEY" = "$expected_key"
  test "$BASELINE_KEY" = "$expected_baseline"
}

assert_keys main-5190 main-baseline main 5190 '' 317f4a21bbed0fb8a4ac48ddef68356268c79394 refs/heads/pull-request/5190 push pull-request/5190
assert_keys dev-5190 dev-baseline refs/heads/dev 5190 '' 317f4a21bbed0fb8a4ac48ddef68356268c79394 refs/heads/pull-request/5190 push pull-request/5190
assert_keys main-mcore-d2cf5974cdc0-317f4a21bbed main-baseline main '' d2cf5974cdc04280b0840ba72382ae8fc88f5dfe 317f4a21bbed0fb8a4ac48ddef68356268c79394 refs/heads/mcore-testing-30545783237 workflow_dispatch mcore-testing-30545783237
assert_keys main-baseline main-baseline main '' '' 317f4a21bbed0fb8a4ac48ddef68356268c79394 refs/heads/main push main
assert_keys main-baseline main-baseline main '' '' 317f4a21bbed0fb8a4ac48ddef68356268c79394 refs/heads/main schedule main
assert_keys main-deploy-release-1.2 main-baseline main '' '' 317f4a21bbed0fb8a4ac48ddef68356268c79394 refs/heads/deploy-release/1.2 push deploy-release/1.2
