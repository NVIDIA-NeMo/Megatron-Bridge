#!/usr/bin/env bash
set -euo pipefail

select_cache_donor() {
  local run_cache=$1
  local baseline_cache=$2
  local legacy_cache=$3
  local donor_cache=$run_cache

  if ! docker buildx imagetools inspect "$donor_cache" >/dev/null 2>&1; then
    donor_cache=$baseline_cache
  fi
  if ! docker buildx imagetools inspect "$donor_cache" >/dev/null 2>&1; then
    donor_cache=$legacy_cache
  fi
  printf '%s\n' "$donor_cache"
}

select_cache_donor "$@"
