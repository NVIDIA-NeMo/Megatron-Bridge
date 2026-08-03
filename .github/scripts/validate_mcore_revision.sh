#!/usr/bin/env bash
set -euo pipefail

repo="${1:-}"
revision="${2:-}"

if ! .github/scripts/validate_mcore_repo.sh "$repo"; then
  exit 1
fi
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  exit 1
fi

refs=$(git ls-remote "$repo" "refs/heads/main" "refs/heads/pull-request/*" "refs/pull/*/merge")
while IFS=$'\t' read -r sha ref; do
  if [[ "$sha" == "$revision" && "$ref" == "refs/heads/main" ]]; then
    exit 0
  fi
  if [[ "$sha" == "$revision" && "$ref" =~ ^refs/heads/pull-request/([0-9]+)$ ]]; then
    exit 0
  fi
  if [[ "$sha" == "$revision" && "$ref" =~ ^refs/pull/([0-9]+)/merge$ ]]; then
    pr_number="${BASH_REMATCH[1]}"
    if grep -qE "^[0-9a-f]{40}[[:space:]]+refs/heads/pull-request/${pr_number}$" <<<"$refs"; then
      exit 0
    fi
  fi
done <<<"$refs"

exit 1
