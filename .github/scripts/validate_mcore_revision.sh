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
  if [[ "$sha" == "$revision" && "$ref" =~ ^refs/heads/pull-request/[0-9]+$ ]]; then
    exit 0
  fi
done <<<"$refs"

while IFS=$'\t' read -r merge_sha merge_ref; do
  if [[ "$merge_sha" != "$revision" || ! "$merge_ref" =~ ^refs/pull/([0-9]+)/merge$ ]]; then
    continue
  fi
  pr_number="${BASH_REMATCH[1]}"
  mirror_sha=$(awk -v ref="refs/heads/pull-request/${pr_number}" '$2 == ref {print $1}' <<<"$refs")
  [[ -n "$mirror_sha" ]] || continue
  git fetch --quiet --depth 1 "$repo" "$merge_sha" "$mirror_sha"
  parents=$(git rev-list --parents -n 1 "$merge_sha")
  if grep -qw "$mirror_sha" <<<"$parents"; then
    exit 0
  fi
done <<<"$refs"

exit 1
