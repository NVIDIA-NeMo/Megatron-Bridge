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

refs=$(git ls-remote \
  "$repo" \
  "refs/heads/main" \
  "refs/heads/dev" \
  "refs/heads/pull-request/*" \
  "refs/heads/gh-readonly-queue/main/pr-*" \
  "refs/pull/*/merge")
object_store=$(mktemp -d)
trap 'rm -rf "$object_store"' EXIT
git -C "$object_store" init --quiet

main_sha=$(awk '$2 == "refs/heads/main" {print $1}' <<<"$refs")
if [[ -n "$main_sha" ]] && \
  git -C "$object_store" fetch --quiet --filter=blob:none --no-tags "$repo" "$main_sha" "$revision"; then
  if git -C "$object_store" merge-base --is-ancestor "$revision" "$main_sha"; then
    exit 0
  fi
fi

dev_sha=$(awk '$2 == "refs/heads/dev" {print $1}' <<<"$refs")
if [[ -n "$dev_sha" ]] && \
  git -C "$object_store" fetch --quiet --filter=blob:none --no-tags "$repo" "$dev_sha" "$revision"; then
  if git -C "$object_store" merge-base --is-ancestor "$revision" "$dev_sha"; then
    exit 0
  fi
fi

while IFS=$'\t' read -r sha ref; do
  if [[ "$sha" != "$revision" ]]; then
    continue
  fi
  if [[ "$ref" =~ ^refs/heads/pull-request/[0-9]+$ ]] || \
    [[ "$ref" =~ ^refs/heads/gh-readonly-queue/main/pr-[0-9]+-[0-9a-f]{40}$ ]]; then
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
  git -C "$object_store" fetch --quiet --filter=blob:none --no-tags "$repo" "$merge_sha" "$mirror_sha"
  parents=$(git -C "$object_store" rev-list --parents -n 1 "$merge_sha")
  if grep -qw "$mirror_sha" <<<"$parents"; then
    exit 0
  fi
done <<<"$refs"

exit 1
