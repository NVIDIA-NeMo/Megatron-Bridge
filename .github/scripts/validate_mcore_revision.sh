#!/usr/bin/env bash
set -euo pipefail

repo="${1:-}"
revision="${2:-}"
triggered_by="${3:-}"

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
  "refs/heads/gh-readonly-queue/*/pr-*" \
  "refs/pull/*/merge")
object_store=$(mktemp -d)
trap 'rm -rf "$object_store"' EXIT
git -C "$object_store" init --quiet

main_sha=$(awk '$2 == "refs/heads/main" {print $1}' <<<"$refs")
dev_sha=$(awk '$2 == "refs/heads/dev" {print $1}' <<<"$refs")
for protected_sha in "$main_sha" "$dev_sha"; do
  if [[ -n "$protected_sha" ]] && \
    git -C "$object_store" fetch --quiet --filter=blob:none --no-tags "$repo" "$protected_sha" "$revision"; then
    if git -C "$object_store" merge-base --is-ancestor "$revision" "$protected_sha"; then
      exit 0
    fi
  fi
done

while IFS=$'\t' read -r sha ref; do
  if [[ "$sha" != "$revision" ]]; then
    continue
  fi
  if [[ "$ref" == "refs/heads/dev" ]] || \
    [[ "$ref" =~ ^refs/heads/pull-request/[0-9]+$ ]] || \
    [[ "$ref" =~ ^refs/heads/gh-readonly-queue/(main|dev)/pr-[0-9]+-[0-9a-f]{40}$ ]]; then
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

# GitHub deletes merge-queue refs when a group leaves the queue, which can
# happen before MCore's remotely triggered MBridge run starts. In that case,
# authenticate the immutable source run and require it to describe this exact
# official merge-group SHA. Arbitrary workflow-dispatch inputs and fork SHAs
# remain rejected.
if [[ "$triggered_by" =~ ^https://github.com/NVIDIA/Megatron-LM/actions/runs/([1-9][0-9]*)$ ]]; then
  run_id="${BASH_REMATCH[1]}"
  run_metadata=$(gh api "repos/NVIDIA/Megatron-LM/actions/runs/$run_id" --jq \
    '[.repository.full_name, .event, .head_sha, .head_branch] | @tsv' 2>/dev/null) || run_metadata=""
  IFS=$'\t' read -r run_repo run_event run_sha run_branch <<<"$run_metadata"
  if [[ "$run_repo" == "NVIDIA/Megatron-LM" && \
    "$run_event" == "merge_group" && \
    "$run_sha" == "$revision" && \
    "$run_branch" =~ ^gh-readonly-queue/(main|dev)/pr-[0-9]+-[0-9a-f]{40}$ ]]; then
    exit 0
  fi
fi

exit 1
