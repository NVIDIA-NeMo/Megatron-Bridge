#!/usr/bin/env bash
set -euo pipefail

dockerfile="${1:-docker/Dockerfile.ci}"
workflow="${2:-.github/workflows/cicd-main.yml}"

baseline_arg_line=$(grep -n '^ARG BASELINE_MCORE_REF$' "$dockerfile" | cut -d: -f1)
baseline_clone_line=$(grep -n 'git clone --filter=blob:none --no-checkout' "$dockerfile" | cut -d: -f1)
baseline_line=$(grep -n 'Install the main-branch environment from an immutable baseline context' "$dockerfile" | cut -d: -f1)
dispatched_copy_line=$(grep -n '^COPY 3rdparty/Megatron-LM /opt/Megatron-Bridge/3rdparty/Megatron-LM$' "$dockerfile" | cut -d: -f1)
delta_line=$(grep -n 'syncing the dispatched dependency delta' "$dockerfile" | cut -d: -f1)
repo_validator=".github/scripts/validate_mcore_repo.sh"
revision_validator=".github/scripts/validate_mcore_revision.sh"
temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

[[ -n "$baseline_arg_line" ]]
[[ -n "$baseline_clone_line" ]]
[[ -n "$baseline_line" ]]
[[ -n "$dispatched_copy_line" ]]
[[ -n "$delta_line" ]]
((baseline_arg_line < baseline_clone_line))
((baseline_clone_line < baseline_line))
((baseline_line < dispatched_copy_line))
((dispatched_copy_line < delta_line))

grep -q -- '--mount=type=cache,target=/root/.cache/uv' "$dockerfile"
if grep -q -- '--mount=type=secret,id=GH_TOKEN' "$dockerfile"; then
  echo "Baseline MCore clone must not require a GitHub token" >&2
  exit 1
fi
mcore_reinstall_line=$(grep -n 'uv pip install --no-deps --reinstall -e 3rdparty/Megatron-LM' "$dockerfile" | cut -d: -f1)
helper_assertion_line=$(grep -n "find 3rdparty/Megatron-LM/megatron/core/datasets -maxdepth 1 -name 'helpers_cpp\*\.so'" "$dockerfile" | cut -d: -f1)
final_copy_line=$(grep -n '^COPY --chmod=644 \. /opt/Megatron-Bridge$' "$dockerfile" | cut -d: -f1)
[[ -n "$mcore_reinstall_line" ]]
[[ -n "$helper_assertion_line" ]]
[[ -n "$final_copy_line" ]]
((delta_line < mcore_reinstall_line))
((mcore_reinstall_line < helper_assertion_line))
((helper_assertion_line < final_copy_line))
grep -q 'BASELINE_MCORE_REF=$(git -C 3rdparty/Megatron-LM rev-parse HEAD)' "$workflow"
grep -q 'validate_mcore_revision.sh' "$workflow"
grep -q 'BASELINE_MCORE_REF=${{ env.BASELINE_MCORE_REF }}' "$workflow"
grep -q 'test "$(git -C 3rdparty/Megatron-LM rev-parse HEAD)" = "$BASELINE_MCORE_REF"' "$dockerfile"
grep -q 'if \[ -n "$BASELINE_MCORE_REF" \]; then' "$dockerfile"
if grep -q '^COPY 3rdparty/Megatron-LM /opt/Megatron-Bridge/baseline/' "$dockerfile"; then
  echo "Mutable MCore must not enter the baseline dependency layer" >&2
  exit 1
fi
test "$(grep -c '^          MCORE_REF: ${{ github.event.inputs.mcore_ref }}$' "$workflow")" = 3
if grep -q 'MCORE_REF="${{ github.event.inputs.mcore_ref }}"' "$workflow"; then
  echo "Dispatch inputs must enter shell steps through env" >&2
  exit 1
fi
test "$(grep -c 'validate_mcore_revision.sh "$MCORE_REPO" "$MCORE_REF" "$TRIGGERED_BY"' "$workflow")" = 2
test "$(grep -c 'git fetch "$MCORE_REPO" "$MCORE_REF"' "$workflow")" = 2
test "$(grep -c 'git checkout "$MCORE_REF"' "$workflow")" = 2
test "$(grep -c 'test "$(git rev-parse HEAD)" = "$MCORE_REF"' "$workflow")" = 1

install_workflow=".github/workflows/install-test.yml"
grep -q '^          MCORE_COMMIT: ${{ github.event.inputs.mcore_commit }}$' "$install_workflow"
grep -q '^          MCORE_REPO: ${{ github.event.inputs.mcore_repo' "$install_workflow"
grep -q 'validate_mcore_revision.sh "$MCORE_REPO" "$MCORE_COMMIT"' "$install_workflow"
grep -q 'git fetch "$MCORE_REPO" "$MCORE_COMMIT"' "$install_workflow"
grep -q 'git checkout "$MCORE_COMMIT"' "$install_workflow"
if grep -qE '(git fetch|EXPECTED_COMMIT=).*\$\{\{ github\.event\.inputs\.mcore_(repo|commit)' "$install_workflow"; then
  echo "Install workflow interpolates dispatch inputs into its shell" >&2
  exit 1
fi

"$repo_validator" https://github.com/NVIDIA/Megatron-LM.git
if "$repo_validator" https://github.com/example-contributor/Megatron-LM.git; then
  echo "MCore repository validation accepted an untrusted contributor fork" >&2
  exit 1
fi
if "$repo_validator" https://github.com/not-a-fork/Megatron-LM.git; then
  echo "MCore repository validation accepted an untrusted repository" >&2
  exit 1
fi
if "$repo_validator" 'https://github.com/example/Megatron-LM.git;touch /tmp/injected'; then
  echo "MCore repository validation accepted shell metacharacters" >&2
  exit 1
fi
if "$repo_validator" https://example.com/example/Megatron-LM.git; then
  echo "MCore repository validation accepted a non-GitHub host" >&2
  exit 1
fi

revision_repo="file://$temporary_dir/Megatron-LM.git"
revision_worktree="$temporary_dir/revision-worktree"
git init --bare -q "$temporary_dir/Megatron-LM.git"
git init -q "$revision_worktree"
git -C "$revision_worktree" config user.name test
git -C "$revision_worktree" config user.email test@example.com
printf 'base\n' >"$revision_worktree/file"
git -C "$revision_worktree" add file
git -C "$revision_worktree" commit -q -m base
ancestor_sha=$(git -C "$revision_worktree" rev-parse HEAD)
printf 'main\n' >>"$revision_worktree/file"
git -C "$revision_worktree" commit -qam main
main_sha=$(git -C "$revision_worktree" rev-parse HEAD)
git -C "$revision_worktree" branch contributor "$ancestor_sha"
git -C "$revision_worktree" switch -q contributor
printf 'approved\n' >>"$revision_worktree/file"
git -C "$revision_worktree" commit -qam approved
mirror_sha=$(git -C "$revision_worktree" rev-parse HEAD)
git -C "$revision_worktree" switch -q --detach "$main_sha"
git -C "$revision_worktree" merge -q --no-ff -s ours "$mirror_sha" -m merge
merge_sha=$(git -C "$revision_worktree" rev-parse HEAD)
git -C "$revision_worktree" switch -q --detach "$main_sha"
printf 'queue\n' >"$revision_worktree/queue"
git -C "$revision_worktree" add queue
git -C "$revision_worktree" commit -q -m queue
queue_sha=$(git -C "$revision_worktree" rev-parse HEAD)
git -C "$revision_worktree" switch -q --detach "$main_sha"
printf 'orphan\n' >"$revision_worktree/orphan"
git -C "$revision_worktree" add orphan
git -C "$revision_worktree" commit -q -m orphan
orphan_sha=$(git -C "$revision_worktree" rev-parse HEAD)
printf 'unapproved\n' >"$revision_worktree/unapproved"
git -C "$revision_worktree" add unapproved
git -C "$revision_worktree" commit -q -m unapproved
unapproved_sha=$(git -C "$revision_worktree" rev-parse HEAD)
git -C "$revision_worktree" push -q "$revision_repo" "$main_sha:refs/heads/main"
git -C "$revision_worktree" push -q "$revision_repo" "$mirror_sha:refs/heads/pull-request/123"
git -C "$revision_worktree" push -q "$revision_repo" "$merge_sha:refs/pull/123/merge"
git -C "$revision_worktree" push -q \
  "$revision_repo" \
  "$queue_sha:refs/heads/gh-readonly-queue/main/pr-123-$main_sha"
git -C "$revision_worktree" push -q "$revision_repo" "$orphan_sha:refs/pull/456/merge"
git -C "$revision_worktree" push -q "$revision_repo" "$unapproved_sha:refs/heads/unapproved"

mkdir -p "$temporary_dir/revision-bin"
cat >"$temporary_dir/revision-bin/repo-validator" <<EOF
#!/usr/bin/env bash
[[ "\${1:-}" == "$revision_repo" ]]
EOF
chmod +x "$temporary_dir/revision-bin/repo-validator"
sed "s#\.github/scripts/validate_mcore_repo\.sh#$temporary_dir/revision-bin/repo-validator#" \
  "$revision_validator" >"$temporary_dir/revision-validator"
chmod +x "$temporary_dir/revision-validator"
"$temporary_dir/revision-validator" "$revision_repo" "$ancestor_sha"
"$temporary_dir/revision-validator" "$revision_repo" "$mirror_sha"
"$temporary_dir/revision-validator" "$revision_repo" "$merge_sha"
"$temporary_dir/revision-validator" "$revision_repo" "$queue_sha"
git -C "$revision_worktree" push -q \
  "$revision_repo" \
  ":refs/heads/gh-readonly-queue/main/pr-123-$main_sha"
if "$temporary_dir/revision-validator" "$revision_repo" "$queue_sha"; then
  echo "MCore revision validation accepted a deleted queue ref without source-run evidence" >&2
  exit 1
fi
cat >"$temporary_dir/revision-bin/gh" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == "api" && "$2" == "repos/NVIDIA/Megatron-LM/actions/runs/$TEST_RUN_ID" ]] || exit 1
printf '%s\t%s\t%s\t%s\n' \
  "${RUN_REPO:-NVIDIA/Megatron-LM}" \
  "${RUN_EVENT:-merge_group}" \
  "${RUN_SHA:-}" \
  "${RUN_BRANCH:-gh-readonly-queue/main/pr-123-0000000000000000000000000000000000000000}"
EOF
chmod +x "$temporary_dir/revision-bin/gh"
TEST_RUN_ID=123456
trigger_url="https://github.com/NVIDIA/Megatron-LM/actions/runs/$TEST_RUN_ID"
PATH="$temporary_dir/revision-bin:$PATH" \
  TEST_RUN_ID="$TEST_RUN_ID" \
  RUN_SHA="$queue_sha" \
  RUN_BRANCH="gh-readonly-queue/main/pr-123-$main_sha" \
  "$temporary_dir/revision-validator" "$revision_repo" "$queue_sha" "$trigger_url"
if PATH="$temporary_dir/revision-bin:$PATH" \
  TEST_RUN_ID="$TEST_RUN_ID" \
  RUN_SHA="$unapproved_sha" \
  RUN_BRANCH="gh-readonly-queue/main/pr-123-$main_sha" \
  "$temporary_dir/revision-validator" "$revision_repo" "$queue_sha" "$trigger_url"; then
  echo "MCore revision validation accepted source-run metadata for a different SHA" >&2
  exit 1
fi
if PATH="$temporary_dir/revision-bin:$PATH" \
  TEST_RUN_ID="$TEST_RUN_ID" \
  RUN_SHA="$queue_sha" \
  RUN_BRANCH="pull-request/123" \
  "$temporary_dir/revision-validator" "$revision_repo" "$queue_sha" "$trigger_url"; then
  echo "MCore revision validation accepted a non-queue source run" >&2
  exit 1
fi
if PATH="$temporary_dir/revision-bin:$PATH" \
  TEST_RUN_ID="$TEST_RUN_ID" \
  RUN_REPO="example/Megatron-LM" \
  RUN_SHA="$queue_sha" \
  RUN_BRANCH="gh-readonly-queue/main/pr-123-$main_sha" \
  "$temporary_dir/revision-validator" "$revision_repo" "$queue_sha" "$trigger_url"; then
  echo "MCore revision validation accepted an untrusted source-run repository" >&2
  exit 1
fi
if PATH="$temporary_dir/revision-bin:$PATH" \
  TEST_RUN_ID="$TEST_RUN_ID" \
  RUN_EVENT="push" \
  RUN_SHA="$queue_sha" \
  RUN_BRANCH="gh-readonly-queue/main/pr-123-$main_sha" \
  "$temporary_dir/revision-validator" "$revision_repo" "$queue_sha" "$trigger_url"; then
  echo "MCore revision validation accepted a non-merge-group source run" >&2
  exit 1
fi
if "$temporary_dir/revision-validator" "$revision_repo" "$queue_sha"x; then
  echo "MCore revision validation accepted a malformed queue SHA" >&2
  exit 1
fi
if "$temporary_dir/revision-validator" "$revision_repo" "$orphan_sha"; then
  echo "MCore revision validation accepted a PR merge without an approved mirror parent" >&2
  exit 1
fi
if "$temporary_dir/revision-validator" "$revision_repo" "$unapproved_sha"; then
  echo "MCore revision validation accepted an existing commit on an unapproved ref" >&2
  exit 1
fi
if "$temporary_dir/revision-validator" "$revision_repo" 0000000000000000000000000000000000000000; then
  echo "MCore revision validation accepted a nonexistent SHA" >&2
  exit 1
fi
if "$temporary_dir/revision-validator" "$revision_repo" not-a-full-sha; then
  echo "MCore revision validation accepted a malformed SHA" >&2
  exit 1
fi

composite_action=".github/actions/test-template/action.yml"
if grep -qE 'mcore_(commit|ref)|MCORE_COMMIT|uv sync --all-extras --all-groups' "$composite_action"; then
  echo "Test template must use the already-validated MCore in the built image" >&2
  exit 1
fi

# The baseline dependency layer must be structurally independent of the mutable
# dispatched checkout. CI validates the ordering statically so this regression
# check never pulls or executes an external container image.
assert_baseline_precedes_dispatched_copy() {
  local candidate="$1"
  local baseline_sync_line
  local mutable_copy_line

  baseline_sync_line=$(grep -n 'uv sync --link-mode copy --locked --all-extras --all-groups --no-group diffusion' "$candidate" | head -1 | cut -d: -f1)
  mutable_copy_line=$(grep -n '^COPY 3rdparty/Megatron-LM /opt/Megatron-Bridge/3rdparty/Megatron-LM$' "$candidate" | head -1 | cut -d: -f1)
  [[ -n "$baseline_sync_line" && -n "$mutable_copy_line" ]] || return 1
  ((baseline_sync_line < mutable_copy_line))
}

if ! assert_baseline_precedes_dispatched_copy "$dockerfile"; then
  echo "Mutable MCore enters the Dockerfile before the baseline layer is complete" >&2
  exit 1
fi

cat >"$temporary_dir/early-copy.Dockerfile" <<'EOF'
COPY 3rdparty/Megatron-LM /opt/Megatron-Bridge/3rdparty/Megatron-LM
RUN uv sync --link-mode copy --locked --all-extras --all-groups --no-group diffusion
EOF
if assert_baseline_precedes_dispatched_copy "$temporary_dir/early-copy.Dockerfile"; then
  echo "Cache-order regression accepted an early mutable MCore copy" >&2
  exit 1
fi
