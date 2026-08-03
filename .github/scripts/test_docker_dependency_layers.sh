#!/usr/bin/env bash
set -euo pipefail

dockerfile="${1:-docker/Dockerfile.ci}"
workflow="${2:-.github/workflows/cicd-main.yml}"

baseline_arg_line=$(grep -n '^ARG BASELINE_MCORE_REF$' "$dockerfile" | cut -d: -f1)
baseline_clone_line=$(grep -n 'git clone --filter=blob:none --no-checkout' "$dockerfile" | cut -d: -f1)
baseline_line=$(grep -n 'Install the main-branch environment from an immutable baseline context' "$dockerfile" | cut -d: -f1)
dispatched_copy_line=$(grep -n '^COPY 3rdparty/Megatron-LM /opt/Megatron-Bridge/3rdparty/Megatron-LM$' "$dockerfile" | cut -d: -f1)
delta_line=$(grep -n 'syncing the dispatched dependency delta' "$dockerfile" | cut -d: -f1)
validator=".github/scripts/validate_mcore_repo.sh"
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
grep -q 'BASELINE_MCORE_REF=${{ env.BASELINE_MCORE_REF }}' "$workflow"
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
test "$(grep -c 'validate_mcore_repo.sh "$MCORE_REPO"' "$workflow")" = 2
test "$(grep -c 'if \[\[ ! "$MCORE_REF" =~ ^\[0-9a-f\]{40}\$ \]\]; then' "$workflow")" = 2
test "$(grep -c 'git fetch "$MCORE_REPO" "$MCORE_REF"' "$workflow")" = 2
test "$(grep -c 'git checkout "$MCORE_REF"' "$workflow")" = 2

install_workflow=".github/workflows/install-test.yml"
grep -q '^          MCORE_COMMIT: ${{ github.event.inputs.mcore_commit }}$' "$install_workflow"
grep -q '^          MCORE_REPO: ${{ github.event.inputs.mcore_repo' "$install_workflow"
grep -q 'if \[\[ ! "$MCORE_COMMIT" =~ ^\[0-9a-f\]{40}\$ \]\]; then' "$install_workflow"
grep -q 'validate_mcore_repo.sh "$MCORE_REPO"' "$install_workflow"
grep -q 'git fetch "$MCORE_REPO" "$MCORE_COMMIT"' "$install_workflow"
grep -q 'git checkout "$MCORE_COMMIT"' "$install_workflow"
if grep -q 'git fetch ${{ github.event.inputs.mcore_repo' "$install_workflow"; then
  echo "Install workflow interpolates dispatch inputs into its shell" >&2
  exit 1
fi

"$validator" https://github.com/NVIDIA/Megatron-LM.git
if "$validator" https://github.com/example-contributor/Megatron-LM.git; then
  echo "MCore repository validation accepted an untrusted contributor fork" >&2
  exit 1
fi
if "$validator" https://github.com/not-a-fork/Megatron-LM.git; then
  echo "MCore repository validation accepted an untrusted repository" >&2
  exit 1
fi
if "$validator" 'https://github.com/example/Megatron-LM.git;touch /tmp/injected'; then
  echo "MCore repository validation accepted shell metacharacters" >&2
  exit 1
fi
if "$validator" https://example.com/example/Megatron-LM.git; then
  echo "MCore repository validation accepted a non-GitHub host" >&2
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
