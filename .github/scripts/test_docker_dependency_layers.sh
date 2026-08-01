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
grep -q 'uv pip install --no-deps --reinstall -e 3rdparty/Megatron-LM' "$dockerfile"
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

"$validator" https://github.com/NVIDIA/Megatron-LM.git
"$validator" https://github.com/example-contributor/Megatron-LM.git
if "$validator" 'https://github.com/example/Megatron-LM.git;touch /tmp/injected'; then
  echo "MCore repository validation accepted shell metacharacters" >&2
  exit 1
fi
if "$validator" https://example.com/example/Megatron-LM.git; then
  echo "MCore repository validation accepted a non-GitHub host" >&2
  exit 1
fi

if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  temporary_dir=$(mktemp -d)
  trap 'rm -rf "$temporary_dir"' EXIT
  cat >"$temporary_dir/Dockerfile" <<'EOF'
FROM alpine:3.22
ARG BASELINE_REF
RUN echo "baseline-$BASELINE_REF" >/baseline
COPY dispatched-source.txt /dispatched-source.txt
EOF
  echo first >"$temporary_dir/dispatched-source.txt"
  DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg BASELINE_REF=main -t mbridge-cache-order-test "$temporary_dir" \
    >"$temporary_dir/first.log" 2>&1
  echo second >"$temporary_dir/dispatched-source.txt"
  DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg BASELINE_REF=main -t mbridge-cache-order-test "$temporary_dir" \
    >"$temporary_dir/second.log" 2>&1
  grep -A2 'baseline-$BASELINE_REF' "$temporary_dir/second.log" | grep -q CACHED
fi
