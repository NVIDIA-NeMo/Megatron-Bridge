#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mock_dir=$(mktemp -d)
trap 'rm -rf "$mock_dir"' EXIT
mkdir "$mock_dir/bin"
cat > "$mock_dir/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$*" == "run view 123 --json jobs" ]]
calls=$(cat "$MOCK_DIR/calls")
calls=$((calls + 1))
echo "$calls" > "$MOCK_DIR/calls"
if [ "$calls" -le "$MOCK_FAILURES" ]; then
  echo '{"jobs":['
  echo 'HTTP 502: Server Error' >&2
  exit 1
fi
cat "$MOCK_DIR/jobs.json"
EOF
cat > "$mock_dir/bin/sleep" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "$1" >> "$MOCK_DIR/delays"
EOF
chmod +x "$mock_dir/bin/gh" "$mock_dir/bin/sleep"
export MOCK_DIR="$mock_dir" PATH="$mock_dir/bin:$PATH"
export GITHUB_RUN_ID=123 GH_REPO=NVIDIA-NeMo/Megatron-Bridge

# Exercise the command actually wired into the required summary job.
summary_command=$(awk '
  /^  Nemo_CICD_Test:/ { in_job=1; next }
  in_job && /^  [[:alnum:]_-]+:/ { in_job=0 }
  in_job && /^        run: \|/ { in_run=1; next }
  in_run && /^          / { sub(/^          /, ""); print; next }
  in_run { exit }
' .github/workflows/cicd-main.yml)
[[ "$summary_command" == '.github/scripts/check_workflow_result.sh' ]]
grep -q '^          .github/scripts/test_workflow_result.sh$' .github/workflows/cicd-main.yml

run_case() {
  local label="$1" expected_exit="$2" expected_calls="$3" expected_delays="$4"
  echo 0 > "$mock_dir/calls"
  : > "$mock_dir/delays"
  local actual_exit=0
  bash -e -u -o pipefail -c "$summary_command" > "$mock_dir/output" 2>&1 || actual_exit=$?
  if [[ "$actual_exit" != "$expected_exit" || "$(cat "$mock_dir/calls")" != "$expected_calls" \
      || "$(tr '\n' ' ' < "$mock_dir/delays")" != "$expected_delays" ]]; then
    echo "FAIL: $label (exit=$actual_exit, calls=$(cat "$mock_dir/calls"))" >&2
    cat "$mock_dir/output" >&2
    exit 1
  fi
  if [ "$expected_exit" -eq 0 ]; then
    grep -q 'All previous jobs completed successfully' "$mock_dir/output"
  elif grep -q 'All previous jobs completed successfully' "$mock_dir/output"; then
    echo "FAIL: $label reported success" >&2
    exit 1
  fi
  echo "PASS: $label"
}

export EXPECT_L0=true EXPECT_L1=false EXPECT_L2=false IS_MEMBER=true DISABLE_GB200_TESTS=false MOCK_FAILURES=0
cat > "$mock_dir/jobs.json" <<'EOF'
{"jobs":[{"name":"L0_test","status":"completed","conclusion":"success"},{"name":"L1_optional","status":"completed","conclusion":"skipped"},{"name":"Nemo_CICD_Test","status":"in_progress","conclusion":""}]}
EOF
run_case 'passing jobs with disabled tiers and summary in progress' 0 1 ''
export MOCK_FAILURES=1
run_case 'transient API failure then success' 0 2 '5 '
export MOCK_FAILURES=2
run_case 'two API failures then success' 0 3 '5 10 '
export MOCK_FAILURES=3
run_case 'API retry exhaustion fails closed' 1 3 '5 10 '
grep -q 'Unable to retrieve workflow jobs after 3 attempts' "$mock_dir/output"
export MOCK_FAILURES=0

for conclusion in failure cancelled timed_out action_required stale startup_failure; do
  printf '{"jobs":[{"name":"failed_test","status":"completed","conclusion":"%s"}]}\n' "$conclusion" > "$mock_dir/jobs.json"
  run_case "$conclusion is still a failure" 1 1 ''
  grep -q 'failed_test' "$mock_dir/output"
done

for tier in L0 L1 L2; do
  export EXPECT_L0=false EXPECT_L1=false EXPECT_L2=false
  export "EXPECT_$tier=true"
  printf '{"jobs":[{"name":"%s_test","status":"completed","conclusion":"skipped"}]}\n' "$tier" > "$mock_dir/jobs.json"
  run_case "$tier unexpected skip" 1 1 ''
  grep -q "${tier}_test" "$mock_dir/output"
  export "EXPECT_$tier=false"
  run_case "$tier intentional skip" 0 1 ''

  printf '{"jobs":[{"name":"gb200_%s_test","status":"completed","conclusion":"skipped"}]}\n' "$tier" > "$mock_dir/jobs.json"
  export "EXPECT_$tier=true"
  run_case "GB200 $tier unexpected skip" 1 1 ''
  grep -q "gb200_${tier}_test" "$mock_dir/output"
  export DISABLE_GB200_TESTS=true
  run_case "GB200 $tier disabled" 0 1 ''
  export DISABLE_GB200_TESTS=false IS_MEMBER=false
  run_case "GB200 $tier non-member" 0 1 ''
  export IS_MEMBER=true
done

for response in '' 'not json' '{}' '{"jobs":null}' '{"jobs":[]}' '{"jobs":{}}'; do
  printf '%s\n' "$response" > "$mock_dir/jobs.json"
  run_case "invalid response: $response" 1 1 ''
done
