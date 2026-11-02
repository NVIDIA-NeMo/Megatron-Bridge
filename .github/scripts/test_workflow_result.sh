#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mock_dir=$(mktemp -d)
trap 'rm -rf "$mock_dir"' EXIT
mkdir "$mock_dir/bin"
# A summary must not depend on GitHub API availability, even for diagnostics.
cat > "$mock_dir/bin/gh" <<'MOCK'
#!/usr/bin/env bash
echo 'Unexpected GitHub API call' >&2
touch "$MOCK_DIR/api-called"
exit 1
MOCK
chmod +x "$mock_dir/bin/gh"
export MOCK_DIR="$mock_dir" PATH="$mock_dir/bin:$PATH"
unset GITHUB_RUN_ID GH_TOKEN GH_REPO

# Exercise the command actually wired into the required summary job.
summary_job=$(awk '
  /^  Nemo_CICD_Test:/ { in_job=1; next }
  in_job && /^  [[:alnum:]_-]+:/ { exit }
  in_job { print }
' .github/workflows/cicd-main.yml)
summary_command=$(awk '
  /^        run: \|/ { in_run=1; next }
  in_run && /^          / { sub(/^          /, ""); print; next }
  in_run { exit }
' <<< "$summary_job")
[[ "$summary_command" == '.github/scripts/check_workflow_result.sh' ]]
grep -Fq 'NEEDS_JSON: ${{ toJSON(needs) }}' <<< "$summary_job"
if grep -Eq 'GH_TOKEN:|GH_REPO:|permissions: write-all' <<< "$summary_job"; then
  echo 'The summary must not request API credentials or write permissions' >&2
  exit 1
fi
grep -q '^          .github/scripts/test_workflow_result.sh$' .github/workflows/cicd-main.yml

# needs omits transitive dependencies. Keep every preceding job in the summary,
# including matrix generators, setup jobs, and opt-in flaky suites.
upstream_jobs=$(awk '
  /^jobs:/ { in_jobs=1; next }
  /^  Nemo_CICD_Test:/ { exit }
  in_jobs && /^  [[:alnum:]_-]+:/ { sub(":$", "", $1); print $1 }
' .github/workflows/cicd-main.yml)
summary_needs=$(awk '/^    if:/ { exit } /^      - / { print $2 }' <<< "$summary_job")
[[ "$(sort <<< "$upstream_jobs")" == "$(sort <<< "$summary_needs")" ]]
base_needs=$(printf '%s\n' "$summary_needs" | jq -Rn '
  [inputs | {key: ., value: {result: "success", outputs: {}}}] | from_entries
')

run_case() {
  local label="$1" expected_exit="$2"
  local actual_exit=0
  bash -e -u -o pipefail -c "$summary_command" > "$mock_dir/output" 2>&1 || actual_exit=$?
  if [[ "$actual_exit" != "$expected_exit" || -e "$mock_dir/api-called" ]]; then
    echo "FAIL: $label (exit=$actual_exit)" >&2
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

export EXPECT_L0=true EXPECT_L1=true EXPECT_L2=true IS_MEMBER=true DISABLE_GB200_TESTS=false
export NEEDS_JSON="$base_needs"
run_case 'all dependency and matrix results pass without GitHub access' 0

# Check every upstream failure, not just direct test dependencies. In particular,
# a failed matrix shard rolls up to failure even when other shards pass.
for job in $summary_needs; do
  for result in failure cancelled; do
    NEEDS_JSON=$(jq --arg job "$job" --arg result "$result" '.[$job].result = $result' <<< "$base_needs")
    run_case "$job $result blocks the summary" 1
    grep -Fq "$job: $result" "$mock_dir/output"
  done
done

for tier in l0 l1 l2; do
  variable="EXPECT_${tier^^}"
  for prefix in cicd-functional-tests cicd-functional-tests-gb200; do
    job="$prefix-$tier"
    export EXPECT_L0=false EXPECT_L1=false EXPECT_L2=false "$variable=true"
    NEEDS_JSON=$(jq --arg job "$job" '.[$job].result = "skipped"' <<< "$base_needs")
    run_case "$job unexpected skip" 1
    grep -Fq "$job" "$mock_dir/output"
    export "$variable=false"
    run_case "$job intentional skip" 0
    export "$variable=true"
    NEEDS_JSON=$(jq --arg job "$job" 'del(.[$job])' <<< "$base_needs")
    run_case "$job missing expected result" 1
  done
  NEEDS_JSON=$(jq --arg job "cicd-functional-tests-gb200-$tier" '.[$job].result = "skipped"' <<< "$base_needs")
  export DISABLE_GB200_TESTS=true
  run_case "GB200 $tier disabled" 0
  export DISABLE_GB200_TESTS=false IS_MEMBER=false
  run_case "GB200 $tier non-member" 0
  export IS_MEMBER=true
done

export EXPECT_L0=false EXPECT_L1=false EXPECT_L2=false
NEEDS_JSON=$(jq 'with_entries(.value.result = "skipped") | .["pre-flight"].result = "success" | .configure.result = "success"' <<< "$base_needs")
run_case 'docs-only or deployment intentionally skips tests' 0
NEEDS_JSON=$(jq '.["cicd-wait-in-queue"].result = "skipped" | .["cicd-functional-tests-flaky"].result = "skipped" | .["cicd-functional-tests-gb200-flaky"].result = "skipped"' <<< "$base_needs")
export EXPECT_L0=true EXPECT_L1=true EXPECT_L2=true
run_case 'queue bypass and disabled flaky suites' 0

for response in '' 'not json' '{}' 'null' '[]' '1' '{"job":null}' '{"job":[]}' '{"job":{}}' '{"job":{"result":null}}' '{"job":{"result":"pending"}}' '{"job":{"result":"neutral"}}'; do
  export NEEDS_JSON="$response"
  run_case "invalid dependency results: $response" 1
  grep -q 'Invalid workflow dependency results' "$mock_dir/output"
done
unset NEEDS_JSON
run_case 'missing dependency results' 1
