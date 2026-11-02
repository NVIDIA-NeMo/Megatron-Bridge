#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mock_dir=$(mktemp -d)
trap 'rm -rf "$mock_dir"' EXIT
mkdir "$mock_dir/bin"
printf '%s\n' '#!/usr/bin/env bash' 'echo "Unexpected GitHub API call" >&2' 'exit 99' > "$mock_dir/bin/gh"
chmod +x "$mock_dir/bin/gh"
export PATH="$mock_dir/bin:$PATH"

summary_command=$(awk '
  /^  Nemo_CICD_Test:/ { in_job=1; next }
  in_job && /^  [[:alnum:]_-]+:/ { in_job=0 }
  in_job && /^        run: \|/ { in_run=1; next }
  in_run && /^          / { sub(/^          /, ""); print; next }
  in_run { exit }
' .github/workflows/cicd-main.yml)
[[ "$summary_command" == '.github/scripts/check_workflow_result.sh' ]]
grep -Fq 'NEEDS_JSON: ${{ toJSON(needs) }}' .github/workflows/cicd-main.yml
grep -q '^          .github/scripts/test_workflow_result.sh$' .github/workflows/cicd-main.yml

dependency_ids=$(awk '
  /^  Nemo_CICD_Test:/ { in_job=1; next }
  in_job && /^      - / { print $2; next }
  in_job && /^    if:/ { exit }
' .github/workflows/cicd-main.yml)
base_results=$(jq -n --arg jobs "$dependency_ids" '$jobs | split("\n") | map({key: ., value: {result: "success"}}) | from_entries')
for job in $(awk '
  /^jobs:/ { in_jobs=1; next }
  /^  Nemo_CICD_Test:/ { exit }
  in_jobs && /^  [[:alnum:]_-]+:/ { sub(/:$/, "", $1); print $1 }
' .github/workflows/cicd-main.yml); do
  [[ "$job" == *-flaky ]] && continue
  jq -e --arg job "$job" 'has($job)' <<< "$base_results" > /dev/null
done

run_case() {
  local label="$1" expected_exit="$2"
  local actual_exit=0
  bash -e -u -o pipefail -c "$summary_command" > "$mock_dir/output" 2>&1 || actual_exit=$?
  if [[ "$actual_exit" != "$expected_exit" ]]; then
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

export EXPECT_L0=true EXPECT_L1=false EXPECT_L2=false IS_MEMBER=true DISABLE_GB200_TESTS=false
export NEEDS_JSON="$base_results"
run_case 'passing dependencies without GitHub API access' 0
NEEDS_JSON=$(jq 'with_entries(if .key | test("functional-tests.*-l[12]$") then .value.result = "skipped" else . end)' <<< "$base_results")
run_case 'disabled tiers may be skipped' 0

for job in $dependency_ids; do
  for result in failure cancelled; do
    NEEDS_JSON=$(jq --arg job "$job" --arg result "$result" '.[$job].result = $result' <<< "$base_results")
    run_case "$job $result blocks summary, including matrix aggregate failures" 1
    grep -q "$job" "$mock_dir/output"
  done
done

for tier in L0 L1 L2; do
  export EXPECT_L0=false EXPECT_L1=false EXPECT_L2=false
  export "EXPECT_$tier=true"
  for job in "cicd-functional-tests-${tier,,}" "cicd-functional-tests-gb200-${tier,,}"; do
    NEEDS_JSON=$(jq --arg job "$job" '.[$job].result = "skipped"' <<< "$base_results")
    run_case "$job unexpected skip" 1
    grep -q "$job" "$mock_dir/output"
    NEEDS_JSON=$(jq --arg job "$job" 'del(.[$job])' <<< "$base_results")
    run_case "$job missing result" 1
  done
  NEEDS_JSON=$(jq --arg job "cicd-functional-tests-gb200-${tier,,}" '.[$job].result = "skipped"' <<< "$base_results")
  export DISABLE_GB200_TESTS=true
  run_case "GB200 $tier disabled" 0
  export DISABLE_GB200_TESTS=false IS_MEMBER=false
  run_case "GB200 $tier non-member" 0
  export IS_MEMBER=true
done

for job in ephemeral-runner-routing pre-flight configure; do
  NEEDS_JSON=$(jq --arg job "$job" '.[$job].result = "skipped"' <<< "$base_results")
  run_case "$job cannot be skipped" 1
done

for response in '' 'not json' '{}' 'null' '[]' '{"job":null}' '{"job":{}}' '{"job":{"result":"in_progress"}}'; do
  export NEEDS_JSON="$response"
  run_case "invalid dependency results: $response" 1
done
