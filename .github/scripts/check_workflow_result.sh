#!/usr/bin/env bash
set -euo pipefail

# GitHub supplies the completed dependency results, including matrix rollups.
# Do not query the jobs API here: an API outage must not turn passing tests red.
needs_json=${NEEDS_JSON:-}
if ! jq -e '
  type == "object" and length > 0 and
  all(.[];
    type == "object" and
    (.result | IN("success", "failure", "cancelled", "skipped")))
' <<< "$needs_json" > /dev/null; then
  echo "Invalid workflow dependency results" >&2
  exit 1
fi

failed_jobs=$(jq '[to_entries[] | select(.value.result == "failure" or .value.result == "cancelled")] | length' <<< "$needs_json")
unexpected_skips=0
check_tier_skips() {
  local job="$1" expect="$2"
  if [ "$expect" != "true" ]; then
    return 0
  fi
  local result
  result=$(jq -r --arg job "$job" '.[$job].result // "missing"' <<< "$needs_json")
  if [[ "$result" == "skipped" || "$result" == "missing" ]]; then
    echo "❌ Expected $job to run, but its result is $result"
    unexpected_skips=$((unexpected_skips + 1))
  fi
}

check_tier_skips "cicd-functional-tests-l0" "$EXPECT_L0"
check_tier_skips "cicd-functional-tests-l1" "$EXPECT_L1"
check_tier_skips "cicd-functional-tests-l2" "$EXPECT_L2"
if [[ "$IS_MEMBER" == "true" && "$DISABLE_GB200_TESTS" != "true" ]]; then
  check_tier_skips "cicd-functional-tests-gb200-l0" "$EXPECT_L0"
  check_tier_skips "cicd-functional-tests-gb200-l1" "$EXPECT_L1"
  check_tier_skips "cicd-functional-tests-gb200-l2" "$EXPECT_L2"
fi

if [ "$failed_jobs" -gt 0 ]; then
  echo "❌ Found $failed_jobs failed job(s):"
  jq -r 'to_entries[] | select(.value.result == "failure" or .value.result == "cancelled") | "\(.key): \(.value.result)"' <<< "$needs_json"
fi

if [ "$failed_jobs" -gt 0 ] || [ "$unexpected_skips" -gt 0 ]; then
  exit 1
fi

echo "✅ All previous jobs completed successfully"
