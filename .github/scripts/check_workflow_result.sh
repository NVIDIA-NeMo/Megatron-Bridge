#!/usr/bin/env bash
set -euo pipefail

# Reuse one complete snapshot for every assertion and diagnostic. An API failure
# must never be interpreted as zero failures, including after retry exhaustion.
for attempt in 1 2 3; do
  if jobs_json=$(gh run view "$GITHUB_RUN_ID" --json jobs); then
    break
  fi
  if [ "$attempt" -eq 3 ]; then
    echo "Unable to retrieve workflow jobs after $attempt attempts" >&2
    exit 1
  fi
  echo "Unable to retrieve workflow jobs (attempt $attempt); retrying" >&2
  sleep "$((attempt * 5))"
done

# Reject empty or malformed responses rather than reporting a false success.
if ! jq -e '.jobs | type == "array" and length > 0' <<< "$jobs_json" > /dev/null; then
  echo "Invalid workflow jobs response" >&2
  exit 1
fi
failed_jobs=$(jq '[.jobs[] | select(.status == "completed" and .conclusion != "success" and .conclusion != "skipped")] | length' <<< "$jobs_json")

unexpected_skips=0
check_tier_skips() {
  local tier="$1" expect="$2"
  if [ "$expect" != "true" ]; then
    return 0
  fi
  local n
  n=$(jq --arg p "${tier}_" '[.jobs[] | select(.name | startswith($p)) | select(.conclusion == "skipped")] | length' <<< "$jobs_json")
  if [ "$n" -gt 0 ]; then
    echo "❌ Found $n unexpectedly skipped ${tier} job(s):"
    jq -r --arg p "${tier}_" '.jobs[] | select(.name | startswith($p)) | select(.conclusion == "skipped") | .name' <<< "$jobs_json"
    unexpected_skips=$((unexpected_skips + n))
  fi
}

check_tier_skips "L0" "$EXPECT_L0"
check_tier_skips "L1" "$EXPECT_L1"
check_tier_skips "L2" "$EXPECT_L2"
if [[ "$IS_MEMBER" == "true" && "$DISABLE_GB200_TESTS" != "true" ]]; then
  check_tier_skips "gb200_L0" "$EXPECT_L0"
  check_tier_skips "gb200_L1" "$EXPECT_L1"
  check_tier_skips "gb200_L2" "$EXPECT_L2"
fi

if [ "$failed_jobs" -gt 0 ]; then
  echo "❌ Found $failed_jobs failed job(s):"
  jq -r '.jobs[] | select(.status == "completed" and .conclusion != "success" and .conclusion != "skipped") | .name' <<< "$jobs_json"
fi

if [ "$failed_jobs" -gt 0 ] || [ "$unexpected_skips" -gt 0 ]; then
  exit 1
fi

echo "✅ All previous jobs completed successfully"
