#!/usr/bin/env bash
set -euo pipefail

if ! jq -e 'type == "object" and length > 0 and all(.[];
  type == "object" and (.result | IN("success", "failure", "cancelled", "skipped")))' \
  <<< "${NEEDS_JSON:-}" > /dev/null; then
  echo "Invalid workflow dependency results" >&2
  exit 1
fi

failed_jobs=$(jq '[to_entries[] | select(.value.result != "success" and .value.result != "skipped")] | length' \
  <<< "$NEEDS_JSON")
unexpected_skips=0

require_success() {
  local job="$1"
  if ! jq -e --arg job "$job" '.[$job].result == "success"' <<< "$NEEDS_JSON" > /dev/null; then
    echo "❌ Required dependency $job did not succeed"
    unexpected_skips=$((unexpected_skips + 1))
  fi
}

require_success "ephemeral-runner-routing"
require_success "pre-flight"
require_success "configure"

for tier in L0 L1 L2; do
  expectation="EXPECT_$tier"
  if [[ "${!expectation}" != "true" ]]; then
    continue
  fi
  require_success "cicd-functional-tests-${tier,,}"
  if [[ "$IS_MEMBER" == "true" && "$DISABLE_GB200_TESTS" != "true" ]]; then
    require_success "cicd-functional-tests-gb200-${tier,,}"
  fi
done

if [ "$failed_jobs" -gt 0 ]; then
  echo "❌ Found $failed_jobs failed dependency job(s):"
  jq -r 'to_entries[] | select(.value.result != "success" and .value.result != "skipped")
    | "\(.key): \(.value.result)"' <<< "$NEEDS_JSON"
fi

if [ "$failed_jobs" -gt 0 ] || [ "$unexpected_skips" -gt 0 ]; then
  exit 1
fi

echo "✅ All previous jobs completed successfully"
