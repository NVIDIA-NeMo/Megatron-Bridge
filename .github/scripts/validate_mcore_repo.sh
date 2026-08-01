#!/usr/bin/env bash
set -euo pipefail

repo="${1:-}"
if [[ ! "$repo" =~ ^https://github\.com/([A-Za-z0-9_.-]+)/Megatron-LM\.git$ ]]; then
  exit 1
fi

owner="${BASH_REMATCH[1]}"
if [[ "$owner" == "NVIDIA" ]]; then
  exit 0
fi

gh api "repos/$owner/Megatron-LM" \
  --jq 'select(.fork == true and .parent.full_name == "NVIDIA/Megatron-LM") | .full_name' | \
  grep -Fxi "$owner/Megatron-LM" >/dev/null
