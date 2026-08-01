#!/usr/bin/env bash
set -euo pipefail

repo="${1:-}"
[[ "$repo" =~ ^https://github\.com/[A-Za-z0-9_.-]+/Megatron-LM\.git$ ]]
