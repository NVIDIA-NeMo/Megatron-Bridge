# CI_TIMEOUT=15
#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0

# Each case launches its own fresh single-rank worker; do not wrap pytest in torchrun.
uv run python -m pytest -v -s -x --tb=short \
  tests/functional_tests/test_groups/training/test_determinism_startup.py
