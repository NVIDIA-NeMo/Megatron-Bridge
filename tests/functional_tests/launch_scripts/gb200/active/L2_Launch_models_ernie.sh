#!/bin/bash
set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)

export CUDA_VISIBLE_DEVICES="0,1"

uv run coverage run \
    --data-file="${REPO_ROOT}/.coverage" \
    --source="${REPO_ROOT}" \
    --parallel-mode \
    -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -v -s -x \
    -m "not pleasefixme" \
    --tb=short -rA \
    tests/functional_tests/test_groups/models/ernie/test_ernie45_moe_conversion.py

coverage combine -q
