#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

: "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the mounted Megatron-Bridge checkout}"
: "${NEMO_RL_ROOT:?Set NEMO_RL_ROOT to the pinned NeMo-RL checkout}"
: "${EXPECTED_BRIDGE_SHA:?Set EXPECTED_BRIDGE_SHA to the tested Bridge commit}"
: "${EXPECTED_NEMO_RL_SHA:?Set EXPECTED_NEMO_RL_SHA to the pinned NeMo-RL commit}"

MCORE_ROOT="${BRIDGE_ROOT}/3rdparty/Megatron-LM"
POLICY_PYTHON=/usr/local/bin/python-MegatronPolicyWorker
SUITE=grpo-qwen3-8b-base-1n8g-megatron-lora
SUITE_DIR="${NEMO_RL_ROOT}/tests/test_suites/llm"
OUTPUT_DIR="${SUITE_DIR}/${SUITE}"
METRICS="${OUTPUT_DIR}/metrics.json"

bridge_sha=$(git -C "$BRIDGE_ROOT" rev-parse HEAD)
mcore_sha=$(git -C "$MCORE_ROOT" rev-parse HEAD)
nemo_rl_sha=$(git -C "$NEMO_RL_ROOT" rev-parse HEAD)
expected_mcore_sha=$(git -C "$BRIDGE_ROOT" ls-tree HEAD 3rdparty/Megatron-LM | awk '{print $3}')

test "$bridge_sha" = "$EXPECTED_BRIDGE_SHA"
test "$mcore_sha" = "$expected_mcore_sha"
test "$nemo_rl_sha" = "$EXPECTED_NEMO_RL_SHA"
test -x "$POLICY_PYTHON"

echo "Megatron-Bridge SHA: ${bridge_sha}"
echo "Megatron-Core SHA: ${mcore_sha}"
echo "NeMo-RL SHA: ${nemo_rl_sha}"
echo "Version injection: mounted checkout through PYTHONPATH (worker venv rebuild disabled)"

export NRL_FORCE_REBUILD_VENVS=false
export PYTHONPATH="${BRIDGE_ROOT}/src:${MCORE_ROOT}:${NEMO_RL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export UV_NO_SYNC=1
export WANDB_MODE=disabled

BRIDGE_ROOT="$BRIDGE_ROOT" MCORE_ROOT="$MCORE_ROOT" "$POLICY_PYTHON" - <<'PY'
import os
from pathlib import Path

import megatron.bridge
import megatron.core


def assert_from_checkout(module_file: str, checkout_root: str) -> Path:
    resolved = Path(module_file).resolve()
    resolved.relative_to(Path(checkout_root).resolve())
    return resolved


bridge_file = assert_from_checkout(megatron.bridge.__file__, os.environ["BRIDGE_ROOT"])
mcore_file = assert_from_checkout(megatron.core.__file__, os.environ["MCORE_ROOT"])
print(f"Megatron policy worker Bridge import: {bridge_file}")
print(f"Megatron policy worker MCore import: {mcore_file}")
PY

rm -rf "$OUTPUT_DIR"
cleanup() {
    rm -rf "$OUTPUT_DIR"
}
trap cleanup EXIT

start_seconds=$SECONDS
cd "$NEMO_RL_ROOT"
bash "${SUITE_DIR}/${SUITE}.sh" logger.wandb_enabled=false
runtime_seconds=$((SECONDS - start_seconds))

test -s "$METRICS"
last_step=$(jq -er '.["train/loss"] | keys | map(tonumber) | max' "$METRICS")
if ((last_step < 20)); then
    echo "NeMo-RL suite ended at step ${last_step}; expected step 20" >&2
    exit 1
fi

# The upstream suite conditionally checks these only after step 20. Re-run them
# after the unconditional step assertion so an incomplete run cannot pass.
uv run --no-sync tests/check_metrics.py "$METRICS" \
    'mean(data["train/gen_kl_error"], 20) < 0.002' \
    'data["train/gen_kl_error"]["20"] < 0.002' \
    'max(data["train/reward"]) > 0.35' \
    'mean(data["timing/train/total_step_time"], 2) < 80'

jq '{
    final_step: (."train/loss" | keys | map(tonumber) | max),
    final_gen_kl_error: ."train/gen_kl_error"."20",
    max_reward: (."train/reward" | to_entries | map(.value) | max),
    final_step_time_seconds: (."timing/train/total_step_time" | to_entries | max_by(.key | tonumber) | .value)
}' "$METRICS"
echo "NeMo-RL suite runtime: ${runtime_seconds} seconds"
