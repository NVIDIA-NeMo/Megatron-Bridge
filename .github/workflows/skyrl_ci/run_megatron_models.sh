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
#
# Runs SkyRL's Megatron <-> vLLM logprob-parity test
# (tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py) against
# the Megatron-Bridge checkout under test and its 3rdparty/Megatron-LM submodule.
# See README.md in this directory.
set -euo pipefail

: "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the mounted Megatron-Bridge checkout}"
: "${SKYRL_ROOT:?Set SKYRL_ROOT to the pinned SkyRL checkout}"
: "${EXPECTED_BRIDGE_SHA:?Set EXPECTED_BRIDGE_SHA to the tested Bridge commit}"
: "${EXPECTED_SKYRL_SHA:?Set EXPECTED_SKYRL_SHA to the pinned SkyRL commit}"

MCORE_ROOT="${BRIDGE_ROOT}/3rdparty/Megatron-LM"
TEST_FILE=tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py

# Rows of test_logprobs_matching_roundtrip whose HF <-> Megatron conversion goes
# through bridges that ship in Megatron-Bridge (directly, or via a thin SkyRL
# subclass that only re-roots `model.language_model.*`). Rows built on code SkyRL
# carries itself (GLM-5.3-Flash, Kimi K2.5 INT4 QAT) or that need SM100 (MXFP8)
# are left out; see README.md. Override with SKYRL_TEST_ROWS="id1 id2 ...".
DEFAULT_ROWS=(
    qwen3-moe_tp2_ep2
    qwen3-moe_pp2_cp2
    qwen3.5-0.8b-dense_tp2
    qwen3.5-35b-a3b_h100_tp4_ep4
    nemotron3.5-lightning_tp4_ep4_h100
)
if [ -n "${SKYRL_TEST_ROWS:-}" ]; then
    read -r -a ROWS <<<"$SKYRL_TEST_ROWS"
else
    ROWS=("${DEFAULT_ROWS[@]}")
fi
NODE_IDS=()
for row in "${ROWS[@]}"; do
    NODE_IDS+=("${TEST_FILE}::test_logprobs_matching_roundtrip[${row}]")
done

# The SkyRL image runs as `ray`, not as the owner of the mounted checkouts.
git config --global --add safe.directory '*'

bridge_sha=$(git -C "$BRIDGE_ROOT" rev-parse HEAD)
mcore_sha=$(git -C "$MCORE_ROOT" rev-parse HEAD)
skyrl_sha=$(git -C "$SKYRL_ROOT" rev-parse HEAD)
expected_mcore_sha=$(git -C "$BRIDGE_ROOT" ls-tree HEAD 3rdparty/Megatron-LM | awk '{print $3}')

test "$bridge_sha" = "$EXPECTED_BRIDGE_SHA"
test "$mcore_sha" = "$expected_mcore_sha"
test "$skyrl_sha" = "$EXPECTED_SKYRL_SHA"

echo "Megatron-Bridge SHA: ${bridge_sha}"
echo "Megatron-Core SHA: ${mcore_sha}"
echo "SkyRL SHA: ${skyrl_sha}"
echo "Version injection: mounted checkouts first on PYTHONPATH, exported to Ray workers"

export CI=true
export WANDB_MODE=disabled
export NVTE_FLASH_ATTN=0
# SkyRL's Ray workers re-run the driver's `uv run --isolated ...` command through
# Ray's uv runtime-env hook. The image only sets this in ~/.bashrc, which
# `docker exec bash script.sh` does not read.
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# SkyRL's GPU test conftest forwards the driver's PYTHONPATH into every Ray
# worker's runtime env when this is set, so the workers import the checkouts
# below instead of the megatron-bridge / megatron-core pinned in SkyRL's uv.lock.
export SKYRL_PYTHONPATH_EXPORT=1
export PYTHONPATH="${BRIDGE_ROOT}/src:${MCORE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

UV_RUN=(uv run --isolated --extra dev --extra megatron)

cd "$SKYRL_ROOT"

# Prove the injection reached both the driver and a Ray worker, which builds its
# own environment from SkyRL's lockfile.
check_script=$(mktemp --suffix=.py)
cleanup() {
    ray stop --force >/dev/null 2>&1 || true
    rm -f "$check_script"
}
trap cleanup EXIT

cat >"$check_script" <<'PY'
import os
from pathlib import Path

import ray


def import_roots() -> dict[str, str]:
    import megatron.bridge
    import megatron.core

    return {"bridge": megatron.bridge.__file__, "mcore": megatron.core.__file__}


def assert_under(module_file: str, root: str) -> Path:
    resolved = Path(module_file).resolve()
    resolved.relative_to(Path(root).resolve())
    return resolved


expected = {"bridge": os.environ["BRIDGE_ROOT"], "mcore": os.environ["MCORE_ROOT"]}
ray.init(runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}})
try:
    for where, roots in (
        ("driver", import_roots()),
        ("Ray worker", ray.get(ray.remote(import_roots).remote())),
    ):
        for name, root in expected.items():
            print(f"{where} {name} import: {assert_under(roots[name], root)}")
finally:
    ray.shutdown()
PY
BRIDGE_ROOT="$BRIDGE_ROOT" MCORE_ROOT="$MCORE_ROOT" "${UV_RUN[@]}" python "$check_script"

# get_test_generator_input reads ~/data/gsm8k/validation.parquet.
"${UV_RUN[@]}" python examples/train/gsm8k/gsm8k_dataset.py --output_dir "${HOME}/data/gsm8k"

# SkyRL's conftest skips h100-marked rows unless "h100" appears in -m. The
# expression below keeps the unmarked rows selected too; which rows run is
# decided by the node IDs.
"${UV_RUN[@]}" pytest -s -vv -m "h100 or not h100" "${NODE_IDS[@]}"
