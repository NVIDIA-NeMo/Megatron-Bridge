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

"""Tests for scripts/performance/perf_plugins.py PerfEnvPlugin determinism wiring."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run  # noqa: F401

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

pytestmark = pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")

if HAS_NEMO_RUN:
    from perf_plugins import PerfEnvPlugin
    from utils.utils import WorkloadBaseConfig


def test_set_determinism_env_vars_writes_three_keys():
    plugin = PerfEnvPlugin(
        deterministic=True,
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        gpu="h100",
        compute_dtype="bf16",
        train_task="pretrain",
    )
    executor = MagicMock()
    executor.env_vars = {}

    plugin._set_determinism_env_vars(executor)

    assert executor.env_vars["NCCL_ALGO"] == "Ring"
    assert executor.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert executor.env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_nemodiag_mxfp8_environment_preserves_determinism_and_cudnn_layernorm():
    plugin = PerfEnvPlugin(
        model_family_name="nemodiag",
        model_recipe_name="nemodiag_v0",
        gpu="gb300",
        compute_dtype="fp8_mx",
        train_task="pretrain",
    )
    executor = MagicMock()
    executor.env_vars = {
        "NVTE_NORM_FWD_USE_CUDNN": "1",
        "NVTE_NORM_BWD_USE_CUDNN": "1",
    }

    plugin._set_model_specific_environment_variables(
        MagicMock(),
        executor,
        WorkloadBaseConfig(),
        model_family_name="nemodiag",
        model_recipe_name="nemodiag_v0",
        gpu="gb300",
        compute_dtype="fp8_mx",
        train_task="pretrain",
    )

    assert executor.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert executor.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == "1"
    assert executor.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == "1"
