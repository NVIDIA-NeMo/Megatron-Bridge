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


def test_recipe_environment_defaults_preserve_explicit_executor_values(monkeypatch):
    """Recipe defaults and plugin fallbacks must not replace explicit launcher values."""
    plugin = PerfEnvPlugin(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb200",
        compute_dtype="bf16",
        train_task="pretrain",
    )
    executor = MagicMock()
    executor.env_vars = {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "48",
        "USE_MNNVL": "custom",
    }
    workload_config = MagicMock()
    workload_config.env_vars = {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
        "TORCHINDUCTOR_WORKER_START": "fork",
    }
    workload_config.tensor_model_parallel_size = 1
    workload_config.pipeline_model_parallel_size = 1
    workload_config.context_parallel_size = 1
    workload_config.expert_model_parallel_size = 64
    workload_config.moe_flex_dispatcher_backend = "hybridep"
    monkeypatch.setattr("perf_plugins.get_workload_base_config", lambda *args: workload_config)
    monkeypatch.setattr(plugin, "_set_num_cuda_device_max_connections", MagicMock())
    monkeypatch.setattr(plugin, "_set_manual_gc", MagicMock())
    monkeypatch.setattr(plugin, "_set_vboost", MagicMock())
    monkeypatch.setattr(plugin, "_set_lock_gpu_freq", MagicMock())
    monkeypatch.setattr(plugin, "_set_model_specific_environment_variables", MagicMock())

    plugin.setup(MagicMock(), executor)

    assert executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "48"
    assert executor.env_vars["TORCHINDUCTOR_WORKER_START"] == "fork"
    assert executor.env_vars["USE_MNNVL"] == "custom"
    assert executor.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "64"
