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

"""Tests for launcher-side library recipe environment resolution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from utils import utils


def _add_environment(monkeypatch, config, custom_env_vars, **overrides):
    monkeypatch.setattr(utils, "get_library_recipe", lambda **kwargs: config)
    args = {
        "custom_env_vars": custom_env_vars,
        "model_family_name": "deepseek",
        "model_recipe_name": "deepseek_v3",
        "train_task": "pretrain",
        "gpu": "h100",
        "experiment_name": "test-experiment",
        "expert_model_parallel_size": None,
    }
    args.update(overrides)
    utils.add_library_recipe_environment_variables(**args)


def test_library_recipe_environment_mutates_existing_mapping_and_preserves_explicit_values(monkeypatch):
    config = SimpleNamespace(
        env_vars={
            "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
            "TORCHINDUCTOR_WORKER_START": "fork",
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend=None),
    )
    custom_env_vars = {"NVTE_FWD_LAYERNORM_SM_MARGIN": "48"}
    original_mapping = custom_env_vars

    _add_environment(monkeypatch, config, custom_env_vars)

    assert custom_env_vars is original_mapping
    assert custom_env_vars == {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "48",
        "TORCHINDUCTOR_WORKER_START": "fork",
    }


@pytest.mark.parametrize(
    ("gpu", "ep_size", "expected_ranks", "expected_mnnvl"),
    [("h100", 4, "4", "0"), ("gb200", 32, "32", "1")],
)
def test_library_hybridep_topology_environment_tracks_ep_override(
    monkeypatch, gpu, ep_size, expected_ranks, expected_mnnvl
):
    config = SimpleNamespace(
        env_vars={
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
            "USE_MNNVL": 1,
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep"),
    )
    custom_env_vars = {}

    _add_environment(
        monkeypatch,
        config,
        custom_env_vars,
        gpu=gpu,
        expert_model_parallel_size=ep_size,
    )

    assert custom_env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_ranks
    assert custom_env_vars["USE_MNNVL"] == expected_mnnvl


def test_library_recipe_environment_rejects_non_scalar_values(monkeypatch):
    config = SimpleNamespace(
        env_vars={"INVALID": ["value"]},
        model=SimpleNamespace(moe_flex_dispatcher_backend=None),
    )

    with pytest.raises(TypeError, match="must have a scalar value"):
        _add_environment(monkeypatch, config, {})
