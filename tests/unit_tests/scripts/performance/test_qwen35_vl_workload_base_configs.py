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

"""Tests for Qwen3.5-VL performance workload presets."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from megatron.bridge.perf_recipes.qwen_vl.gb200.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config,
)
from megatron.bridge.perf_recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))


@pytest.mark.parametrize(
    ("recipe_fn", "expected_pp_size", "expected_vp_size"),
    [
        (
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
            8,
            2,
        ),
        (
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
            8,
            2,
        ),
    ],
)
def test_qwen35_vl_122b_h100_pipeline_layout(
    recipe_fn: Callable,
    expected_pp_size: int,
    expected_vp_size: int,
) -> None:
    num_layers = 48
    config = recipe_fn()
    pp_size = config.model.pipeline_model_parallel_size
    vp_size = config.model.virtual_pipeline_model_parallel_size

    assert num_layers % pp_size == 0
    assert vp_size is not None
    assert (num_layers // pp_size) % vp_size == 0
    assert pp_size == expected_pp_size
    assert vp_size == expected_vp_size


@pytest.mark.parametrize(
    ("recipe_fn", "expected_graph_modules"),
    [
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config,
            [],
        ),
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config,
            [],
        ),
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config,
            ["moe_router", "moe_preprocess"],
        ),
    ],
)
def test_qwen35_vl_35b_gb200_measured_performance_defaults(
    recipe_fn: Callable,
    expected_graph_modules: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real GB200 factories should retain their measured batch and graph defaults."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    assert config.train.micro_batch_size == 3
    assert config.train.global_batch_size == 480
    assert config.model.moe_router_force_load_balancing is True
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    expected_graph_impl = "transformer_engine" if expected_graph_modules else "none"
    assert config.model.cuda_graph_impl == expected_graph_impl
    assert config.model.cuda_graph_scope is None
    assert bool(config.model.cuda_graph_modules) is bool(expected_graph_modules)
    assert cuda_graph_module_names(config.model) == expected_graph_modules
    assert config.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert config.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1
