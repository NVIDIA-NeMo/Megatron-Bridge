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

"""Tests for DeepSeek V3 VR200 performance recipe topology."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config


pytestmark = pytest.mark.unit


def _load_vr200_recipe_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    common = ModuleType("megatron.bridge.perf_recipes.deepseek.common")
    common.ConfigContainer = SimpleNamespace
    common._benchmark_common = lambda _cfg: None
    common._deepseek_v3_common = lambda _cfg: None
    common._enable_deepseek_full_iteration_mxfp8 = lambda _cfg: None
    common._perf_precision = lambda _precision: SimpleNamespace()
    common.deepseek_v3_pretrain_config = lambda: SimpleNamespace(
        model=SimpleNamespace(expert_tensor_parallel_size=1),
        train=SimpleNamespace(),
        ddp=SimpleNamespace(),
        comm_overlap=SimpleNamespace(),
    )
    common.set_deepseek_v3_pipeline_model_parallel_layout = lambda _model: None
    monkeypatch.setitem(sys.modules, common.__name__, common)

    gb300 = ModuleType("megatron.bridge.perf_recipes.deepseek.gb300.deepseek_v3")
    for precision in ("bf16", "fp8cs", "fp8mx", "nvfp4"):
        setattr(gb300, f"deepseek_v3_pretrain_256gpu_gb300_{precision}_config", lambda: SimpleNamespace())
    monkeypatch.setitem(sys.modules, gb300.__name__, gb300)

    environment = ModuleType("megatron.bridge.perf_recipes.environment")
    environment.COMMON_PERF_ENV_VARS = {}
    monkeypatch.setitem(sys.modules, environment.__name__, environment)

    path = Path(__file__).resolve().parents[3] / "src/megatron/bridge/perf_recipes/deepseek/vr200/deepseek_v3.py"
    spec = importlib.util.spec_from_file_location("_deepseek_vr200_recipe_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "recipe_factory_name",
    (
        "deepseek_v3_pretrain_128gpu_vr200_bf16_config",
        "deepseek_v3_pretrain_128gpu_vr200_fp8cs_config",
        "deepseek_v3_pretrain_128gpu_vr200_fp8mx_config",
        "deepseek_v3_pretrain_128gpu_vr200_nvfp4_config",
    ),
)
def test_deepseek_v3_128gpu_vr200_topology_is_valid(recipe_factory_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_vr200_recipe_module(monkeypatch)
    recipe_factory = getattr(module, recipe_factory_name)
    cfg = recipe_factory()

    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.virtual_pipeline_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size * cfg.model.virtual_pipeline_model_parallel_size == 16
    assert data_parallel_size(num_gpus=128, topology=topology_from_config(cfg.model)) == 64
