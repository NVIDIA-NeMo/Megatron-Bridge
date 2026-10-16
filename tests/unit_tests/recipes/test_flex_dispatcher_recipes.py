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

"""Backend choices must survive recipe inheritance independently of the build host."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.bridge.training.flex_dispatcher_backend import validate_flex_dispatcher_backend
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("package", "factory_name", "backend"),
    [
        ("recipes.qwen", "qwen3_30b_a3b_pretrain_config", "hybridep"),
        ("recipes.qwen", "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_235b_a22b_pretrain_64gpu_gb200_bf16_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config", "hybridep"),
        ("perf_recipes.qwen", "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config", "hybridep"),
    ],
)
@pytest.mark.parametrize("cuda_available", [False, True])
def test_recipe_selects_backend_before_hardware_validation(
    monkeypatch, package, factory_name, backend, cuda_available
):
    factory = getattr(importlib.import_module(f"megatron.bridge.{package}"), factory_name)
    patch_recipe_construction_dependencies(monkeypatch)
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch("torch.cuda.get_device_properties", side_effect=AssertionError("recipe probed the build host")),
    ):
        cfg = factory()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == backend
    assert cfg.model.moe_shared_expert_overlap is False

    with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(major=10, name="NVIDIA GB200")):
        # GB200 children can replace DeepEP selected by an H100 parent.
        validate_flex_dispatcher_backend(cfg.model)
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == backend


@pytest.mark.parametrize(
    ("major", "name"),
    [(9, "NVIDIA H100 80GB HBM3"), (10, "NVIDIA B200"), (10, "NVIDIA GB200")],
)
def test_qwen3_235b_pretrain_recipe_selects_alltoall(monkeypatch, major, name):
    factory = importlib.import_module("megatron.bridge.recipes.qwen").qwen3_235b_a22b_pretrain_config
    patch_recipe_construction_dependencies(monkeypatch)
    with patch("torch.cuda.get_device_properties", side_effect=AssertionError("recipe probed the build host")):
        cfg = factory()

    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.moe_flex_dispatcher_backend is None

    # The generic recipe is used unchanged by the launcher on every GPU target.
    with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(major=major, name=name)):
        validate_flex_dispatcher_backend(cfg.model)
