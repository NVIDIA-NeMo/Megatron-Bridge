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

"""Tests for explicit hardware recipe environment settings."""

import ast
import re
from pathlib import Path

import pytest

from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS


_CANONICAL_RECIPE_NAME = re.compile(
    r".+_(?:pretrain|sft|peft)_\d+gpu_[a-z0-9]+_(?:bf16|fp8cs|fp8mx|fp8sc|nvfp4)(?:_.+)?_config"
)
_RECIPE_ROOT = Path(__file__).resolve().parents[3] / "src" / "megatron" / "bridge" / "recipes"
_HYBRID_EP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}
_DEEPSEEK_NON_BASELINE_ENV_NAMES = {
    "QUANTIZATION_TYPE_DEBUG",
    "TORCHINDUCTOR_WORKER_START",
}
_DEEPSEEK_V3_ENVIRONMENT_RECIPE_NAMES = {
    "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
    "deepseek_v3_pretrain_256gpu_h100_bf16_32nodes_config",
}


def _function(path: Path, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _explicit_environment(path: Path, function_name: str) -> dict[str, str | int | float | bool]:
    function = _function(path, function_name)
    assignments = [
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "cfg"
        and node.targets[0].attr == "env_vars"
    ]
    assert len(assignments) == 1
    mapping = assignments[0].value
    assert isinstance(mapping, ast.Dict)

    result = COMMON_RECIPE_ENV_VARS.copy()
    common_expansions = 0
    for index, (key, value) in enumerate(zip(mapping.keys, mapping.values)):
        if key is None:
            assert index == 0
            assert isinstance(value, ast.Name) and value.id == "COMMON_RECIPE_ENV_VARS"
            common_expansions += 1
            continue
        result[ast.literal_eval(key)] = ast.literal_eval(value)
    assert common_expansions == 1
    return result


def _explicit_environments():
    for path in _RECIPE_ROOT.glob("*/h100/*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _CANONICAL_RECIPE_NAME.fullmatch(node.name) is not None:
                returns = [statement for statement in node.body if isinstance(statement, ast.Return)]
                if not returns:
                    continue
                yield path, node.name, _explicit_environment(path, node.name)


def test_common_recipe_environment_is_small_and_universal():
    assert COMMON_RECIPE_ENV_VARS == {
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "TORCH_NCCL_HIGH_PRIORITY": 1,
    }


def test_every_supported_hardware_recipe_declares_its_environment_inline():
    supported = []
    unsupported = []

    for path in _RECIPE_ROOT.glob("*/h100/*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or _CANONICAL_RECIPE_NAME.fullmatch(node.name) is None:
                continue
            assert not node.decorator_list
            returns = [statement for statement in node.body if isinstance(statement, ast.Return)]
            if not returns:
                unsupported.append(f"{path.relative_to(_RECIPE_ROOT)}:{node.name}")
                continue
            supported.append(f"{path.relative_to(_RECIPE_ROOT)}:{node.name}")
            _explicit_environment(path, node.name)

            assignment_index = next(
                index
                for index, statement in enumerate(node.body)
                if isinstance(statement, ast.Assign)
                and isinstance(statement.targets[0], ast.Attribute)
                and statement.targets[0].attr == "env_vars"
            )
            assert isinstance(node.body[assignment_index + 1], ast.Return)

    assert len(supported) == 259
    assert unsupported == ["qwen/h100/qwen3_next.py:qwen3_next_80b_a3b_peft_1gpu_h100_bf16_config"]


def test_explicit_recipe_environment_invariants():
    recipes = list(_explicit_environments())
    hybrid_ep_count = 0
    deepseek_v3_environment_recipe_names = set()

    for path, function_name, environment in recipes:
        hybrid_ep_names = environment.keys() & _HYBRID_EP_ENV_NAMES
        assert not hybrid_ep_names or hybrid_ep_names == _HYBRID_EP_ENV_NAMES
        if hybrid_ep_names:
            hybrid_ep_count += 1
            assert environment["NVLINK_DOMAIN_SIZE"] == 8
            assert environment["USE_MNNVL"] == 0
            assert environment["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] in {1, 8}

        if path.parts[-3] == "deepseek":
            assert environment.keys().isdisjoint(_DEEPSEEK_NON_BASELINE_ENV_NAMES)
            layernorm_margin_names = {
                "NVTE_BWD_LAYERNORM_SM_MARGIN",
                "NVTE_FWD_LAYERNORM_SM_MARGIN",
            }
            configured_margin_names = environment.keys() & layernorm_margin_names
            assert not configured_margin_names or configured_margin_names == layernorm_margin_names
            if function_name in _DEEPSEEK_V3_ENVIRONMENT_RECIPE_NAMES:
                assert configured_margin_names == layernorm_margin_names
                deepseek_v3_environment_recipe_names.add(function_name)
            if configured_margin_names:
                assert environment["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
                assert environment["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20

    assert len(recipes) == 259
    assert hybrid_ep_count == 8
    assert deepseek_v3_environment_recipe_names == _DEEPSEEK_V3_ENVIRONMENT_RECIPE_NAMES


@pytest.mark.parametrize(
    ("relative_path", "function_name", "expected"),
    [
        (
            "deepseek/h100/deepseek_v3.py",
            "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
            {
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NVLINK_DOMAIN_SIZE": 8,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "USE_MNNVL": 0,
            },
        ),
        (
            "nemotronh/h100/nemotron_3_super.py",
            "nemotron_3_super_peft_1gpu_h100_bf16_config",
            {
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 1,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
            },
        ),
        (
            "qwen/h100/qwen3.py",
            "qwen3_600m_pretrain_1gpu_h100_bf16_config",
            {},
        ),
    ],
)
def test_representative_recipe_environment_is_visible(relative_path, function_name, expected):
    environment = _explicit_environment(_RECIPE_ROOT / relative_path, function_name)

    assert environment.items() >= expected.items()
