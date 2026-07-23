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

"""Tests for explicit flat performance recipe environment settings."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.perf_recipes._common import _benchmark_common
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS


_CANONICAL_RECIPE_NAME = re.compile(
    r".+_(?:pretrain|sft|peft)_\d+gpu_[a-z0-9]+_(?:bf16|fp8cs|fp8mx|fp8sc|nvfp4)(?:_.+)?_config"
)
_RECIPE_ROOT = Path(__file__).resolve().parents[3] / "src" / "megatron" / "bridge" / "perf_recipes"
# Deliberately lock the discovered public inventory; update this for intentional recipe additions or removals.
_EXPECTED_FLAT_RECIPE_COUNT = 411
_EXPECTED_DEEPSEEK_RECIPE_COUNT = 40
_EXPECTED_DEEPSEEK_HYBRID_EP_RECIPE_COUNT = 38
_INLINE_CORE_ENV_NAMES = {
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "NVTE_BWD_LAYERNORM_SM_MARGIN",
    "NVTE_FWD_LAYERNORM_SM_MARGIN",
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
}
_HYBRID_EP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}
_DEEPSEEK_NON_BASELINE_ENV_NAMES = {
    "QUANTIZATION_TYPE_DEBUG",
    "TORCHINDUCTOR_WORKER_START",
}
_DEEPSEEK_WITHOUT_HYBRID_EP_RECIPES = {
    ("b200", "deepseek_v3_pretrain_256gpu_b200_fp8mx_config"),
    ("b200", "deepseek_v3_pretrain_256gpu_b200_nvfp4_config"),
}


def _function(path: Path, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _explicit_environment(path: Path, function_name: str) -> dict[str, str | int | float | bool]:
    """Read the literal env mapping written in a flat recipe builder."""
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

    result = COMMON_PERF_ENV_VARS.copy()
    common_expansions = 0
    for key, value in zip(mapping.keys, mapping.values):
        if key is None:
            assert isinstance(value, ast.Name) and value.id == "COMMON_PERF_ENV_VARS"
            common_expansions += 1
            continue
        result[ast.literal_eval(key)] = ast.literal_eval(value)
    assert common_expansions == 1
    return result


def _explicit_environments():
    for path in _RECIPE_ROOT.glob("*/*/*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _CANONICAL_RECIPE_NAME.fullmatch(node.name) is not None:
                yield path, node.name, _explicit_environment(path, node.name)


def test_common_environment_defaults_are_small_and_universal():
    assert COMMON_PERF_ENV_VARS == {"TORCH_NCCL_HIGH_PRIORITY": 1}


def test_benchmark_common_preserves_legacy_manual_gc_defaults():
    cfg = SimpleNamespace(
        train=SimpleNamespace(train_iters=0, eval_iters=1, manual_gc=False, manual_gc_interval=0),
        checkpoint=SimpleNamespace(save="checkpoint"),
        logger=SimpleNamespace(log_interval=10, tensorboard_dir="tensorboard"),
        ddp=SimpleNamespace(check_for_nan_in_grad=True, check_for_large_grads=True, grad_reduce_in_fp32=True),
        rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=True),
        scheduler=SimpleNamespace(lr_decay_iters=0, lr_warmup_iters=0),
        model=SimpleNamespace(
            use_transformer_engine_op_fuser=False,
            apply_rope_fusion=False,
            cross_entropy_fusion_impl="native",
            cuda_graph_impl=None,
            cuda_graph_scope=[],
            moe_flex_dispatcher_backend=None,
        ),
        mixed_precision=SimpleNamespace(grad_reduce_in_fp32=True),
    )

    _benchmark_common(cfg)

    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100


def test_every_flat_recipe_builder_declares_its_environment_inline():
    builders = []
    invalid = []

    for path in _RECIPE_ROOT.glob("*/*/*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or _CANONICAL_RECIPE_NAME.fullmatch(node.name) is None:
                continue
            builders.append(f"{path.relative_to(_RECIPE_ROOT)}:{node.name}")
            try:
                _explicit_environment(path, node.name)
            except AssertionError:
                invalid.append(builders[-1])
            assert not any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "perf_recipe_environment"
                for decorator in node.decorator_list
            )

    assert len(builders) == _EXPECTED_FLAT_RECIPE_COUNT
    assert not invalid


def test_explicit_environment_invariants_across_all_flat_recipes():
    """Keep duplicated inline settings complete without deriving them at runtime."""
    recipes = list(_explicit_environments())
    deepseek_recipe_count = 0
    deepseek_hybrid_ep_count = 0

    for path, function_name, environment in recipes:
        assert environment.keys() >= _INLINE_CORE_ENV_NAMES

        cudnn_names = {"NVTE_NORM_BWD_USE_CUDNN", "NVTE_NORM_FWD_USE_CUDNN"}
        assert environment.keys().isdisjoint(cudnn_names) or environment.keys() >= cudnn_names

        hybrid_ep_names = environment.keys() & _HYBRID_EP_ENV_NAMES
        assert not hybrid_ep_names or hybrid_ep_names == _HYBRID_EP_ENV_NAMES
        if hybrid_ep_names:
            gpu = path.parent.name
            nvlink_domain_size = 72 if gpu in {"gb200", "gb300", "vr200"} else 8
            assert environment["NVLINK_DOMAIN_SIZE"] == nvlink_domain_size
            assert environment["USE_MNNVL"] == int(nvlink_domain_size == 72)
            assert environment["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] <= nvlink_domain_size

        if "_nvfp4" in function_name:
            assert environment["NVTE_USE_FAST_MATH"] == 1
        if path.parts[-3] == "deepseek":
            deepseek_recipe_count += 1
            assert environment.keys().isdisjoint(_DEEPSEEK_NON_BASELINE_ENV_NAMES)
            assert environment["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
            assert environment["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20
            assert environment["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == 0

            recipe_id = (path.parent.name, function_name)
            if recipe_id in _DEEPSEEK_WITHOUT_HYBRID_EP_RECIPES:
                assert not hybrid_ep_names
            else:
                assert hybrid_ep_names == _HYBRID_EP_ENV_NAMES
                deepseek_hybrid_ep_count += 1

    assert len(recipes) == _EXPECTED_FLAT_RECIPE_COUNT
    assert deepseek_recipe_count == _EXPECTED_DEEPSEEK_RECIPE_COUNT
    assert deepseek_hybrid_ep_count == _EXPECTED_DEEPSEEK_HYBRID_EP_RECIPE_COUNT


@pytest.mark.parametrize(
    ("relative_path", "function_name", "expected"),
    [
        (
            "qwen/h100/qwen3_moe.py",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            {
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
            },
        ),
        (
            "deepseek/gb200/deepseek_v3.py",
            "deepseek_v3_pretrain_256gpu_gb200_bf16_config",
            {
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
                "NVLINK_DOMAIN_SIZE": 72,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                "USE_MNNVL": 1,
            },
        ),
        (
            "llama/h100/llama3.py",
            "llama3_8b_pretrain_8gpu_h100_fp8cs_config",
            {
                "NCCL_CTA_POLICY": 1,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            },
        ),
        (
            "wan/h100/wan.py",
            "wan_14b_pretrain_32gpu_h100_bf16_config",
            {"CUDA_DEVICE_MAX_CONNECTIONS": 1},
        ),
    ],
)
def test_representative_recipe_specific_environment_is_visible(relative_path, function_name, expected):
    environment = _explicit_environment(_RECIPE_ROOT / relative_path, function_name)

    assert environment.items() >= expected.items()
