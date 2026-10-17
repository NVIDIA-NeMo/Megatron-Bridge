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

"""Regression tests for MLPerf Llama 2 70B LoRA performance recipes.

Pin the GB200/GB300 layouts from PRs #5735 and #6266 so a later helper rewrite
cannot silently change parallelism, batch, eval, CUDA-graph, or CPU-offload
settings.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config

from megatron.bridge.perf_recipes.llama import (
    llama2_70b_peft_4gpu_gb200_fp8ds_config,
    llama2_70b_peft_4gpu_gb300_fp8ds_config,
    llama2_70b_peft_8gpu_gb200_fp8ds_config,
    llama2_70b_peft_8gpu_gb200_nvfp4_config,
    llama2_70b_peft_8gpu_gb300_fp8ds_config,
    llama2_70b_peft_72gpu_gb200_fp8ds_config,
    llama2_70b_peft_72gpu_gb300_fp8ds_config,
    llama2_70b_peft_512gpu_gb200_fp8ds_config,
    llama2_70b_peft_512gpu_gb300_fp8ds_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import (
    discover_recipe_factories,
    patch_recipe_construction_dependencies,
)


pytestmark = pytest.mark.unit

_LLAMA_PACKAGE = importlib.import_module("megatron.bridge.perf_recipes.llama")
_LLAMA2_70B_LORA_RECIPES = tuple(
    factory for factory in discover_recipe_factories(_LLAMA_PACKAGE) if "llama2_70b_peft_" in factory.__name__
)


@dataclass(frozen=True)
class _MlperfLlama2_70bLoraExpected:
    """Pinned MLPerf Llama 2 70B LoRA layout for one GB200/GB300 recipe."""

    context_parallel_size: int
    global_batch_size: int
    micro_batch_size: int
    lr: float
    eval_global_batch_size: int
    eval_interval: int
    eval_iters: int
    cuda_graph_impl: str
    cuda_graph_scope: str | list[str] | None
    cpu_offloading: bool
    fp8_param_gather: bool
    seed: int
    cpu_offloading_num_layers: int | None = None


_MLPERF_LLAMA2_70B_LORA_RECIPES: tuple[tuple[Callable[[], ConfigContainer], _MlperfLlama2_70bLoraExpected], ...] = (
    (
        llama2_70b_peft_4gpu_gb200_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=1,
            global_batch_size=8,
            micro_batch_size=1,
            lr=0.0005,
            eval_global_batch_size=4,
            eval_interval=48,
            eval_iters=44,
            cuda_graph_impl="none",
            cuda_graph_scope=None,
            cpu_offloading=True,
            cpu_offloading_num_layers=20,
            fp8_param_gather=True,
            seed=30339,
        ),
    ),
    (
        llama2_70b_peft_8gpu_gb200_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=1,
            global_batch_size=8,
            micro_batch_size=1,
            lr=0.0005,
            eval_global_batch_size=8,
            eval_interval=48,
            eval_iters=22,
            cuda_graph_impl="local",
            cuda_graph_scope=["full_iteration"],
            cpu_offloading=True,
            cpu_offloading_num_layers=11,
            fp8_param_gather=True,
            seed=27208,
        ),
    ),
    (
        llama2_70b_peft_8gpu_gb200_nvfp4_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=2,
            global_batch_size=8,
            micro_batch_size=1,
            lr=0.0006,
            eval_global_batch_size=4,
            eval_interval=48,
            eval_iters=44,
            cuda_graph_impl="local",
            cuda_graph_scope=["full_iteration"],
            cpu_offloading=False,
            fp8_param_gather=False,
            seed=23829,
        ),
    ),
    (
        llama2_70b_peft_72gpu_gb200_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=8,
            global_batch_size=9,
            micro_batch_size=1,
            lr=0.0005,
            eval_global_batch_size=36,
            eval_interval=43,
            eval_iters=5,
            cuda_graph_impl="local",
            cuda_graph_scope=["full_iteration"],
            cpu_offloading=False,
            fp8_param_gather=True,
            seed=16584,
        ),
    ),
    (
        llama2_70b_peft_512gpu_gb200_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=8,
            global_batch_size=64,
            micro_batch_size=1,
            lr=0.0006,
            eval_global_batch_size=64,
            eval_interval=6,
            eval_iters=3,
            cuda_graph_impl="local",
            cuda_graph_scope=["full_iteration"],
            cpu_offloading=False,
            fp8_param_gather=True,
            seed=22205,
        ),
    ),
    (
        llama2_70b_peft_4gpu_gb300_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=1,
            global_batch_size=8,
            micro_batch_size=1,
            lr=0.0005,
            eval_global_batch_size=4,
            eval_interval=48,
            eval_iters=44,
            cuda_graph_impl="local",
            cuda_graph_scope="full_iteration",
            cpu_offloading=False,
            fp8_param_gather=True,
            seed=10710,
        ),
    ),
    (
        llama2_70b_peft_8gpu_gb300_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=1,
            global_batch_size=8,
            micro_batch_size=1,
            lr=0.0005,
            eval_global_batch_size=8,
            eval_interval=48,
            eval_iters=22,
            cuda_graph_impl="local",
            cuda_graph_scope="full_iteration",
            cpu_offloading=False,
            fp8_param_gather=True,
            seed=27208,
        ),
    ),
    (
        llama2_70b_peft_72gpu_gb300_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=8,
            global_batch_size=9,
            micro_batch_size=1,
            lr=0.0005,
            eval_global_batch_size=36,
            eval_interval=43,
            eval_iters=5,
            cuda_graph_impl="local",
            cuda_graph_scope="full_iteration",
            cpu_offloading=False,
            fp8_param_gather=True,
            seed=14954,
        ),
    ),
    (
        llama2_70b_peft_512gpu_gb300_fp8ds_config,
        _MlperfLlama2_70bLoraExpected(
            context_parallel_size=8,
            global_batch_size=64,
            micro_batch_size=1,
            lr=0.0006,
            eval_global_batch_size=64,
            eval_interval=6,
            eval_iters=3,
            cuda_graph_impl="local",
            cuda_graph_scope="full_iteration",
            cpu_offloading=False,
            fp8_param_gather=True,
            seed=8353,
        ),
    ),
)


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def _gpu_count_from_recipe_name(recipe_name: str) -> int:
    match = re.search(r"_(\d+)gpu_", recipe_name)
    if match is None:
        raise AssertionError(f"Could not parse GPU count from {recipe_name}")
    return int(match.group(1))


@pytest.mark.parametrize("recipe_factory", _LLAMA2_70B_LORA_RECIPES, ids=lambda factory: factory.__name__)
def test_llama2_70b_lora_perf_recipes_apply_common_defaults(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """Every Llama 2 70B LoRA perf recipe keeps the MLPerf helper defaults."""
    cfg = recipe_factory()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model.hidden_size == 8192
    assert cfg.model.ffn_hidden_size == 28672
    assert cfg.model.num_layers == 80
    assert cfg.model.num_attention_heads == 64
    assert cfg.model.num_query_groups == 8
    assert cfg.model.seq_length == 8192
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.sequence_parallel is False
    assert cfg.model.recompute_modules == ["core_attn"]
    assert cfg.model.fused_single_qkv_rope is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.ddp.check_for_nan_in_grad is False
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.dataset.seq_length == 8192
    assert cfg.dataset.enable_offline_packing is True
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.peft is not None
    assert cfg.peft.dim == 16
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]

    gpu_count = _gpu_count_from_recipe_name(recipe_factory.__name__)
    assert data_parallel_size(num_gpus=gpu_count, topology=topology_from_config(cfg.model)) >= 1


@pytest.mark.parametrize(
    ("recipe_factory", "expected"),
    _MLPERF_LLAMA2_70B_LORA_RECIPES,
    ids=[recipe_factory.__name__ for recipe_factory, _ in _MLPERF_LLAMA2_70B_LORA_RECIPES],
)
def test_llama2_70b_lora_mlperf_recipes_pin_layout(
    recipe_factory: Callable[[], ConfigContainer],
    expected: _MlperfLlama2_70bLoraExpected,
) -> None:
    """GB200/GB300 70B LoRA recipes keep the restored MLPerf parallelism and offload settings."""
    cfg = recipe_factory()
    is_nvfp4 = "nvfp4" in recipe_factory.__name__

    assert cfg.model.context_parallel_size == expected.context_parallel_size
    assert cfg.train.global_batch_size == expected.global_batch_size
    assert cfg.train.micro_batch_size == expected.micro_batch_size
    assert cfg.optimizer.lr == pytest.approx(expected.lr)
    assert cfg.validation.eval_global_batch_size == expected.eval_global_batch_size
    assert cfg.validation.eval_interval == expected.eval_interval
    assert cfg.validation.eval_iters == expected.eval_iters
    assert cfg.model.cuda_graph_impl == expected.cuda_graph_impl
    assert cfg.model.cuda_graph_scope == expected.cuda_graph_scope
    assert cfg.model.cpu_offloading is expected.cpu_offloading
    if expected.cpu_offloading_num_layers is not None:
        assert cfg.model.cpu_offloading_num_layers == expected.cpu_offloading_num_layers
    assert cfg.mixed_precision.fp8_param_gather is expected.fp8_param_gather
    assert cfg.rng.seed == expected.seed
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1
    assert cfg.env_vars["NCCL_NVLS_ENABLE"] == 1

    if is_nvfp4:
        assert cfg.mixed_precision.fp4_recipe == "nvfp4"
        assert cfg.mixed_precision.fp8_param is False
    else:
        assert cfg.mixed_precision.fp8_recipe == "delayed"
        assert cfg.mixed_precision.fp8_param is True
