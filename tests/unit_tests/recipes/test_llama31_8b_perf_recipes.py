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

"""Regression tests for MLPerf Llama 3.1 8B performance recipes.

Pin the GB200/GB300 layouts from the MLPerf recipe restore so a later helper
rewrite cannot silently change parallelism, batch, LR, eval, or CUDA-graph
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
    llama31_8b_pretrain_8gpu_gb200_nvfp4_config,
    llama31_8b_pretrain_8gpu_gb300_nvfp4_config,
    llama31_8b_pretrain_72gpu_gb200_nvfp4_config,
    llama31_8b_pretrain_72gpu_gb300_nvfp4_config,
    llama31_8b_pretrain_512gpu_gb200_fp8cs_config,
    llama31_8b_pretrain_512gpu_gb300_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import (
    discover_recipe_factories,
    patch_recipe_construction_dependencies,
)


pytestmark = pytest.mark.unit

_LLAMA_PACKAGE = importlib.import_module("megatron.bridge.perf_recipes.llama")
_LLAMA31_8B_RECIPES = tuple(
    factory for factory in discover_recipe_factories(_LLAMA_PACKAGE) if "llama31_8b_" in factory.__name__
)


@dataclass(frozen=True)
class _MlperfLlama31_8bExpected:
    """Pinned MLPerf Llama 3.1 8B layout for one GB200/GB300 recipe."""

    global_batch_size: int
    micro_batch_size: int
    lr: float
    min_lr: float
    eval_interval: int
    eval_iters: int
    lr_warmup_iters: int
    tp_comm_bootstrap_backend: str
    load_main_params_from_ckpt: bool
    fp8_param_gather: bool = False
    tp_comm_overlap: bool = False
    fp8_dot_product_attention: bool = False


_MLPERF_LLAMA31_8B_RECIPES: tuple[tuple[Callable[[], ConfigContainer], _MlperfLlama31_8bExpected], ...] = (
    (
        llama31_8b_pretrain_8gpu_gb200_nvfp4_config,
        _MlperfLlama31_8bExpected(
            global_batch_size=16,
            micro_batch_size=2,
            lr=0.0004,
            min_lr=4e-5,
            eval_interval=768,
            eval_iters=64,
            lr_warmup_iters=16,
            tp_comm_bootstrap_backend="mpi",
            load_main_params_from_ckpt=False,
            fp8_dot_product_attention=True,
        ),
    ),
    (
        llama31_8b_pretrain_72gpu_gb200_nvfp4_config,
        _MlperfLlama31_8bExpected(
            global_batch_size=72,
            micro_batch_size=1,
            lr=0.0008,
            min_lr=8e-5,
            eval_interval=171,
            eval_iters=15,
            lr_warmup_iters=64,
            tp_comm_bootstrap_backend="mpi",
            load_main_params_from_ckpt=False,
            fp8_dot_product_attention=True,
        ),
    ),
    (
        llama31_8b_pretrain_512gpu_gb200_fp8cs_config,
        _MlperfLlama31_8bExpected(
            global_batch_size=64,
            micro_batch_size=1,
            lr=0.0008,
            min_lr=8e-5,
            eval_interval=192,
            eval_iters=16,
            lr_warmup_iters=64,
            tp_comm_bootstrap_backend="gloo",
            load_main_params_from_ckpt=True,
            fp8_param_gather=True,
            tp_comm_overlap=True,
        ),
    ),
    (
        llama31_8b_pretrain_8gpu_gb300_nvfp4_config,
        _MlperfLlama31_8bExpected(
            global_batch_size=16,
            micro_batch_size=2,
            lr=0.0004,
            min_lr=4e-5,
            eval_interval=768,
            eval_iters=64,
            lr_warmup_iters=16,
            tp_comm_bootstrap_backend="mpi",
            load_main_params_from_ckpt=False,
            fp8_dot_product_attention=True,
        ),
    ),
    (
        llama31_8b_pretrain_72gpu_gb300_nvfp4_config,
        _MlperfLlama31_8bExpected(
            global_batch_size=72,
            micro_batch_size=1,
            lr=0.0008,
            min_lr=8e-5,
            eval_interval=171,
            eval_iters=15,
            lr_warmup_iters=64,
            tp_comm_bootstrap_backend="mpi",
            load_main_params_from_ckpt=False,
            fp8_dot_product_attention=True,
        ),
    ),
    (
        llama31_8b_pretrain_512gpu_gb300_fp8cs_config,
        _MlperfLlama31_8bExpected(
            global_batch_size=64,
            micro_batch_size=1,
            lr=0.0008,
            min_lr=8e-5,
            eval_interval=192,
            eval_iters=16,
            lr_warmup_iters=64,
            tp_comm_bootstrap_backend="gloo",
            load_main_params_from_ckpt=True,
            fp8_param_gather=True,
            tp_comm_overlap=True,
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


@pytest.mark.parametrize("recipe_factory", _LLAMA31_8B_RECIPES, ids=lambda factory: factory.__name__)
def test_llama31_8b_perf_recipes_apply_common_defaults(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """Every Llama 3.1 8B perf recipe keeps the MLPerf helper defaults."""
    cfg = recipe_factory()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model.hidden_size == 4096
    assert cfg.model.num_layers == 32
    assert cfg.model.num_attention_heads == 32
    assert cfg.model.num_query_groups == 8
    assert cfg.model.seq_length == 8192
    assert cfg.model.vocab_size == 128256
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.sequence_parallel is False
    assert cfg.model.recompute_modules == ["core_attn"]
    assert cfg.model.cuda_graph_impl == "local"
    assert cfg.model.cuda_graph_scope == ["full_iteration"]
    assert cfg.ddp.check_for_nan_in_grad is False
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.optimizer.overlap_param_gather is True
    assert cfg.dataset.seq_length == 8192
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    gpu_count = _gpu_count_from_recipe_name(recipe_factory.__name__)
    assert data_parallel_size(num_gpus=gpu_count, topology=topology_from_config(cfg.model)) >= 1


@pytest.mark.parametrize(
    ("recipe_factory", "expected"),
    _MLPERF_LLAMA31_8B_RECIPES,
    ids=[recipe_factory.__name__ for recipe_factory, _ in _MLPERF_LLAMA31_8B_RECIPES],
)
def test_llama31_8b_mlperf_recipes_pin_layout(
    recipe_factory: Callable[[], ConfigContainer],
    expected: _MlperfLlama31_8bExpected,
) -> None:
    """GB200/GB300 8B recipes keep the restored MLPerf parallelism and schedule."""
    cfg = recipe_factory()
    is_nvfp4 = "nvfp4" in recipe_factory.__name__

    assert cfg.train.global_batch_size == expected.global_batch_size
    assert cfg.train.micro_batch_size == expected.micro_batch_size
    assert cfg.optimizer.lr == pytest.approx(expected.lr)
    assert cfg.optimizer.min_lr == pytest.approx(expected.min_lr)
    assert cfg.validation.eval_interval == expected.eval_interval
    assert cfg.validation.eval_iters == expected.eval_iters
    assert cfg.validation.eval_global_batch_size == expected.global_batch_size
    assert cfg.validation.eval_micro_batch_size == expected.micro_batch_size
    assert cfg.scheduler.lr_warmup_iters == expected.lr_warmup_iters
    assert cfg.model.tp_comm_bootstrap_backend == expected.tp_comm_bootstrap_backend
    assert cfg.load_main_params_from_ckpt is expected.load_main_params_from_ckpt
    assert cfg.ddp.fp8_param_gather is expected.fp8_param_gather
    assert cfg.model.tp_comm_overlap is expected.tp_comm_overlap
    assert cfg.mixed_precision.fp8_dot_product_attention is expected.fp8_dot_product_attention
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1
    assert cfg.env_vars["NVTE_USE_FAST_MATH"] == 0

    if is_nvfp4:
        assert cfg.mixed_precision.fp4_recipe == "nvfp4"
        assert cfg.mixed_precision.fp4_param is False
        assert cfg.model.use_transformer_engine_op_fuser is True
    else:
        assert cfg.mixed_precision.fp8 is not None
        assert cfg.mixed_precision.fp4_param is False
