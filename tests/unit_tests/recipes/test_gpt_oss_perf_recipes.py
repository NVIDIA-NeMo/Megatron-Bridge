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

"""Regression tests for GPT-OSS 20B performance recipes.

Pin the MLPerf GB200/GB300 layouts restored after the 26.08 recipe refactor so a
later helper rewrite cannot silently change parallelism, batch, LR, eval, or
CUDA-graph settings.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config

from megatron.bridge.perf_recipes.gpt_oss import (
    gpt_oss_20b_pretrain_8gpu_gb200_fp8mx_config,
    gpt_oss_20b_pretrain_8gpu_gb200_nvfp4_config,
    gpt_oss_20b_pretrain_8gpu_gb300_fp8mx_config,
    gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config,
    gpt_oss_20b_pretrain_72gpu_gb200_fp8mx_config,
    gpt_oss_20b_pretrain_72gpu_gb200_nvfp4_config,
    gpt_oss_20b_pretrain_72gpu_gb300_fp8mx_config,
    gpt_oss_20b_pretrain_72gpu_gb300_nvfp4_config,
    gpt_oss_20b_pretrain_512gpu_gb200_fp8mx_config,
    gpt_oss_20b_pretrain_512gpu_gb300_fp8mx_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import (
    discover_recipe_factories,
    patch_recipe_construction_dependencies,
)


pytestmark = pytest.mark.unit

_TE_GRAPH_SCOPE = ["attn", "moe_router", "moe_preprocess"]
_GPT_OSS_PACKAGE = importlib.import_module("megatron.bridge.perf_recipes.gpt_oss")
_GPT_OSS_20B_RECIPES = tuple(
    factory for factory in discover_recipe_factories(_GPT_OSS_PACKAGE) if "_20b_" in factory.__name__
)


@dataclass(frozen=True)
class _MlperfGptOss20bExpected:
    """Pinned MLPerf GPT-OSS 20B layout for one GB200/GB300 recipe."""

    tp: int
    cp: int
    ep: int
    etp: int
    sequence_parallel: bool
    global_batch_size: int
    micro_batch_size: int
    lr: float
    min_lr: float
    eval_interval: int
    eval_iters: int
    lr_warmup_iters: int
    cuda_graph_impl: str
    hybridep_ranks: int
    moe_expert_rank_capacity_factor: float | None = None
    cuda_graph_warmup_steps: int = 2
    fp8_param_gather: bool | None = None
    average_in_collective: bool | None = None


_MLPERF_GB200_GB300_20B_RECIPES: tuple[tuple[Callable[[], ConfigContainer], _MlperfGptOss20bExpected], ...] = (
    (
        gpt_oss_20b_pretrain_8gpu_gb200_nvfp4_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=1,
            ep=2,
            etp=1,
            sequence_parallel=False,
            global_batch_size=16,
            micro_batch_size=2,
            lr=0.0006,
            min_lr=6e-5,
            eval_interval=768,
            eval_iters=64,
            lr_warmup_iters=128,
            cuda_graph_impl="transformer_engine",
            hybridep_ranks=2,
        ),
    ),
    (
        gpt_oss_20b_pretrain_8gpu_gb200_fp8mx_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=1,
            ep=1,
            etp=1,
            sequence_parallel=False,
            global_batch_size=16,
            micro_batch_size=2,
            lr=0.0004,
            min_lr=4e-5,
            eval_interval=768,
            eval_iters=64,
            lr_warmup_iters=128,
            cuda_graph_impl="local",
            hybridep_ranks=1,
            moe_expert_rank_capacity_factor=1.2,
            cuda_graph_warmup_steps=5,
            fp8_param_gather=True,
            average_in_collective=True,
        ),
    ),
    (
        gpt_oss_20b_pretrain_72gpu_gb200_nvfp4_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=2,
            ep=4,
            etp=1,
            sequence_parallel=False,
            global_batch_size=36,
            micro_batch_size=1,
            lr=0.0006,
            min_lr=4e-5,
            eval_interval=341,
            eval_iters=29,
            lr_warmup_iters=64,
            cuda_graph_impl="transformer_engine",
            hybridep_ranks=4,
        ),
    ),
    (
        gpt_oss_20b_pretrain_72gpu_gb200_fp8mx_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=2,
            ep=4,
            etp=1,
            sequence_parallel=False,
            global_batch_size=36,
            micro_batch_size=1,
            lr=0.0004,
            min_lr=4e-5,
            eval_interval=341,
            eval_iters=29,
            lr_warmup_iters=256,
            cuda_graph_impl="local",
            hybridep_ranks=4,
            moe_expert_rank_capacity_factor=5,
        ),
    ),
    (
        gpt_oss_20b_pretrain_512gpu_gb200_fp8mx_config,
        _MlperfGptOss20bExpected(
            tp=2,
            cp=4,
            ep=8,
            etp=1,
            sequence_parallel=True,
            global_batch_size=64,
            micro_batch_size=1,
            lr=0.00052,
            min_lr=5.2e-5,
            eval_interval=192,
            eval_iters=16,
            lr_warmup_iters=32,
            cuda_graph_impl="local",
            hybridep_ranks=8,
            moe_expert_rank_capacity_factor=7,
        ),
    ),
    (
        gpt_oss_20b_pretrain_8gpu_gb300_fp8mx_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=1,
            ep=1,
            etp=1,
            sequence_parallel=False,
            global_batch_size=24,
            micro_batch_size=3,
            lr=0.0005,
            min_lr=5e-5,
            eval_interval=512,
            eval_iters=43,
            lr_warmup_iters=256,
            cuda_graph_impl="local",
            hybridep_ranks=1,
            moe_expert_rank_capacity_factor=1.5,
        ),
    ),
    (
        gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=1,
            ep=2,
            etp=1,
            sequence_parallel=False,
            global_batch_size=24,
            micro_batch_size=3,
            lr=0.0004,
            min_lr=4e-5,
            eval_interval=512,
            eval_iters=43,
            lr_warmup_iters=192,
            cuda_graph_impl="transformer_engine",
            hybridep_ranks=2,
        ),
    ),
    (
        gpt_oss_20b_pretrain_72gpu_gb300_fp8mx_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=2,
            ep=4,
            etp=1,
            sequence_parallel=False,
            global_batch_size=36,
            micro_batch_size=1,
            lr=0.0004,
            min_lr=4e-5,
            eval_interval=341,
            eval_iters=29,
            lr_warmup_iters=256,
            cuda_graph_impl="local",
            hybridep_ranks=4,
            moe_expert_rank_capacity_factor=5,
        ),
    ),
    (
        gpt_oss_20b_pretrain_72gpu_gb300_nvfp4_config,
        _MlperfGptOss20bExpected(
            tp=1,
            cp=2,
            ep=4,
            etp=1,
            sequence_parallel=False,
            global_batch_size=36,
            micro_batch_size=1,
            lr=0.0006,
            min_lr=6e-5,
            eval_interval=341,
            eval_iters=29,
            lr_warmup_iters=64,
            cuda_graph_impl="transformer_engine",
            hybridep_ranks=4,
        ),
    ),
    (
        gpt_oss_20b_pretrain_512gpu_gb300_fp8mx_config,
        _MlperfGptOss20bExpected(
            tp=2,
            cp=4,
            ep=8,
            etp=1,
            sequence_parallel=True,
            global_batch_size=64,
            micro_batch_size=1,
            lr=0.00052,
            min_lr=5.2e-5,
            eval_interval=192,
            eval_iters=16,
            lr_warmup_iters=32,
            cuda_graph_impl="local",
            hybridep_ranks=8,
            moe_expert_rank_capacity_factor=7,
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


@pytest.mark.parametrize("recipe_factory", _GPT_OSS_20B_RECIPES, ids=lambda factory: factory.__name__)
def test_gpt_oss_20b_perf_recipes_apply_common_defaults(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """Every 20B perf recipe keeps the HybridEP helper defaults used by MLPerf."""
    cfg = recipe_factory()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model.seq_length == 8192
    assert cfg.model.vocab_size == 128256
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.fsdp_double_buffer is True
    assert cfg.ddp.nccl_ub is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.optimizer.overlap_param_gather is False
    assert cfg.rng.te_rng_tracker is True

    gpu_count = _gpu_count_from_recipe_name(recipe_factory.__name__)
    assert data_parallel_size(num_gpus=gpu_count, topology=topology_from_config(cfg.model)) >= 1


@pytest.mark.parametrize(
    ("recipe_factory", "expected"),
    _MLPERF_GB200_GB300_20B_RECIPES,
    ids=[recipe_factory.__name__ for recipe_factory, _ in _MLPERF_GB200_GB300_20B_RECIPES],
)
def test_gpt_oss_20b_mlperf_recipes_pin_layout(
    recipe_factory: Callable[[], ConfigContainer],
    expected: _MlperfGptOss20bExpected,
) -> None:
    """GB200/GB300 20B recipes keep the restored MLPerf parallelism and schedule."""
    cfg = recipe_factory()
    is_fp8mx = "fp8mx" in recipe_factory.__name__

    assert cfg.model.tensor_model_parallel_size == expected.tp
    assert cfg.model.context_parallel_size == expected.cp
    assert cfg.model.expert_model_parallel_size == expected.ep
    assert cfg.model.expert_tensor_parallel_size == expected.etp
    assert cfg.model.sequence_parallel is expected.sequence_parallel
    assert cfg.train.global_batch_size == expected.global_batch_size
    assert cfg.train.micro_batch_size == expected.micro_batch_size
    assert cfg.optimizer.lr == pytest.approx(expected.lr)
    assert cfg.optimizer.min_lr == pytest.approx(expected.min_lr)
    assert cfg.validation.eval_interval == expected.eval_interval
    assert cfg.validation.eval_iters == expected.eval_iters
    assert cfg.scheduler.lr_warmup_iters == expected.lr_warmup_iters
    assert cfg.model.cuda_graph_impl == expected.cuda_graph_impl
    assert cfg.model.cuda_graph_warmup_steps == expected.cuda_graph_warmup_steps
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected.hybridep_ranks
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1

    if expected.cuda_graph_impl == "transformer_engine":
        assert cfg.model.cuda_graph_scope == _TE_GRAPH_SCOPE
    else:
        assert cfg.model.cuda_graph_modules == "full_iteration"
        assert cfg.model.cuda_graph_scope is None
        assert cfg.model.use_transformer_engine_op_fuser is True
        assert cfg.model.moe_mlp_glu_interleave_size == 32

    if expected.moe_expert_rank_capacity_factor is not None:
        assert cfg.model.moe_expert_rank_capacity_factor == expected.moe_expert_rank_capacity_factor
    if expected.fp8_param_gather is not None:
        assert cfg.mixed_precision.fp8_param_gather is expected.fp8_param_gather
        assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is expected.fp8_param_gather
    if expected.average_in_collective is not None:
        assert cfg.ddp.average_in_collective is expected.average_in_collective

    if is_fp8mx:
        assert cfg.mixed_precision.fp8_recipe == "mxfp8"
        assert cfg.mixed_precision.fp8 == "e4m3"
    else:
        assert cfg.mixed_precision.fp4_recipe == "nvfp4"
        assert cfg.env_vars["NVTE_USE_FAST_MATH"] == 1
