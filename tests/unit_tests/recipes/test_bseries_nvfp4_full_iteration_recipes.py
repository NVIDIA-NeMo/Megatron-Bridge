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

"""Configuration checks for experimental B-series NVFP4 full-iteration recipes."""

import pytest

from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_super_pretrain_64gpu_b200_nvfp4_full_iteration_config,
    nemotron_3_super_pretrain_64gpu_b300_nvfp4_full_iteration_config,
)
from megatron.bridge.perf_recipes.qwen import (
    qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_full_iteration_config,
    qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_full_iteration_config,
)
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    ("recipe", "expected_parallelism"),
    [
        (
            qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_full_iteration_config,
            (1, 8, 3, 8),
        ),
        (
            qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_full_iteration_config,
            (1, 8, 3, 8),
        ),
        (
            nemotron_3_super_pretrain_64gpu_b200_nvfp4_full_iteration_config,
            (2, 1, None, 64),
        ),
        (
            nemotron_3_super_pretrain_64gpu_b300_nvfp4_full_iteration_config,
            (1, 1, None, 8),
        ),
    ],
)
def test_bseries_nvfp4_full_iteration_stack(recipe, expected_parallelism) -> None:
    cfg = recipe()

    assert cfg.mixed_precision.fp4 == "e2m1"
    assert cfg.mixed_precision.fp4_param_gather is True
    assert cfg.mixed_precision.fp8_dot_product_attention is False
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True

    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.high_priority_a2a_comm_stream is False
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32

    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True

    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["NVTE_USE_FAST_MATH"] == 1
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1
    assert "graph_capture_record_stream_reuse:True" in cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"]

    actual_parallelism = (
        cfg.model.tensor_model_parallel_size,
        cfg.model.pipeline_model_parallel_size,
        cfg.model.virtual_pipeline_model_parallel_size,
        cfg.model.expert_model_parallel_size,
    )
    assert actual_parallelism == expected_parallelism


@pytest.mark.parametrize(
    "recipe",
    [
        nemotron_3_super_pretrain_64gpu_b200_nvfp4_full_iteration_config,
        nemotron_3_super_pretrain_64gpu_b300_nvfp4_full_iteration_config,
    ],
)
def test_nemotron_3_super_full_iteration_keeps_mtp_in_bf16(recipe) -> None:
    cfg = recipe()

    assert cfg.model.keep_mtp_spec_in_bf16 is True
    assert cfg.model.quant_recipe is not None
