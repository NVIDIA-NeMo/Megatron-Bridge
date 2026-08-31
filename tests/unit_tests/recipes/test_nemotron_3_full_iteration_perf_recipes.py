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

"""Full-iteration CUDA graph coverage for Nemotron 3 GB-series MXFP8 recipes."""

import pytest

from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names, is_full_iteration_cuda_graph
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_FULL_ITERATION_MXFP8_RECIPES = (
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config,
)


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("recipe_factory", _FULL_ITERATION_MXFP8_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gb_mxfp8_recipes_enable_capture_safe_full_iteration_graphs(recipe_factory) -> None:
    cfg = recipe_factory()

    assert is_full_iteration_cuda_graph(cfg.model)
    assert cuda_graph_module_names(cfg.model) == []
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is False
    assert cfg.ddp.check_for_nan_in_grad is False
    assert cfg.optimizer.overlap_param_gather_with_optimizer_step is False
    if cfg.comm_overlap is not None:
        assert cfg.comm_overlap.overlap_param_gather_with_optimizer_step is False

    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules == []
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.moe_expert_rank_capacity_factor == 1.5
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
    assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 1.0

    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.mixed_precision.fp8_dot_product_attention is True

    assert cfg.env_vars["NCCL_GRAPH_REGISTER"] == 0
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 0
    assert "graph_capture_record_stream_reuse:True" in cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"]
