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

"""Guard the 64-GPU debugging proxies against drift from their 256-GPU parents."""

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining, get_forward_backward_func
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.perf_recipes import deepseek
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("precision", ["fp8mx", "nvfp4"])
def test_proxy_preserves_parent_except_depth_and_layout(precision: str) -> None:
    parent_fn: Callable[[], ConfigContainer] = getattr(
        deepseek, f"deepseek_v3_pretrain_256gpu_vr200_{precision}_config"
    )
    proxy_fn: Callable[[], ConfigContainer] = getattr(
        deepseek, f"deepseek_v3_pretrain_64gpu_vr200_{precision}_proxy_config"
    )
    parent = parent_fn()
    proxy = proxy_fn()

    assert parent.train.global_batch_size == proxy.train.global_batch_size == 4096
    assert parent.train.micro_batch_size == proxy.train.micro_batch_size == 1
    assert parent.train.train_iters == proxy.train.train_iters == 50
    # Current MCore requires an explicit backend; None fails during model construction.
    assert parent.model.attention_backend is proxy.model.attention_backend is AttnBackend.auto
    depth_fields = {
        "num_layers",
        "moe_layer_freq",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "pipeline_model_parallel_layout",
    }
    assert {key: value for key, value in vars(proxy.model).items() if key not in depth_fields} == {
        key: value for key, value in vars(parent.model).items() if key not in depth_fields
    }
    for field in fields(parent):
        if field.name != "model":
            assert getattr(proxy, field.name) == getattr(parent, field.name), field.name

    model = proxy.model
    assert model.num_layers == 13
    assert model.moe_layer_freq == [0] * 3 + [1] * 10
    assert (model.tensor_model_parallel_size, model.context_parallel_size) == (1, 1)
    assert (model.pipeline_model_parallel_size, model.virtual_pipeline_model_parallel_size) == (1, None)
    assert model.pipeline_model_parallel_layout is None
    assert (parent.model.pipeline_model_parallel_size, parent.model.virtual_pipeline_model_parallel_size) == (2, 8)
    assert (model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (32, 1)
    for cfg, world, expected_dp, expected_edp, microbatches in ((proxy, 64, 64, 2, 64), (parent, 256, 128, 4, 32)):
        mesh_model = cfg.model
        dense_mesh = (
            mesh_model.tensor_model_parallel_size
            * mesh_model.context_parallel_size
            * mesh_model.pipeline_model_parallel_size
        )
        expert_mesh = (
            mesh_model.expert_tensor_parallel_size
            * mesh_model.expert_model_parallel_size
            * mesh_model.pipeline_model_parallel_size
        )
        assert world % dense_mesh == world % expert_mesh == 0
        assert world // dense_mesh == expected_dp
        assert world // expert_mesh == expected_edp
        assert cfg.train.global_batch_size % (expected_dp * cfg.train.micro_batch_size) == 0
        assert cfg.train.global_batch_size // (expected_dp * cfg.train.micro_batch_size) == microbatches

    assert model.cuda_graph_impl == "full_iteration"
    assert proxy.comm_overlap is not None
    assert proxy.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert proxy.comm_overlap.delay_wgrad_compute is True
    assert proxy.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert model.moe_flex_dispatcher_backend == "hybridep"
    assert model.mtp_num_layers == 1
    assert model.recompute_modules == []

    # A stale VPP/layout must not send the proxy through an interleaved PP schedule.
    assert (
        get_forward_backward_func(
            pp_size=model.pipeline_model_parallel_size, vp_size=model.virtual_pipeline_model_parallel_size
        )
        is forward_backward_no_pipelining
    )


@pytest.mark.parametrize("precision", ["fp8mx", "nvfp4"])
def test_proxy_overlap_setup_without_pipeline_parallelism(monkeypatch: pytest.MonkeyPatch, precision: str) -> None:
    cfg = getattr(deepseek, f"deepseek_v3_pretrain_64gpu_vr200_{precision}_proxy_config")()
    # Validate the topology without requiring the GPU libraries in this unit test.
    monkeypatch.setattr("megatron.bridge.training.comm_overlap.is_te_min_version", lambda *_: True)
    monkeypatch.setattr("megatron.bridge.training.comm_overlap.is_torch_min_version", lambda *_: True)
    cfg.mixed_precision.setup(cfg.model, cfg.optimizer, cfg.ddp)
    assert cfg.comm_overlap is not None
    cfg.comm_overlap.data_parallel_size = cfg.get_data_parallel_size(64)
    cfg.comm_overlap.finalize()
    cfg.comm_overlap.setup(cfg.model, cfg.optimizer, cfg.ddp)

    assert cfg.model.gradient_accumulation_fusion is True
    assert cfg.model.tp_comm_overlap is False
    assert cfg.model.overlap_p2p_comm is False
    assert cfg.model.batch_p2p_comm is False
    assert cfg.model.overlap_moe_expert_parallel_comm is True
    assert cfg.model.delay_wgrad_compute is True
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.ddp.align_param_gather is False


@pytest.mark.parametrize("precision", ["fp8_mx", "nvfp4"])
def test_proxy_is_available_through_performance_selector(monkeypatch: pytest.MonkeyPatch, precision: str) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts" / "performance"))
    from utils.utils import get_perf_recipe_by_name, list_available_config_variants

    cfg = get_perf_recipe_by_name("deepseek_v3", "pretrain", 64, "vr200", precision, config_variant="proxy")
    assert cfg.train.global_batch_size == 4096
    assert cfg.model.num_layers == 13
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.model.attention_backend is AttnBackend.auto
    assert "proxy" in list_available_config_variants(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="vr200",
        compute_dtype=precision,
        task="pretrain",
    )
