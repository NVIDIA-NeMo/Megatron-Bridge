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

"""Unit tests for Nemotron 3.5 Lightning performance recipe exports."""

import importlib
from collections.abc import Callable
from inspect import signature

import pytest
import torch

from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_fsdp_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_nvfp4_config,
    nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config,
    nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config,
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
    nemotron_3_nano_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config,
    nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit

_NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"  # pragma: allowlist secret

_H100_RECIPES = (
    nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config,
    nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config,
)
_GB200_RECIPES = (
    nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4_config,
)
_GB300_RECIPES = (
    nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_nvfp4_config,
)
_GB_FSDP_RECIPES = (
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_fsdp_config,
)
_NEMOTRON_3_RECIPES = (
    nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
    nemotron_3_nano_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config,
)
_NEMOTRON_3_5_BASE_RECIPE_PAIRS = (
    (
        nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config,
        nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    ),
    (
        nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config,
        nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
    ),
    (
        nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config,
        nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    ),
    (
        nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
        nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    ),
    (
        nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4_config,
        nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
    ),
    (
        nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config,
        nemotron_3_nano_pretrain_8gpu_gb300_bf16_config,
    ),
    (
        nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
        nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config,
    ),
    (
        nemotron_3_5_lightning_pretrain_8gpu_gb300_nvfp4_config,
        nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config,
    ),
)
_NEMOTRON_NANO_PERF_FACTORIES = (
    ("megatron.bridge.perf_recipes.nemotronh.h100.nemotronh", "nemotron_3_nano_pretrain_16gpu_h100_bf16_config"),
    ("megatron.bridge.perf_recipes.nemotronh.h100.nemotronh", "nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config"),
    ("megatron.bridge.perf_recipes.nemotronh.b200.nemotronh", "nemotron_3_nano_pretrain_8gpu_b200_bf16_config"),
    ("megatron.bridge.perf_recipes.nemotronh.b200.nemotronh", "nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config"),
    ("megatron.bridge.perf_recipes.nemotronh.b200.nemotronh", "nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config"),
    ("megatron.bridge.perf_recipes.nemotronh.b300.nemotronh", "nemotron_3_nano_pretrain_8gpu_b300_bf16_config"),
    ("megatron.bridge.perf_recipes.nemotronh.b300.nemotronh", "nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config"),
    ("megatron.bridge.perf_recipes.nemotronh.b300.nemotronh", "nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config"),
    ("megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config"),
    ("megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config"),
    ("megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config"),
    ("megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb300_bf16_config"),
    ("megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config"),
    ("megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config"),
    ("megatron.bridge.perf_recipes.nemotronh.vr200.nemotronh", "nemotron_3_nano_pretrain_8gpu_vr200_bf16_config"),
    ("megatron.bridge.perf_recipes.nemotronh.vr200.nemotronh", "nemotron_3_nano_pretrain_8gpu_vr200_fp8mx_config"),
    ("megatron.bridge.perf_recipes.nemotronh.vr200.nemotronh", "nemotron_3_nano_pretrain_8gpu_vr200_nvfp4_config"),
    (
        "megatron.bridge.perf_recipes.nemotronh.h100.nemotronh",
        "nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.h100.nemotronh",
        "nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_fsdp_config",
    ),
    (
        "megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh",
        "nemotron_3_5_lightning_pretrain_8gpu_gb300_nvfp4_config",
    ),
)


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    _NEMOTRON_NANO_PERF_FACTORIES,
    ids=[factory_name for _, factory_name in _NEMOTRON_NANO_PERF_FACTORIES],
)
def test_nemotron_nano_perf_recipe_model_finalizes(module_name: str, factory_name: str) -> None:
    """Nano perf recipes use one non-conflicting CUDA graph configuration API."""
    module = importlib.import_module(module_name)
    cfg = getattr(module, factory_name)()

    cfg.model.finalize()


@pytest.mark.parametrize("recipe_factory", _NEMOTRON_3_RECIPES, ids=lambda recipe: recipe.__name__)
def test_standard_perf_recipes_do_not_expose_mtp_flag(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """Standard performance recipes remain parameterless and non-MTP."""
    assert "enable_mtp" not in signature(recipe_factory).parameters
    assert recipe_factory().model.mtp_num_layers == 0


@pytest.mark.parametrize(
    "recipe_factory",
    (*_H100_RECIPES, *_GB200_RECIPES, *_GB300_RECIPES, *_GB_FSDP_RECIPES),
    ids=lambda recipe: recipe.__name__,
)
def test_perf_recipes_enable_mtp(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """Each Nemotron 3.5 performance recipe preserves the shared MTP block."""
    cfg = recipe_factory()

    assert cfg.model.mtp_num_layers == 2
    assert cfg.model.mtp_hybrid_override_pattern == "*E"
    assert cfg.model.mtp_use_repeated_layer is True
    assert cfg.model.keep_mtp_spec_in_bf16 is True
    assert cfg.model.mtp_loss_scaling_factor == 0.3
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.hf_model_id == _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    assert cfg.model.hf_model_revision == _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    assert cfg.tokenizer.tokenizer_model == _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}


@pytest.mark.parametrize(
    ("recipe_factory", "base_recipe_factory"),
    _NEMOTRON_3_5_BASE_RECIPE_PAIRS,
    ids=[recipe.__name__ for recipe, _ in _NEMOTRON_3_5_BASE_RECIPE_PAIRS],
)
def test_nemotron_3_5_perf_recipes_inherit_nemotron_3_policy(
    recipe_factory: Callable[[], ConfigContainer],
    base_recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """Nemotron 3.5 variants inherit shared loss normalization and RNG policy."""
    cfg = recipe_factory()
    base_cfg = base_recipe_factory()

    if recipe_factory is not nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config:
        assert cfg.env_vars == base_cfg.env_vars
    assert cfg.model.calculate_per_token_loss == base_cfg.model.calculate_per_token_loss
    assert cfg.model.use_te_rng_tracker == base_cfg.model.use_te_rng_tracker
    assert cfg.tokenizer.tokenizer_model != base_cfg.tokenizer.tokenizer_model


@pytest.mark.parametrize(
    "recipe_factory",
    (
        nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
        nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
    ),
    ids=lambda recipe: recipe.__name__,
)
def test_gb_mxfp8_enables_cutedsl_fusion(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """The non-FSDP Lightning GB recipes enable CutDSL without MoE A2A overlap."""
    cfg = recipe_factory()

    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["CUDNNFE_CLUSTER_OVERLAP_MARGIN"] == 8
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size is None
    assert cfg.model.high_priority_a2a_comm_stream is False
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 108
    assert cfg.mixed_precision.fp8_dot_product_attention is True
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False


@pytest.mark.parametrize("recipe_factory", _GB_FSDP_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gb_mxfp8_fsdp_skips_cutedsl_fusion(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """The Lightning GB MXFP8 FSDP variants remain outside the CutDSL tuning scope."""
    cfg = recipe_factory()

    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars
    assert "CUDNNFE_CLUSTER_OVERLAP_MARGIN" not in cfg.env_vars
    assert getattr(cfg.model, "use_transformer_engine_op_fuser", False) is False
    assert getattr(cfg.model, "moe_mlp_glu_interleave_size", None) is None
    assert getattr(cfg.model, "high_priority_a2a_comm_stream", False) is False
    assert getattr(cfg.model, "moe_hybridep_num_sms_preprocessing", None) != 32
    assert cfg.mixed_precision.fp8_dot_product_attention is False
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False


@pytest.mark.parametrize("recipe_factory", _H100_RECIPES, ids=lambda recipe: recipe.__name__)
def test_h100_perf_recipe_topology(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """H100 Nemotron 3.5 Lightning variants retain the established performance topology."""
    cfg = recipe_factory()

    assert cfg.model.expert_model_parallel_size == 8
    expected_global_batch_size = (
        512 if recipe_factory is nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config else 1024
    )
    assert cfg.train.global_batch_size == expected_global_batch_size
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0


def test_h100_bf16_perf_recipe_uses_measured_benchmark_tuning() -> None:
    """The H100 BF16 benchmark preserves the measured memory and HybridEP tuning."""
    cfg = nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config()

    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16

    assert cfg.model.recompute_modules == ["moe_act", "layernorm"]
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["expert_fc1"]
    assert cfg.model.activation_offload_fraction == 1.0
    assert cfg.model.delay_offload_until_cuda_graph is True

    assert cfg.model.moe_router_fusion is True
    assert cfg.model.moe_permute_fusion_into_hybridep is True
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 64
    assert cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 10
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1
    assert cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 10


def test_h100_fp8_perf_recipe_retains_fp32_optimizer_state() -> None:
    """The unrelated H100 FP8 benchmark retains its existing optimizer and dispatcher policy."""
    cfg = nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config()

    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.model.moe_hybridep_num_sms == 16
    assert cfg.model.moe_flex_dispatcher_num_sms is None


def test_bf16_perf_recipes_share_training_workload() -> None:
    """H100 and GB200 BF16 recipes otherwise share the same model workload."""
    h100_cfg = nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config()
    gb200_cfg = nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config()

    gb200_cfg.train.micro_batch_size = h100_cfg.train.micro_batch_size
    gb200_cfg.model.recompute_granularity = h100_cfg.model.recompute_granularity
    gb200_cfg.model.recompute_modules = h100_cfg.model.recompute_modules
    gb200_cfg.model.cuda_graph_impl = h100_cfg.model.cuda_graph_impl
    gb200_cfg.model.cuda_graph_scope = h100_cfg.model.cuda_graph_scope
    gb200_cfg.model.cuda_graph_modules = h100_cfg.model.cuda_graph_modules
    gb200_cfg.model.fine_grained_activation_offloading = h100_cfg.model.fine_grained_activation_offloading
    gb200_cfg.model.offload_modules = h100_cfg.model.offload_modules
    gb200_cfg.model.activation_offload_fraction = h100_cfg.model.activation_offload_fraction
    gb200_cfg.model.delay_offload_until_cuda_graph = h100_cfg.model.delay_offload_until_cuda_graph
    gb200_cfg.model.moe_router_fusion = h100_cfg.model.moe_router_fusion
    gb200_cfg.model.moe_permute_fusion_into_hybridep = h100_cfg.model.moe_permute_fusion_into_hybridep
    gb200_cfg.model.moe_hybridep_num_sms = h100_cfg.model.moe_hybridep_num_sms
    gb200_cfg.model.moe_flex_dispatcher_num_sms = h100_cfg.model.moe_flex_dispatcher_num_sms
    gb200_cfg.optimizer.use_precision_aware_optimizer = h100_cfg.optimizer.use_precision_aware_optimizer
    gb200_cfg.optimizer.exp_avg_dtype = h100_cfg.optimizer.exp_avg_dtype
    gb200_cfg.optimizer.exp_avg_sq_dtype = h100_cfg.optimizer.exp_avg_sq_dtype
    gb200_cfg.env_vars = h100_cfg.env_vars

    assert gb200_cfg == h100_cfg


@pytest.mark.parametrize("recipe_factory", _GB200_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gb200_perf_recipe_topology(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """GB200 Nemotron 3.5 Lightning variants retain the established performance topology."""
    cfg = recipe_factory()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 2
    assert cfg.model.recompute_granularity is None
    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.model.moe_hybridep_num_sms == 16
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


@pytest.mark.parametrize("recipe_factory", _GB300_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gb300_perf_recipe_topology(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """GB300 Nemotron 3.5 Lightning variants retain the established performance topology."""
    cfg = recipe_factory()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 4
    assert cfg.model.recompute_granularity is None
    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.model.moe_hybridep_num_sms == 16
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


@pytest.mark.parametrize("recipe_factory", _GB_FSDP_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gb_fsdp_perf_recipe_defaults(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """The GB FSDP variants retain their measured 8-GPU performance settings."""
    cfg = recipe_factory()

    assert cfg.train.global_batch_size == 384
    assert cfg.train.micro_batch_size == 3
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope is None
    assert cfg.model.cuda_graph_modules == []
    assert cfg.model.init_model_with_meta_device is True

    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is False
    assert cfg.dist.use_megatron_fsdp is True
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.num_distributed_optimizer_instances == 1
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.outer_dp_sharding_strategy == "no_shard"
    assert cfg.ddp.average_in_collective is False
    assert cfg.ddp.keep_fp8_transpose_cache is False
    assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag is False
    assert cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag is False
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"
