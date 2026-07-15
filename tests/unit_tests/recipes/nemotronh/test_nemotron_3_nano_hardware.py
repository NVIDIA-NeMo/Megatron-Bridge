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

"""Hardware parity and convergence-safety tests for Nemotron 3 Nano recipes."""

from collections.abc import Callable

import pytest
import torch

import megatron.bridge.recipes as recipes
from megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh import (
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config as gb200_perf_config,
)
from megatron.bridge.perf_recipes.nemotronh.h100.nemotronh import (
    nemotron_3_nano_pretrain_16gpu_h100_bf16_config as h100_perf_config,
)
from megatron.bridge.recipes.nemotronh.gb200.nemotron_3_nano import (
    nemotron_3_nano_peft_8gpu_gb200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_sft_8gpu_gb200_bf16_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano import (
    nemotron_3_nano_peft_8gpu_h100_bf16_config,
    nemotron_3_nano_pretrain_8gpu_h100_bf16_config,
    nemotron_3_nano_sft_8gpu_h100_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


_APPROVED_PERF_FIELDS = (
    "model.pipeline_model_parallel_size",
    "model.virtual_pipeline_model_parallel_size",
    "model.context_parallel_size",
    "model.expert_tensor_parallel_size",
    "model.expert_model_parallel_size",
    "model.moe_token_dispatcher_type",
    "model.moe_flex_dispatcher_backend",
    "model.moe_shared_expert_overlap",
    "model.moe_permute_fusion_into_hybridep",
    "model.moe_hybridep_num_blocks_permute",
    "model.moe_hybridep_num_blocks_unpermute",
    "model.moe_hybridep_num_sms_preprocessing",
    "model.transformer_impl",
    "model.attention_backend",
    "model.moe_router_fusion",
    "model.moe_permute_fusion",
    "model.moe_grouped_gemm",
    "model.masked_softmax_fusion",
    "model.use_fused_weighted_squared_relu",
    "model.fine_grained_activation_offloading",
    "model.offload_modules",
    "ddp.overlap_grad_reduce",
    "ddp.overlap_param_gather",
    "ddp.use_distributed_optimizer",
    "comm_overlap.overlap_moe_expert_parallel_comm",
    "comm_overlap.delay_wgrad_compute",
    "comm_overlap.tp_comm_bootstrap_backend",
    "comm_overlap.overlap_p2p_comm",
    "comm_overlap.batch_p2p_comm",
)


def _get_field(config: ConfigContainer, path: str):
    value = config
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _flex_dispatcher_num_sms(config: ConfigContainer) -> int | None:
    return config.model.moe_flex_dispatcher_num_sms or config.model.moe_hybridep_num_sms


def _assert_convergence_sensitive_model_contract(config: ConfigContainer) -> None:
    """Assert routing, capacity, objective, initialization, and dropout invariants."""
    assert config.model.moe_router_force_load_balancing is False
    assert config.model.moe_router_force_biased is None
    assert config.model.moe_router_load_balancing_type == "seq_aux_loss"
    assert config.model.moe_aux_loss_coeff == 0.0001
    assert config.model.moe_z_loss_coeff is None
    assert config.model.moe_input_jitter_eps is None
    assert config.model.moe_router_topk == 6
    assert config.model.moe_router_topk_scaling_factor == 2.5
    assert config.model.moe_router_num_groups == 1
    assert config.model.moe_router_group_topk == 1
    assert config.model.moe_router_score_function == "sigmoid"
    assert config.model.moe_router_pre_softmax is False
    assert config.model.moe_router_topk_limited_devices is None
    assert config.model.moe_router_dtype == "fp32"
    assert config.model.moe_router_enable_expert_bias is True
    assert config.model.moe_router_bias_update_rate == 1e-3
    assert config.model.moe_enable_routing_replay is False
    assert config.model.moe_apply_probs_on_input is False
    assert config.model.moe_shared_expert_gate is False

    assert config.model.moe_token_dropping is False
    assert config.model.moe_expert_capacity_factor is None
    assert config.model.moe_expert_rank_capacity_factor is None
    assert config.model.moe_pad_expert_input_to_capacity is False

    assert config.model.init_method_std == 0.0173
    assert config.model.hidden_dropout == 0.0
    assert config.model.attention_dropout == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("library_factory", "perf_factory"),
    [
        (nemotron_3_nano_pretrain_8gpu_h100_bf16_config, h100_perf_config),
        (nemotron_3_nano_pretrain_8gpu_gb200_bf16_config, gb200_perf_config),
    ],
    ids=["h100", "gb200"],
)
def test_pretrain_approved_fields_match_perf_reference(
    library_factory: Callable[[], ConfigContainer],
    perf_factory: Callable[[], ConfigContainer],
) -> None:
    """Approved execution fields should stay aligned with the perf references."""
    library_config = library_factory()
    perf_config = perf_factory()

    for field in _APPROVED_PERF_FIELDS:
        assert _get_field(library_config, field) == _get_field(perf_config, field), field
    assert _flex_dispatcher_num_sms(library_config) == _flex_dispatcher_num_sms(perf_config) == 16


@pytest.mark.unit
def test_h100_pretrain_uses_8gpu_memory_execution_config() -> None:
    """The 8-GPU library needs TP and memory-safe overlap beyond its perf reference."""
    library_config = nemotron_3_nano_pretrain_8gpu_h100_bf16_config()
    perf_config = h100_perf_config()

    assert perf_config.model.tensor_model_parallel_size == 1
    assert perf_config.model.sequence_parallel is False
    assert perf_config.model.recompute_granularity == "selective"
    assert perf_config.model.recompute_modules == ["moe", "layernorm"]
    assert library_config.model.tensor_model_parallel_size == 8
    assert library_config.model.sequence_parallel is True
    assert library_config.model.recompute_granularity == perf_config.model.recompute_granularity == "selective"
    assert library_config.model.recompute_method is perf_config.model.recompute_method is None
    assert library_config.model.recompute_num_layers is perf_config.model.recompute_num_layers is None
    assert library_config.model.recompute_modules == perf_config.model.recompute_modules == ["moe", "layernorm"]
    assert library_config.model.cross_entropy_loss_fusion is False
    assert perf_config.model.cross_entropy_loss_fusion is True
    assert library_config.model.cross_entropy_fusion_impl == "native"
    assert library_config.train.empty_unused_memory_level == 2
    assert perf_config.train.empty_unused_memory_level == 0
    assert library_config.validation.eval_micro_batch_size == 1
    assert perf_config.validation.eval_micro_batch_size is None
    assert library_config.validation.eval_global_batch_size is perf_config.validation.eval_global_batch_size is None
    assert library_config.optimizer.optimizer_cpu_offload is perf_config.optimizer.optimizer_cpu_offload is False
    assert library_config.comm_overlap.tp_comm_overlap is False
    assert perf_config.comm_overlap.tp_comm_overlap is True
    assert perf_config.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(perf_config.model) == ["attn", "mamba"]
    assert perf_config.model.use_te_rng_tracker is True
    assert perf_config.rng.te_rng_tracker is True
    assert library_config.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(library_config.model) == []
    assert library_config.model.use_te_rng_tracker is False
    assert library_config.rng.te_rng_tracker is False


@pytest.mark.unit
def test_gb200_pretrain_uses_memory_safe_execution_config() -> None:
    """The 8-GPU GB200 library retains topology but excludes graph memory."""
    library_config = nemotron_3_nano_pretrain_8gpu_gb200_bf16_config()
    perf_config = gb200_perf_config()

    assert library_config.model.tensor_model_parallel_size == perf_config.model.tensor_model_parallel_size == 1
    assert library_config.model.sequence_parallel is perf_config.model.sequence_parallel is False
    assert library_config.model.recompute_granularity is perf_config.model.recompute_granularity is None
    assert library_config.model.recompute_method is perf_config.model.recompute_method is None
    assert library_config.model.recompute_num_layers is perf_config.model.recompute_num_layers is None
    assert library_config.model.recompute_modules == perf_config.model.recompute_modules is None
    assert library_config.model.cross_entropy_loss_fusion is perf_config.model.cross_entropy_loss_fusion is True
    assert library_config.model.cross_entropy_fusion_impl == "native"
    assert library_config.train.empty_unused_memory_level is perf_config.train.empty_unused_memory_level == 0
    assert library_config.validation.eval_micro_batch_size is perf_config.validation.eval_micro_batch_size is None
    assert library_config.optimizer.optimizer_cpu_offload is False
    assert library_config.comm_overlap.tp_comm_overlap is False
    assert perf_config.comm_overlap.tp_comm_overlap is True
    assert perf_config.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(perf_config.model) == [
        "attn",
        "mamba",
        "moe_router",
        "moe_preprocess",
    ]
    assert perf_config.model.use_te_rng_tracker is True
    assert perf_config.rng.te_rng_tracker is True
    assert perf_config.model.cuda_graph_warmup_steps == 3
    assert library_config.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(library_config.model) == []
    assert library_config.model.use_te_rng_tracker is False
    assert library_config.rng.te_rng_tracker is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("library_factory", "perf_factory"),
    [
        (nemotron_3_nano_pretrain_8gpu_h100_bf16_config, h100_perf_config),
        (nemotron_3_nano_pretrain_8gpu_gb200_bf16_config, gb200_perf_config),
    ],
    ids=["h100", "gb200"],
)
def test_pretrain_excludes_benchmark_and_convergence_sensitive_overrides(
    library_factory: Callable[[], ConfigContainer],
    perf_factory: Callable[[], ConfigContainer],
) -> None:
    """Library recipes must not inherit benchmark-only or convergence-sensitive settings."""
    library_config = library_factory()
    perf_config = perf_factory()

    _assert_convergence_sensitive_model_contract(library_config)
    assert perf_config.model.moe_router_force_load_balancing is True
    assert library_config.model.moe_router_padding_for_fp8 is False
    assert library_config.model.calculate_per_token_loss is False

    assert library_config.train.train_iters == 39735
    assert library_config.train.global_batch_size == 3072
    assert library_config.train.micro_batch_size == 2
    assert perf_config.train.train_iters == 50

    assert library_config.optimizer.lr == 1.6e-3
    assert library_config.optimizer.min_lr == 1.6e-5
    assert library_config.scheduler.lr_warmup_iters == 333
    assert library_config.optimizer.main_grads_dtype == torch.float32
    assert library_config.optimizer.main_params_dtype == torch.float32
    assert library_config.optimizer.exp_avg_dtype == torch.float32
    assert library_config.optimizer.exp_avg_sq_dtype == torch.float32
    assert library_config.mixed_precision == "bf16_mixed"
    assert perf_config.mixed_precision.bf16 is True
    assert perf_config.mixed_precision.fp8 is None
    assert library_config.ddp.grad_reduce_in_fp32 is True
    assert perf_config.ddp.grad_reduce_in_fp32 is False

    assert library_config.model.apply_rope_fusion is False
    assert library_config.model.cross_entropy_fusion_impl == "native"
    assert perf_config.model.apply_rope_fusion is True
    assert perf_config.model.cross_entropy_fusion_impl == "te"

    assert library_config.ddp.check_for_nan_in_grad is True
    assert library_config.rerun_state_machine.check_for_nan_in_loss is True
    assert perf_config.ddp.check_for_nan_in_grad is False
    assert perf_config.rerun_state_machine.check_for_nan_in_loss is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("recipe_factory", "tp", "sequence_parallel", "dispatcher", "backend", "dispatcher_sms", "learning_rate"),
    [
        (nemotron_3_nano_sft_8gpu_h100_bf16_config, 4, True, "flex", "deepep", 16, 5e-6),
        (nemotron_3_nano_peft_8gpu_h100_bf16_config, 1, False, "flex", "deepep", 16, 1e-4),
        (nemotron_3_nano_sft_8gpu_gb200_bf16_config, 1, False, "alltoall", None, None, 5e-6),
        (nemotron_3_nano_peft_8gpu_gb200_bf16_config, 1, False, "alltoall", None, None, 1e-4),
    ],
)
def test_finetune_recipes_retain_safe_execution_defaults(
    recipe_factory: Callable[[], ConfigContainer],
    tp: int,
    sequence_parallel: bool,
    dispatcher: str,
    backend: str | None,
    dispatcher_sms: int | None,
    learning_rate: float,
) -> None:
    """Packed finetuning uses supported dispatch and keeps graphs/recompute disabled."""
    config = recipe_factory()

    assert config.model.tensor_model_parallel_size == tp
    assert config.model.pipeline_model_parallel_size == 1
    assert config.model.virtual_pipeline_model_parallel_size is None
    assert config.model.context_parallel_size == 1
    assert config.model.sequence_parallel is sequence_parallel
    assert config.model.expert_tensor_parallel_size == 1
    assert config.model.expert_model_parallel_size == 8

    assert config.model.moe_token_dispatcher_type == dispatcher
    assert config.model.moe_flex_dispatcher_backend == backend
    assert config.model.moe_flex_dispatcher_num_sms == dispatcher_sms
    assert config.model.moe_hybridep_num_sms is None
    assert config.model.moe_shared_expert_overlap is False
    _assert_convergence_sensitive_model_contract(config)

    assert config.model.cuda_graph_impl == "none"
    assert config.model.recompute_granularity is None
    assert config.model.recompute_modules is None
    assert config.dataset.enable_offline_packing is True
    assert config.dataset.seq_length == 2048
    assert config.model.seq_length == 2048
    assert config.mixed_precision == "bf16_mixed"
    assert config.model.calculate_per_token_loss is True

    assert config.train.train_iters == 1000
    assert config.train.global_batch_size == 128
    assert config.train.micro_batch_size == 1
    assert config.optimizer.lr == learning_rate
    assert config.optimizer.min_lr == 0.0
    assert config.scheduler.lr_warmup_iters == 50
    assert config.optimizer.optimizer_cpu_offload is False

    assert config.ddp.overlap_grad_reduce is True
    assert config.ddp.overlap_param_gather is True
    assert config.ddp.use_distributed_optimizer is True
    assert config.comm_overlap is None


@pytest.mark.unit
def test_gb200_recipes_are_exported_for_discovery() -> None:
    """Top-level recipe discovery should expose every GB200 variant."""
    recipe_names = (
        "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config",
        "nemotron_3_nano_sft_8gpu_gb200_bf16_config",
        "nemotron_3_nano_peft_8gpu_gb200_bf16_config",
    )

    for recipe_name in recipe_names:
        assert callable(getattr(recipes, recipe_name))
