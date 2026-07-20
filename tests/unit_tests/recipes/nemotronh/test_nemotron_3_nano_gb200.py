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

import pytest
import torch

import megatron.bridge.recipes as recipes
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.recipes.nemotronh import (
    nemotron_3_nano_gb200_pretrain_config,
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
)
from megatron.bridge.training import config as training_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


def test_nemotron_3_nano_gb200_pretrain_config() -> None:
    """The release backport should expose the validated GB200 execution contract."""
    config = nemotron_3_nano_pretrain_8gpu_gb200_bf16_config()

    assert isinstance(config, ConfigContainer)
    assert isinstance(config.model, HybridModelProvider)

    assert config.model.seq_length == 4096
    assert config.dataset.seq_length == 4096
    assert config.model.tensor_model_parallel_size == 1
    assert config.model.pipeline_model_parallel_size == 1
    assert config.model.virtual_pipeline_model_parallel_size is None
    assert config.model.context_parallel_size == 1
    assert config.model.sequence_parallel is False
    assert config.model.expert_tensor_parallel_size == 1
    assert config.model.expert_model_parallel_size == 8

    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_flex_dispatcher_num_sms == 16
    assert config.model.moe_hybridep_num_sms is None
    assert config.model.moe_shared_expert_overlap is False
    assert config.model.moe_router_force_load_balancing is False

    assert config.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(config.model) == ["attn", "mamba", "moe_router", "moe_preprocess"]
    assert config.model.cuda_graph_warmup_steps == 3
    assert config.model.use_te_rng_tracker is True
    assert config.rng.te_rng_tracker is True
    assert config.model.apply_rope_fusion is True
    assert config.model.cross_entropy_fusion_impl == "native"
    assert config.rerun_state_machine.check_for_nan_in_loss is False
    assert config.ddp.check_for_nan_in_grad is False
    assert isinstance(config.mixed_precision, MixedPrecisionConfig)
    assert config.mixed_precision.bf16 is True
    assert config.mixed_precision.grad_reduce_in_fp32 is False
    assert config.ddp.grad_reduce_in_fp32 is False
    assert config.comm_overlap is not None
    assert config.comm_overlap.tp_comm_overlap is False


def test_nemotron_3_nano_gb200_retains_training_contract() -> None:
    """The GB200 wrapper must not import convergence-sensitive perf settings."""
    config = nemotron_3_nano_pretrain_8gpu_gb200_bf16_config()

    assert config.train.train_iters == 39735
    assert config.train.global_batch_size == 3072
    assert config.train.micro_batch_size == 2
    assert config.optimizer.lr == 1.6e-3
    assert config.optimizer.min_lr == 1.6e-5
    assert config.scheduler.lr_warmup_iters == 333
    assert config.optimizer.main_grads_dtype == torch.float32
    assert config.optimizer.main_params_dtype == torch.float32
    assert config.optimizer.exp_avg_dtype == torch.float32
    assert config.optimizer.exp_avg_sq_dtype == torch.float32
    assert isinstance(config.mixed_precision, MixedPrecisionConfig)
    assert config.mixed_precision.bf16 is True
    assert config.mixed_precision.fp8 is None
    assert config.mixed_precision.grad_reduce_in_fp32 is False

    assert config.model.moe_router_load_balancing_type == "seq_aux_loss"
    assert config.model.moe_aux_loss_coeff == 0.0001
    assert config.model.moe_router_topk == 6
    assert config.model.moe_router_force_load_balancing is False
    assert config.model.moe_router_padding_for_fp8 is False
    assert config.model.cross_entropy_loss_fusion is True
    assert config.model.cross_entropy_fusion_impl == "native"
    assert config.model.apply_rope_fusion is True
    assert config.rerun_state_machine.check_for_nan_in_loss is False
    assert config.ddp.check_for_nan_in_grad is False
    assert config.ddp.grad_reduce_in_fp32 is False
    assert config.ddp.use_distributed_optimizer is True


def test_nemotron_3_nano_gb200_aliases_are_discoverable() -> None:
    """Direct users and NeMo-CI should both resolve the GB200 recipe."""
    assert nemotron_3_nano_gb200_pretrain_config is nemotron_3_nano_pretrain_8gpu_gb200_bf16_config
    assert recipes.nemotron_3_nano_gb200_pretrain_config is nemotron_3_nano_pretrain_8gpu_gb200_bf16_config
    assert recipes.nemotron_3_nano_pretrain_8gpu_gb200_bf16_config is (nemotron_3_nano_pretrain_8gpu_gb200_bf16_config)


def test_nemotron_3_nano_gb200_validates_with_nemo_ci_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 64-GPU runtime update should preserve topology and precision contracts."""
    config = nemotron_3_nano_gb200_pretrain_config()
    config.train.train_iters = 48000
    config.train.global_batch_size = 512
    config.train.micro_batch_size = 2

    monkeypatch.setattr(training_config, "get_world_size_safe", lambda: 64)
    # The real release run performs this capability check on GB200. Unit
    # finalization only verifies the combined configuration contract.
    monkeypatch.setattr(training_config, "validate_flex_dispatcher_backend", lambda _model: None)

    training_config.runtime_config_update(config)

    assert config.data_parallel_size == 64
    assert config.train.train_iters == 48000
    assert config.train.global_batch_size == 512
    assert config.train.micro_batch_size == 2
    assert config.model.seq_length == 4096
    assert config.dataset.seq_length == 4096
    assert config.validation.eval_global_batch_size == 512
    assert config.validation.eval_micro_batch_size == 2
    assert config.model.bf16 is True
    assert config.model.params_dtype == torch.bfloat16
    assert config.ddp.grad_reduce_in_fp32 is False
    assert config.optimizer.main_grads_dtype == torch.float32
    assert config.optimizer.main_params_dtype == torch.float32
    assert config.optimizer.exp_avg_dtype == torch.float32
    assert config.optimizer.exp_avg_sq_dtype == torch.float32
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_flex_dispatcher_num_sms == 16
    assert config.model.moe_hybridep_num_sms is None
    assert config.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(config.model) == ["attn", "mamba", "moe_router", "moe_preprocess"]
