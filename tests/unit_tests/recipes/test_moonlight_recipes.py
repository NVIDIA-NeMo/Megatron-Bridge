# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#
# Test purpose:
# - Parametrize over all exported Moonlight recipe functions in `megatron.bridge.recipes.moonlight`.
# - For each recipe, monkeypatch `AutoBridge` with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection and sanity-check parallelism fields.
#

import importlib
from typing import Callable

import pytest
import torch

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_moonlight_module = importlib.import_module("megatron.bridge.recipes.moonlight")
_MOONLIGHT_RECIPE_FUNCS = [
    getattr(_moonlight_module, name)
    for name in getattr(_moonlight_module, "__all__", [])
    if callable(getattr(_moonlight_module, name, None))
]

# Moonlight SFT-specific tests
_MOONLIGHT_SFT_FUNCS = [
    getattr(_moonlight_module, name)
    for name in ["moonlight_16b_sft_config", "moonlight_16b_sft_8k_config"]
    if callable(getattr(_moonlight_module, name, None))
]

# Moonlight PEFT-specific tests
_MOONLIGHT_PEFT_FUNCS = [
    getattr(_moonlight_module, name)
    for name in ["moonlight_16b_peft_config"]
    if callable(getattr(_moonlight_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    All configs now use the new parameterless API (return empty dict).
    """
    return {}


def _apply_test_overrides(cfg, name: str):
    """Apply test-friendly overrides to a config after creation."""
    # Apply common test overrides
    cfg.train.train_iters = 10
    cfg.train.micro_batch_size = 1
    cfg.dataset.seq_length = 64
    cfg.optimizer.min_lr = 1e-5
    cfg.scheduler.lr_warmup_iters = 2
    cfg.optimizer.lr = 1e-4
    cfg.logger.name = f"unit_{name}"
    cfg.logger.dir = "."
    cfg.train.global_batch_size = 2
    cfg.tokenizer.tokenizer_model = "moonshotai/Moonlight-16B-A3B"

    return cfg


class _FakeMoonlightModelProvider:
    """Fake Moonlight model provider for testing without model I/O."""

    def __init__(self, *args, **kwargs):
        # Store all the kwargs that would be passed to the real provider
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set required attributes
        self.vocab_size = 163840
        self.kv_channels = 128
        self.multi_latent_attention = True
        self.q_lora_rank = None
        self.num_moe_experts = 64
        self.moe_router_topk = 6
        self.moe_router_num_groups = 1
        self.moe_router_group_topk = 1
        self.moe_router_topk_scaling_factor = 2.446
        self.moe_aux_loss_coeff = 0.001
        self.moe_router_pre_softmax = True
        self.moe_router_load_balancing_type = "seq_aux_loss"
        self.moe_router_score_function = "sigmoid"
        self.moe_router_enable_expert_bias = True
        self.moe_router_bias_update_rate = 1e-3
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False
        self.num_layers_in_first_pipeline_stage = None
        self.num_layers_in_last_pipeline_stage = None
        self.moe_permute_fusion = True
        self.apply_rope_fusion = False
        self.pipeline_model_parallel_layout = None
        self.moe_token_dispatcher_type = "alltoall"
        self.moe_enable_deepep = False
        self.moe_shared_expert_overlap = True

        # Set parallelism defaults if not provided
        if not hasattr(self, "tensor_model_parallel_size"):
            self.tensor_model_parallel_size = 1
        if not hasattr(self, "pipeline_model_parallel_size"):
            self.pipeline_model_parallel_size = 1
        if not hasattr(self, "context_parallel_size"):
            self.context_parallel_size = 1
        if not hasattr(self, "expert_model_parallel_size"):
            self.expert_model_parallel_size = 1

    def finalize(self):
        return None


class _FakeBridge:
    """Return the checkpoint-compatible fake provider without model I/O."""

    @classmethod
    def from_hf_pretrained(cls, model_id: str, **kwargs) -> "_FakeBridge":
        assert model_id == "moonshotai/Moonlight-16B-A3B"
        assert kwargs == {
            "revision": "476b36a473d4467f94469414bef6cee75c9c8172"  # pragma: allowlist secret
        }
        return cls()

    def to_megatron_provider(self, *, load_weights: bool) -> _FakeMoonlightModelProvider:
        assert load_weights is False
        return _FakeMoonlightModelProvider()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1
    assert cfg.dataset.seq_length >= 1


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_RECIPE_FUNCS)
def test_each_moonlight_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)

    # Keep every recipe construction offline.
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    func_name = recipe_func.__name__
    is_peft = "peft" in func_name.lower()
    is_sft = "sft" in func_name.lower()

    # New API: SFT configs are parameterless, PEFT has optional peft_scheme
    if is_peft:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    is_sft_or_peft = is_sft or is_peft
    if is_sft_or_peft:
        # SFT/PEFT recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use NullTokenizer
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Check parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_SFT_FUNCS)
def test_moonlight_sft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Moonlight SFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func()
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # SFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1

    # SFT should not have PEFT config
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_PEFT_FUNCS)
def test_moonlight_peft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Moonlight PEFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func(peft_scheme="lora")
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # PEFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1

    # PEFT should have PEFT config
    assert cfg.peft is not None


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_moonlight_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configurations are correctly applied with different schemes."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func(peft_scheme=peft_scheme)
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    assert cfg.peft is not None


def _assert_moonlight_router_identity(cfg, *, aux_loss_coeff: float):
    """Assert that cohort alignment did not replace Moonlight routing semantics."""
    assert cfg.model.num_moe_experts == 64
    assert cfg.model.moe_router_topk == 6
    assert cfg.model.moe_router_num_groups == 1
    assert cfg.model.moe_router_group_topk == 1
    assert cfg.model.moe_router_topk_scaling_factor == 2.446
    assert cfg.model.moe_aux_loss_coeff == aux_loss_coeff
    assert cfg.model.moe_router_pre_softmax is True
    assert cfg.model.moe_router_load_balancing_type == "seq_aux_loss"
    assert cfg.model.moe_router_score_function == "sigmoid"
    assert cfg.model.moe_router_enable_expert_bias is True
    assert cfg.model.moe_router_bias_update_rate == 1e-3
    assert cfg.model.moe_router_force_load_balancing is False


def _assert_finetuning_optimizer_contract(cfg, *, precision_aware: bool):
    """Assert the shared SFT/PEFT optimizer and arithmetic contract."""
    assert cfg.optimizer.optimizer == "adam"
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.95
    assert cfg.optimizer.adam_eps == 1e-8
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.optimizer.clip_grad == 1.0
    assert cfg.optimizer.use_distributed_optimizer is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    if precision_aware:
        assert cfg.optimizer.use_precision_aware_optimizer is True
        assert cfg.optimizer.main_grads_dtype == torch.bfloat16
        assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
        assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    else:
        assert cfg.optimizer.use_precision_aware_optimizer is False
        assert cfg.optimizer.main_grads_dtype == torch.float32
        assert cfg.optimizer.exp_avg_dtype == torch.float32
        assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.scheduler.start_weight_decay == 0.033
    assert cfg.scheduler.end_weight_decay == 0.033
    assert cfg.scheduler.weight_decay_incr_style == "constant"
    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.scheduler.lr_warmup_init == 0.0
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.params_dtype == torch.bfloat16
    assert cfg.mixed_precision.pipeline_dtype == torch.bfloat16
    reduce_in_fp32 = not precision_aware
    assert cfg.mixed_precision.grad_reduce_in_fp32 is reduce_in_fp32
    assert cfg.ddp.grad_reduce_in_fp32 is reduce_in_fp32


def test_moonlight_16b_pretrain_convergence_contract(monkeypatch: pytest.MonkeyPatch):
    """Test the exact 16-GPU pretrain convergence contract."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_pretrain_config

    patch_recipe_module_global(monkeypatch, moonlight_16b_pretrain_config, "AutoBridge", _FakeBridge)
    cfg = moonlight_16b_pretrain_config()

    assert moonlight_16b_pretrain_config.__name__ == "moonlight_16b_pretrain_16gpu_h100_bf16_config"
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.seq_length == 4096
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 2
    assert cfg.train.global_batch_size // 16 == 64
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * 16) == 32
    assert cfg.dataset.random_seed == 1234
    assert cfg.rng.seed == 1234
    assert cfg.optimizer.optimizer == "adam"
    assert cfg.optimizer.lr == 3e-4
    assert cfg.optimizer.min_lr == 3e-5
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.95
    assert cfg.optimizer.adam_eps == 1e-8
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.optimizer.clip_grad == 1.0
    assert cfg.optimizer.use_distributed_optimizer is True
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.scheduler.lr_warmup_iters == 40
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.scheduler.lr_warmup_init == 0.0
    assert cfg.scheduler.start_weight_decay == 0.033
    assert cfg.scheduler.end_weight_decay == 0.033
    assert cfg.scheduler.weight_decay_incr_style == "constant"
    assert cfg.checkpoint.save_interval == 50
    assert cfg.checkpoint.load is None
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.params_dtype == torch.bfloat16
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.moe_router_fusion is True
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_deepep_num_sms is None
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.high_priority_a2a_comm_stream is True
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 128
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20
    assert cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
    assert cfg.env_vars["USE_MNNVL"] == 0
    _assert_moonlight_router_identity(cfg, aux_loss_coeff=0.001)


def test_moonlight_16b_sft_convergence_contract(monkeypatch: pytest.MonkeyPatch):
    """Test the exact 8-GPU full-SFT convergence contract."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_sft_config

    patch_recipe_module_global(monkeypatch, moonlight_16b_sft_config, "AutoBridge", _FakeBridge)
    cfg = moonlight_16b_sft_config()

    assert moonlight_16b_sft_config.__name__ == "moonlight_16b_sft_8gpu_h100_bf16_tp1_config"
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.vocab_size == 163842
    assert cfg.get_data_parallel_size(8) == 8
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * cfg.get_data_parallel_size(8)) == 4
    assert cfg.model.num_moe_experts // cfg.model.expert_model_parallel_size == 8
    assert cfg.model.seq_length == 2048
    assert cfg.dataset.seq_length == 2048
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 2048
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 1
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seed == 1234
    assert cfg.rng.seed == 5678
    assert cfg.optimizer.lr == 5e-6
    assert cfg.optimizer.min_lr == 0.0
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.load is None
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == "moonshotai/Moonlight-16B-A3B"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "476b36a473d4467f94469414bef6cee75c9c8172",  # pragma: allowlist secret
        "trust_remote_code": True,
    }
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    process_config_with_overrides(
        cfg.tokenizer,
        cli_overrides=[
            '++hf_tokenizer_kwargs.revision="476b36a473d4467f94469414bef6cee75c9c8172"'  # pragma: allowlist secret
        ],
    )
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_deepep_num_sms is None
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_a2a_overlap is False
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.high_priority_a2a_comm_stream is True
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 128
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20
    assert cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
    assert cfg.env_vars["USE_MNNVL"] == 0
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.moe_router_fusion is True
    _assert_finetuning_optimizer_contract(cfg, precision_aware=True)
    _assert_moonlight_router_identity(cfg, aux_loss_coeff=0.001)


def test_moonlight_16b_legacy_sft_contract_remains_available(monkeypatch: pytest.MonkeyPatch):
    """Keep the existing TP4/PP2 support topology available to callers."""
    from megatron.bridge.recipes.moonlight.h100.moonlight_16b import (
        moonlight_16b_sft_8gpu_h100_bf16_config,
    )

    patch_recipe_module_global(monkeypatch, moonlight_16b_sft_8gpu_h100_bf16_config, "AutoBridge", _FakeBridge)
    cfg = moonlight_16b_sft_8gpu_h100_bf16_config()

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_layout == [
        ["embedding"] + ["decoder"] * 14,
        ["decoder"] * 13 + ["loss"],
    ]
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    assert cfg.model.vocab_size == 163844
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4


def test_moonlight_16b_peft_convergence_contract(monkeypatch: pytest.MonkeyPatch):
    """Test the exact 4-GPU LoRA convergence contract and frozen base."""
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.peft.lora_layers import LinearAdapter
    from megatron.bridge.recipes.moonlight import moonlight_16b_peft_config

    patch_recipe_module_global(monkeypatch, moonlight_16b_peft_config, "AutoBridge", _FakeBridge)
    cfg = moonlight_16b_peft_config(peft_scheme="lora")

    assert moonlight_16b_peft_config.__name__ == "moonlight_16b_peft_4gpu_h100_bf16_config"
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.vocab_size == 163842
    assert cfg.get_data_parallel_size(4) == 4
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * cfg.get_data_parallel_size(4)) == 8
    assert cfg.model.seq_length == 2048
    assert cfg.dataset.seq_length == 2048
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 2048
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seed == 1234
    assert cfg.rng.seed == 5678
    assert cfg.optimizer.lr == 1e-4
    assert cfg.optimizer.min_lr == 0.0
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.load is None
    assert cfg.tokenizer.tokenizer_model == "moonshotai/Moonlight-16B-A3B"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "476b36a473d4467f94469414bef6cee75c9c8172",  # pragma: allowlist secret
        "trust_remote_code": True,
    }
    assert isinstance(cfg.peft, LoRA)
    expected_targets = ["linear_q_proj", "linear_kv_down_proj", "linear_kv_up_proj", "linear_proj"]
    assert cfg.peft.target_modules == expected_targets
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.dropout == 0.0
    base_model = torch.nn.Module()
    for target in expected_targets + ["linear_qkv"]:
        setattr(base_model, target, torch.nn.Linear(2, 2))
    cfg.peft(base_model)
    assert all(isinstance(getattr(base_model, target), LinearAdapter) for target in expected_targets)
    assert isinstance(base_model.linear_qkv, torch.nn.Linear)
    assert all(not parameter.requires_grad for parameter in base_model.linear_qkv.parameters())
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.moe_router_fusion is True
    _assert_finetuning_optimizer_contract(cfg, precision_aware=False)
    _assert_moonlight_router_identity(cfg, aux_loss_coeff=0.001)


def test_moonlight_16b_sft_8k_contract_is_separate(monkeypatch: pytest.MonkeyPatch):
    """Test that 8K/CP2 SFT retains its prior batch, topology, and precision."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_sft_8k_config

    patch_recipe_module_global(monkeypatch, moonlight_16b_sft_8k_config, "AutoBridge", _FakeBridge)
    cfg = moonlight_16b_sft_8k_config()

    assert moonlight_16b_sft_8k_config.__name__ == "moonlight_16b_sft_8gpu_h100_bf16_8k_config"
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.sequence_parallel is True
    assert cfg.model.vocab_size == 163842
    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 8192
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert cfg.train.train_iters == 20
    assert cfg.train.global_batch_size == 128
    assert cfg.train.micro_batch_size == 1
    assert cfg.optimizer.lr == 1e-6
    assert cfg.optimizer.min_lr == 0.0
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.98
    assert cfg.optimizer.adam_eps == 1e-5
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.scheduler.lr_warmup_iters == 2
    assert cfg.scheduler.lr_decay_iters == 20
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.model.cross_entropy_loss_fusion is False
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    _assert_moonlight_router_identity(cfg, aux_loss_coeff=0.001)


def test_moonlight_compatibility_recipes_remain_exported(monkeypatch: pytest.MonkeyPatch):
    """Test that the prior explicit pretrain and PEFT entry points remain available."""
    from megatron.bridge.recipes.moonlight.h100 import (
        moonlight_16b_peft_2gpu_h100_bf16_config,
        moonlight_16b_pretrain_8gpu_h100_bf16_config,
    )

    patch_recipe_module_global(monkeypatch, moonlight_16b_pretrain_8gpu_h100_bf16_config, "AutoBridge", _FakeBridge)
    pretrain_cfg = moonlight_16b_pretrain_8gpu_h100_bf16_config()
    peft_cfg = moonlight_16b_peft_2gpu_h100_bf16_config(peft_scheme="lora")

    assert pretrain_cfg.model.tensor_model_parallel_size == 2
    assert pretrain_cfg.model.pipeline_model_parallel_size == 1
    assert pretrain_cfg.model.expert_model_parallel_size == 8
    assert pretrain_cfg.train.global_batch_size == 2048
    assert pretrain_cfg.optimizer.use_precision_aware_optimizer is True
    assert pretrain_cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert peft_cfg.model.tensor_model_parallel_size == 1
    assert peft_cfg.model.pipeline_model_parallel_size == 1
    assert peft_cfg.model.expert_model_parallel_size == 2
    assert peft_cfg.train.global_batch_size == 128
    assert peft_cfg.optimizer.use_precision_aware_optimizer is True
    assert peft_cfg.optimizer.main_grads_dtype == torch.bfloat16
