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
# - Parametrize over all exported Qwen recipe functions in `megatron.bridge.recipes.qwen`.
# - For each recipe, monkeypatch `AutoBridge` with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection: pretrain recipes honor `use_null_tokenizer`, sft/peft recipes always use HF tokenizer.
# - Sanity-check parallelism fields and finetuning-specific requirements.
#

import importlib
from typing import Callable

import pytest
import torch

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_qwen_module = importlib.import_module("megatron.bridge.recipes.qwen")
_QWEN_RECIPE_FUNCS = [
    getattr(_qwen_module, name)
    for name in getattr(_qwen_module, "__all__", [])
    if callable(getattr(_qwen_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    All configs (pretrain, sft, peft) use the new parameterless API.
    For peft configs, only peft_scheme can be passed as a parameter.
    """
    # All configs now use parameterless API (or peft_scheme only for peft)
    return {}


class _FakeModelCfg:
    # Minimal provider to accept attribute assignments used in recipes

    def __init__(self):
        self.cross_entropy_fusion_impl = "native"
        self.context_parallel_size = 1

    def finalize(self):
        # qwen3 recipe may call finalize(); make it a no-op
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        expected_revisions = {
            "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",  # pragma: allowlist secret
            "Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",  # pragma: allowlist secret
        }
        if hf_path in expected_revisions:
            assert kwargs == {"revision": expected_revisions[hf_path]}
        return _FakeBridge()

    @staticmethod
    def from_hf_config(hf_config):
        # Ignore hf_config; return a bridge that yields a fake provider
        return _FakeBridge()


class _FakeTextConfig:
    architectures = None


class _FakeRootConfig:
    text_config = _FakeTextConfig()


class _FakeAutoConfig:
    @staticmethod
    def from_pretrained(hf_path: str, **kwargs):
        expected_revisions = {
            "Qwen/Qwen3.5-35B-A3B-Base": "0f0813072d2358973511097385626f21fcb6d422",  # pragma: allowlist secret
        }
        assert kwargs == ({"revision": expected_revisions[hf_path]} if hf_path in expected_revisions else {})
        return _FakeRootConfig()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    # Required top-level sections
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    # A few critical fields
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1

    if hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _QWEN_RECIPE_FUNCS)
def test_each_qwen_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    # Always patch AutoBridge in qwen3_moe (where base configs actually call it)
    qwen3_moe_mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, qwen3_moe_mod, "AutoBridge", _FakeBridge)
    # Also patch in the recipe function's own module if it directly references AutoBridge
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    if hasattr(mod, "AutoBridge"):
        patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)
    if hasattr(mod, "AutoConfig"):
        patch_recipe_module_global(monkeypatch, mod, "AutoConfig", _FakeAutoConfig)

    overrides = _safe_overrides_for(recipe_func.__name__)

    # qwen3_next PEFT is intentionally not implemented.
    if recipe_func.__name__ in {
        "qwen3_next_80b_a3b_peft_config",
        "qwen3_next_80b_a3b_peft_1gpu_h100_bf16_config",
    }:
        with pytest.raises(NotImplementedError):
            recipe_func(**overrides)
        return

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer is properly configured
    recipe_name = recipe_func.__name__.lower()
    is_sft_or_peft = "sft" in recipe_name or "peft" in recipe_name
    if is_sft_or_peft:
        # SFT and PEFT recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        # depending on the model (qwen2/qwen25 use NullTokenizer, qwen3 uses HuggingFaceTokenizer)
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    # Parallelism and shaping
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    if (
        "qwen3" in recipe_name
        and "pretrain" in recipe_name
        and "next" not in recipe_name
        and "qwen35" not in recipe_name
    ):
        assert cfg.model.cross_entropy_fusion_impl == "te"

    # SFT and PEFT-specific assertions
    if is_sft_or_peft:
        # New parameterless API - pretrained_checkpoint is set by user after config creation
        # Just verify the checkpoint config exists
        assert cfg.checkpoint is not None
        # Should have PEFT config (or None if SFT)
        assert hasattr(cfg, "peft")  # peft field should exist
        # Dataset should be configured (SQuAD by default)
        assert cfg.dataset is not None


def _patch_qwen3_dense_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    qwen3_mod = importlib.import_module("megatron.bridge.recipes.qwen.h100.qwen3")
    patch_recipe_module_global(monkeypatch, qwen3_mod, "AutoBridge", _FakeBridge)


def _assert_qwen_finetune_optimizer_contract(cfg, *, expected_lr: float) -> None:
    assert cfg.optimizer.optimizer == "adam"
    assert cfg.optimizer.lr == expected_lr
    assert cfg.optimizer.min_lr == 0.0
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.95
    assert cfg.optimizer.adam_eps == 1.0e-8
    assert cfg.optimizer.clip_grad == 1.0
    assert cfg.scheduler.start_weight_decay == 0.033
    assert cfg.scheduler.end_weight_decay == 0.033
    assert cfg.scheduler.weight_decay_incr_style == "constant"
    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.scheduler.lr_warmup_init == 0.0
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.optimizer.use_distributed_optimizer is True
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.params_dtype == torch.bfloat16
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_qwen3_8b_pretrain_convergence_contract(monkeypatch: pytest.MonkeyPatch):
    """The generic 8B pretrain recipe should select the 16-GPU convergence cohort."""
    from megatron.bridge.recipes.qwen import qwen3_8b_pretrain_config
    from megatron.bridge.recipes.qwen.h100.qwen3 import qwen3_8b_pretrain_16gpu_h100_bf16_config

    _patch_qwen3_dense_bridge(monkeypatch)

    assert qwen3_8b_pretrain_config is qwen3_8b_pretrain_16gpu_h100_bf16_config
    cfg = qwen3_8b_pretrain_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.seq_length == 4096
    assert cfg.dataset.seq_length == 4096
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.random_seed == 1234
    assert cfg.rng.seed == 1234
    assert cfg.optimizer.optimizer == "adam"
    assert cfg.optimizer.lr == 3.0e-4
    assert cfg.optimizer.min_lr == 3.0e-5
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.95
    assert cfg.optimizer.adam_eps == 1.0e-8
    assert cfg.optimizer.clip_grad == 1.0
    assert cfg.scheduler.start_weight_decay == 0.033
    assert cfg.scheduler.end_weight_decay == 0.033
    assert cfg.scheduler.weight_decay_incr_style == "constant"
    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.scheduler.lr_warmup_init == 0.0
    assert cfg.scheduler.lr_warmup_iters == 40
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.optimizer.use_distributed_optimizer is True
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.params_dtype == torch.bfloat16
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.checkpoint.save_interval == 50
    assert cfg.checkpoint.load is None
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret
    }
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    process_config_with_overrides(
        cfg.tokenizer,
        cli_overrides=[
            '++hf_tokenizer_kwargs.revision="b968826d9c46dd6066d109eabc6255188de91218"'  # pragma: allowlist secret
        ],
    )


def test_qwen3_8b_sft_convergence_contract(monkeypatch: pytest.MonkeyPatch):
    """The bounded 8B SFT recipe should own the shared finetuning contract."""
    from megatron.bridge.recipes.qwen import qwen3_8b_sft_config

    _patch_qwen3_dense_bridge(monkeypatch)
    cfg = qwen3_8b_sft_config()

    _assert_basic_config(cfg)
    assert cfg.model.seq_length == 2048
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seed == 1234
    assert cfg.rng.seed == 5678
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 1
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.load is None
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret
    }
    _assert_qwen_finetune_optimizer_contract(cfg, expected_lr=5.0e-6)


def test_qwen3_8b_peft_convergence_contract(monkeypatch: pytest.MonkeyPatch):
    """The bounded 8B PEFT recipe should own the optimizer and LoRA contracts."""
    from megatron.bridge.recipes.qwen import qwen3_8b_peft_config

    _patch_qwen3_dense_bridge(monkeypatch)
    cfg = qwen3_8b_peft_config()

    _assert_basic_config(cfg)
    assert cfg.model.seq_length == 2048
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seed == 1234
    assert cfg.rng.seed == 5678
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.load is None
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret
    }
    _assert_qwen_finetune_optimizer_contract(cfg, expected_lr=1.0e-4)
    assert cfg.peft is not None
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.dropout == 0.0


def test_qwen3_8b_32k_sft_preserves_separate_batch_contract(monkeypatch: pytest.MonkeyPatch):
    """The 32K SFT cohort should not inherit the bounded 2K recipe's GBS=32."""
    from megatron.bridge.recipes.qwen import qwen3_8b_sft_32k_config
    from megatron.bridge.recipes.qwen.h100.qwen3 import qwen3_8b_sft_8gpu_h100_bf16_32k_config

    _patch_qwen3_dense_bridge(monkeypatch)

    assert qwen3_8b_sft_32k_config is qwen3_8b_sft_8gpu_h100_bf16_32k_config
    cfg = qwen3_8b_sft_32k_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.sequence_parallel is True
    assert cfg.model.cp_comm_type == "a2a"
    assert cfg.model.seq_length == 32768
    assert cfg.dataset.seq_length == 32768
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 32768
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 8
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.cross_entropy_loss_fusion is False
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False


# Qwen3 MoE SFT and PEFT-specific tests
_QWEN3_MOE_SFT_FUNCS = [
    getattr(_qwen_module, name)
    for name in [
        "qwen3_30b_a3b_sft_config",
        "qwen3_235b_a22b_sft_config",
    ]
    if callable(getattr(_qwen_module, name, None))
]

_QWEN3_MOE_PEFT_FUNCS = [
    getattr(_qwen_module, name)
    for name in [
        "qwen3_30b_a3b_peft_config",
        "qwen3_235b_a22b_peft_config",
    ]
    if callable(getattr(_qwen_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _QWEN3_MOE_SFT_FUNCS)
def test_qwen3_moe_sft_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that full SFT configurations are correctly applied for Qwen3 MoE models."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Full SFT should not have PEFT config
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _QWEN3_MOE_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_qwen3_moe_peft_config(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configurations are correctly applied for Qwen3 MoE models."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)

    # PEFT config should be present
    assert cfg.peft is not None


def test_qwen3_30b_a3b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, 30B-A3B should use TP=4, PP=1, EP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.dropout == 0.0
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]
    assert cfg.model.seq_length == 2048
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seed == 1234
    assert cfg.rng.seed == 5678
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.load is None
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"  # pragma: allowlist secret
    }
    _assert_qwen_finetune_optimizer_contract(cfg, expected_lr=1.0e-4)


def test_qwen3_30b_a3b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B DoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, 30B-A3B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]


def test_qwen3_30b_a3b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that generic 30B-A3B full SFT uses the verified 16-GPU config."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.get_data_parallel_size(16) == 16
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * cfg.get_data_parallel_size(16)) == 2
    assert cfg.model.seq_length == 2048
    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.moe_flex_dispatcher_backend is None
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.model.moe_a2a_overlap is False
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.bias_activation_fusion is True
    assert cfg.model.apply_rope_fusion is True
    assert cfg.model.moe_router_fusion is True
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.comm_overlap is None
    assert cfg.train.train_iters == 100
    assert cfg.dataset.seed == 1234
    assert cfg.rng.seed == 5678
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 1
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.load is None
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"  # pragma: allowlist secret
    }
    assert cfg.peft is None
    _assert_qwen_finetune_optimizer_contract(cfg, expected_lr=5.0e-6)


def test_qwen3_30b_a3b_legacy_8gpu_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Keep the explicit 8-GPU SFT topology available for existing callers."""
    from megatron.bridge.recipes.qwen.h100.qwen3_moe import qwen3_30b_a3b_sft_8gpu_h100_bf16_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.h100.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_sft_8gpu_h100_bf16_config()

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    assert cfg.model.moe_flex_dispatcher_backend == "deepep"
    assert cfg.peft is None
    assert cfg.model.seq_length == 2048
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seed == 1234
    assert cfg.rng.seed == 5678
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.load is None
    _assert_qwen_finetune_optimizer_contract(cfg, expected_lr=5.0e-6)


def test_qwen3_30b_a3b_sft_generic_alias_uses_16gpu_factory():
    """Keep the generic SFT alias attached to the verified 16-GPU factory."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_sft_config
    from megatron.bridge.recipes.qwen.h100.qwen3_moe import qwen3_30b_a3b_sft_16gpu_h100_bf16_config

    assert qwen3_30b_a3b_sft_config is qwen3_30b_a3b_sft_16gpu_h100_bf16_config


def test_qwen3_30b_a3b_pretrain_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that the generic 30B-A3B pretrain recipe uses the verified 16-GPU topology."""
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_pretrain_config()

    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.random_seed == 1234
    assert cfg.rng.seed == 1234
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.moe_hybridep_num_sms == 32
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["moe_router", "moe_preprocess"]
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.optimizer == "adam"
    assert cfg.optimizer.lr == 3.0e-4
    assert cfg.optimizer.min_lr == 3.0e-5
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.95
    assert cfg.optimizer.adam_eps == 1.0e-8
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.optimizer.clip_grad == 1.0
    assert cfg.scheduler.start_weight_decay == 0.033
    assert cfg.scheduler.end_weight_decay == 0.033
    assert cfg.scheduler.weight_decay_incr_style == "constant"
    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.scheduler.lr_warmup_init == 0.0
    assert cfg.scheduler.lr_warmup_iters == 40
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.optimizer.use_distributed_optimizer is True
    assert cfg.mixed_precision.bf16 is True
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"  # pragma: allowlist secret
    }
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    process_config_with_overrides(
        cfg.tokenizer,
        cli_overrides=[
            '++hf_tokenizer_kwargs.revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"'  # pragma: allowlist secret
        ],
    )
    assert cfg.mixed_precision.params_dtype == torch.bfloat16
    assert cfg.checkpoint.save_interval == 50
    assert cfg.checkpoint.load is None
    assert cfg.comm_overlap.tp_comm_overlap is True


def test_qwen3_30b_a3b_bf16_perf_recipe_uses_default_functional_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the H100 BF16 perf recipe only adds benchmark-specific overrides."""
    from megatron.bridge.perf_recipes.qwen.h100.qwen3_moe import (
        qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config,
    )
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    default_cfg = qwen3_30b_a3b_pretrain_config()
    perf_cfg = qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config()

    assert perf_cfg.model.tensor_model_parallel_size == default_cfg.model.tensor_model_parallel_size
    assert perf_cfg.model.pipeline_model_parallel_size == default_cfg.model.pipeline_model_parallel_size
    assert perf_cfg.model.expert_model_parallel_size == default_cfg.model.expert_model_parallel_size
    assert perf_cfg.model.sequence_parallel == default_cfg.model.sequence_parallel
    assert perf_cfg.train.global_batch_size == default_cfg.train.global_batch_size
    assert perf_cfg.train.micro_batch_size == default_cfg.train.micro_batch_size
    assert perf_cfg.model.moe_flex_dispatcher_backend == default_cfg.model.moe_flex_dispatcher_backend
    assert perf_cfg.model.moe_token_dispatcher_type == default_cfg.model.moe_token_dispatcher_type
    assert perf_cfg.model.cuda_graph_impl == default_cfg.model.cuda_graph_impl
    assert perf_cfg.model.cuda_graph_scope == default_cfg.model.cuda_graph_scope
    assert perf_cfg.comm_overlap.tp_comm_overlap == default_cfg.comm_overlap.tp_comm_overlap
    assert perf_cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert default_cfg.comm_overlap.overlap_moe_expert_parallel_comm is None
    assert perf_cfg.comm_overlap.delay_wgrad_compute is False
    assert perf_cfg.optimizer.use_precision_aware_optimizer == default_cfg.optimizer.use_precision_aware_optimizer
    assert perf_cfg.model.moe_router_force_load_balancing is True
    assert default_cfg.model.moe_router_force_load_balancing is False


def test_qwen3_30b_a3b_perf_base_remains_legacy_8gpu_recipe():
    """Keep non-H100-BF16 perf recipes isolated from the new generic default."""
    from megatron.bridge.perf_recipes.qwen.common import qwen3_30b_a3b_pretrain_config as perf_base
    from megatron.bridge.recipes.qwen.h100.qwen3_moe import (
        qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config as legacy_base,
    )

    assert perf_base is legacy_base


def test_qwen35_h100_perf_recipe_requires_flash_qla_mcore_fields():
    """Test that the recipe selects FlashQLA and rejects an incompatible MCore."""
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.qwen.h100.qwen35 import (
        _configure_flash_qla_model_fields,
    )

    new_mcore_model = SimpleNamespace(
        gdn_pre_gated_delta_rule_fusion=False,
        gated_delta_rule_backend="fla",
    )
    _configure_flash_qla_model_fields(new_mcore_model)
    assert new_mcore_model.gdn_pre_gated_delta_rule_fusion is True
    assert new_mcore_model.gated_delta_rule_backend == "flash_qla"

    with pytest.raises(RuntimeError, match="requires the pinned MCore GDN performance fields"):
        _configure_flash_qla_model_fields(SimpleNamespace())


def test_qwen35_text_35b_a3b_h100_bf16_perf_recipe(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test the text-only Qwen3.5 H100 benchmark topology and performance features."""
    from megatron.bridge.perf_recipes.qwen.h100.qwen35 import (
        qwen35_text_35b_a3b_pretrain_16gpu_h100_bf16_config,
    )
    from megatron.bridge.perf_recipes.qwen.h100.qwen35_runtime import (
        qwen35_h100_transformer_block_spec,
    )
    from megatron.bridge.utils.cuda_graph import cuda_graph_module_names

    mod = importlib.import_module("megatron.bridge.recipes.qwen.gb200.qwen35")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)
    patch_recipe_module_global(monkeypatch, mod, "AutoConfig", _FakeAutoConfig)

    cfg = qwen35_text_35b_a3b_pretrain_16gpu_h100_bf16_config()

    assert cfg.tokenizer.tokenizer_model == "Qwen/Qwen3.5-35B-A3B"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {
        "revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",  # pragma: allowlist secret
    }
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules == []
    assert cfg.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(cfg.model) == []
    assert cfg.model.transformer_layer_spec is qwen35_h100_transformer_block_spec
    assert cfg.model.gdn_pre_gated_delta_rule_fusion is True
    assert cfg.model.gated_delta_rule_backend == "flash_qla"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 16
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 108
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_shared_expert_overlap is True
    assert cfg.model.moe_expert_rank_capacity_factor == 1.05
    assert cfg.model.moe_permute_fusion_into_hybridep is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.mixed_precision.bf16 is True
    assert cfg.train.train_iters == 50
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API"] == 64
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API"] == 64
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 64
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8


def test_qwen35_h100_perf_spec_replaces_grouped_expert_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the H100 block factory replaces the measured expert and GDN runtimes."""
    from functools import partial
    from types import SimpleNamespace

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    expert_builder = partial(TEGroupedMLP)
    moe_builder = partial(
        MoELayer,
        submodules=MoESubmodules(experts=expert_builder),
    )
    gdn_spec = ModuleSpec(module=GatedDeltaNet)
    layer_specs = [
        SimpleNamespace(submodules=SimpleNamespace(mlp=moe_builder, self_attention=gdn_spec)),
        SimpleNamespace(submodules=SimpleNamespace(mlp=moe_builder, self_attention=gdn_spec)),
    ]
    block_spec = SimpleNamespace(layer_specs=layer_specs)
    monkeypatch.setattr(
        qwen35_runtime,
        "get_transformer_block_with_experimental_attention_variant_spec",
        lambda config, vp_stage=None: block_spec,
    )

    result = qwen35_runtime.qwen35_h100_transformer_block_spec(object())

    assert result is block_spec
    first_custom_moe = result.layer_specs[0].submodules.mlp
    second_custom_moe = result.layer_specs[1].submodules.mlp
    assert first_custom_moe is second_custom_moe
    assert first_custom_moe.func is qwen35_runtime._Qwen35H100MoELayer
    assert first_custom_moe.keywords["submodules"].experts.func is qwen35_runtime._Qwen35H100TorchGroupedMLP
    assert result.layer_specs[0].submodules.self_attention.module is qwen35_runtime._Qwen35H100FlashQLAGatedDeltaNet
    assert result.layer_specs[1].submodules.self_attention.module is qwen35_runtime._Qwen35H100FlashQLAGatedDeltaNet


def test_qwen35_h100_flash_qla_runtime_requires_exact_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the measured GDN runtime rejects a different FlashQLA version."""
    import sys
    from types import ModuleType

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    flash_qla = ModuleType("flash_qla")
    flash_qla.__version__ = "0.1.1"
    flash_qla.chunk_gated_delta_rule = object()
    monkeypatch.setitem(sys.modules, "flash_qla", flash_qla)

    with pytest.raises(ImportError, match=r"requires flash_qla==0\.1\.2; found 0\.1\.1"):
        qwen35_runtime._load_flash_qla_gated_delta_rule()


def test_qwen35_h100_flash_qla_runtime_loads_pinned_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the measured GDN runtime accepts the pinned FlashQLA version."""
    import sys
    from types import ModuleType

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    gated_delta_rule = object()
    flash_qla = ModuleType("flash_qla")
    flash_qla.__version__ = "0.1.2"
    flash_qla.chunk_gated_delta_rule = gated_delta_rule
    monkeypatch.setitem(sys.modules, "flash_qla", flash_qla)

    assert qwen35_runtime._load_flash_qla_gated_delta_rule() is gated_delta_rule


def test_qwen35_h100_flash_qla_runtime_configures_pinned_mcore(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that pinned MCore receives the raw FlashQLA kernel."""
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    gated_delta_rule = object()
    module = SimpleNamespace(config=SimpleNamespace(), gated_delta_rule=object())
    monkeypatch.setattr(qwen35_runtime, "_load_flash_qla_gated_delta_rule", lambda: gated_delta_rule)

    qwen35_runtime._configure_flash_qla_gated_delta_rule(module)

    assert module.gated_delta_rule is gated_delta_rule


def test_qwen35_h100_flash_qla_runtime_preserves_new_mcore_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that newer MCore keeps its keyword-compatible backend adapter."""
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    native_adapter = object()
    module = SimpleNamespace(
        config=SimpleNamespace(gated_delta_rule_backend="flash_qla"),
        gated_delta_rule=native_adapter,
    )
    monkeypatch.setattr(qwen35_runtime, "_load_flash_qla_gated_delta_rule", lambda: object())

    qwen35_runtime._configure_flash_qla_gated_delta_rule(module)

    assert module.gated_delta_rule is native_adapter


def test_qwen35_h100_flash_qla_runtime_rejects_new_mcore_fla_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that newer MCore cannot silently retain a different GDN backend."""
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    module = SimpleNamespace(
        config=SimpleNamespace(gated_delta_rule_backend="fla"),
        gated_delta_rule=object(),
    )
    monkeypatch.setattr(qwen35_runtime, "_load_flash_qla_gated_delta_rule", lambda: object())

    with pytest.raises(RuntimeError, match="requires gated_delta_rule_backend='flash_qla'"):
        qwen35_runtime._configure_flash_qla_gated_delta_rule(module)


def test_qwen35_h100_fused_gated_rms_norm_requires_exact_fla_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the measured fused norm rejects a different FLA version."""
    import sys
    from types import ModuleType

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    fused_norm_gate = ModuleType("fla.modules.fused_norm_gate")
    fused_norm_gate.rms_norm_gated = object()
    monkeypatch.setitem(sys.modules, "fla.modules.fused_norm_gate", fused_norm_gate)
    monkeypatch.setattr(qwen35_runtime.metadata, "version", lambda package: "0.4.1")

    with pytest.raises(ImportError, match=r"requires flash-linear-attention==0\.4\.2; found 0\.4\.1"):
        qwen35_runtime._load_fused_gated_rms_norm()


def test_qwen35_h100_fused_gated_rms_norm_loads_pinned_fla_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the measured fused norm accepts the pinned FLA version."""
    import sys
    from types import ModuleType

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    fused_gated_rms_norm = object()
    fused_norm_gate = ModuleType("fla.modules.fused_norm_gate")
    fused_norm_gate.rms_norm_gated = fused_gated_rms_norm
    monkeypatch.setitem(sys.modules, "fla.modules.fused_norm_gate", fused_norm_gate)
    monkeypatch.setattr(qwen35_runtime.metadata, "version", lambda package: "0.4.2")

    assert qwen35_runtime._load_fused_gated_rms_norm() is fused_gated_rms_norm


def test_qwen35_h100_fused_gated_rms_norm_applies_measured_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test the fused norm's version, shape, epsilon, activation, and gamma contract."""
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    kernel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    result = torch.empty(2, 3)

    def fused_gated_rms_norm(*args: object, **kwargs: object) -> torch.Tensor:
        kernel_calls.append((args, kwargs))
        return result

    monkeypatch.setattr(
        qwen35_runtime,
        "_load_fused_gated_rms_norm",
        lambda: fused_gated_rms_norm,
    )
    weight = torch.tensor([1.0, 2.0, 3.0])
    module = SimpleNamespace(
        activation="silu",
        config=SimpleNamespace(
            normalization="RMSNorm",
            layernorm_epsilon=1.0e-5,
            layernorm_zero_centered_gamma=False,
        ),
        out_norm=SimpleNamespace(
            weight=weight,
            bias=None,
            eps=1.0e-6,
            zero_centered_gamma=True,
        ),
    )

    qwen35_runtime._configure_fused_gated_rms_norm(module)
    x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    gate = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    output = qwen35_runtime._apply_fused_gated_rms_norm(module, x, gate)

    assert output is result
    assert len(kernel_calls) == 1
    args, kwargs = kernel_calls[0]
    torch.testing.assert_close(args[0], x.reshape(2, 3))
    torch.testing.assert_close(args[1], gate.reshape(2, 3))
    torch.testing.assert_close(args[2], weight + 1.0)
    assert args[3] is None
    assert kwargs == {"activation": "swish", "eps": 1.0e-6}


@pytest.mark.parametrize(
    ("activation", "normalization", "weight", "bias", "match"),
    [
        ("gelu", "RMSNorm", torch.ones(3), None, "requires SiLU activation"),
        ("silu", "LayerNorm", torch.ones(3), None, "requires RMSNorm"),
        ("silu", "RMSNorm", None, None, "requires a norm weight and no bias"),
        ("silu", "RMSNorm", torch.ones(3), torch.zeros(3), "requires a norm weight and no bias"),
    ],
)
def test_qwen35_h100_fused_gated_rms_norm_rejects_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
    normalization: str,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    match: str,
):
    """Test that the fused norm fails closed outside the measured contract."""
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    monkeypatch.setattr(qwen35_runtime, "_load_fused_gated_rms_norm", lambda: object())
    module = SimpleNamespace(
        activation=activation,
        config=SimpleNamespace(normalization=normalization),
        out_norm=SimpleNamespace(weight=weight, bias=bias),
    )

    with pytest.raises(RuntimeError, match=match):
        qwen35_runtime._configure_fused_gated_rms_norm(module)


@pytest.mark.parametrize(
    ("supports_clamp_value", "expected_argument_count"),
    [(False, 3), (True, 4)],
)
def test_qwen35_h100_weighted_swiglu_runtime_supports_mcore_api_versions(
    monkeypatch: pytest.MonkeyPatch,
    supports_clamp_value: bool,
    expected_argument_count: int,
):
    """Test pinned and newer MCore weighted-SwiGLU call signatures."""
    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    result = object()
    applied_args: tuple[object, ...] = ()

    def fake_apply(*args: object) -> object:
        nonlocal applied_args
        applied_args = args
        return result

    monkeypatch.setattr(
        qwen35_runtime,
        "_WEIGHTED_SWIGLU_FORWARD_HAS_CLAMP_VALUE",
        supports_clamp_value,
    )
    monkeypatch.setattr(qwen35_runtime.WeightedSwiGLUFunction, "apply", fake_apply)

    input_tensor = object()
    token_weights = object()
    output = qwen35_runtime._apply_weighted_swiglu(
        input_tensor,
        token_weights,
        fp8_input_store=False,
        clamp_value=None,
    )

    assert output is result
    assert len(applied_args) == expected_argument_count
    assert applied_args[:2] == (input_tensor, token_weights)
    assert applied_args[2:] == ((False, None) if supports_clamp_value else (False,))


def test_qwen35_h100_static_hybridep_metadata_uses_bf16_alignment():
    """Test the static H100 rank budget without the SM100 op-fuser alignment."""
    from types import SimpleNamespace

    from megatron.bridge.perf_recipes.qwen.h100 import qwen35_runtime

    manager = SimpleNamespace(
        config=SimpleNamespace(
            fp8=None,
            fp4=None,
            moe_router_topk=8,
        ),
        drop_and_pad=False,
        moe_expert_rank_capacity_factor=1.05,
        num_experts=16,
    )
    routing_map = torch.zeros((3, 2, 8), dtype=torch.bool)
    probs = torch.zeros((3, 2, 8), dtype=torch.float32)

    qwen35_runtime._setup_h100_static_hybridep_metadata(
        manager,
        routing_map,
        probs,
    )

    assert manager.routing_map.shape == (3, 16)
    assert manager.token_probs.shape == (3, 16)
    assert manager.num_permuted_tokens == 32


def test_qwen3_30b_a3b_h100_fp8_perf_recipe_keeps_cuda_graphs_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that changing the generic default does not alter the H100 FP8 recipe."""
    from megatron.bridge.perf_recipes.qwen.h100.qwen3_moe import (
        qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config,
    )

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config()

    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == []


def test_qwen3_235b_a22b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_235b_a22b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_235b_a22b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, 235B-A22B should use TP=4, PP=4, EP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]


def test_qwen3_235b_a22b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B DoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_235b_a22b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_235b_a22b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, 235B-A22B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]


def test_qwen3_235b_a22b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen3_235b_a22b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen3_moe")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = qwen3_235b_a22b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 235B-A22B should use TP=4, PP=16, EP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 16
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True

    assert cfg.peft is None
