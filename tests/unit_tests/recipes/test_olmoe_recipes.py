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
# - Parametrize over all exported OLMoE recipe functions in `megatron.bridge.recipes.olmoe`.
# - For each recipe, monkeypatch `OlMoEModelProvider` with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection and sanity-check parallelism fields.
#

import importlib
from collections.abc import Callable

import pytest
import torch


_olmoe_module = importlib.import_module("megatron.bridge.recipes.olmoe")
_OLMOE_RECIPE_FUNCS = [
    getattr(_olmoe_module, name)
    for name in getattr(_olmoe_module, "__all__", [])
    if callable(getattr(_olmoe_module, name, None))
]

# OLMoE finetune-specific tests
_OLMOE_FINETUNE_FUNCS = [
    getattr(_olmoe_module, name)
    for name in ["olmoe_7b_finetune_config"]
    if callable(getattr(_olmoe_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    Pretrain configs use the new parameterless API (return empty dict).
    Finetune configs still accept parameters.
    """
    is_finetune = "finetune" in name.lower()

    if is_finetune:
        # Finetuning-specific overrides - finetune configs still accept parameters
        overrides = {
            "name": f"unit_{name}",
            "dir": ".",
            "train_iters": 10,
            "micro_batch_size": 1,
            "seq_length": 64,
            "min_lr": 1e-5,
            "lr_warmup_iters": 2,
            "tokenizer_path": "allenai/OLMoE-1B-7B-0125",
            "finetune_lr": 1e-4,
            "global_batch_size": 2,
        }
    else:
        # Pretrain configs use the new parameterless API
        overrides = {}

    return overrides


class _FakeOlMoEModelProvider:
    """Fake OlMoEModelProvider for testing without model I/O."""

    def __init__(self, *args, **kwargs):
        # Store all the kwargs that would be passed to the real provider
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set required attributes
        self.vocab_size = 50304  # Default vocab size for OLMoE
        self.moe_permute_fusion = True
        self.apply_rope_fusion = False
        self.pipeline_model_parallel_layout = None
        self.seq_length = 4096

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

    # Check sequence length (different attribute names for different dataset types)
    if hasattr(cfg.dataset, "sequence_length"):
        assert cfg.dataset.sequence_length >= 1  # GPTDatasetConfig
    elif hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1  # FinetuningDatasetConfig / HFDatasetConfig
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _OLMOE_RECIPE_FUNCS)
def test_each_olmoe_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)

    # Monkeypatch the OlMoEModelProvider class
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    is_finetune = "finetune" in recipe_func.__name__.lower()
    if is_finetune:
        # Finetuning recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use NullTokenizer
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Check parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1


@pytest.mark.parametrize("recipe_func", _OLMOE_FINETUNE_FUNCS)
def test_olmoe_finetune_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each OLMoE finetune recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for(recipe_func.__name__)
    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Finetuning always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1


@pytest.mark.parametrize("recipe_func", _OLMOE_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "dora", None])
def test_olmoe_finetune_peft_vs_full_sft(recipe_func: Callable, peft, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT and full SFT configurations are correctly applied."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    if peft in ["lora", "dora"]:
        assert cfg.peft is not None
    elif peft is None:
        assert cfg.peft is None


def test_olmoe_7b_pretrain_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B pretrain has correct default parallelism."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    # Pretrain configs use the new parameterless API
    cfg = olmoe_7b_pretrain_config()

    _assert_basic_config(cfg)

    # For pretrain, OLMoE-7B defaults - check actual default values
    assert cfg.model.tensor_model_parallel_size >= 1
    assert cfg.model.pipeline_model_parallel_size >= 1
    assert cfg.model.expert_model_parallel_size >= 1

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 5

    # Check NullTokenizer for pretraining
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.tokenizer.vocab_size == cfg.model.vocab_size


def test_olmoe_7b_finetune_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for("olmoe_7b_finetune_config")
    overrides["peft"] = "lora"

    cfg = olmoe_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, OLMoE-7B should use TP=1, PP=1, EP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 5


def test_olmoe_7b_finetune_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B DoRA has correct default parallelism."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for("olmoe_7b_finetune_config")
    overrides["peft"] = "dora"

    cfg = olmoe_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For DoRA, OLMoE-7B should use TP=1, PP=1, EP=1 (same as LoRA)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 5


def test_olmoe_7b_finetune_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for("olmoe_7b_finetune_config")
    overrides["peft"] = None

    cfg = olmoe_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, OLMoE-7B should use TP=1, PP=1, EP=8
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.sequence_parallel is False

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 5


def test_olmoe_7b_finetune_precision_aware_optimizer(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B finetune uses precision-aware optimizer settings."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for("olmoe_7b_finetune_config")
    cfg = olmoe_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check precision-aware optimizer settings
    assert cfg.optimizer.use_precision_aware_optimizer is True

    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16


def test_olmoe_7b_finetune_tokenizer_with_trust_remote_code(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B finetune uses HF tokenizer with trust_remote_code."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for("olmoe_7b_finetune_config")
    cfg = olmoe_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check tokenizer settings
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == "allenai/OLMoE-1B-7B-0125"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"trust_remote_code": True}


def test_olmoe_7b_pretrain_optimizer_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B pretrain has correct optimizer settings."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    # Pretrain configs use the new parameterless API
    cfg = olmoe_7b_pretrain_config()

    _assert_basic_config(cfg)

    # Check optimizer is using precision-aware settings
    assert cfg.optimizer.use_precision_aware_optimizer is True

    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16


def test_olmoe_7b_pretrain_mixed_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B pretrain has correct mixed precision settings."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    # Pretrain configs use the new parameterless API
    cfg = olmoe_7b_pretrain_config()

    _assert_basic_config(cfg)

    # Check mixed precision config
    assert cfg.mixed_precision is not None
    assert cfg.mixed_precision.bf16 is True

    assert cfg.mixed_precision.params_dtype == torch.bfloat16
    assert cfg.mixed_precision.pipeline_dtype == torch.bfloat16
    assert cfg.mixed_precision.autocast_enabled is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False


def test_olmoe_7b_finetune_mixed_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B finetune has correct mixed precision settings."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    overrides = _safe_overrides_for("olmoe_7b_finetune_config")
    cfg = olmoe_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check mixed precision config
    assert cfg.mixed_precision is not None
    assert cfg.mixed_precision.bf16 is True

    assert cfg.mixed_precision.params_dtype == torch.bfloat16
    assert cfg.mixed_precision.pipeline_dtype == torch.bfloat16
    assert cfg.mixed_precision.autocast_enabled is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False


def test_olmoe_7b_moe_optimizations_enabled(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B has MoE optimizations enabled."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    # Pretrain configs use the new parameterless API
    cfg = olmoe_7b_pretrain_config()

    _assert_basic_config(cfg)

    # Check MoE optimizations
    assert cfg.model.moe_permute_fusion is True


def test_olmoe_7b_comm_overlap_config(monkeypatch: pytest.MonkeyPatch):
    """Test that OLMoE-7B has comm overlap config set."""
    from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.olmoe.olmoe_7b")
    monkeypatch.setattr(mod, "OlMoEModelProvider", _FakeOlMoEModelProvider)

    # Pretrain configs use the new parameterless API
    cfg = olmoe_7b_pretrain_config()

    _assert_basic_config(cfg)

    # Check comm overlap config
    assert cfg.comm_overlap is not None
    assert cfg.comm_overlap.tp_comm_overlap is False
