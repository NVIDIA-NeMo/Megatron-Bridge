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
# - Parametrize over all exported NemotronH recipe functions in `megatron.bridge.recipes.nemotronh`.
# - Build each config without I/O and assert its batch, sequence, parallelism, tokenizer, and task contracts.
# - Verify stale CLI override fields fail without creating phantom config attributes.
#

import importlib
from typing import Callable

import pytest

from megatron.bridge.training.utils.omegaconf_utils import OverridesError, process_config_with_overrides
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_nemotronh_module = importlib.import_module("megatron.bridge.recipes.nemotronh")
_NEMOTRONH_RECIPE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in getattr(_nemotronh_module, "__all__", [])
    if callable(getattr(_nemotronh_module, name, None)) and not name.startswith("nemotronh_")
]


class _FakeModelProvider:
    """Lightweight mutable provider for recipe construction without HF Hub access."""

    def __init__(self) -> None:
        self.vocab_size = 256

    def finalize(self) -> None:
        return None


class _FakeAutoBridge:
    """Return a local model provider without loading a Hugging Face config."""

    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        return cls()

    def to_megatron_provider(self, *args, **kwargs):
        return _FakeModelProvider()


@pytest.fixture(autouse=True)
def _patch_hf_backed_recipe_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep AutoBridge-backed recipe construction deterministic and offline."""
    for module_name in (
        "megatron.bridge.recipes.nemotronh.gb200.nemotron_3_nano",
        "megatron.bridge.recipes.nemotronh.nemotron_3_super",
        "megatron.bridge.recipes.nemotronh.nemotron_3_ultra",
    ):
        module = importlib.import_module(module_name)
        patch_recipe_module_global(monkeypatch, module, "AutoBridge", _FakeAutoBridge)


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.validation is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.ddp is not None
    assert cfg.mixed_precision is not None

    assert cfg.train.micro_batch_size >= 1
    assert cfg.train.global_batch_size >= cfg.train.micro_batch_size
    assert cfg.train.global_batch_size % cfg.train.micro_batch_size == 0

    assert 1 <= cfg.dataset.seq_length <= cfg.model.seq_length

    assert cfg.model.tensor_model_parallel_size >= 1
    assert cfg.model.pipeline_model_parallel_size >= 1
    assert cfg.model.context_parallel_size >= 1
    assert cfg.model.expert_model_parallel_size >= 1


@pytest.mark.parametrize("recipe_func", _NEMOTRONH_RECIPE_FUNCS)
def test_each_nemotronh_recipe_builds_config(recipe_func: Callable):
    """Test that each NemotronH recipe builds a valid config."""
    # All configs use parameterless API (peft configs have optional peft_scheme)
    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    is_sft = "sft" in recipe_func.__name__.lower()
    is_peft = "peft" in recipe_func.__name__.lower()
    is_finetune = is_sft or is_peft

    if is_finetune:
        # Finetuning recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    if is_peft:
        assert cfg.peft is not None
    else:
        assert cfg.peft is None


def test_nemotronh_recipe_rejects_unknown_cli_override():
    """A stale recipe override must fail instead of creating a phantom field."""
    cfg = _nemotronh_module.nemotron_3_nano_pretrain_config()

    with pytest.raises(OverridesError, match="Failed to parse Hydra overrides"):
        process_config_with_overrides(cfg, cli_overrides=["model.not_a_real_field=true"])

    assert not hasattr(cfg.model, "not_a_real_field")


def test_nemotron_3_nano_gb200_defers_vocab_size_to_training_tokenizer():
    """The GB200 pretraining model vocabulary must follow its runtime tokenizer."""
    cfg = _nemotronh_module.nemotron_3_nano_pretrain_8gpu_gb200_bf16_config()

    assert cfg.model.vocab_size is None


def test_nemotron_nano_9b_v2_lora_defaults():
    """Test that Nemotron Nano 9B v2 LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_peft_config

    cfg = nemotron_nano_9b_v2_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, Nemotron Nano 9B v2 should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotron_nano_9b_v2_full_sft_defaults():
    """Test that Nemotron Nano 9B v2 full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_sft_config

    cfg = nemotron_nano_9b_v2_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, Nemotron Nano 9B v2 should use TP=2, PP=1
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None


def test_nemotron_nano_12b_v2_lora_defaults():
    """Test that Nemotron Nano 12B v2 LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_12b_v2_peft_config

    cfg = nemotron_nano_12b_v2_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, Nemotron Nano 12B v2 should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotron_nano_12b_v2_full_sft_defaults():
    """Test that Nemotron Nano 12B v2 full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_12b_v2_sft_config

    cfg = nemotron_nano_12b_v2_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, Nemotron Nano 12B v2 should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None


# --- Nemotron 3 Super tests ---


def test_nemotron_3_super_pretrain_defaults():
    """Test that Nemotron 3 Super pretrain has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_super_pretrain_config

    cfg = nemotron_3_super_pretrain_config()

    _assert_basic_config(cfg)

    # Pretrain should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.expert_model_parallel_size == 8


def test_nemotron_3_super_peft_lora_defaults():
    """Test that Nemotron 3 Super PEFT with LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_super_peft_config

    cfg = nemotron_3_super_peft_config()

    _assert_basic_config(cfg)

    # For LoRA, should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotron_3_super_sft_defaults():
    """Test that Nemotron 3 Super SFT has correct defaults."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_super_sft_config

    cfg = nemotron_3_super_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, should use TP=1, PP=1, EP=8
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.peft is None
