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

#
# Test purpose:
# - Cover the previously untested qwen3_vl recipes (issue #3177 sweep).
# - Parametrize over all exported pretrain / SFT / PEFT functions.
# - Monkeypatch AutoBridge and the flex-dispatcher helper to avoid HF Hub
#   I/O and torch.cuda dependencies.
# - Verify each recipe returns a valid ConfigContainer with the expected
#   parallelism, freeze flags, dataset provider, and PEFT branching.
# - Special-case the energon variant by mocking AutoTokenizer +
#   Qwen3VLProcessor so the recipe can build without HF Hub access.
#

import importlib
from typing import Callable
from unittest import mock

import pytest


_qwen3_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen3_vl")


# Recipe groups — each entry is the raw function reference. Tests parametrize
# over these so any new recipe added to qwen3_vl.py picks up coverage just by
# being added to the right list.
_PRETRAIN_FUNCS: list[Callable] = [
    _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config,
    _qwen3_vl_module.qwen3_vl_30b_a3b_pretrain_mock_config,
    _qwen3_vl_module.qwen3_vl_235b_a22b_pretrain_mock_config,
]

_SFT_FUNCS: list[Callable] = [
    _qwen3_vl_module.qwen3_vl_8b_sft_config,
    _qwen3_vl_module.qwen3_vl_30b_a3b_sft_config,
    _qwen3_vl_module.qwen3_vl_235b_a22b_sft_config,
]

_PEFT_FUNCS: list[Callable] = [
    _qwen3_vl_module.qwen3_vl_8b_peft_config,
    _qwen3_vl_module.qwen3_vl_30b_a3b_peft_config,
    _qwen3_vl_module.qwen3_vl_235b_a22b_peft_config,
]


class _FakeQwen3VLProvider:
    """Fake provider returned by AutoBridge.to_megatron_provider.

    Attribute access is permissive — recipes set a long list of attrs on
    the model config (parallelism, freeze flags, MoE knobs, kernel flags),
    and the existing provider instance just absorbs them. Only attrs that
    the recipe READS need explicit defaults here.

    `num_moe_experts` is intentionally left unset so
    `apply_flex_dispatcher_backend` short-circuits before touching
    `torch.cuda.get_device_properties` — see flex_dispatcher_backend.py.
    """

    def __init__(self):
        # Vocab size is not read by qwen3_vl recipes (they use NullTokenizer
        # with DEFAULT_NULL_TOKENIZER_VOCAB_SIZE), but mirror the real
        # provider shape for safety.
        self.vocab_size = 152000
        # Some recipes interrogate apply_rope_fusion; default to False so
        # the experimental dist flag stays at its default.
        self.apply_rope_fusion = False

    def finalize(self):
        return None


class _FakeAutoBridge:
    """AutoBridge stub that bypasses HuggingFace Hub network access."""

    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        return cls()

    def to_megatron_provider(self, *args, **kwargs):
        return _FakeQwen3VLProvider()


@pytest.fixture(autouse=True)
def _patch_recipe_module(monkeypatch):
    """Bypass HF Hub and CUDA for all qwen3_vl recipe tests.

    - `AutoBridge` returns the fake provider above (no HF download).
    - `apply_flex_dispatcher_backend` is replaced with a no-op so the
      MoE recipes don't hit `torch.cuda.get_device_properties` on CPU
      runners. The function would already short-circuit on a fake
      provider (no `num_moe_experts`), but mocking it keeps the test
      independent of that internal early-return.
    """
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)
    monkeypatch.setattr(_qwen3_vl_module, "apply_flex_dispatcher_backend", lambda *a, **kw: None)


def _assert_config_shape(cfg) -> None:
    """Verify the structural contract every recipe must satisfy."""
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1


@pytest.mark.parametrize("recipe_func", _PRETRAIN_FUNCS, ids=lambda f: f.__name__)
def test_pretrain_recipe_builds(recipe_func: Callable):
    """Every pretrain recipe builds without HF Hub or CUDA access."""
    cfg = recipe_func()
    _assert_config_shape(cfg)

    # Pretrain configs use NullTokenizer with the project's default vocab.
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Parallelism fields must be set to positive integers.
    assert cfg.model.tensor_model_parallel_size >= 1
    assert cfg.model.pipeline_model_parallel_size >= 1
    assert cfg.model.context_parallel_size >= 1

    # Pretrain VLM defaults: language + vision frozen, projection trainable.
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False

    # Pretrain configs use mock VLM data by default.
    from megatron.bridge.data.vlm_datasets import MockVLMConversationProvider

    assert isinstance(cfg.dataset, MockVLMConversationProvider)


@pytest.mark.parametrize("recipe_func", _SFT_FUNCS, ids=lambda f: f.__name__)
def test_sft_recipe_builds(recipe_func: Callable):
    """Every SFT recipe builds without HF Hub or CUDA access."""
    cfg = recipe_func()
    _assert_config_shape(cfg)

    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # SFT recipes train all components (no freeze).
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False

    # SFT configs do not attach a PEFT scheme.
    assert cfg.peft is None

    # Sequence length is harmonized between model and dataset.
    assert cfg.dataset.seq_length == cfg.model.seq_length


@pytest.mark.parametrize("recipe_func", _PEFT_FUNCS, ids=lambda f: f.__name__)
def test_peft_recipe_builds_with_default_lora(recipe_func: Callable):
    """Every PEFT recipe builds with the default scheme (lora)."""
    cfg = recipe_func()
    _assert_config_shape(cfg)

    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Default peft_scheme is "lora"; the resulting config exposes dim/alpha.
    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


@pytest.mark.parametrize("recipe_func", _PEFT_FUNCS, ids=lambda f: f.__name__)
def test_peft_recipe_with_dora(recipe_func: Callable):
    """Each PEFT recipe accepts peft_scheme='dora' and attaches a DoRA config."""
    from megatron.bridge.peft.dora import DoRA

    cfg = recipe_func(peft_scheme="dora")
    _assert_config_shape(cfg)

    assert isinstance(cfg.peft, DoRA)


def test_pretrain_30b_a3b_uses_recommended_moe_parallelism():
    """Spot-check the 30B-A3B MoE recipe wires EP/SP correctly."""
    cfg = _qwen3_vl_module.qwen3_vl_30b_a3b_pretrain_mock_config()
    _assert_config_shape(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True


def test_pretrain_235b_a22b_uses_recommended_parallelism():
    """Spot-check the 235B-A22B recipe wires the largest parallel layout."""
    cfg = _qwen3_vl_module.qwen3_vl_235b_a22b_pretrain_mock_config()
    _assert_config_shape(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 16
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.sequence_parallel is True


def test_pretrain_overrides_apply():
    """User overrides flow through `_qwen3_vl_common` to the model config."""
    cfg = _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        sequence_parallel=True,
        seq_length=2048,
    )
    _assert_config_shape(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.sequence_parallel is True
    assert cfg.model.seq_length == 2048
    assert cfg.dataset.seq_length == 2048


def test_pretrain_non_mock_dataset_raises():
    """`mock=False` is intentionally not yet supported; the recipe must raise."""
    with pytest.raises(ValueError, match="Non-mock dataset not yet supported"):
        _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config(mock=False)


def test_sft_8b_default_parallelism():
    """qwen3_vl_8b_sft_config uses TP=2, PP=1 (single-node 8 GPU)."""
    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.sequence_parallel is False


def test_sft_30b_a3b_uses_ep_for_moe():
    """qwen3_vl_30b_a3b_sft_config enables EP for the MoE model."""
    cfg = _qwen3_vl_module.qwen3_vl_30b_a3b_sft_config()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_token_dispatcher_type == "alltoall"


def test_sft_uses_hf_dataset_provider():
    """SFT configs inherit the HF conversation provider from `_sft_common_vlm`."""
    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()
    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_peft_energon_config_builds_with_mocked_processor():
    """The energon PEFT variant builds when AutoTokenizer + Qwen3VLProcessor are mocked.

    The energon config calls into HF directly (`AutoTokenizer.from_pretrained`,
    `Qwen3VLProcessor.from_pretrained`) inside `_make_energon_dataset`. Without
    a mock these would attempt network access. We mock both to return cheap
    sentinels and verify the resulting config still has the energon dataset.
    """
    from megatron.bridge.data.energon.energon_provider import EnergonProvider

    with (
        mock.patch.object(_qwen3_vl_module, "AutoTokenizer") as mock_tok,
        mock.patch.object(_qwen3_vl_module, "Qwen3VLProcessor") as mock_proc,
    ):
        mock_tok.from_pretrained.return_value = mock.MagicMock(name="tokenizer")
        mock_proc.from_pretrained.return_value = mock.MagicMock(name="processor")

        cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    _assert_config_shape(cfg)
    assert isinstance(cfg.dataset, EnergonProvider)
    # The energon path must still resolve a PEFT scheme (default lora).
    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
