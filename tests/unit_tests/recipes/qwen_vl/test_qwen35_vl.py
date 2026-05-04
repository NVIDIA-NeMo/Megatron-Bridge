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
# - Cover the previously untested qwen35_vl recipes (issue #3177 sweep).
# - Parametrize over all dense + MoE pretrain / SFT / PEFT functions.
# - Monkeypatch AutoBridge in BOTH the qwen35_vl module and the qwen3_vl
#   module that pretrain configs delegate into.
# - Patch `apply_flex_dispatcher_backend` in the qwen3_vl module so the
#   shared common helper does not touch torch.cuda on a CPU runner.
# - Sanity-check the dense vs MoE wiring (EP only set on MoE recipes,
#   recompute enabled on the largest two MoE SFT recipes), and the
#   FSDP-specific wiring on the dedicated FSDP variant.
#

import importlib
from typing import Callable

import pytest


_qwen35_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen35_vl")
_qwen3_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen3_vl")


# -----------------------------------------------------------------------------
# Recipe groups
# -----------------------------------------------------------------------------

_PRETRAIN_FUNCS: list[Callable] = [
    _qwen35_vl_module.qwen35_vl_9b_pretrain_mock_config,
    _qwen35_vl_module.qwen35_vl_35b_a3b_pretrain_mock_config,
    _qwen35_vl_module.qwen35_vl_122b_a10b_pretrain_mock_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_pretrain_mock_config,
]

_DENSE_SFT_FUNCS: list[Callable] = [
    _qwen35_vl_module.qwen35_vl_800m_sft_config,
    _qwen35_vl_module.qwen35_vl_2b_sft_config,
    _qwen35_vl_module.qwen35_vl_4b_sft_config,
    _qwen35_vl_module.qwen35_vl_9b_sft_config,
    _qwen35_vl_module.qwen35_vl_27b_sft_config,
]

_MOE_SFT_FUNCS: list[Callable] = [
    _qwen35_vl_module.qwen35_vl_35b_a3b_sft_config,
    _qwen35_vl_module.qwen35_vl_122b_a10b_sft_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_sft_config,
]

_DENSE_PEFT_FUNCS: list[Callable] = [
    _qwen35_vl_module.qwen35_vl_800m_peft_config,
    _qwen35_vl_module.qwen35_vl_2b_peft_config,
    _qwen35_vl_module.qwen35_vl_4b_peft_config,
    _qwen35_vl_module.qwen35_vl_9b_peft_config,
    _qwen35_vl_module.qwen35_vl_27b_peft_config,
]

_MOE_PEFT_FUNCS: list[Callable] = [
    _qwen35_vl_module.qwen35_vl_35b_a3b_peft_config,
    _qwen35_vl_module.qwen35_vl_122b_a10b_peft_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_peft_config,
]

_RECOMPUTE_SFT_FUNCS: list[Callable] = [
    _qwen35_vl_module.qwen35_vl_122b_a10b_sft_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_sft_config,
]


# -----------------------------------------------------------------------------
# Stubs
# -----------------------------------------------------------------------------


class _FakeQwen35VLProvider:
    """Permissive provider stub.

    The qwen35_vl recipes touch a long list of attrs on `cfg.model`
    (parallelism, MTP, kernel, MoE knobs, recompute, freeze flags). The
    stub exposes only the attrs that are READ by the recipes; the rest
    are absorbed via ordinary `setattr`.

    `num_moe_experts` is intentionally omitted so that any incidental
    call to `apply_flex_dispatcher_backend` short-circuits before
    touching `torch.cuda`. Pretrain recipes also pass through this stub
    via `_qwen3_vl_common`; the test fixture additionally patches the
    flex helper to a no-op for belt-and-suspenders.
    """

    def __init__(self):
        self.vocab_size = 152000
        self.apply_rope_fusion = False

    def finalize(self):
        return None


class _FakeAutoBridge:
    """AutoBridge stub that bypasses HuggingFace Hub network access."""

    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        return cls()

    def to_megatron_provider(self, *args, **kwargs):
        return _FakeQwen35VLProvider()


@pytest.fixture(autouse=True)
def _patch_recipe_modules(monkeypatch):
    """Patch AutoBridge in BOTH qwen35_vl and qwen3_vl modules.

    The qwen35_vl pretrain configs delegate into `_qwen3_vl_common`, which
    references the qwen3_vl module's `AutoBridge` symbol. SFT/PEFT recipes
    reference qwen35_vl's own `AutoBridge` symbol.
    """
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)
    # Belt-and-suspenders: even though our fake provider has no
    # num_moe_experts attribute (so apply_flex_dispatcher_backend would
    # short-circuit), explicitly no-op the helper so the test does not
    # depend on that internal behavior.
    monkeypatch.setattr(_qwen3_vl_module, "apply_flex_dispatcher_backend", lambda *a, **kw: None)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _assert_config_shape(cfg) -> None:
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


# -----------------------------------------------------------------------------
# Pretrain
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _PRETRAIN_FUNCS, ids=lambda f: f.__name__)
def test_pretrain_recipe_builds(recipe_func: Callable):
    """Every qwen35_vl pretrain recipe builds without HF Hub or CUDA access."""
    cfg = recipe_func()
    _assert_config_shape(cfg)

    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.model.tensor_model_parallel_size >= 1
    assert cfg.model.pipeline_model_parallel_size >= 1
    assert cfg.model.context_parallel_size >= 1

    # Pretrain VLM defaults inherited from `_qwen3_vl_common`: language +
    # vision frozen, projection trainable.
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False

    # Pretrain configs use mock VLM data by default.
    from megatron.bridge.data.vlm_datasets import MockVLMConversationProvider

    assert isinstance(cfg.dataset, MockVLMConversationProvider)


# -----------------------------------------------------------------------------
# SFT (dense + MoE)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _DENSE_SFT_FUNCS + _MOE_SFT_FUNCS, ids=lambda f: f.__name__)
def test_sft_recipe_builds(recipe_func: Callable):
    """Every qwen35_vl SFT recipe builds without HF Hub or CUDA access."""
    cfg = recipe_func()
    _assert_config_shape(cfg)

    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # SFT recipes train all components.
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False

    # SFT recipes don't attach a PEFT scheme.
    assert cfg.peft is None

    # Sequence length is harmonized between model and dataset.
    assert cfg.dataset.seq_length == cfg.model.seq_length

    # MTP is on by default for qwen35_vl.
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.mtp_loss_scaling_factor == 0.1


@pytest.mark.parametrize("recipe_func", _DENSE_SFT_FUNCS, ids=lambda f: f.__name__)
def test_dense_sft_does_not_set_ep(recipe_func: Callable):
    """Dense SFT recipes do not touch expert parallelism."""
    cfg = recipe_func()
    # `_qwen35_vl_apply_moe` is the only place that sets EP; dense recipes
    # never invoke it, so the attr should remain unset on the fake provider.
    assert not hasattr(cfg.model, "expert_model_parallel_size")


@pytest.mark.parametrize("recipe_func", _MOE_SFT_FUNCS, ids=lambda f: f.__name__)
def test_moe_sft_enables_ep_and_alltoall(recipe_func: Callable):
    """MoE SFT recipes set EP, sequence parallel, and the alltoall dispatcher."""
    cfg = recipe_func()

    # All three MoE SFT recipes go through `_qwen35_vl_apply_moe`.
    assert getattr(cfg.model, "expert_model_parallel_size", 1) > 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.moe_grouped_gemm is True


@pytest.mark.parametrize("recipe_func", _RECOMPUTE_SFT_FUNCS, ids=lambda f: f.__name__)
def test_large_moe_sft_enables_full_recompute(recipe_func: Callable):
    """The 122B and 397B MoE SFT recipes call `_qwen35_vl_enable_recompute`."""
    cfg = recipe_func()

    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1


def test_35b_a3b_fsdp_sft_specifics():
    """The Megatron-FSDP variant overrides DDP and disables SP (SP needs TP>1)."""
    cfg = _qwen35_vl_module.qwen35_vl_35b_a3b_fsdp_sft_config()
    _assert_config_shape(cfg)

    # FSDP-specific wiring.
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.fsdp_double_buffer is True
    assert cfg.ddp.nccl_ub is False
    assert cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail is True
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.ddp.num_distributed_optimizer_instances == 1

    # FSDP variant uses TP=1, so SP must be disabled even though
    # `_qwen35_vl_apply_moe` would normally turn it on.
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # FSDP variant still uses MoE EP.
    assert cfg.model.expert_model_parallel_size == 2


# -----------------------------------------------------------------------------
# PEFT (dense + MoE)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _DENSE_PEFT_FUNCS + _MOE_PEFT_FUNCS, ids=lambda f: f.__name__)
def test_peft_recipe_builds_with_default_lora(recipe_func: Callable):
    """Every qwen35_vl PEFT recipe builds with the default scheme (lora)."""
    cfg = recipe_func()
    _assert_config_shape(cfg)

    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


@pytest.mark.parametrize("recipe_func", _DENSE_PEFT_FUNCS + _MOE_PEFT_FUNCS, ids=lambda f: f.__name__)
def test_peft_recipe_with_dora(recipe_func: Callable):
    """Every PEFT recipe accepts peft_scheme='dora'."""
    from megatron.bridge.peft.dora import DoRA

    cfg = recipe_func(peft_scheme="dora")
    _assert_config_shape(cfg)
    assert isinstance(cfg.peft, DoRA)


@pytest.mark.parametrize("recipe_func", _MOE_PEFT_FUNCS, ids=lambda f: f.__name__)
def test_moe_peft_enables_ep(recipe_func: Callable):
    """MoE PEFT recipes wire EP just like their SFT siblings."""
    cfg = recipe_func()
    assert getattr(cfg.model, "expert_model_parallel_size", 1) > 1
    assert cfg.model.moe_token_dispatcher_type == "alltoall"


# -----------------------------------------------------------------------------
# Spot checks for the largest recipes
# -----------------------------------------------------------------------------


def test_sft_27b_uses_pp_for_dense_27b():
    """27B dense SFT spans 2 nodes via TP=4, PP=4."""
    cfg = _qwen35_vl_module.qwen35_vl_27b_sft_config()

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4


def test_sft_122b_a10b_specifics():
    """122B-A10B SFT wires TP=2, PP=6, EP=8 with full recompute."""
    cfg = _qwen35_vl_module.qwen35_vl_122b_a10b_sft_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 6
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.recompute_granularity == "full"


def test_sft_397b_a17b_specifics():
    """397B-A17B SFT wires the largest documented EP layout."""
    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_sft_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.recompute_granularity == "full"
