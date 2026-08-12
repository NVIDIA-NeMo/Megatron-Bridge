# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Focused tests for Muse Glimmer verification recipes."""

from __future__ import annotations

import pytest

from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.muse_glimmer.h100 import (
    muse_glimmer_30b_peft_8gpu_h100_bf16_config,
    muse_glimmer_30b_pretrain_128gpu_h100_bf16_config,
    muse_glimmer_30b_sft_32gpu_h100_bf16_config,
    muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    ("recipe", "seq_length", "context_parallel_size", "global_batch_size"),
    [
        (muse_glimmer_30b_pretrain_128gpu_h100_bf16_config, 4096, 1, 1024),
        (muse_glimmer_30b_sft_32gpu_h100_bf16_config, 4096, 1, 8),
        (muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config, 8192, 4, 8),
        (muse_glimmer_30b_peft_8gpu_h100_bf16_config, 8192, 1, 8),
    ],
)
def test_muse_glimmer_recipe_contracts(recipe, seq_length, context_parallel_size, global_batch_size) -> None:
    cfg = recipe()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == context_parallel_size
    assert cfg.model.sequence_parallel is True
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["core_attn"]
    assert cfg.model.seq_length == seq_length
    assert cfg.dataset.seq_length == seq_length
    assert cfg.dataset.hf_processor_kwargs == {
        "revision": "f84ecc3a0ea984a4c04542a84269e3d065350a6e"  # pragma: allowlist secret
    }
    assert cfg.dataset.source.load_kwargs == {
        "revision": "7f0115a4b758a71d6473b8d085751692da2fef98"  # pragma: allowlist secret
    }
    assert cfg.dataset.pad_to_max_length is True
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == global_batch_size
    assert cfg.train.micro_batch_size == 1
    assert cfg.validation.eval_iters == 0
    assert cfg.validation.eval_interval == 0
    assert cfg.logger.log_throughput is True


def test_muse_glimmer_pretrain_owns_resume_checkpoint_contract() -> None:
    cfg = muse_glimmer_30b_pretrain_128gpu_h100_bf16_config()

    assert cfg.rng.seed == 1234
    assert cfg.scheduler.lr_warmup_iters == 40
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.optimizer.lr == pytest.approx(3e-4)
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.checkpoint.save_interval == 50
    assert cfg.checkpoint.load is None


def test_muse_glimmer_lora_targets_native_attention_projections() -> None:
    cfg = muse_glimmer_30b_peft_8gpu_h100_bf16_config()

    assert isinstance(cfg.peft, LoRA)
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.dropout == 0.0
    assert cfg.rng.seed == 5678
    assert cfg.optimizer.lr == pytest.approx(1e-4)
