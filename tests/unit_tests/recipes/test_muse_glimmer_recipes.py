# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Focused tests for Muse Glimmer verification recipes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.peft.lora import LoRA
from megatron.bridge.perf_recipes.muse_glimmer.h100 import (
    muse_glimmer_30b_pretrain_32gpu_h100_bf16_config,
    muse_glimmer_30b_pretrain_32gpu_h100_fp8cs_config,
)
from megatron.bridge.recipes.muse_glimmer.h100 import muse_glimmer as muse_glimmer_recipes
from megatron.bridge.recipes.muse_glimmer.h100 import (
    muse_glimmer_30b_peft_8gpu_h100_bf16_config,
    muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config,
    muse_glimmer_30b_sft_32gpu_h100_bf16_config,
    muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config,
)
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    (
        "recipe",
        "seq_length",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "global_batch_size",
    ),
    [
        (muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config, 4096, 8, 2, 1, 2),
        (muse_glimmer_30b_sft_32gpu_h100_bf16_config, 4096, 8, 2, 1, 8),
        (muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config, 8192, 1, 4, 2, 8),
        (muse_glimmer_30b_peft_8gpu_h100_bf16_config, 8192, 8, 1, 1, 8),
    ],
)
def test_muse_glimmer_recipe_contracts(
    recipe,
    seq_length,
    tensor_parallel_size,
    pipeline_parallel_size,
    context_parallel_size,
    global_batch_size,
) -> None:
    cfg = recipe()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model.tensor_model_parallel_size == tensor_parallel_size
    assert cfg.model.pipeline_model_parallel_size == pipeline_parallel_size
    assert cfg.model.pipeline_dtype is (torch.bfloat16 if pipeline_parallel_size > 1 else None)
    expected_pattern = {
        1: "*" * 52,
        2: f"{'*' * 20}|{'*' * 32}",
        4: "|".join("*" * layers for layers in (9, 15, 15, 13)),
    }[pipeline_parallel_size]
    assert getattr(cfg.model, "hybrid_layer_pattern", "*" * 52) == expected_pattern
    assert cfg.model.recompute_vision_layers is True
    assert cfg.dataset.hf_processor_kwargs == {
        "revision": muse_glimmer_recipes._MODEL_REVISION,
        "max_image_tokens": 256,
    }
    assert getattr(cfg.model, "pipeline_model_parallel_layout", None) is None
    assert cfg.model.context_parallel_size == context_parallel_size
    assert getattr(cfg.model, "cp_comm_type", "p2p") == ("a2a" if context_parallel_size > 1 else "p2p")
    assert cfg.model.sequence_parallel is True
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["core_attn"]
    assert cfg.model.seq_length == seq_length
    assert cfg.dataset.seq_length == seq_length
    assert cfg.dataset.source.load_kwargs == {
        "revision": "7f0115a4b758a71d6473b8d085751692da2fef98"  # pragma: allowlist secret
    }
    is_long_context = recipe is muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config
    assert cfg.dataset.pad_to_max_length is not is_long_context
    assert cfg.dataset.enable_in_batch_packing is is_long_context
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == global_batch_size
    assert cfg.train.micro_batch_size == (2 if is_long_context else 1)
    assert cfg.validation.eval_iters == 0
    assert cfg.validation.eval_interval == 0
    assert cfg.logger.log_throughput is True


def test_muse_glimmer_pretrain_owns_resume_checkpoint_contract() -> None:
    cfg = muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config()

    assert cfg.rng.seed == 1234
    assert cfg.scheduler.lr_warmup_iters == 40
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.optimizer.lr == pytest.approx(3e-4)
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.dataset.num_workers == 0
    assert cfg.checkpoint.save_interval == 50
    assert cfg.checkpoint.load is None


def test_muse_glimmer_long_context_uses_te_supported_packed_cp_layout() -> None:
    cfg = muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.cp_comm_type == "a2a"
    assert cfg.model.hybrid_layer_pattern == "|".join("*" * layers for layers in (9, 15, 15, 13))
    assert cfg.train.micro_batch_size == 2


def test_muse_glimmer_performance_recipe_is_dense_decoder_only() -> None:
    cfg = muse_glimmer_30b_pretrain_32gpu_h100_bf16_config()

    assert isinstance(cfg.dataset, MockGPTDatasetConfig)
    assert cfg.dataset.seq_length == 4096
    assert cfg.model.hybrid_layer_pattern == "|".join("*" * layers for layers in (9, 15, 15, 13))
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is True
    assert cfg.model.recompute_vision_layers is False
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.cp_comm_type == "all_gather"
    assert cfg.train.train_iters == 50
    assert cfg.train.global_batch_size == 96
    assert cfg.train.micro_batch_size == 3
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 50
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.comm_overlap.tp_comm_bootstrap_backend == "nccl"
    assert cfg.comm_overlap.tp_comm_overlap is True
    assert cfg.checkpoint.save is None
    assert cfg.checkpoint.load is None
    assert cfg.env_vars == {
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "TORCH_NCCL_HIGH_PRIORITY": 1,
    }


def test_muse_glimmer_fp8_performance_recipe_uses_current_scaling() -> None:
    cfg = muse_glimmer_30b_pretrain_32gpu_h100_fp8cs_config()

    assert cfg.mixed_precision.fp8 == "hybrid"
    assert cfg.mixed_precision.fp8_recipe == "tensorwise"
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.context_parallel_size == 2
    assert cfg.train.global_batch_size == 192
    assert cfg.train.micro_batch_size == 6


def test_muse_glimmer_lora_targets_native_attention_projections() -> None:
    cfg = muse_glimmer_30b_peft_8gpu_h100_bf16_config()

    assert isinstance(cfg.peft, LoRA)
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.dropout == 0.0
    assert cfg.rng.seed == 5678
    assert cfg.optimizer.lr == pytest.approx(1e-4)


def test_muse_glimmer_recipes_use_builder_model_config_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Bridge:
        def get_model_config(self):
            return SimpleNamespace()

        def to_megatron_provider(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("legacy GPT provider path used")

    class _AutoBridge:
        @staticmethod
        def from_hf_pretrained(*args, **kwargs):
            del args, kwargs
            return _Bridge()

    monkeypatch.setattr(muse_glimmer_recipes, "AutoBridge", _AutoBridge)

    cfg = muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config()

    assert cfg.model is not None
