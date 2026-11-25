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

"""Tests for the 16-GPU Qwen3.5-35B-A3B 128K dynamic-CP recipe on CoderForge."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5MoeConfig

from megatron.bridge.recipes.qwen.gb200 import qwen35


pytestmark = pytest.mark.unit

_STATIC = qwen35.qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config
_DYNAMIC = qwen35.qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_dynamic_cp_config

# Text configuration of Qwen/Qwen3.5-35B-A3B at the recipe's pinned revision.
_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "moe_intermediate_size": 512,
    "mtp_num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 512,
    "vocab_size": 248320,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
}


@pytest.fixture(autouse=True)
def published_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build recipes on the published 35B-A3B text config without reaching the Hub."""
    monkeypatch.setattr(
        qwen35.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: Qwen3_5MoeConfig(text_config=_TEXT_CONFIG, tie_word_embeddings=False),
    )


def _validated(cfg, monkeypatch: pytest.MonkeyPatch, *, world_size: int):
    """Mirror runtime_config_update on ``world_size`` GB200 GPUs: set up mixed precision, then validate."""
    training_module = importlib.import_module("megatron.bridge.training.config")
    monkeypatch.setattr(training_module, "get_world_size_safe", lambda: world_size)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(major=10, name="NVIDIA GB200")
    )
    cfg.mixed_precision.finalize()
    cfg.mixed_precision.setup(cfg.model, cfg.optimizer, cfg.ddp)
    cfg.validate()
    return cfg


# Settings the dynamic-CP recipe takes unchanged from the static long-context recipe.
_SHARED_SETTINGS = (
    "model.seq_length",
    "model.tensor_model_parallel_size",
    "model.context_parallel_size",
    "model.expert_model_parallel_size",
    "model.mtp_num_layers",
    "model.recompute_granularity",
    "model.recompute_modules",
    "model.moe_token_dispatcher_type",
    "model.moe_flex_dispatcher_backend",
    "model.gdn_kernel_backend",
    "model.calculate_per_token_loss",
    "dataset.seq_length",
    "dataset.hf_dataset",
    "dataset.enable_offline_packing",
    "dataset.offline_packing_specs",
    "dataset.dataset_kwargs",
    "dataset.do_validation",
    "dataset.hf_validation_proportion",
    "dataset.seed",
    "train.global_batch_size",
    "train.micro_batch_size",
    "train.train_iters",
    "validation.eval_interval",
    "optimizer.lr",
    "optimizer.min_lr",
    "scheduler.lr_warmup_iters",
    "scheduler.lr_decay_iters",
    "mixed_precision.fp8_recipe",
    "checkpoint.load_main_params_from_ckpt",
    "tokenizer.tokenizer_model",
    "tokenizer.hf_tokenizer_kwargs",
    "rng.seed",
    "env_vars",
)


def _setting(cfg, path: str):
    value = cfg
    for name in path.split("."):
        value = getattr(value, name)
    return value


def test_keeps_the_static_recipe_and_adds_packing_and_dynamic_cp():
    static, dynamic = _STATIC(), _DYNAMIC()

    for path in _SHARED_SETTINGS:
        assert _setting(dynamic, path) == _setting(static, path), path
    assert (static.dataset.enable_global_batch_packing, dynamic.dataset.enable_global_batch_packing) == (False, True)
    assert not getattr(static.model, "dynamic_context_parallel", False)
    assert dynamic.model.dynamic_context_parallel is True
    assert dynamic.model.min_dynamic_context_parallel_size == 1
    # Fixed per-rank length: every bin is padded to its group's capacity.
    assert dynamic.model.pad_packed_seq_alignment == "max"
    assert dynamic.dataset.dataloader_type == "single"
    assert dynamic.dataset.fold_alignment_padding is True  # 256-wide heads train on FlashAttention only
    assert dynamic.ddp.average_in_collective is False
    # The fused pre-delta-rule kernels cannot run a bin on the whole 16-rank pool.
    assert (static.model.gdn_pre_gated_delta_rule_fusion, dynamic.model.gdn_pre_gated_delta_rule_fusion) == (
        True,
        False,
    )
    # Every evaluation of the run draws new validation rows, so evaluations are shorter than the static recipe's.
    assert (static.validation.eval_iters, dynamic.validation.eval_iters) == (8, 2)


def test_runs_the_long_context_recipe_on_a_16_rank_dynamic_cp_pool(monkeypatch: pytest.MonkeyPatch):
    cfg = _validated(_DYNAMIC(), monkeypatch, world_size=16)
    model, dataset = cfg.model, cfg.dataset

    assert model.seq_length == dataset.seq_length == 131072
    assert (model.tensor_model_parallel_size, model.context_parallel_size, model.expert_model_parallel_size) == (
        1,
        8,
        16,
    )
    assert model.sequence_packing_scheduler == "default_dynamic_cp"
    assert model.max_seqlen_per_dp_cp_rank == 16384
    # Selective recompute and NCCL EP come from the static long-context recipe.
    assert (model.recompute_granularity, model.recompute_modules) == ("selective", ["gdn_norm_out", "moe"])
    assert model.moe_flex_dispatcher_backend == "ncclep"
    assert model.transformer.max_seq_length_per_sample == 131072
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    # The real CoderForge rows, re-binned by the scheduler; the global batch counts rows.
    assert dataset.hf_dataset.dataset_name == "coderforge"
    assert dataset.enable_offline_packing is True and dataset.yields_unpacked_samples is True
    specs = dataset.offline_packing_specs
    assert (specs.packed_sequence_size, specs.pad_seq_to_mult, specs.pad_cu_seqlens) == (131072, 16, False)
    assert (cfg.train.micro_batch_size, cfg.train.global_batch_size) == (1, 32)
    assert cfg.checkpoint.pretrained_checkpoint is None


def test_static_cp_override_keeps_the_packing_path(monkeypatch: pytest.MonkeyPatch):
    cfg = _DYNAMIC()
    cfg.model.dynamic_context_parallel = False
    cfg = _validated(cfg, monkeypatch, world_size=16)

    assert cfg.model.sequence_packing_scheduler == "dp_balanced"
    assert cfg.dataset.enable_global_batch_packing is True


def test_pool_beyond_the_gated_delta_net_key_heads_is_rejected(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="Gated DeltaNet"):
        _validated(_DYNAMIC(), monkeypatch, world_size=32)


def test_sequence_length_sets_the_tokens_per_rank():
    cfg = _DYNAMIC(65536)

    assert cfg.model.seq_length == cfg.dataset.seq_length == 65536
    assert cfg.model.max_seqlen_per_dp_cp_rank == 8192
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 65536
