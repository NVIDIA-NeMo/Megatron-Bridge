# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""K3 recipe integration without checkpoint downloads or model allocations."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

import megatron.bridge.recipes as recipes
from megatron.bridge.models.kimi.kimi_k3_provider import KimiK3ModelProvider
from megatron.bridge.recipes.kimi import kimi_k3
from megatron.bridge.recipes.kimi.kimi_k3_runtime import configure_kimi_k3_kernels
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


@pytest.fixture
def offline_bridge(monkeypatch, tmp_path):
    monkeypatch.setattr(kimi_k3, "hf_hub_download", lambda *args: str(tmp_path / "tiktoken.model"))
    provider = KimiK3ModelProvider(num_layers=93, hidden_size=7168, num_attention_heads=96, vocab_size=163840)
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider
    constructor = MagicMock(return_value=bridge)
    monkeypatch.setattr(kimi_k3.AutoBridge, "from_hf_pretrained", constructor)
    return constructor, bridge, provider


def test_registered_recipe_preserves_native_provider_and_pipeline(offline_bridge):
    _, bridge, provider = offline_bridge
    cfg = recipes.kimi_k3_pretrain_config()
    assert cfg.model is provider
    bridge.to_megatron_provider.assert_called_once_with(load_weights=False)
    assert cfg.model.variable_seq_lengths
    assert cfg.model.num_layers == 93
    first = cfg.model.num_layers_in_first_pipeline_stage
    pp = cfg.model.pipeline_model_parallel_size
    assert (first, (cfg.model.num_layers - first) // (pp - 1)) == (24, 23)
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.context_parallel_size == 1
    assert not cfg.model.overlap_moe_expert_parallel_comm
    dp = 512 // cfg.model.tensor_model_parallel_size // pp
    assert dp == 32
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * dp) == 4
    assert configure_kimi_k3_kernels in cfg.model._pre_wrap_hooks


def test_local_snapshot_and_short_run_keep_optimizer_contract(offline_bridge, monkeypatch, tmp_path):
    constructor, _, _ = offline_bridge
    (tmp_path / "snapshot").mkdir()
    snapshot = str(tmp_path / "snapshot")
    monkeypatch.setenv("KIMI_K3_HF_PATH", snapshot)
    cfg = kimi_k3.kimi_k3_pretrain_config()
    constructor.assert_called_once_with(snapshot, trust_remote_code=True)
    assert cfg.tokenizer.tokenizer_model == snapshot
    assert cfg.tokenizer.vocab_file == str(tmp_path / "snapshot" / "tiktoken.model")
    assert not cfg.tokenizer.use_tokenizer_vocab_size
    cfg.train.train_iters = 10
    assert cfg.scheduler.lr_decay_iters == 10000
    assert cfg.scheduler.lr_warmup_iters == 100
    assert cfg.optimizer.use_distributed_optimizer
    assert not cfg.optimizer.use_precision_aware_optimizer
    assert cfg.optimizer.bf16 and not cfg.optimizer.fp16
    assert cfg.ddp.check_for_nan_in_grad and cfg.ddp.grad_reduce_in_fp32
    assert cfg.rerun_state_machine.check_for_nan_in_loss
    assert not cfg.ddp.overlap_grad_reduce and not cfg.ddp.overlap_param_gather
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.dataset.seq_length == cfg.model.seq_length == 8192
    assert "early, perf not optimized" in kimi_k3.kimi_k3_pretrain_config.__doc__


def test_runner_overrides_preserve_k3_tokenizer_and_hook(offline_bridge):
    cfg = kimi_k3.kimi_k3_pretrain_config()
    cfg = process_config_with_overrides(
        cfg,
        cli_overrides=[
            "train.train_iters=10",
            "dataset.seq_length=8192",
            "dataset.split='100,0,0'",
            "checkpoint.save=null",
        ],
    )
    # ConfigContainer.validate reruns initialization for subconfigs without finalize.
    cfg.tokenizer.__post_init__()
    assert cfg.tokenizer.trust_remote_code is True
    assert cfg.tokenizer.vocab_file.endswith("tiktoken.model")
    assert configure_kimi_k3_kernels in cfg.model._pre_wrap_hooks
    assert cfg.scheduler.lr_decay_iters == 10000


@pytest.mark.parametrize("legacy_kwargs", [{}, None])
def test_direct_tokenizer_options_survive_reconstruction(legacy_kwargs):
    cfg = TokenizerConfig(
        trust_remote_code=True,
        tokenizer_hf_no_use_fast=True,
        tokenizer_hf_no_include_special_tokens=True,
        hf_tokenizer_kwargs=legacy_kwargs,
    )
    copied = replace(cfg, tokenizer_model="local-snapshot")
    assert copied.trust_remote_code is True
    assert copied.tokenizer_hf_no_use_fast is True
    assert copied.tokenizer_hf_no_include_special_tokens is True


def test_explicit_legacy_tokenizer_options_keep_precedence():
    cfg = TokenizerConfig(
        trust_remote_code=True,
        tokenizer_hf_no_use_fast=True,
        tokenizer_hf_no_include_special_tokens=True,
        hf_tokenizer_kwargs={"trust_remote_code": False, "use_fast": True, "include_special_tokens": True},
    )
    assert cfg.trust_remote_code is False
    assert cfg.tokenizer_hf_no_use_fast is False
    assert cfg.tokenizer_hf_no_include_special_tokens is False
