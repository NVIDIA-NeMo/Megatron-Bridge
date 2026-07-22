# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the dense Nemotron 3 Nano 4B recipe configurations."""

import pytest

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.nemotronh import (
    nemotron_3_nano_4b_peft_config,
    nemotron_3_nano_4b_pretrain_config,
    nemotron_3_nano_4b_sft_32k_config,
    nemotron_3_nano_4b_sft_config,
)
from megatron.bridge.training.config import ConfigContainer


EXPECTED_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"


def _assert_exact_architecture(config: ConfigContainer) -> None:
    assert isinstance(config, ConfigContainer)
    assert isinstance(config.model, HybridModelProvider)
    assert config.model.hybrid_layer_pattern == EXPECTED_PATTERN
    assert config.model.num_layers == 42
    assert config.model.hidden_size == 3136
    assert config.model.ffn_hidden_size == 12544
    assert config.model.num_attention_heads == 40
    assert config.model.num_query_groups == 8
    assert config.model.kv_channels == 128
    assert config.model.mamba_num_heads == 96
    assert config.model.mamba_head_dim == 80
    assert config.model.mamba_state_dim == 128
    assert config.model.mamba_num_groups == 8
    assert config.model.mamba_chunk_size == 256
    assert config.model.vocab_size == 131072
    assert config.model.share_embeddings_and_output_weights is False
    assert config.model.position_embedding_type == "none"
    assert config.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert config.tokenizer.hf_tokenizer_kwargs["revision"] == (
        "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"  # pragma: allowlist secret
    )


@pytest.mark.unit
def test_pretrain_config_matches_exact_architecture_and_bounded_contract() -> None:
    config = nemotron_3_nano_4b_pretrain_config()

    _assert_exact_architecture(config)
    assert config.model.tensor_model_parallel_size == 1
    assert config.model.context_parallel_size == 1
    assert config.dataset.seq_length == 4096
    assert config.train.train_iters == 100
    assert config.train.global_batch_size == 1024
    assert config.train.micro_batch_size == 1
    assert config.scheduler.lr_warmup_iters == 40
    assert config.scheduler.lr_decay_iters == 100
    assert config.checkpoint.save_interval == 50
    assert config.checkpoint.load is None


@pytest.mark.unit
def test_sft_config_matches_packed_cohort_contract() -> None:
    config = nemotron_3_nano_4b_sft_config()

    _assert_exact_architecture(config)
    assert config.dataset.seq_length == 2048
    assert config.dataset.offline_packing_specs.packed_sequence_size == 2048
    assert config.dataset.offline_packing_specs.pad_seq_to_mult == 1
    assert config.train.train_iters == 100
    assert config.train.global_batch_size == 32
    assert config.train.micro_batch_size == 1
    assert config.optimizer.lr == 5.0e-6
    assert config.checkpoint.save_interval == 100


@pytest.mark.unit
def test_long_context_sft_config_combines_packing_and_context_parallelism() -> None:
    config = nemotron_3_nano_4b_sft_32k_config()

    _assert_exact_architecture(config)
    assert config.model.context_parallel_size == 2
    assert config.model.cp_comm_type == "a2a"
    assert config.model.calculate_per_token_loss is True
    assert config.model.cross_entropy_loss_fusion is False
    assert config.dataset.seq_length == 32768
    assert config.dataset.offline_packing_specs.packed_sequence_size == 32768
    assert config.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert config.train.global_batch_size == 8
    assert config.ddp.average_in_collective is False


@pytest.mark.unit
def test_peft_config_uses_attention_only_cohort_lora() -> None:
    config = nemotron_3_nano_4b_peft_config()

    _assert_exact_architecture(config)
    assert isinstance(config.peft, LoRA)
    assert config.peft.dim == 8
    assert config.peft.alpha == 16
    assert config.peft.dropout == 0.0
    assert config.peft.target_modules == ["linear_qkv", "linear_proj"]
    assert config.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert config.train.train_iters == 100
    assert config.train.global_batch_size == 32
    assert config.optimizer.lr == 1.0e-4


@pytest.mark.unit
def test_peft_config_preserves_custom_peft_instance() -> None:
    custom = LoRA(target_modules=["linear_qkv"], dim=4, alpha=8)

    config = nemotron_3_nano_4b_peft_config(custom)

    assert config.peft is custom
