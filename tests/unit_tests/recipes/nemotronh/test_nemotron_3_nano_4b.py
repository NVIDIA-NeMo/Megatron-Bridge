# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the dense Nemotron 3 Nano 4B recipe configurations."""

import pytest
from transformers import NemotronHConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.hybrid import BridgeHybridModelConfig
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.nemotronh import (
    nemotron_3_nano_4b_peft_config,
    nemotron_3_nano_4b_pretrain_config,
    nemotron_3_nano_4b_sft_32k_config,
    nemotron_3_nano_4b_sft_config,
)
from megatron.bridge.recipes.nemotronh.h100 import nemotron_3_nano_4b as recipe_module
from megatron.bridge.training.config import ConfigContainer


EXPECTED_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"

# The public Hugging Face config of the pinned checkpoint revision.
_HF_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "chunk_size": 256,
    "conv_kernel": 4,
    "dtype": "bfloat16",
    "eos_token_id": 2,
    "expand": 2,
    "head_dim": 128,
    "hidden_dropout": 0.0,
    "hidden_size": 3136,
    "hybrid_override_pattern": EXPECTED_PATTERN,
    "initializer_range": 0.02,
    "intermediate_size": 12544,
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 80,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 96,
    "mamba_proj_bias": False,
    "max_position_embeddings": 262144,
    "mlp_bias": False,
    "mlp_hidden_act": "relu2",
    "n_groups": 8,
    "num_attention_heads": 40,
    "num_hidden_layers": 42,
    "num_key_value_heads": 8,
    "num_logits_to_keep": 1,
    "pad_token_id": 0,
    "rescale_prenorm_residual": True,
    "residual_in_fp32": False,
    "rms_norm_eps": 1e-05,
    "sliding_window": None,
    "ssm_state_size": 128,
    "tie_word_embeddings": False,
    "time_step_floor": 0.0001,
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "time_step_rank": 256,
    "use_bias": False,
    "use_cache": True,
    "use_conv_bias": True,
    "use_mamba_kernels": True,
    "vocab_size": 131072,
}


@pytest.fixture(autouse=True)
def _offline_hf_config(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Map the pinned checkpoint config through the real bridge without Hugging Face Hub access."""
    calls = []

    def from_hf_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        hf_config = NemotronHConfig(**_HF_CONFIG)
        hf_config.architectures = ["NemotronHForCausalLM"]
        hf_config._name_or_path = model_id
        bridge = AutoBridge.from_hf_config(hf_config)
        bridge.hf_model_revision = kwargs.get("revision")
        return bridge

    monkeypatch.setattr(recipe_module.AutoBridge, "from_hf_pretrained", staticmethod(from_hf_pretrained))
    return calls


def _assert_exact_architecture(config: ConfigContainer) -> None:
    assert isinstance(config, ConfigContainer)
    assert isinstance(config.model, BridgeHybridModelConfig)
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
    # The dense model keeps every MoE field at its Megatron Core default.
    assert config.model.num_moe_experts is None
    assert config.model.moe_ffn_hidden_size is None
    assert config.model.moe_shared_expert_intermediate_size is None
    assert config.model.moe_router_num_groups is None
    assert config.model.moe_router_score_function == "softmax"
    assert config.model.moe_token_dispatcher_type == "allgather"
    assert config.model.extra_checkpoint_metadata == {
        "hf_model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "hf_model_revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",  # pragma: allowlist secret
    }
    assert config.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert config.tokenizer.hf_tokenizer_kwargs["revision"] == (
        "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"  # pragma: allowlist secret
    )


@pytest.mark.unit
def test_pretrain_config_matches_exact_architecture_and_bounded_contract(
    _offline_hf_config: list[tuple[str, dict]],
) -> None:
    config = nemotron_3_nano_4b_pretrain_config()

    assert _offline_hf_config == [
        (
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            {"revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"},  # pragma: allowlist secret
        )
    ]
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
