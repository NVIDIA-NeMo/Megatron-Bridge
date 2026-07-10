# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.bailing import bailing_moe2_bridge
from megatron.bridge.models.bailing.bailing_moe2_bridge import (
    BailingMoeV2Bridge,
    bailing_moe2_provider_layer_spec,
)
from megatron.bridge.models.bailing.configuration_bailing_moe_v2 import BailingMoeV2Config
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def _bailing_hf_config() -> BailingMoeV2Config:
    return BailingMoeV2Config(
        num_hidden_layers=4,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        first_k_dense_replace=1,
        use_qkv_bias=True,
        torch_dtype=torch.bfloat16,
    )


def test_model_config_bridge_preserves_bailing_moe2_specialization():
    hf_config = _bailing_hf_config()

    result = BailingMoeV2Bridge().model_config_bridge(SimpleNamespace(config=hf_config))

    assert type(result) is BridgeGPTModelConfig
    assert type(result.transformer) is TransformerConfig
    assert result.transformer_layer_spec is None
    assert result.transformer.normalization == "RMSNorm"
    assert result.transformer.add_qkv_bias is True
    assert result.transformer.moe_grouped_gemm is True
    assert result.transformer.moe_router_score_function == "sigmoid"
    assert result.transformer.moe_router_enable_expert_bias is True
    assert result.transformer.moe_layer_freq == [0, 1, 1, 1]
    assert result.transformer.moe_shared_expert_intermediate_size == 64
    assert "moe_layer_freq" not in result.__dict__

    restored = type(result).from_dict(result.as_dict())

    assert type(restored) is BridgeGPTModelConfig
    assert restored.transformer_layer_spec is None


def test_legacy_provider_preserves_mixed_dense_moe_layer_spec():
    provider = BailingMoeV2Bridge().provider_bridge(SimpleNamespace(config=_bailing_hf_config()))

    assert provider.transformer_layer_spec is bailing_moe2_provider_layer_spec
    assert provider.moe_layer_freq == [0, 1, 1, 1]


def test_model_config_bridge_uses_local_layers_without_transformer_engine(monkeypatch):
    monkeypatch.setattr(bailing_moe2_bridge, "HAVE_TE", False)

    result = BailingMoeV2Bridge().model_config_bridge(SimpleNamespace(config=_bailing_hf_config()))

    assert result.transformer.transformer_impl == "local"
