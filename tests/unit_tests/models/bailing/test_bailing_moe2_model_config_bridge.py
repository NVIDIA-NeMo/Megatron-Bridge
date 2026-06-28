from functools import partial
from types import SimpleNamespace

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.bailing.bailing_moe2_bridge import BailingMoeV2Bridge
from megatron.bridge.models.bailing.configuration_bailing_moe_v2 import BailingMoeV2Config


def test_model_config_bridge_preserves_bailing_moe2_specialization():
    hf_config = BailingMoeV2Config(
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

    result = BailingMoeV2Bridge().model_config_bridge(SimpleNamespace(config=hf_config))

    assert isinstance(result, GPTModelConfig)
    assert type(result.transformer) is TransformerConfig
    assert result.transformer_layer_spec is not None
    assert result.transformer.normalization == "RMSNorm"
    assert result.transformer.add_qkv_bias is True
    assert result.transformer.moe_grouped_gemm is True
    assert result.transformer.moe_router_score_function == "sigmoid"
    assert result.transformer.moe_router_enable_expert_bias is True
    assert result.transformer.moe_layer_freq == [0, 1, 1, 1]
    assert result.transformer.moe_shared_expert_intermediate_size == 64
    assert "moe_layer_freq" not in result.__dict__

    restored = type(result).from_dict(result.as_dict())

    assert type(restored) is type(result)
    assert isinstance(restored.transformer_layer_spec, partial)
    assert restored.transformer_layer_spec.func is result.transformer_layer_spec.func
    assert restored.transformer_layer_spec.keywords == result.transformer_layer_spec.keywords
