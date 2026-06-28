from types import SimpleNamespace

import pytest
from megatron.core.transformer.transformer_config import MLATransformerConfig

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.kimi.kimi_bridge import KimiK2Bridge, KimiK2ModelConfig


pytestmark = pytest.mark.unit


def test_model_config_bridge_maps_kimi_mla_config() -> None:
    hf_config = SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=1,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        kv_lora_rank=512,
        max_position_embeddings=4096,
        moe_intermediate_size=1024,
        n_group=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_attention_heads=16,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        num_key_value_heads=16,
        q_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        tie_word_embeddings=False,
        topk_group=1,
        torch_dtype="float32",
        v_head_dim=64,
        vocab_size=32000,
    )

    model_config = KimiK2Bridge().model_config_bridge(SimpleNamespace(config=hf_config))

    assert isinstance(model_config, BridgeGPTModelConfig)
    assert type(model_config.transformer) is MLATransformerConfig
    assert model_config.transformer.rope_type == "rope"
    assert model_config.transformer.rotary_scaling_factor == 1.0
    assert model_config.transformer.mscale == 1.0
    assert model_config.transformer.mscale_all_dim == 1.0
    assert model_config.transformer.moe_layer_freq == [0, 1, 1, 1]
    assert model_config.transformer.moe_aux_loss_coeff == 1e-3
    assert model_config.make_vocab_size_divisible_by == 1280

    restored = BridgeGPTModelConfig.from_dict(model_config.as_dict())
    assert isinstance(restored, KimiK2ModelConfig)
    assert callable(restored.transformer_layer_spec)
