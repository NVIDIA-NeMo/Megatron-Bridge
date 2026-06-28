from megatron.core.transformer import TransformerConfig

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.stepfun.step37_model_config import Step37ModelConfig


def test_step37_config_roundtrip_preserves_exact_mcore_config_and_family_fields():
    config = Step37ModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size=256,
        layer_types=["full_attention", "sliding_attention"],
        rotary_percents=[0.5, 1.0],
        sliding_attention_setting={
            "window_size": [512, 0],
            "num_attention_heads": 4,
            "num_query_groups": 2,
            "kv_channels": 32,
        },
        image_token_id=42,
    )

    restored = ModelConfig.from_dict(config.as_dict())

    assert type(restored.transformer) is TransformerConfig
    assert restored.layer_types == ["full_attention", "sliding_attention"]
    assert restored.image_token_id == 42
    assert restored.builder == "megatron.bridge.models.stepfun.step37_model_config.Step37ModelBuilder"
