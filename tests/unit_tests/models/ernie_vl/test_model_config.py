from megatron.core.transformer import TransformerConfig

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.ernie_vl.model_config import Ernie45VLModelConfig


def test_ernie_vl_config_roundtrip_preserves_exact_mcore_configs():
    config = Ernie45VLModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vision_transformer=TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4),
        vocab_size=256,
        patch_size=16,
        in_channels=3,
        spatial_merge_size=4,
    )

    restored = ModelConfig.from_dict(config.as_dict())

    assert type(restored.transformer) is TransformerConfig
    assert type(restored.vision_transformer) is TransformerConfig
    assert restored.patch_size == 16
    assert restored.spatial_merge_size == 4
    assert restored.builder == "megatron.bridge.models.ernie_vl.model_config.Ernie45VLModelBuilder"
