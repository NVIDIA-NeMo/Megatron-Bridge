"""Config-only native VL dispatch, checkpoint configuration and tying checks."""

import pytest
from transformers import Lfm2VlConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge, _resolve_pretrained_wrapper_cls
from megatron.bridge.models.d1 import D1VLBridge, D1VLModelProvider
from megatron.bridge.models.hf_pretrained.auto_model import PreTrainedAutoModel


pytestmark = pytest.mark.unit


def tiny_vl_config() -> Lfm2VlConfig:
    return Lfm2VlConfig(
        architectures=["Lfm2VlForConditionalGeneration"],
        auto_map={"AutoModel": "modeling_d1.D1Model"},
        tie_word_embeddings=True,
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 192,
            "block_auto_adjust_ff_dim": False,
            "layer_types": ["conv", "full_attention"],
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "conv_L_cache": 3,
            "conv_bias": False,
        },
        vision_config={
            "model_type": "siglip2_vision_model",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_patches": 16,
            "patch_size": 16,
            "vision_use_head": False,
        },
        projector_hidden_size=64,
        image_token_id=31,
    )


def test_vl_dispatch_and_export_preserve_published_config() -> None:
    config = tiny_vl_config()
    bridge = AutoBridge.from_hf_config(config)
    assert isinstance(bridge._model_bridge, D1VLBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(provider, D1VLModelProvider)
    assert provider.hidden_size == 64
    assert provider.ffn_hidden_size == 192
    assert provider.num_query_groups == 2
    assert provider.rotary_base == 1000000.0
    assert provider.share_embeddings_and_output_weights is True
    assert _resolve_pretrained_wrapper_cls(config) is PreTrainedAutoModel
    exported = D1VLBridge.megatron_to_hf_config(provider)
    assert exported["auto_map"] == config.auto_map
    assert exported["tie_word_embeddings"] is True
    assert exported["vision_config"] == config.to_dict()["vision_config"]
    assert exported["image_token_id"] == 31


def test_vl_rejects_untied_vocabulary() -> None:
    config = tiny_vl_config()
    config.tie_word_embeddings = False
    with pytest.raises(ValueError, match="tied input/output"):
        AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
