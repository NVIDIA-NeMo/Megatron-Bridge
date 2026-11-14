"""Config-only Omni dispatch and provider field validation."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge, _resolve_pretrained_wrapper_cls
from megatron.bridge.models.d1 import D1OmniBridge, D1OmniModelProvider
from megatron.bridge.models.hf_pretrained.auto_model import PreTrainedAutoModel
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit


@pytest.fixture
def omni_config() -> PretrainedConfig:
    config = PretrainedConfig(
        architectures=["D1OmniModel"],
        model_type="d1_omni",
        tie_word_embeddings=False,
        dtype="float32",
        auto_map={"AutoModel": "modeling_d1.D1OmniModel"},
        max_length=128,
        head_layers=2,
        projector_hidden_size=128,
        temperatures={"choice:2": 1.7},
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 192,
            "layer_types": ["conv", "full_attention"],
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 64,
            "conv_L_cache": 3,
            "conv_bias": False,
        },
        vision_config={"hidden_size": 64},
        audio_config={"d_model": 64},
    )
    # Published Omni defines this as model context, while HF's base config drops generation fields.
    config.max_length = 128
    return config


def test_registered_noncausal_architecture_dispatch(omni_config: PretrainedConfig) -> None:
    assert AutoBridge.supports(omni_config)
    bridge = AutoBridge.from_hf_config(omni_config)
    assert isinstance(bridge._model_bridge, D1OmniBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(provider, D1OmniModelProvider)
    assert provider.hidden_size == 64
    assert provider.ffn_hidden_size == 128
    assert provider.num_layers == 2
    assert provider.num_attention_heads == 4
    assert provider.num_query_groups == 2
    assert provider.rotary_base == 1000000.0
    assert provider.d1_layer_types == ("conv", "full_attention")
    assert provider.d1_head_layers == 2
    assert provider.d1_max_length == 128
    assert provider.d1_projector_hidden_size == 128
    assert provider.params_dtype == torch.float32
    assert provider.share_embeddings_and_output_weights is False
    assert _resolve_pretrained_wrapper_cls(omni_config) is PreTrainedAutoModel
    exported = D1OmniBridge.megatron_to_hf_config(provider)
    assert exported["text_config"]["rope_theta"] == 1000000.0
    assert exported["auto_map"] == omni_config.auto_map
    assert exported["temperatures"] == omni_config.temperatures


def test_unregistered_noncausal_architecture_is_rejected() -> None:
    assert not AutoBridge.supports(PretrainedConfig(architectures=["UnknownDecisionModel"]))


def test_remote_auto_model_does_not_override_explicit_causal_loader() -> None:
    config = SimpleNamespace(
        architectures=["SomeForCausalLM"],
        auto_map={"AutoModel": "module.Base", "AutoModelForCausalLM": "module.SomeForCausalLM"},
    )
    assert _resolve_pretrained_wrapper_cls(config) is PreTrainedCausalLM


def test_unsupported_embedding_tying_is_rejected(omni_config: PretrainedConfig) -> None:
    omni_config.tie_word_embeddings = True
    with pytest.raises(ValueError, match="independent marker head"):
        AutoBridge.from_hf_config(omni_config).to_megatron_provider(load_weights=False)
