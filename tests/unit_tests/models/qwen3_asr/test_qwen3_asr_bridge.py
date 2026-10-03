# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from megatron.bridge.models.qwen3_asr.hf_qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRThinkerConfig,
)
from megatron.bridge.models.qwen3_asr.qwen3_asr_bridge import Qwen3ASRBridge
from megatron.bridge.models.qwen3_asr.qwen3_asr_provider import Qwen3ASRModelProvider


pytestmark = pytest.mark.unit


@pytest.fixture
def thinker_dict():
    return {
        "dtype": "bfloat16",
        "audio_token_id": 151676,
        "audio_start_token_id": 151669,
        "audio_config": {
            "d_model": 128,
            "encoder_layers": 2,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 256,
            "output_dim": 128,
        },
        "text_config": {
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "initializer_range": 0.01,
            "rope_theta": 1000000,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 4, 4]},
            "tie_word_embeddings": True,
        },
    }


@pytest.mark.parametrize("form", ["dict", "mapping", "object", "auto_config"])
def test_provider_preserves_legacy_thinker_config(thinker_dict, form):
    original = deepcopy(thinker_dict)
    if form == "object":
        thinker = Qwen3ASRThinkerConfig(**deepcopy(thinker_dict))
        config = Qwen3ASRConfig(thinker_config=thinker)
    elif form == "auto_config":
        # Native Transformers uses a flat schema and leaves this legacy field as a dict.
        config = AutoConfig.for_model("qwen3_asr", thinker_config=deepcopy(thinker_dict))
    else:
        thinker = MappingProxyType(thinker_dict) if form == "mapping" else thinker_dict
        config = SimpleNamespace(thinker_config=thinker)

    provider = Qwen3ASRBridge().provider_bridge(SimpleNamespace(config=config))

    assert isinstance(provider, Qwen3ASRModelProvider)
    assert isinstance(provider.thinker_config, Qwen3ASRThinkerConfig)
    assert provider.thinker_config.audio_config.d_model == 128
    assert provider.thinker_config.audio_config.output_dim == 128
    assert provider.num_layers == 2
    assert provider.hidden_size == 128
    assert provider.ffn_hidden_size == 256
    assert provider.num_attention_heads == 4
    assert provider.num_query_groups == 2
    assert provider.kv_channels == 32
    assert provider.vocab_size == 512
    assert provider.seq_length == 1024
    assert provider.layernorm_epsilon == 1e-6
    assert provider.init_method_std == 0.01
    assert provider.rotary_base == 1000000
    assert provider.mrope_section == [8, 4, 4]
    assert provider.share_embeddings_and_output_weights
    assert provider.params_dtype == torch.bfloat16
    assert provider.bf16 and not provider.fp16
    assert provider.qk_layernorm and not provider.add_qkv_bias
    assert provider.audio_token_id == 151676
    assert provider.audio_start_token_id == 151669
    assert thinker_dict == original
    if form == "object":
        assert provider.thinker_config is thinker


@pytest.mark.parametrize("value", [None, [], "invalid", 42])
def test_provider_rejects_invalid_thinker_config(value):
    with pytest.raises(ValueError, match="thinker_config"):
        Qwen3ASRBridge().provider_bridge(SimpleNamespace(config=SimpleNamespace(thinker_config=value)))


def test_provider_rejects_missing_thinker_config():
    with pytest.raises(ValueError, match="thinker_config"):
        Qwen3ASRBridge().provider_bridge(SimpleNamespace(config=SimpleNamespace()))


@pytest.mark.parametrize("field", ["text_config", "audio_config"])
@pytest.mark.parametrize("value", [None, [], "invalid", {}])
def test_provider_rejects_malformed_nested_config(thinker_dict, field, value):
    thinker_dict[field] = value
    with pytest.raises(ValueError, match=field):
        Qwen3ASRBridge().provider_bridge(SimpleNamespace(config=SimpleNamespace(thinker_config=thinker_dict)))


@pytest.mark.parametrize("field", ["text_config", "audio_config"])
def test_provider_rejects_missing_nested_config(thinker_dict, field):
    del thinker_dict[field]
    with pytest.raises(ValueError, match=field):
        Qwen3ASRBridge().provider_bridge(SimpleNamespace(config=SimpleNamespace(thinker_config=thinker_dict)))
