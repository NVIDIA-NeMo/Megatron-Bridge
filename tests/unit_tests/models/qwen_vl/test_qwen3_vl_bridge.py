# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for Qwen3-VL Dense and MoE provider conversion."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLBridge, Qwen3VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider


pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[
        pytest.param((Qwen3VLBridge, Qwen3VLModelProvider), id="dense"),
        pytest.param((Qwen3VLMoEBridge, Qwen3VLMoEModelProvider), id="moe"),
    ]
)
def bridge_types(request):
    return request.param


@pytest.fixture
def mock_pretrained(bridge_types):
    text_kwargs = dict(
        num_hidden_layers=3,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        attention_bias=True,
        hidden_act="silu",
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [2, 3, 3]},
        bos_token_id=11,
        eos_token_id=12,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )
    if bridge_types[0] is Qwen3VLMoEBridge:
        text_kwargs.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64)
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = SimpleNamespace(
        text_config=SimpleNamespace(**text_kwargs),
        vision_config=Qwen3VLVisionConfig(),
        tie_word_embeddings=False,
        vision_start_token_id=21,
        vision_end_token_id=22,
        image_token_id=23,
        video_token_id=24,
    )
    return pretrained


class TestQwen3VLBridgeProviderBridge:
    def test_provider_bridge_basic_config(self, bridge_types, mock_pretrained):
        bridge_class, provider_class = bridge_types

        provider = bridge_class().provider_bridge(mock_pretrained)

        assert isinstance(provider, provider_class)
        assert provider.num_layers == 3
        assert provider.hidden_size == 64
        assert provider.ffn_hidden_size == 128
        assert provider.num_attention_heads == 4
        assert provider.num_query_groups == 2
        assert provider.vocab_size == 256
        assert provider.seq_length == 128
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.activation_func is torch.nn.functional.silu
        assert provider.add_qkv_bias is True
        assert provider.add_bias_linear is False
        assert provider.qk_layernorm is True
        assert provider.hidden_dropout == 0.0
        if provider_class is Qwen3VLMoEModelProvider:
            assert provider.num_moe_experts == 4
            assert provider.moe_router_topk == 2
            assert provider.moe_ffn_hidden_size == 64

    def test_provider_bridge_vision_config_and_tokens(self, bridge_types, mock_pretrained):
        provider = bridge_types[0]().provider_bridge(mock_pretrained)

        assert provider.vision_config is mock_pretrained.config.vision_config
        assert provider.hf_text_config is mock_pretrained.config.text_config
        assert provider.position_embedding_type == "mrope"
        assert provider.head_dim == 16
        assert provider.bos_token_id == 11
        assert provider.eos_token_id == 12
        assert provider.vision_start_token_id == 21
        assert provider.vision_end_token_id == 22
        assert provider.image_token_id == 23
        assert provider.video_token_id == 24

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_provider_bridge_dtype(self, bridge_types, mock_pretrained, dtype):
        mock_pretrained.config.text_config.torch_dtype = dtype

        provider = bridge_types[0]().provider_bridge(mock_pretrained)

        assert provider.params_dtype is dtype
        assert provider.vision_config.torch_dtype is dtype
        assert provider.fp16 is (dtype is torch.float16)
        assert provider.bf16 is (dtype is torch.bfloat16)

    @pytest.mark.parametrize("tie_embeddings", [False, True])
    def test_provider_bridge_uses_top_level_embedding_sharing(self, bridge_types, mock_pretrained, tie_embeddings):
        mock_pretrained.config.tie_word_embeddings = tie_embeddings
        mock_pretrained.config.text_config.tie_word_embeddings = not tie_embeddings

        provider = bridge_types[0]().provider_bridge(mock_pretrained)

        assert provider.share_embeddings_and_output_weights is tie_embeddings

    @pytest.mark.parametrize("legacy_rope_config", [False, True])
    def test_provider_bridge_rope_settings(self, bridge_types, mock_pretrained, legacy_rope_config):
        if legacy_rope_config:
            text_config = mock_pretrained.config.text_config
            text_config.rope_theta = 10000.0
            text_config.rope_scaling = {"mrope_section": [2, 3, 3]}
            del text_config.rope_parameters

        provider = bridge_types[0]().provider_bridge(mock_pretrained)

        assert provider.rotary_base == 10000.0
        assert provider.mrope_section == [2, 3, 3]
