# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import Qwen3VLVisionConfig

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLBridge, Qwen3VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider


def _text_config(*, moe: bool = False):
    config = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1024,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        tie_word_embeddings=False,
        rope_theta=5_000_000.0,
        rope_scaling={"mrope_section": [4, 6, 6]},
        num_nextn_predict_layers=None,
        torch_dtype=torch.bfloat16,
        bos_token_id=1,
        eos_token_id=2,
    )
    if moe:
        config.moe_intermediate_size = 64
        config.num_experts = 8
        config.num_experts_per_tok = 2
        config.decoder_sparse_step = 1
        config.mlp_only_layers = [0]
    return config


def _pretrained(text_config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = SimpleNamespace(
        text_config=text_config,
        vision_config=Qwen3VLVisionConfig(depth=2, hidden_size=32, num_heads=4),
        tie_word_embeddings=False,
        vision_start_token_id=3,
        vision_end_token_id=4,
        image_token_id=5,
        video_token_id=6,
    )
    return pretrained


def _mapping_names(bridge) -> list[str]:
    names = []
    for mapping in bridge.mapping_registry().mappings:
        megatron_param = getattr(mapping, "megatron_param", None)
        if megatron_param is not None:
            names.append(str(megatron_param))
    return names


def test_dense_provider_uses_two_physical_layers_per_block():
    provider = Qwen3VLBridge().provider_bridge(_pretrained(_text_config()))

    assert isinstance(provider, Qwen3VLModelProvider)
    assert provider.num_layers == 4
    assert provider.hybrid_layer_pattern == "*-*-"


def test_moe_provider_preserves_dense_layer_placement():
    provider = Qwen3VLMoEBridge().provider_bridge(_pretrained(_text_config(moe=True)))

    assert isinstance(provider, Qwen3VLMoEModelProvider)
    assert provider.num_layers == 4
    assert provider.hybrid_layer_pattern == "*-*E"


def test_dense_mappings_use_physical_attention_and_mlp_indices():
    previous = Qwen3VLBridge.hf_config
    Qwen3VLBridge.hf_config = SimpleNamespace(text_config=_text_config())
    try:
        names = _mapping_names(Qwen3VLBridge())
    finally:
        Qwen3VLBridge.hf_config = previous

    assert "language_model.decoder.layers.0.self_attention.linear_qkv.weight" in names
    assert "language_model.decoder.layers.1.mlp.linear_fc1.weight" in names
    assert "language_model.decoder.layers.2.self_attention.linear_qkv.weight" in names
    assert "language_model.decoder.layers.3.mlp.linear_fc1.weight" in names
    assert "language_model.decoder.final_norm.weight" in names


def test_moe_mappings_use_mlp_symbol_for_each_logical_layer():
    previous = Qwen3VLMoEBridge.hf_config
    Qwen3VLMoEBridge.hf_config = SimpleNamespace(text_config=_text_config(moe=True))
    try:
        names = _mapping_names(Qwen3VLMoEBridge())
    finally:
        Qwen3VLMoEBridge.hf_config = previous

    assert "language_model.decoder.layers.1.mlp.linear_fc1.weight" in names
    assert "language_model.decoder.layers.3.mlp.router.weight" in names
