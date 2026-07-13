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

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    GatedMLPMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    QKVMapping,
)
from megatron.bridge.models.falcon_h1.falconh1_bridge import FalconH1Bridge
from megatron.bridge.models.falcon_h1.falconh1_provider import FalconH1ModelProvider


pytestmark = pytest.mark.unit


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Pretrained:
    def __init__(self, config):
        self.config = config


FALCON_H1_500M_CONFIG = {
    "architectures": ["FalconH1ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_in_multiplier": 1.0,
    "attention_out_multiplier": 0.9375,
    "attn_layer_indices": None,
    "embedding_multiplier": 5.656854249492381,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "key_multiplier": 0.39062499999999994,
    "lm_head_multiplier": 0.0390625,
    "mamba_chunk_size": 128,
    "mamba_conv_bias": True,
    "mamba_d_conv": 4,
    "mamba_d_head": 64,
    "mamba_d_state": 128,
    "mamba_expand": 2,
    "mamba_n_groups": 1,
    "mamba_n_heads": 24,
    "mamba_norm_before_gate": False,
    "mamba_proj_bias": False,
    "mamba_rms_norm": False,
    "mamba_use_mlp": True,
    "max_position_embeddings": 16384,
    "mlp_bias": False,
    "mlp_multipliers": [0.8838834764831844, 0.5859375],
    "model_type": "falcon_h1",
    "num_attention_heads": 8,
    "num_hidden_layers": 36,
    "num_key_value_heads": 2,
    "projectors_bias": False,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 100000000000.0,
    "ssm_in_multiplier": 1.25,
    "ssm_multipliers": [0.3535533905932738, 0.25, 0.3535533905932738, 0.5, 0.3535533905932738],
    "ssm_out_multiplier": 0.23570226039551587,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 32784,
}


@pytest.fixture
def falcon_h1_pretrained():
    return _Pretrained(_Config(**FALCON_H1_500M_CONFIG))


def test_bridge_registration_uses_falcon_provider():
    assert issubclass(FalconH1Bridge, MegatronModelBridge)
    assert FalconH1Bridge.PROVIDER_CLASS is FalconH1ModelProvider
    assert FalconH1Bridge.SOURCE_NAME == "FalconH1ForCausalLM"
    assert FalconH1Bridge.MODEL_TYPE == "falcon_h1"


def test_provider_bridge_maps_shared_and_falcon_config(falcon_h1_pretrained):
    provider = FalconH1Bridge().provider_bridge(falcon_h1_pretrained)
    hf_config = falcon_h1_pretrained.config

    assert isinstance(provider, FalconH1ModelProvider)
    assert provider.num_layers == hf_config.num_hidden_layers
    assert provider.hidden_size == hf_config.hidden_size
    assert provider.ffn_hidden_size == hf_config.intermediate_size
    assert provider.num_attention_heads == hf_config.num_attention_heads
    assert provider.num_query_groups == hf_config.num_key_value_heads
    assert provider.kv_channels == hf_config.head_dim
    assert provider.vocab_size == hf_config.vocab_size
    assert provider.seq_length == hf_config.max_position_embeddings
    assert provider.layernorm_epsilon == hf_config.rms_norm_eps
    assert provider.rotary_base == int(hf_config.rope_theta)
    assert provider.make_vocab_size_divisible_by == 16
    assert provider.params_dtype == torch.bfloat16
    assert provider.bf16 is True
    assert provider.fp16 is False

    assert provider.mamba_state_dim == hf_config.mamba_d_state
    assert provider.mamba_head_dim == hf_config.mamba_d_head
    assert provider.mamba_num_heads == hf_config.mamba_n_heads
    assert provider.mamba_num_groups == hf_config.mamba_n_groups
    assert provider.expand == hf_config.mamba_expand
    assert provider.d_conv == hf_config.mamba_d_conv
    assert provider.conv_bias == hf_config.mamba_conv_bias
    assert provider.chunk_size == hf_config.mamba_chunk_size
    assert provider.rmsnorm == hf_config.mamba_rms_norm
    assert provider.norm_before_gate == hf_config.mamba_norm_before_gate


def test_provider_bridge_applies_falcon_h1_defaults(falcon_h1_pretrained):
    provider = FalconH1Bridge().provider_bridge(falcon_h1_pretrained)

    assert provider.position_embedding_type == "rope"
    assert provider.rotary_percent == 1.0
    assert provider.normalization == "RMSNorm"
    assert provider.gated_linear_unit is True
    assert provider.add_bias_linear is False
    assert provider.share_embeddings_and_output_weights is False
    assert provider.falconh1_ratio == 1.0
    assert provider.use_mamba is True
    assert provider.use_attention is True
    assert provider.use_mlp is True


def test_provider_bridge_maps_mup_multipliers(falcon_h1_pretrained):
    provider = FalconH1Bridge().provider_bridge(falcon_h1_pretrained)
    hf_config = falcon_h1_pretrained.config

    assert provider.embedding_multiplier == hf_config.embedding_multiplier
    assert provider.lm_head_multiplier == hf_config.lm_head_multiplier
    assert provider.key_multiplier == hf_config.key_multiplier
    assert provider.attention_in_multiplier == hf_config.attention_in_multiplier
    assert provider.attention_out_multiplier == hf_config.attention_out_multiplier
    assert provider.ssm_in_multiplier == hf_config.ssm_in_multiplier
    assert provider.ssm_out_multiplier == hf_config.ssm_out_multiplier
    assert provider.mlp_multipliers == tuple(hf_config.mlp_multipliers)
    assert provider.ssm_multipliers == tuple(hf_config.ssm_multipliers)


def test_megatron_to_hf_config_preserves_falcon_fields(falcon_h1_pretrained):
    provider = FalconH1Bridge().provider_bridge(falcon_h1_pretrained)

    hf_config = FalconH1Bridge.megatron_to_hf_config(provider)

    assert hf_config["architectures"] == ["FalconH1ForCausalLM"]
    assert hf_config["model_type"] == "falcon_h1"
    assert hf_config["mamba_d_state"] == provider.mamba_state_dim
    assert hf_config["mamba_d_head"] == provider.mamba_head_dim
    assert hf_config["mamba_n_heads"] == provider.mamba_num_heads
    assert hf_config["mamba_n_groups"] == provider.mamba_num_groups
    assert hf_config["mamba_expand"] == provider.expand
    assert hf_config["mamba_d_conv"] == provider.d_conv
    assert hf_config["mamba_conv_bias"] == provider.conv_bias
    assert hf_config["mamba_chunk_size"] == provider.chunk_size
    assert hf_config["mamba_rms_norm"] == provider.rmsnorm
    assert hf_config["mamba_norm_before_gate"] == provider.norm_before_gate
    assert hf_config["mamba_d_ssm"] == provider.mamba_num_heads * provider.mamba_head_dim
    assert hf_config["mamba_proj_bias"] == provider.add_bias_linear
    assert hf_config["mamba_use_mlp"] == provider.use_mlp
    assert hf_config["projectors_bias"] == provider.add_bias_linear
    assert tuple(hf_config["mlp_multipliers"]) == provider.mlp_multipliers
    assert tuple(hf_config["ssm_multipliers"]) == provider.ssm_multipliers


def test_mapping_registry_contains_falcon_h1_weight_families():
    registry = FalconH1Bridge().mapping_registry()
    megatron_params = [str(mapping.megatron_param) for mapping in registry]
    hf_params = [str(mapping.hf_param) for mapping in registry]

    assert "embedding.word_embeddings.weight" in megatron_params
    assert "output_layer.weight" in megatron_params
    assert "decoder.final_norm.weight" in megatron_params
    assert "decoder.layers.*.mamba_mixer.in_proj.weight" in megatron_params
    assert "decoder.layers.*.mamba_mixer.conv1d_weight" in megatron_params
    assert "decoder.layers.*.mamba_mixer.conv1d_bias" in megatron_params
    assert "decoder.layers.*.mamba_mixer.conv1d.weight" in megatron_params
    assert "decoder.layers.*.mamba_mixer.conv1d.bias" in megatron_params
    assert megatron_params.count("decoder.layers.*.norm.weight") == 1
    assert hf_params.count("model.layers.*.input_layernorm.weight") == 1
    assert "decoder.layers.*.mamba_mixer.in_proj.layer_norm_weight" not in megatron_params
    assert "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight" not in megatron_params
    assert "decoder.layers.*.self_attention.linear_qkv.weight" in megatron_params
    assert "decoder.layers.*.mlp.linear_fc1.weight" in megatron_params
    assert "decoder.layers.*.mlp.linear_fc2.weight" in megatron_params

    reverse_weight = registry.hf_to_megatron_lookup("model.layers.0.mamba.conv1d.weight")
    reverse_bias = registry.hf_to_megatron_lookup("model.layers.0.mamba.conv1d.bias")
    assert reverse_weight.megatron_param == "decoder.layers.0.mamba_mixer.conv1d_weight"
    assert reverse_bias.megatron_param == "decoder.layers.0.mamba_mixer.conv1d_bias"

    assert any(isinstance(mapping, MambaInProjMapping) for mapping in registry)
    assert any(isinstance(mapping, MambaConv1dMapping) for mapping in registry)
    assert any(isinstance(mapping, QKVMapping) for mapping in registry)
    assert any(isinstance(mapping, GatedMLPMapping) for mapping in registry)
