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

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from transformers import ApertusConfig, ApertusForCausalLM

from megatron.bridge.models.apertus import apertus_bridge
from megatron.bridge.models.apertus.apertus_bridge import ApertusBridge, ApertusMLP
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit

XIELU_PARAMS = ("alpha_p", "alpha_n", "beta", "eps")


def _apertus_8b_config(**overrides) -> ApertusConfig:
    """Apertus-8B shape with a small layer count."""
    kwargs = dict(
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=21504,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="xielu",
        max_position_embeddings=65536,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        attention_bias=False,
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 12000000.0,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
        torch_dtype="bfloat16",
    )
    kwargs.update(overrides)
    return ApertusConfig(**kwargs)


def _pretrained(config) -> Mock:
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    pretrained.generation_config = Mock()
    pretrained.model = Mock(spec=ApertusForCausalLM)
    return pretrained


class TestApertusBridge:
    def test_provider_matches_hf_config(self):
        provider = ApertusBridge().provider_bridge(_pretrained(_apertus_8b_config()))

        assert provider.num_layers == 2
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 21504
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.normalization == "RMSNorm"
        assert provider.qk_layernorm is True
        assert provider.gated_linear_unit is False
        assert provider.use_te_activation_func is False
        assert provider.bias_activation_fusion is False
        assert provider.add_bias_linear is False
        assert provider.rotary_base == 12000000.0
        assert provider.rope_scaling is True
        assert provider.rope_scaling_factor == 8.0
        assert provider.transformer_layer_spec is apertus_bridge.get_apertus_decoder_block_spec

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"hidden_act": "silu"}, "hidden_act='xielu'"),
            ({"attention_bias": True}, "attention_bias=True"),
        ],
    )
    def test_provider_rejects_unsupported_configs(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            ApertusBridge().provider_bridge(_pretrained(_apertus_8b_config(**overrides)))

    def test_hf_config_round_trip_keeps_xielu_and_rope(self):
        bridge = ApertusBridge()
        provider = bridge.provider_bridge(_pretrained(_apertus_8b_config()))
        hf_config = bridge.megatron_to_hf_config(provider)

        assert hf_config["hidden_act"] == "xielu"
        assert hf_config["attention_bias"] is False
        assert hf_config["rope_theta"] == 12000000.0
        assert hf_config["rope_scaling"]["rope_type"] == "llama3"
        assert hf_config["rope_scaling"]["factor"] == 8.0
        assert hf_config["rope_parameters"]["rope_theta"] == 12000000.0

    def test_activation_name_round_trips(self):
        activation = ApertusBridge.hf_to_megatron_activation("xielu")
        assert activation is apertus_bridge.xielu_placeholder
        assert ApertusBridge.megatron_to_hf_activation(activation) == "xielu"
        assert ApertusBridge.megatron_to_hf_activation(torch.nn.functional.silu) == "silu"

    @pytest.mark.parametrize("name", XIELU_PARAMS)
    def test_xielu_parameters_are_replicated(self, name):
        mapping = (
            ApertusBridge().mapping_registry().megatron_to_hf_lookup(f"decoder.layers.3.mlp.activation_func.{name}")
        )
        assert isinstance(mapping, ReplicatedMapping)
        assert mapping.hf_param == f"model.layers.3.mlp.act_fn.{name}"

    def test_mlp_has_no_gate_projection(self):
        mapping = ApertusBridge().mapping_registry().megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc1.weight")
        assert mapping.hf_param == "model.layers.0.mlp.up_proj.weight"


class _Scaled(torch.nn.Module):
    """Stands in for MCoreXIELU: an activation with a learnable parameter."""

    def __init__(self, *, config):
        super().__init__()
        self.alpha_p = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return self.alpha_p * x


class TestApertusMLP:
    @pytest.fixture
    def light_mlp_init(self, monkeypatch):
        """Replace MLP.__init__ with the part ApertusMLP builds on: config and activation_func."""

        def init(self, config, submodules, *args, **kwargs):
            torch.nn.Module.__init__(self)
            self.config = config
            self.activation_func = config.activation_func

        monkeypatch.setattr(MLP, "__init__", init)

    def test_builds_activation_module(self, light_mlp_init):
        config = SimpleNamespace(activation_func=torch.nn.functional.gelu)
        mlp = ApertusMLP(config, MLPSubmodules(linear_fc1=None, linear_fc2=None, activation_func=_Scaled))

        assert isinstance(mlp.activation_func, _Scaled)
        assert "activation_func.alpha_p" in dict(mlp.named_parameters())
        assert "activation_func" in mlp._modules

    def test_without_activation_module_keeps_config_function(self, light_mlp_init):
        config = SimpleNamespace(activation_func=torch.nn.functional.gelu)
        mlp = ApertusMLP(config, MLPSubmodules(linear_fc1=None, linear_fc2=None))

        assert mlp.activation_func is torch.nn.functional.gelu

    def test_block_spec_swaps_in_apertus_mlp(self, monkeypatch):
        fc1, fc2 = object(), object()

        def layer():
            attention = SimpleNamespace(submodules=SimpleNamespace(q_layernorm=None, k_layernorm=None))
            mlp = partial(MLP.as_mlp_submodule, submodules=MLPSubmodules(linear_fc1=fc1, linear_fc2=fc2))
            return SimpleNamespace(submodules=SimpleNamespace(self_attention=attention, mlp=mlp))

        block = SimpleNamespace(layer_specs=[layer(), layer()])
        monkeypatch.setattr(apertus_bridge, "get_gpt_decoder_block_spec", lambda *args, **kwargs: block)

        spec = apertus_bridge.get_apertus_decoder_block_spec(SimpleNamespace())

        for layer_spec in spec.layer_specs:
            mlp = layer_spec.submodules.mlp
            assert mlp.func == ApertusMLP.as_mlp_submodule
            submodules = mlp.keywords["submodules"]
            assert submodules.activation_func is apertus_bridge.MCoreXIELU
            assert submodules.linear_fc1 is fc1 and submodules.linear_fc2 is fc2
            norm = layer_spec.submodules.self_attention.submodules.q_layernorm(config=None, hidden_size=8)
            assert isinstance(norm, torch.nn.RMSNorm)
