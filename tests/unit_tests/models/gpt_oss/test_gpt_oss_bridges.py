#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from megatron.bridge.models.conversion.gtp import _gather_gtp_weight, _slice_gtp_weight
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge, GPTOSSMLPGateUpProjMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestGptOssBridge:
    """Unit tests for GPT-OSS bridge provider mapping."""

    @pytest.fixture
    def gpt_oss_cfg(self):
        return {
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_local_experts": 32,
            "torch_dtype": "bfloat16",
            "vocab_size": 201088,
            "hidden_act": "silu",
            "sliding_window": 4096,
            "attention_bias": True,
        }

    @pytest.fixture
    def mock_pretrained(self, gpt_oss_cfg):
        # Use spec to prevent Mock from auto-creating undefined attributes
        cfg = Mock(spec=list(gpt_oss_cfg.keys()))
        for k, v in gpt_oss_cfg.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock()
        return m

    def test_registration(self):
        assert issubclass(GPTOSSBridge, MegatronModelBridge)

    def test_provider_bridge_maps_config(self, mock_pretrained):
        bridge = GPTOSSBridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, GPTModelProvider)
        # Key fields mapped from HF config
        assert provider.num_layers == mock_pretrained.config.num_hidden_layers
        assert provider.num_moe_experts == mock_pretrained.config.num_local_experts
        assert provider.add_qkv_bias == mock_pretrained.config.attention_bias
        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16
        # The mock config has no rope_scaling; GPT-OSS still uses YaRN.
        assert provider.position_embedding_type == "yarn"

    def test_megatron_to_hf_config_preserves_attention_bias(self, mock_pretrained):
        bridge = GPTOSSBridge()
        provider = bridge.provider_bridge(mock_pretrained)

        hf_config = bridge.megatron_to_hf_config(provider)

        assert hf_config["attention_bias"] is True


def _gpt_oss_config(**overrides):
    from transformers import GptOssConfig

    values = dict(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        sliding_window=32,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    config = GptOssConfig(**values)
    config.architectures = ["GptOssForCausalLM"]
    return config


class TestGptOssBuilderConfig:
    """Builder-backed ModelConfig path for GPT-OSS."""

    def test_provider_bridge_is_inherited_compatibility_only(self):
        assert "provider_bridge" not in GPTOSSBridge.__dict__

    def test_conversion_uses_builder_config(self):
        from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig

        assert GPTOSSBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
        assert GPTOSSBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_hf_config_to_model_config_uses_direct_mapping(self):
        from unittest.mock import patch

        from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
        from megatron.bridge.models.gpt_oss.gpt_oss_bridge import quick_gelu

        config = _gpt_oss_config()
        bridge = GPTOSSBridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(config)

        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.num_moe_experts == config.num_local_experts
        assert model_config.moe_ffn_hidden_size == config.intermediate_size
        assert model_config.activation_func is quick_gelu
        assert model_config.activation_func_clamp_value == 7.0
        assert model_config.glu_linear_offset == 1.0
        assert model_config.softmax_type == "learnable"
        assert model_config.window_size == (config.sliding_window - 1, 0)
        assert model_config.window_attn_skip_freq == 2
        assert model_config.position_embedding_type == "yarn"
        assert model_config.yarn_rotary_scaling_factor == 32.0
        assert model_config.yarn_correction_range_round_to_int is False
        assert model_config.add_qkv_bias is True
        assert model_config.share_embeddings_and_output_weights is False
        assert model_config.params_dtype == torch.bfloat16

    def test_model_config_matches_provider_runtime_config(self):
        from dataclasses import fields

        config = _gpt_oss_config()
        bridge = GPTOSSBridge()
        model_config = bridge.hf_config_to_model_config(config)
        pretrained = Mock(spec=PreTrainedCausalLM)
        pretrained.config = config
        with pytest.warns(FutureWarning, match=r"deprecated.*get_model_config.*get_model"):
            provider = bridge.provider_bridge(pretrained)

        model_config_fields = {field.name for field in fields(model_config)}
        model_config_fields.update(field.name for field in fields(model_config.transformer))
        shared = ({field.name for field in fields(provider)} & model_config_fields) - {"transformer_layer_spec"}
        for field_name in sorted(shared):
            assert getattr(model_config, field_name) == getattr(provider, field_name), field_name
        assert model_config.transformer_layer_spec is None

    @pytest.mark.parametrize(
        "rope_scaling",
        [
            "missing",
            None,
            {},
            {"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 4096, "truncate": False},
        ],
        ids=["missing", "none", "empty", "hf_yarn"],
    )
    def test_both_paths_use_yarn_regardless_of_hf_rope_scaling(self, rope_scaling):
        config = SimpleNamespace(
            architectures=["GptOssForCausalLM"],
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=128,
            max_position_embeddings=4096,
            sliding_window=32,
            hidden_act="silu",
            attention_bias=True,
            torch_dtype="bfloat16",
        )
        if rope_scaling != "missing":
            config.rope_scaling = rope_scaling
        bridge = GPTOSSBridge()
        model_config = bridge.hf_config_to_model_config(config)
        pretrained = Mock(spec=PreTrainedCausalLM)
        pretrained.config = config
        with pytest.warns(FutureWarning):
            provider = bridge.provider_bridge(pretrained)

        for built in (model_config, provider):
            assert built.position_embedding_type == "yarn"
            assert built.yarn_rotary_scaling_factor == 32.0
            assert built.yarn_original_max_position_embeddings == 4096

    def test_megatron_to_hf_config_from_model_config(self):
        model_config = GPTOSSBridge().hf_config_to_model_config(_gpt_oss_config())

        hf_config = GPTOSSBridge.megatron_to_hf_config(model_config)

        assert hf_config["attention_bias"] is True
        assert hf_config["num_hidden_layers"] == 2


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("etp_size", [1, 2])
@pytest.mark.parametrize("gtp_size", [1, 2])
def test_gate_up_mapping_preserves_expert_rows_across_etp_and_gtp(monkeypatch, bias, etp_size, gtp_size):
    """Distinct gate/up rows detect a transpose or a split of the fused TP matrix."""
    gate = torch.arange(4 if bias else 12).reshape((4,) if bias else (4, 3)).float() + 10
    up = gate + 100
    interleaved = torch.empty((8,) if bias else (8, 3))
    interleaved[::2], interleaved[1::2] = gate, up
    hf_expert = interleaved if bias else interleaved.t()
    # Import expert 1, so extracting the wrong expert is also observable.
    hf_weights = torch.stack((hf_expert + 1000, hf_expert))
    hf_name = "model.layers.0.mlp.experts.gate_up_proj" + ("_bias" if bias else "")
    param_name = "bias0" if bias else "weight0"
    megatron_name = "decoder.layers.0.mlp.experts.linear_fc1." + ("bias1" if bias else "weight1")
    local_rows = 8 // etp_size
    mapping_path = "megatron.bridge.models.conversion.param_mapping"
    monkeypatch.setattr(f"{mapping_path}.get_pg_size", lambda group: group.size() if group is not None else 1)
    monkeypatch.setattr(f"{mapping_path}.get_pg_rank", lambda group: group.rank() if group is not None else 0)
    expected_tp = [torch.cat((g, u)) for g, u in zip(gate.chunk(etp_size), up.chunk(etp_size))]

    for etp_rank in range(etp_size):
        module = torch.nn.Module()
        padding = 2 if gtp_size > 1 else 0
        shape = ((local_rows + padding) // gtp_size, *gate.shape[1:])
        parameter = torch.nn.Parameter(torch.zeros(shape))
        if gtp_size > 1:
            parameter.is_gtp_weight_remat = True
            parameter.pad_length = padding
            parameter.group = SimpleNamespace(size=lambda: gtp_size, rank=lambda: 1)
        module.register_parameter(param_name, parameter)
        mapping = GPTOSSMLPGateUpProjMapping(megatron_name, hf_name)
        group = SimpleNamespace(size=lambda: etp_size, rank=lambda: etp_rank)
        mapping.set_process_groups_from_pg_collection(SimpleNamespace(expt_tp=group))
        delegate = mapping._gated_mapping

        def scatter(splits, output_shape, dtype, device):
            assert output_shape == expected_tp[etp_rank].shape
            if splits is not None:
                assert all(torch.equal(actual, expected) for actual, expected in zip(splits, expected_tp))
            return expected_tp[etp_rank].clone()

        monkeypatch.setattr(delegate, "scatter_to_tp_ranks", scatter)
        converted = mapping.hf_to_megatron(hf_weights, module)
        assert torch.equal(converted, expected_tp[etp_rank])
        local = _slice_gtp_weight(converted, parameter)
        if gtp_size > 1:
            padded = torch.cat((expected_tp[etp_rank], torch.zeros((padding, *gate.shape[1:]))))
            assert torch.equal(local, padded[shape[0] :])
            with torch.no_grad():
                parameter.copy_(local)
            monkeypatch.setattr(
                torch.distributed, "all_gather_into_tensor", lambda output, local, **kwargs: output.copy_(padded)
            )
            converted = _gather_gtp_weight(parameter)

        monkeypatch.setattr(delegate, "broadcast_from_pp_rank", lambda weight, **kwargs: weight)
        monkeypatch.setattr(delegate, "gather_from_tp_ranks", lambda weight: expected_tp)
        exported = mapping.megatron_to_hf(converted, module)
        assert torch.equal(exported[hf_name], hf_expert)
