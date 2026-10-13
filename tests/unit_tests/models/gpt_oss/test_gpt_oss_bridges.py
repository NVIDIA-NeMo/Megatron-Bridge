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

    def test_megatron_to_hf_config_preserves_attention_bias(self, mock_pretrained):
        bridge = GPTOSSBridge()
        provider = bridge.provider_bridge(mock_pretrained)

        hf_config = bridge.megatron_to_hf_config(provider)

        assert hf_config["attention_bias"] is True


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
