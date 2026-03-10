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

from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mimo.mimo_bridge import MimoBridge


class TestMimoBridge:
    """Test cases for MimoBridge."""

    @pytest.fixture
    def mimo_config(self):
        return {
            "architectures": ["MiMoForCausalLM"],
            "attention_bias": True,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 32768,
            "model_type": "mimo",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "num_nextn_predict_layers": 1,
            "rms_norm_eps": 1e-05,
            "rope_theta": 640000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "vocab_size": 151680,
        }

    @pytest.fixture
    def mock_pretrained_mimo(self, mimo_config):
        cfg = Mock(spec=list(mimo_config.keys()))
        for key, value in mimo_config.items():
            setattr(cfg, key, value)

        model = Mock(spec=PreTrainedCausalLM)
        model.config = cfg
        model.generation_config = Mock(spec=GenerationConfig)
        return model

    def test_registration(self):
        assert issubclass(MimoBridge, MegatronModelBridge)

    def test_provider_bridge_maps_mtp_config(self, mock_pretrained_mimo):
        bridge = MimoBridge()
        provider = bridge.provider_bridge(mock_pretrained_mimo)

        assert isinstance(provider, GPTModelProvider)
        assert provider.hidden_size == mock_pretrained_mimo.config.hidden_size
        assert provider.num_attention_heads == mock_pretrained_mimo.config.num_attention_heads
        assert provider.ffn_hidden_size == mock_pretrained_mimo.config.intermediate_size
        assert provider.vocab_size == mock_pretrained_mimo.config.vocab_size
        assert provider.qk_layernorm is False
        assert provider.add_qkv_bias is True
        assert provider.mtp_num_layers == mock_pretrained_mimo.config.num_nextn_predict_layers
        assert provider.mtp_loss_scaling_factor == 0.1
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_mapping_registry_includes_mtp_paths(self):
        bridge = MimoBridge()
        registry = bridge.mapping_registry()

        mapping = registry.megatron_to_hf_lookup("mtp.layers.0.eh_proj.weight")
        assert mapping is not None
        assert mapping.hf_param == "model.mtp_layers.0.input_proj.weight"

        transformer_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight"
        )
        assert transformer_mapping is not None
        assert transformer_mapping.hf_param["q"] == "model.mtp_layers.0.self_attn.q_proj.weight"

        mtp_model_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"
        )
        assert mtp_model_mapping is not None
        assert mtp_model_mapping.hf_param["q"] == "model.mtp_layers.0.self_attn.q_proj.weight"

    def test_mtp_input_proj_swap_on_hf_load(self):
        bridge = MimoBridge()
        weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        hf_key = "model.mtp_layers.0.input_proj.weight"

        modified = bridge.maybe_modify_loaded_hf_weight(hf_key, {hf_key: weight})

        expected = torch.cat((weight[:, 4:], weight[:, :4]), dim=1)
        assert torch.equal(modified, expected)

    def test_mtp_input_proj_swap_on_hf_export(self):
        bridge = MimoBridge()
        weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)

        task = WeightConversionTask(
            param_name="mtp.layers.0.eh_proj.weight",
            global_param_name="mtp.layers.0.eh_proj.weight",
            mapping=Mock(),
        )
        converted = {"model.mtp_layers.0.input_proj.weight": weight}

        modified = bridge.maybe_modify_converted_hf_weight(task, dict(converted), {})

        expected = torch.cat((weight[:, 4:], weight[:, :4]), dim=1)
        assert torch.equal(modified["model.mtp_layers.0.input_proj.weight"], expected)
