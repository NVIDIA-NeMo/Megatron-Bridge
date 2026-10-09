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

from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from transformers import HYV3Config

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hy_v3.hy_v3_bridge import HYV3Bridge


def _hy_v3_config(**overrides) -> HYV3Config:
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        router_scaling_factor=2.5,
        max_position_embeddings=4096,
        vocab_size=128,
        torch_dtype="bfloat16",
    )
    kwargs.update(overrides)
    config = HYV3Config(**kwargs)
    config.architectures = ["HYV3ForCausalLM"]
    return config


class TestHYV3Bridge:
    def test_bridge_registration(self):
        assert issubclass(HYV3Bridge, MegatronModelBridge)

    def test_provider_bridge_is_inherited_compatibility_only(self):
        assert "provider_bridge" not in HYV3Bridge.__dict__

    def test_conversion_uses_builder_config(self):
        assert HYV3Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_hf_config_to_model_config_uses_direct_mapping(self):
        config = _hy_v3_config()
        bridge = HYV3Bridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            result = bridge.hf_config_to_model_config(config)

        assert isinstance(result, BridgeGPTModelConfig)
        # The builder's default MoE spec is the decoder block spec, so none is configured here.
        assert result.transformer_layer_spec is None
        assert result.moe_layer_freq == [0, 1, 1, 1]
        assert result.moe_shared_expert_intermediate_size == config.moe_intermediate_size * config.num_shared_experts
        assert result.moe_router_topk_scaling_factor == 2.5
        assert result.moe_router_score_function == "sigmoid"
        assert result.moe_router_enable_expert_bias is True
        assert result.qk_layernorm is True
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 is True
        assert result.position_embedding_type == "rope"

    @pytest.mark.parametrize("torch_dtype", ["bfloat16", "float16"])
    def test_model_config_matches_provider_runtime_config(self, torch_dtype):
        config = _hy_v3_config(torch_dtype=torch_dtype)
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        bridge = HYV3Bridge()

        model_config = bridge.hf_config_to_model_config(config)
        with pytest.warns(FutureWarning, match=r"deprecated.*get_model_config.*get_model"):
            provider = bridge.provider_bridge(mock_pretrained)

        provider_fields = {field.name for field in fields(provider)}
        model_config_fields = {field.name for field in fields(model_config)}
        model_config_fields.update(field.name for field in fields(model_config.transformer))
        for field_name in sorted((provider_fields & model_config_fields) - {"transformer_layer_spec"}):
            assert getattr(model_config, field_name) == getattr(provider, field_name), field_name
        # Hy V3 always builds in bf16, whatever the source checkpoint dtype.
        assert model_config.params_dtype == provider.params_dtype == torch.bfloat16
        # The provider keeps its explicit decoder block spec for the dense-first layer pattern.
        assert provider.transformer_layer_spec.func is get_gpt_decoder_block_spec

    def test_megatron_to_hf_config_from_model_config(self):
        model_config = HYV3Bridge().hf_config_to_model_config(_hy_v3_config())

        hf_config = HYV3Bridge.megatron_to_hf_config(model_config)

        assert hf_config["first_k_dense_replace"] == 1
        assert hf_config["num_shared_experts"] == 1
        assert hf_config["router_scaling_factor"] == 2.5
        assert hf_config["moe_intermediate_size"] == 32
