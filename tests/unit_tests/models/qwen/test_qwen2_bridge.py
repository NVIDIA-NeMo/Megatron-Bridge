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
import torch.nn.functional as F
from transformers import Qwen2Config

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen2_bridge import Qwen2Bridge


def _qwen2_config(**overrides) -> Qwen2Config:
    kwargs = dict(
        architectures=["Qwen2ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        max_position_embeddings=32768,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=128,
    )
    kwargs.update(overrides)
    return Qwen2Config(**kwargs)


class TestQwen2Bridge:
    def test_bridge_registration(self):
        assert issubclass(Qwen2Bridge, MegatronModelBridge)

    def test_provider_bridge_is_inherited_compatibility_only(self):
        """Qwen2 should not maintain a separate provider construction path."""
        assert "provider_bridge" not in Qwen2Bridge.__dict__

    def test_conversion_uses_builder_config(self):
        """Checkpoint conversion constructs Qwen2 through its ModelBuilder."""
        assert Qwen2Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_hf_config_to_model_config_uses_direct_mapping(self):
        """The builder config path must not route through the legacy provider."""
        config = _qwen2_config(tie_word_embeddings=True)
        bridge = Qwen2Bridge()

        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(
                bridge,
                "hf_config_to_provider_kwargs",
                side_effect=AssertionError("provider kwargs path used"),
            ),
        ):
            result = bridge.hf_config_to_model_config(config)

        assert isinstance(result, BridgeGPTModelConfig)
        assert result.num_layers == config.num_hidden_layers
        assert result.hidden_size == config.hidden_size
        assert result.ffn_hidden_size == config.intermediate_size
        assert result.num_attention_heads == config.num_attention_heads
        assert result.num_query_groups == config.num_key_value_heads
        assert result.seq_length == config.max_position_embeddings
        assert result.rotary_base == rope_theta_from_hf(config)
        assert result.vocab_size == config.vocab_size
        assert result.share_embeddings_and_output_weights is True
        assert result.layernorm_epsilon == config.rms_norm_eps
        assert result.activation_func is F.silu
        assert result.normalization == "RMSNorm"
        assert result.gated_linear_unit is True
        assert result.add_bias_linear is False
        assert result.add_qkv_bias is True
        assert result.qk_layernorm is False
        assert result.hidden_dropout == 0.0
        assert result.position_embedding_type == "rope"

    @pytest.mark.parametrize(
        ("tie_word_embeddings", "torch_dtype", "rope_scaling"),
        [
            (True, "bfloat16", None),
            (False, "float16", None),
            (
                False,
                "bfloat16",
                {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                },
            ),
        ],
        ids=["tied-bf16", "untied-fp16", "yarn"],
    )
    def test_model_config_matches_provider_runtime_config(self, tie_word_embeddings, torch_dtype, rope_scaling):
        """Builder and provider paths agree on every comparable runtime field."""
        config = _qwen2_config(
            tie_word_embeddings=tie_word_embeddings, torch_dtype=torch_dtype, rope_scaling=rope_scaling
        )
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        bridge = Qwen2Bridge()

        mapped_kwargs = bridge.hf_config_to_model_config_kwargs(config)
        model_config = bridge.hf_config_to_model_config(config)
        with pytest.warns(FutureWarning, match=r"deprecated.*get_model_config.*get_model"):
            provider = bridge.provider_bridge(mock_pretrained)

        provider_fields = {field.name for field in fields(provider)}
        model_config_fields = {field.name for field in fields(model_config)}
        model_config_fields.update(field.name for field in fields(model_config.transformer))
        comparable_fields = sorted((provider_fields & model_config_fields) - {"transformer_layer_spec"})

        assert set(mapped_kwargs) <= set(comparable_fields)
        assert len(comparable_fields) > len(mapped_kwargs)
        for field_name in comparable_fields:
            assert getattr(model_config, field_name) == getattr(provider, field_name), field_name

        # The provider stores its model-construction callable directly; the new
        # config intentionally delegates layer-spec selection to its builder.
        assert model_config.transformer_layer_spec is None
        assert callable(provider.transformer_layer_spec)

    def test_model_config_maps_yarn_to_transformer_config(self):
        """YaRN checkpoints keep YaRN and place its parameters where Megatron Core reads them."""
        config = _qwen2_config(
            max_position_embeddings=131072,
            rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        )

        result = Qwen2Bridge().hf_config_to_model_config(config)

        assert result.position_embedding_type == "yarn"
        assert result.transformer.yarn_rotary_scaling_factor == 4.0
        assert result.transformer.yarn_original_max_position_embeddings == 32768

        hf_config = Qwen2Bridge.megatron_to_hf_config(result)
        assert hf_config["rope_scaling"]["rope_type"] == "yarn"
        assert hf_config["rope_scaling"]["factor"] == 4.0
