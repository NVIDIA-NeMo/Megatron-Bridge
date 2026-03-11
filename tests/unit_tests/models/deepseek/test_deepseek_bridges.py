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

"""
Unit tests for DeepSeek bridges.
"""

from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.deepseek.deepseek_v2_bridge import DeepSeekV2Bridge
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


class TestDeepSeekV2Bridge:
    """Test cases for DeepSeekV2Bridge."""

    @pytest.fixture
    def ds_v2_config(self):
        return {
            "architectures": ["DeepseekV2ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_deepseek.DeepseekV2Config",
                "AutoModel": "modeling_deepseek.DeepseekV2Model",
                "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM",
            },
            "aux_loss_alpha": 0.001,
            "bos_token_id": 100000,
            "eos_token_id": 100001,
            "first_k_dense_replace": 1,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "kv_lora_rank": 512,
            "max_position_embeddings": 163840,
            "model_type": "deepseek_v2",
            "moe_intermediate_size": 1536,
            "moe_layer_freq": 1,
            "n_group": 8,
            "n_routed_experts": 160,
            "n_shared_experts": 2,
            "norm_topk_prob": False,
            "num_attention_heads": 128,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 60,
            "num_key_value_heads": 128,
            "pretraining_tp": 1,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 0.707,
                "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            "rope_theta": 10000,
            "routed_scaling_factor": 16.0,
            "scoring_func": "softmax",
            "seq_aux": True,
            "tie_word_embeddings": False,
            "topk_group": 3,
            "topk_method": "group_limited_greedy",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.39.3",
            "use_cache": True,
            "v_head_dim": 128,
            "vocab_size": 102400,
        }

    @pytest.fixture
    def mock_pretrained_v2(self, ds_v2_config):
        # Use spec to prevent Mock from auto-creating undefined attributes
        cfg = Mock(spec=list(ds_v2_config.keys()))
        for k, v in ds_v2_config.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)
        return m

    def test_registration(self):
        assert issubclass(DeepSeekV2Bridge, MegatronModelBridge)

    def test_provider_bridge_maps_config(self, mock_pretrained_v2):
        bridge = DeepSeekV2Bridge()
        provider = bridge.provider_bridge(mock_pretrained_v2)
        assert isinstance(provider, MLAModelProvider)
        assert provider.hidden_size == mock_pretrained_v2.config.hidden_size
        assert provider.num_attention_heads == mock_pretrained_v2.config.num_attention_heads
        assert provider.ffn_hidden_size == mock_pretrained_v2.config.intermediate_size
        assert provider.vocab_size == mock_pretrained_v2.config.vocab_size
        assert provider.layernorm_epsilon == mock_pretrained_v2.config.rms_norm_eps
        assert provider.rotary_base == mock_pretrained_v2.config.rope_theta
        assert provider.moe_aux_loss_coeff == mock_pretrained_v2.config.aux_loss_alpha
        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_hf_config_to_provider_kwargs_preserves_none_q_lora_rank(self, mock_pretrained_v2):
        mock_pretrained_v2.config.q_lora_rank = None
        bridge = DeepSeekV2Bridge()

        provider_kwargs = bridge.hf_config_to_provider_kwargs(mock_pretrained_v2.config)

        assert "q_lora_rank" in provider_kwargs
        assert provider_kwargs["q_lora_rank"] is None

    def test_provider_bridge_preserves_none_q_lora_rank(self, mock_pretrained_v2):
        mock_pretrained_v2.config.q_lora_rank = None
        bridge = DeepSeekV2Bridge()

        provider = bridge.provider_bridge(mock_pretrained_v2)

        assert provider.q_lora_rank is None

    def test_megatron_to_hf_config_preserves_none_q_lora_rank(self, mock_pretrained_v2):
        mock_pretrained_v2.config.q_lora_rank = None
        bridge = DeepSeekV2Bridge()
        provider = bridge.provider_bridge(mock_pretrained_v2)

        hf_config = bridge.megatron_to_hf_config(provider)

        assert "q_lora_rank" in hf_config
        assert hf_config["q_lora_rank"] is None

    def test_hf_config_to_provider_kwargs_nested_dot_notation(self, mock_pretrained_v2):
        """Test that dot-notation CONFIG_MAPPING reads nested dict values (including None)."""
        bridge = DeepSeekV2Bridge()
        # Patch CONFIG_MAPPING with a dot-notation entry pointing into rope_scaling dict
        original = bridge.CONFIG_MAPPING
        bridge.CONFIG_MAPPING = list(original) + [("rope_scaling.factor", "yarn_rotary_scaling_factor")]
        mock_pretrained_v2.config.rope_scaling = {"factor": 40, "type": "yarn"}

        kwargs = bridge.hf_config_to_provider_kwargs(mock_pretrained_v2.config)

        bridge.CONFIG_MAPPING = original
        assert kwargs.get("yarn_rotary_scaling_factor") == 40

    def test_hf_config_to_provider_kwargs_nested_dot_notation_none_value(self, mock_pretrained_v2):
        """Test that dot-notation CONFIG_MAPPING preserves None values from nested dicts."""
        bridge = DeepSeekV2Bridge()
        original = bridge.CONFIG_MAPPING
        bridge.CONFIG_MAPPING = list(original) + [("rope_scaling.factor", "yarn_rotary_scaling_factor")]
        mock_pretrained_v2.config.rope_scaling = {"factor": None, "type": "yarn"}

        kwargs = bridge.hf_config_to_provider_kwargs(mock_pretrained_v2.config)

        bridge.CONFIG_MAPPING = original
        assert "yarn_rotary_scaling_factor" in kwargs
        assert kwargs["yarn_rotary_scaling_factor"] is None

    def test_megatron_to_hf_config_yarn_none_value(self, mock_pretrained_v2):
        """Test that YARN_ROPE_SCALING_MAPPING preserves None values on provider."""
        bridge = DeepSeekV2Bridge()
        provider = bridge.provider_bridge(mock_pretrained_v2)
        # Ensure YARN rope_scaling block is emitted
        provider.yarn_rotary_scaling_factor = 40
        # Set a YARN key to None — should still appear in hf_config["rope_scaling"]
        provider.yarn_mscale = None

        hf_config = bridge.megatron_to_hf_config(provider)

        assert "rope_scaling" in hf_config
        assert "mscale" in hf_config["rope_scaling"]
        assert hf_config["rope_scaling"]["mscale"] is None


class TestDeepSeekV3Bridge:
    """Test cases for DeepSeekV3Bridge."""

    @pytest.fixture
    def ds_v3_config(self):
        return {
            "architectures": ["DeepseekV3ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_deepseek.DeepseekV3Config",
                "AutoModel": "modeling_deepseek.DeepseekV3Model",
                "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
            },
            "bos_token_id": 0,
            "eos_token_id": 1,
            "ep_size": 1,
            "first_k_dense_replace": 3,
            "hidden_act": "silu",
            "hidden_size": 7168,
            "initializer_range": 0.02,
            "intermediate_size": 18432,
            "kv_lora_rank": 512,
            "max_position_embeddings": 163840,
            "model_type": "deepseek_v3",
            "moe_intermediate_size": 2048,
            "moe_layer_freq": 1,
            "n_group": 8,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "norm_topk_prob": True,
            "num_attention_heads": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "num_nextn_predict_layers": 1,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            "rope_theta": 10000,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "tie_word_embeddings": False,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "aux_loss_alpha": 0.0001,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.33.1",
            "use_cache": True,
            "v_head_dim": 128,
            "vocab_size": 129280,
        }

    @pytest.fixture
    def mock_pretrained_v3(self, ds_v3_config):
        # Use spec to prevent Mock from auto-creating undefined attributes
        cfg = Mock(spec=list(ds_v3_config.keys()))
        for k, v in ds_v3_config.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)
        return m

    def test_registration(self):
        assert issubclass(DeepSeekV3Bridge, MegatronModelBridge)

    def test_provider_bridge_maps_config(self, mock_pretrained_v3):
        bridge = DeepSeekV3Bridge()
        provider = bridge.provider_bridge(mock_pretrained_v3)
        assert isinstance(provider, MLAModelProvider)
        assert provider.hidden_size == mock_pretrained_v3.config.hidden_size
        assert provider.num_attention_heads == mock_pretrained_v3.config.num_attention_heads
        assert provider.ffn_hidden_size == mock_pretrained_v3.config.intermediate_size
        assert provider.vocab_size == mock_pretrained_v3.config.vocab_size
        assert provider.layernorm_epsilon == mock_pretrained_v3.config.rms_norm_eps
        assert provider.rotary_base == mock_pretrained_v3.config.rope_theta
        assert provider.moe_aux_loss_coeff == mock_pretrained_v3.config.aux_loss_alpha
        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_hf_config_to_provider_kwargs_preserves_none_q_lora_rank(self, mock_pretrained_v3):
        mock_pretrained_v3.config.q_lora_rank = None
        bridge = DeepSeekV3Bridge()

        provider_kwargs = bridge.hf_config_to_provider_kwargs(mock_pretrained_v3.config)

        assert "q_lora_rank" in provider_kwargs
        assert provider_kwargs["q_lora_rank"] is None

    def test_provider_bridge_preserves_none_q_lora_rank(self, mock_pretrained_v3):
        mock_pretrained_v3.config.q_lora_rank = None
        bridge = DeepSeekV3Bridge()

        provider = bridge.provider_bridge(mock_pretrained_v3)

        assert provider.q_lora_rank is None

    def test_megatron_to_hf_config_preserves_none_q_lora_rank(self, mock_pretrained_v3):
        mock_pretrained_v3.config.q_lora_rank = None
        bridge = DeepSeekV3Bridge()
        provider = bridge.provider_bridge(mock_pretrained_v3)

        hf_config = bridge.megatron_to_hf_config(provider)

        assert "q_lora_rank" in hf_config
        assert hf_config["q_lora_rank"] is None

    def test_export_injects_inv_freq_for_layer(self, mock_pretrained_v3):
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = mock_pretrained_v3.config
        mock_pretrained_v3.state = {"model.layers.1.self_attn.rotary_emb.inv_freq": torch.randn(1)}
        task = WeightConversionTask(
            param_name="decoder.layers.0.input_layernorm.weight",
            global_param_name="decoder.layers.0.input_layernorm.weight",
            mapping=Mock(),
        )
        converted = {"model.layers.0.input_layernorm.weight": torch.randn(1)}
        result = bridge.maybe_modify_converted_hf_weight(task, dict(converted), mock_pretrained_v3.state)

        inv_key = "model.layers.0.self_attn.rotary_emb.inv_freq"
        expected = 1.0 / (
            mock_pretrained_v3.config.rope_theta
            ** (
                torch.arange(0, mock_pretrained_v3.config.qk_rope_head_dim, 2, dtype=torch.float32)
                / mock_pretrained_v3.config.qk_rope_head_dim
            )
        )

        assert inv_key in result
        assert torch.allclose(result[inv_key], expected)

    def test_export_skips_inv_freq_for_non_layernorm(self, mock_pretrained_v3):
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = mock_pretrained_v3.config
        mock_pretrained_v3.state = {"model.layers.1.self_attn.rotary_emb.inv_freq": torch.randn(1)}
        task = WeightConversionTask(
            param_name="decoder.final_layernorm.weight",
            global_param_name="decoder.final_layernorm.weight",
            mapping=Mock(),
        )
        converted = {"model.norm.weight": torch.randn(1)}
        result = bridge.maybe_modify_converted_hf_weight(task, dict(converted), mock_pretrained_v3.state)

        inv_key = "model.layers.0.self_attn.rotary_emb.inv_freq"
        assert inv_key not in result

    def test_export_skips_inv_freq_when_not_expected(self, mock_pretrained_v3):
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = mock_pretrained_v3.config
        mock_pretrained_v3.state = {}
        task = WeightConversionTask(
            param_name="decoder.layers.0.input_layernorm.weight",
            global_param_name="decoder.layers.0.input_layernorm.weight",
            mapping=Mock(),
        )
        converted = {"model.layers.0.input_layernorm.weight": torch.randn(1)}
        result = bridge.maybe_modify_converted_hf_weight(task, dict(converted), mock_pretrained_v3.state)

        inv_key = "model.layers.0.self_attn.rotary_emb.inv_freq"
        assert inv_key not in result
