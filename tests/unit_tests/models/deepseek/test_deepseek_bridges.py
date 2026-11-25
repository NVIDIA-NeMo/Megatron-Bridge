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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import DeepseekV3Config, GenerationConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.utils import carry_rope_into_rope_parameters, conform_config_to_reference
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


class TestDeepSeekV3Bridge:
    """Test cases for DeepSeekV3Bridge."""

    def test_generate_pipeline_layout_balances_uneven_decoder_layers(self):
        layout = DeepSeekV3Bridge.generate_pipeline_layout(num_layers=7, pp=3, mtp_layers=2)

        assert layout == [
            ["embedding", "decoder", "decoder", "decoder"],
            ["decoder", "decoder"],
            ["decoder", "decoder", "mtp", "mtp", "loss"],
        ]

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
        assert provider.moe_aux_loss_coeff == 0.0001
        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_provider_bridge_preserves_model_specific_context_and_aux_loss(self, mock_pretrained_v3):
        mock_pretrained_v3.config.max_position_embeddings = 8192
        mock_pretrained_v3.config.aux_loss_alpha = 0.001
        bridge = DeepSeekV3Bridge()

        provider = bridge.provider_bridge(mock_pretrained_v3)
        exported = bridge.megatron_to_hf_config(provider)

        assert provider.seq_length == 8192
        assert provider.moe_aux_loss_coeff == 0.001
        assert exported["max_position_embeddings"] == 8192
        assert exported["aux_loss_alpha"] == 0.001

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

    @pytest.mark.parametrize(
        ("rope_scaling", "expected"),
        [
            (
                {
                    "type": "yarn",
                    "factor": 8.0,
                    "original_max_position_embeddings": 20480,
                    "beta_fast": 16.0,
                    "beta_slow": 2.0,
                    "mscale": 0.5,
                    "mscale_all_dim": 0.75,
                },
                {
                    "rope_type": "yarn",
                    "factor": 8.0,
                    "original_max_position_embeddings": 20480,
                    "beta_fast": 16.0,
                    "beta_slow": 2.0,
                    "mscale": 0.5,
                    "mscale_all_dim": 0.75,
                },
            ),
            (None, {"rope_type": "default"}),
        ],
        ids=["other_yarn_factor", "unscaled"],
    )
    def test_config_export_keeps_the_model_mla_rope_scaling(
        self, ds_v3_config, mock_pretrained_v3, rope_scaling, expected
    ):
        """A config export conformed to the factor-40 YaRN reference keeps the model's own RoPE scaling."""
        reference = DeepseekV3Config(**ds_v3_config)
        mock_pretrained_v3.config.rope_scaling = rope_scaling
        bridge = DeepSeekV3Bridge()
        provider = bridge.provider_bridge(mock_pretrained_v3)

        exported = bridge.megatron_to_hf_config(provider)
        conformed = carry_rope_into_rope_parameters(
            exported, conform_config_to_reference(exported, reference.to_dict())
        )
        rope_parameters = DeepseekV3Config(**conformed).rope_parameters

        assert exported["rope_scaling"] == expected
        assert {key: rope_parameters[key] for key in expected} == expected
        assert rope_parameters["rope_theta"] == ds_v3_config["rope_theta"]
        if rope_scaling is None:
            assert rope_parameters == {"rope_type": "default", "rope_theta": ds_v3_config["rope_theta"]}

    def test_export_does_not_synthesize_inv_freq_from_another_layer(self, mock_pretrained_v3):
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
        assert inv_key not in result

    def test_export_preserves_source_inv_freq_for_layer(self, mock_pretrained_v3):
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = mock_pretrained_v3.config
        inv_key = "model.layers.0.self_attn.rotary_emb.inv_freq"
        source_inv_freq = torch.linspace(1.0, 0.0, 64, dtype=torch.bfloat16)
        mock_pretrained_v3.state = {inv_key: source_inv_freq}
        task = WeightConversionTask(
            param_name="decoder.layers.0.input_layernorm.weight",
            global_param_name="decoder.layers.0.input_layernorm.weight",
            mapping=Mock(),
        )
        converted = {"model.layers.0.input_layernorm.weight": torch.randn(1)}

        result = bridge.maybe_modify_converted_hf_weight(task, dict(converted), mock_pretrained_v3.state)

        assert result[inv_key].shape == source_inv_freq.shape
        assert result[inv_key].dtype == source_inv_freq.dtype
        assert torch.equal(result[inv_key], source_inv_freq)

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


def _block_scaled(scale_inv, rows, cols, shape):
    """Expand a per-block scale grid to a weight of ``shape``, clipping partial edge blocks."""
    return scale_inv.repeat_interleave(rows, dim=0).repeat_interleave(cols, dim=1)[: shape[0], : shape[1]]


class TestDeepSeekV3LoadHFWeights:
    """FP8 dequantization on import, which DeepSeek-V3 leaves to the base ``_load_hf_weights``."""

    @staticmethod
    def _bridge(weight_block_size=None):
        bridge = DeepSeekV3Bridge()
        quantization_config = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8"}
        if weight_block_size is not None:
            quantization_config["weight_block_size"] = weight_block_size
        bridge.hf_config = SimpleNamespace(quantization_config=quantization_config)
        return bridge

    def test_has_no_fp8_hook_of_its_own(self):
        """A bridge hook would dequantize before the base loader could apply the declared block size."""
        assert "maybe_modify_loaded_hf_weight" not in DeepSeekV3Bridge.__dict__

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_unquantized_weights_pass_through(self, dtype):
        weight = torch.randn(4, 4, dtype=dtype)

        assert self._bridge([128, 128])._load_hf_weights("layer.weight", {"layer.weight": weight}) is weight

    def test_dequantizes_with_the_declared_weight_block_size(self):
        """With 64x64 blocks, each quadrant of a 128x128 weight takes its own scale."""
        weight = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        state = {"layer.weight": weight, "layer.weight_scale_inv": scale_inv}

        result = self._bridge([64, 64])._load_hf_weights("layer.weight", state)

        torch.testing.assert_close(result, _block_scaled(scale_inv, 64, 64, (128, 128)), rtol=0, atol=0)

    @pytest.mark.parametrize("weight_block_size", [[128, 128], None], ids=["declared", "undeclared"])
    def test_dequantizes_partial_edge_blocks_with_128x128_blocks(self, weight_block_size):
        """DeepSeek-V3 declares 128x128 blocks, which are also the default when a config declares none."""
        weight = torch.ones(200, 160, dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        state = {"layer.weight": weight, "layer.weight_scale_inv": scale_inv}

        result = self._bridge(weight_block_size)._load_hf_weights("layer.weight", state)

        torch.testing.assert_close(result, _block_scaled(scale_inv, 128, 128, (200, 160)), rtol=0, atol=0)

    def test_fp8_without_scale_inv_is_left_for_the_mapping(self):
        """Without a ``*_scale_inv`` companion there is nothing to apply; the mapping casts the codes."""
        weight = torch.ones(4, 4, dtype=torch.float8_e4m3fn)

        assert self._bridge([128, 128])._load_hf_weights("layer.weight", {"layer.weight": weight}) is weight

    def test_dict_hf_param_dequantizes_each_fp8_entry(self):
        """Compound (dict) params dequantize every FP8 entry with its own scales."""
        gate = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        up = torch.ones(64, 64, dtype=torch.bfloat16)
        state = {"key1": gate, "key1_scale_inv": torch.full((2, 2), 2.0), "key2": up}

        result = self._bridge([64, 64])._load_hf_weights({"gate": "key1", "up": "key2"}, state)

        assert result.keys() == {"gate", "up"}
        torch.testing.assert_close(result["gate"], torch.full((128, 128), 2.0), rtol=0, atol=0)
        assert result["up"] is up


class TestDeepSeekV3MaybeModifyConvertedHFWeight:
    """Unit tests for DeepSeek V3 source-specific export aliases."""

    @pytest.mark.parametrize(
        ("global_name", "source_key", "alias_key"),
        [
            (
                "embedding.word_embeddings.weight",
                "model.embed_tokens.weight",
                "model.layers.61.embed_tokens.weight",
            ),
            (
                "output_layer.weight",
                "lm_head.weight",
                "model.layers.61.shared_head.head.weight",
            ),
        ],
    )
    def test_restores_shared_mtp_alias(self, global_name, source_key, alias_key):
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = SimpleNamespace(num_hidden_layers=61)
        task = SimpleNamespace(global_param_name=global_name)
        weight = torch.randn(4, 4)

        result = bridge.maybe_modify_converted_hf_weight(task, {source_key: weight}, {alias_key: torch.empty(0)})

        assert result[alias_key] is weight

    def test_does_not_add_mtp_alias_when_source_does_not_expect_it(self):
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = SimpleNamespace(num_hidden_layers=61)
        task = SimpleNamespace(global_param_name="embedding.word_embeddings.weight")
        converted = {"model.embed_tokens.weight": torch.randn(4, 4)}

        result = bridge.maybe_modify_converted_hf_weight(task, converted, {})

        assert result is converted
        assert set(result) == {"model.embed_tokens.weight"}


class TestCommonMappingExpertBias:
    """Router expert_bias coverage in the shared DeepSeek-family mapping list (issue #4199)."""

    def test_decoder_expert_bias_in_common_list(self):
        """The common list must map router expert_bias for the main decoder layers."""
        registry = MegatronMappingRegistry(*get_common_mapping_list())
        mapping = registry.megatron_to_hf_lookup("decoder.layers.5.mlp.router.expert_bias")
        assert mapping is not None
        assert mapping.hf_param == "model.layers.5.mlp.gate.e_score_correction_bias"

    def test_mtp_expert_bias_still_generated(self):
        """The MTP loop must still emit the expert_bias mapping for MTP layers."""
        hf_config = SimpleNamespace(num_nextn_predict_layers=1, num_hidden_layers=61)
        registry = MegatronMappingRegistry(*get_common_mapping_list(hf_config=hf_config))
        mapping = registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.mlp.router.expert_bias")
        assert mapping is not None
        assert mapping.hf_param == "model.layers.61.mlp.gate.e_score_correction_bias"

    def test_v3_bridge_resolves_expert_bias(self):
        """DeepSeekV3Bridge must keep resolving expert_bias via the common list."""
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=0)
        registry = bridge.mapping_registry()
        mapping = registry.megatron_to_hf_lookup("decoder.layers.0.mlp.router.expert_bias")
        assert mapping is not None
        assert mapping.hf_param == "model.layers.0.mlp.gate.e_score_correction_bias"
