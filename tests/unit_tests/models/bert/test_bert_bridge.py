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

from unittest.mock import Mock

import pytest
import torch
from transformers import MegatronBertConfig, MegatronBertForMaskedLM

from megatron.bridge.models.bert.bert_bridge import BertBridge
from megatron.bridge.models.bert.bert_provider import BertModelProvider
from megatron.bridge.models.conversion.model_bridge import (
    MegatronModelBridge,
    WeightConversionTask,
    get_model_bridge,
)
from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM


class TestBertBridge:
    """Test cases for BertBridge class."""

    def _hf_config(self, **overrides):
        defaults = dict(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            hidden_act="gelu",
            tie_word_embeddings=True,
        )
        defaults.update(overrides)
        return MegatronBertConfig(**defaults)

    def _mock_pretrained(self, hf_config):
        mock_pretrained = Mock(spec=PreTrainedMaskedLM)
        mock_pretrained.config = hf_config
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that BertBridge is a registered MegatronModelBridge with the right provider."""
        assert issubclass(BertBridge, MegatronModelBridge)
        assert BertBridge.PROVIDER_CLASS is BertModelProvider
        assert BertBridge.SOURCE_NAME == "MegatronBertForMaskedLM"

    def test_bridge_dispatch_from_architecture(self):
        """Test that the dispatch registry resolves MegatronBertForMaskedLM to BertBridge.

        `AutoBridge` selects the bridge through this dispatcher, so a missing or
        misspelled registration would only surface at conversion time.
        """
        bridge = get_model_bridge("MegatronBertForMaskedLM", hf_config=self._hf_config())

        assert isinstance(bridge, BertBridge)

    def test_provider_bridge_field_mapping(self):
        """Test that HF config fields land correctly on the BertModelProvider."""
        hf_config = self._hf_config(
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.3,
        )
        provider = BertBridge().provider_bridge(self._mock_pretrained(hf_config))

        assert isinstance(provider, BertModelProvider)
        assert provider.num_layers == hf_config.num_hidden_layers
        assert provider.hidden_size == hf_config.hidden_size
        assert provider.ffn_hidden_size == hf_config.intermediate_size
        assert provider.num_attention_heads == hf_config.num_attention_heads
        assert provider.num_query_groups == hf_config.num_attention_heads
        assert provider.layernorm_epsilon == hf_config.layer_norm_eps
        assert provider.max_position_embeddings == hf_config.max_position_embeddings
        assert provider.num_tokentypes == hf_config.type_vocab_size
        assert provider.vocab_size == hf_config.vocab_size
        assert provider.share_embeddings_and_output_weights == hf_config.tie_word_embeddings
        assert provider.add_binary_head is False
        assert provider.activation_func is torch.nn.functional.gelu
        assert provider.hidden_dropout == hf_config.hidden_dropout_prob
        assert provider.attention_dropout == hf_config.attention_probs_dropout_prob

    def test_provider_bridge_enables_megatron_only_vocab_padding(self):
        """Test that non-divisible HF vocabularies can be used with tensor parallelism."""
        hf_config = self._hf_config(vocab_size=30_522)
        provider = BertBridge().provider_bridge(self._mock_pretrained(hf_config))

        assert provider.should_pad_vocab is True
        assert provider.vocab_size == 30_522
        # 30522 = 2 x 3 x 5087, so the largest power-of-two divisor is 2.
        assert provider.make_vocab_size_divisible_by == 2

    @pytest.mark.parametrize(
        ("torch_dtype", "expected_dtype", "expected_fp16", "expected_bf16"),
        [
            ("float32", torch.float32, False, False),
            ("float16", torch.float16, True, False),
            ("bfloat16", torch.bfloat16, False, True),
        ],
    )
    def test_provider_bridge_dtype_mapping(self, torch_dtype, expected_dtype, expected_fp16, expected_bf16):
        """Test that the HF checkpoint dtype selects the matching Megatron precision flags."""
        hf_config = self._hf_config(torch_dtype=torch_dtype)
        provider = BertBridge().provider_bridge(self._mock_pretrained(hf_config))

        assert provider.params_dtype == expected_dtype
        assert provider.fp16 is expected_fp16
        assert provider.bf16 is expected_bf16

    def test_provider_bridge_defaults_to_float32_without_dtype(self):
        """Test that a config that never declares a dtype falls back to float32."""
        provider = BertBridge().provider_bridge(self._mock_pretrained(self._hf_config()))

        assert provider.params_dtype == torch.float32
        assert provider.fp16 is False
        assert provider.bf16 is False

    @pytest.mark.parametrize(
        ("torch_dtype", "expected"),
        [("float32", "float32"), ("float16", "float16"), ("bfloat16", "bfloat16")],
    )
    def test_megatron_to_hf_config_exports_dtype(self, torch_dtype, expected):
        """Test that the exported HF config advertises the provider's precision."""
        hf_config = self._hf_config(torch_dtype=torch_dtype)
        provider = BertBridge().provider_bridge(self._mock_pretrained(hf_config))

        assert BertBridge.megatron_to_hf_config(provider)["torch_dtype"] == expected

    def test_megatron_to_hf_config_roundtrip(self):
        """Test that megatron_to_hf_config maps the provider fields back to HF names."""
        hf_config = self._hf_config(hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.3)
        provider = BertBridge().provider_bridge(self._mock_pretrained(hf_config))

        hf_config_dict = BertBridge.megatron_to_hf_config(provider)

        assert hf_config_dict["num_hidden_layers"] == hf_config.num_hidden_layers
        assert hf_config_dict["hidden_size"] == hf_config.hidden_size
        assert hf_config_dict["intermediate_size"] == hf_config.intermediate_size
        assert hf_config_dict["vocab_size"] == hf_config.vocab_size
        assert hf_config_dict["max_position_embeddings"] == hf_config.max_position_embeddings
        assert hf_config_dict["type_vocab_size"] == hf_config.type_vocab_size
        assert hf_config_dict["layer_norm_eps"] == hf_config.layer_norm_eps
        assert hf_config_dict["tie_word_embeddings"] == hf_config.tie_word_embeddings
        assert hf_config_dict["hidden_act"] == hf_config.hidden_act
        assert hf_config_dict["hidden_dropout_prob"] == hf_config.hidden_dropout_prob
        assert hf_config_dict["attention_probs_dropout_prob"] == hf_config.attention_probs_dropout_prob
        assert hf_config_dict["architectures"] == ["MegatronBertForMaskedLM"]
        assert hf_config_dict["model_type"] == "megatron-bert"

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"hidden_act": "relu"}, "hidden_act='gelu'"),
            ({"hidden_act": "silu"}, "hidden_act='gelu'"),
            ({"is_decoder": True}, "encoder-only"),
            ({"add_cross_attention": True}, "encoder-only"),
            ({"is_decoder": True, "add_cross_attention": True}, "encoder-only"),
            ({"tie_word_embeddings": False}, "tie_word_embeddings=True"),
        ],
    )
    def test_provider_bridge_rejects_unsupported_architectures(self, overrides, match):
        """Test that configs Megatron-Core BERT cannot represent fail explicitly."""
        hf_config = self._hf_config(**overrides)

        with pytest.raises(ValueError, match=match):
            BertBridge().provider_bridge(self._mock_pretrained(hf_config))

    def test_megatron_to_hf_config_rejects_non_gelu_provider(self):
        """Test that export cannot claim parity for an unsupported MLM-head activation."""
        provider = BertBridge().provider_bridge(self._mock_pretrained(self._hf_config()))
        provider.activation_func = torch.nn.functional.relu

        with pytest.raises(ValueError, match="hardcodes GELU"):
            BertBridge.megatron_to_hf_config(provider)

    def test_megatron_to_hf_config_rejects_untied_provider(self):
        """Test that export cannot claim parity for an unrepresentable untied MLM head."""
        provider = BertBridge().provider_bridge(self._mock_pretrained(self._hf_config()))
        provider.share_embeddings_and_output_weights = False

        with pytest.raises(ValueError, match="share_embeddings_and_output_weights=True"):
            BertBridge.megatron_to_hf_config(provider)

    def test_mapping_registry_resolves_known_params(self):
        """Test that mapping_registry resolves representative Megatron param names to
        the exact HF keys of a real MegatronBertForMaskedLM state dict."""
        hf_config = self._hf_config()
        hf_model = MegatronBertForMaskedLM(hf_config)
        hf_param_names = set(hf_model.state_dict().keys())

        registry = BertBridge().mapping_registry()
        cases = {
            "embedding.word_embeddings.weight": "bert.embeddings.word_embeddings.weight",
            "embedding.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
            "embedding.tokentype_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
            "encoder.layers.0.self_attention.linear_qkv.layer_norm_weight": "bert.encoder.layer.0.attention.ln.weight",
            "encoder.layers.1.mlp.linear_fc1.weight": "bert.encoder.layer.1.intermediate.dense.weight",
            "encoder.final_layernorm.weight": "bert.encoder.ln.weight",
            "lm_head.dense.weight": "cls.predictions.transform.dense.weight",
            "output_layer.bias": "cls.predictions.bias",
        }
        for megatron_param, expected_hf_param in cases.items():
            mapping = registry.megatron_to_hf_lookup(megatron_param)
            assert mapping is not None, f"no mapping found for {megatron_param}"
            assert mapping.hf_param == expected_hf_param
            assert expected_hf_param in hf_param_names

    @pytest.mark.parametrize("suffix", ["weight", "bias"])
    def test_mapping_registry_resolves_fused_qkv(self, suffix):
        """Test that the fused Megatron QKV param resolves to the three separate HF projections.

        The QKV fusion is the only mapping where a naming mistake cannot be caught by the
        "every HF parameter is mapped" check, because q/k/v all resolve through one mapping.
        """
        hf_param_names = set(MegatronBertForMaskedLM(self._hf_config()).state_dict().keys())
        registry = BertBridge().mapping_registry()

        mapping = registry.megatron_to_hf_lookup(f"encoder.layers.1.self_attention.linear_qkv.{suffix}")

        assert mapping is not None
        assert mapping.hf_param == {
            "q": f"bert.encoder.layer.1.attention.self.query.{suffix}",
            "k": f"bert.encoder.layer.1.attention.self.key.{suffix}",
            "v": f"bert.encoder.layer.1.attention.self.value.{suffix}",
        }
        assert set(mapping.hf_param.values()) <= hf_param_names

    @pytest.mark.parametrize(
        "hf_param",
        [
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.attention.self.key.weight",
            "bert.encoder.layer.0.attention.self.value.weight",
        ],
    )
    def test_mapping_registry_reverse_resolves_qkv_to_fused_param(self, hf_param):
        """Test that each HF attention projection maps back to the single fused Megatron param."""
        registry = BertBridge().mapping_registry()

        mapping = registry.hf_to_megatron_lookup(hf_param)

        assert mapping is not None
        assert mapping.megatron_param == "encoder.layers.0.self_attention.linear_qkv.weight"

    def test_mapping_registry_covers_all_unique_hf_parameters(self):
        """Test that every independently serialized HF parameter has a reverse mapping."""
        hf_model = MegatronBertForMaskedLM(self._hf_config())
        registry = BertBridge().mapping_registry()

        unmapped = {name for name in hf_model.state_dict() if registry.hf_to_megatron_lookup(name) is None}

        # HF exposes the decoder bias twice in state_dict(), but save_pretrained serializes only
        # cls.predictions.bias because both names reference the same Parameter.
        assert unmapped == {"cls.predictions.decoder.bias"}

    def test_maybe_modify_converted_hf_weight_duplicates_tied_bias(self):
        """Test that the tied `cls.predictions.decoder.bias` key is synthesized on export."""
        bridge = BertBridge()
        task = Mock(spec=WeightConversionTask)
        task.global_param_name = "output_layer.bias"

        bias = torch.randn(128)
        converted = {"cls.predictions.bias": bias}
        hf_state_dict = {"cls.predictions.bias": bias, "cls.predictions.decoder.bias": bias}

        result = bridge.maybe_modify_converted_hf_weight(task, converted, hf_state_dict)

        assert result["cls.predictions.decoder.bias"] is bias

    def test_maybe_modify_converted_hf_weight_respects_serialized_key_set(self):
        """Test that export does not synthesize an alias omitted by safe serialization."""
        bridge = BertBridge()
        task = Mock(spec=WeightConversionTask)
        task.global_param_name = "output_layer.bias"

        bias = torch.randn(128)
        converted = {"cls.predictions.bias": bias}

        result = bridge.maybe_modify_converted_hf_weight(task, converted, {"cls.predictions.bias": bias})

        assert result == {"cls.predictions.bias": bias}

    def test_maybe_modify_converted_hf_weight_ignores_other_params(self):
        """Test that the hook is a no-op for any param other than output_layer.bias."""
        bridge = BertBridge()
        task = Mock(spec=WeightConversionTask)
        task.global_param_name = "encoder.final_layernorm.weight"

        converted = {"bert.encoder.ln.weight": torch.randn(32)}
        result = bridge.maybe_modify_converted_hf_weight(task, converted, hf_state_dict={})

        assert result == converted
