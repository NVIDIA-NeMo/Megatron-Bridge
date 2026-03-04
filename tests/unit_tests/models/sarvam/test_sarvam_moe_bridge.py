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
Unit tests for Sarvam MoE bridge.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.sarvam.sarvam_moe_bridge import SarvamMoEBridge
from megatron.bridge.models.sarvam.sarvam_provider import SarvamMoEModelProvider


class TestSarvamMoEBridge:
    @pytest.fixture
    def sarvam_moe_config_dict_small(self):
        # Minimal but complete for provider_bridge mappings.
        return {
            "architectures": ["SarvamMoEForCausalLM"],
            "auto_map": {"AutoModelForCausalLM": "modeling_sarvam.SarvamMoEForCausalLM"},
            # Common Sarvam fields
            "num_hidden_layers": 19,
            "hidden_size": 4096,
            "intermediate_size": 8192,
            "moe_intermediate_size": 1024,
            "num_attention_heads": 64,
            "num_experts": 128,
            "num_experts_per_tok": 6,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "vocab_size": 262144,
            "max_position_embeddings": 131072,
            "rope_theta": 8_000_000.0,
            # dtype
            "torch_dtype": "bfloat16",
            # GQA fields (MoE bridge specific)
            "num_key_value_heads": 4,
            "head_dim": 64,
        }

    @pytest.fixture
    def sarvam_moe_config_dict_large(self):
        # A second config to ensure mappings don't depend on a single hardcoded set.
        return {
            "architectures": ["SarvamMoEForCausalLM"],
            "auto_map": {"AutoModelForCausalLM": "modeling_sarvam.SarvamMoEForCausalLM"},
            "num_hidden_layers": 32,
            "hidden_size": 5120,
            "intermediate_size": 10240,
            "moe_intermediate_size": 1536,
            "num_attention_heads": 80,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_shared_experts": 2,
            "first_k_dense_replace": 0,
            "vocab_size": 128000,
            "max_position_embeddings": 65536,
            "rope_theta": 5_000_000.0,
            "torch_dtype": "float16",
            "num_key_value_heads": 8,
            "head_dim": 64,
        }

    @pytest.fixture
    def mock_pretrained_moe(self, sarvam_moe_config_dict_small):
        cfg = Mock()
        for k, v in sarvam_moe_config_dict_small.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        return m

    def test_registration(self):
        assert issubclass(SarvamMoEBridge, MegatronModelBridge)

    def test_provider_bridge_maps_common_config(self, mock_pretrained_moe):
        bridge = SarvamMoEBridge()
        provider = bridge.provider_bridge(mock_pretrained_moe)

        assert isinstance(provider, SarvamMoEModelProvider)
        assert provider.num_layers == mock_pretrained_moe.config.num_hidden_layers
        assert provider.hidden_size == mock_pretrained_moe.config.hidden_size
        assert provider.ffn_hidden_size == mock_pretrained_moe.config.intermediate_size
        assert provider.moe_ffn_hidden_size == mock_pretrained_moe.config.moe_intermediate_size
        assert provider.num_attention_heads == mock_pretrained_moe.config.num_attention_heads
        assert provider.num_moe_experts == mock_pretrained_moe.config.num_experts
        assert provider.moe_router_topk == mock_pretrained_moe.config.num_experts_per_tok
        assert (
            provider.moe_shared_expert_intermediate_size
            == mock_pretrained_moe.config.num_shared_experts * mock_pretrained_moe.config.moe_intermediate_size
        )
        assert provider.vocab_size == mock_pretrained_moe.config.vocab_size
        assert provider.seq_length == mock_pretrained_moe.config.max_position_embeddings
        assert provider.rotary_base == mock_pretrained_moe.config.rope_theta

        expected_freq = [0] * mock_pretrained_moe.config.first_k_dense_replace + [1] * (
            mock_pretrained_moe.config.num_hidden_layers - mock_pretrained_moe.config.first_k_dense_replace
        )
        assert provider.moe_layer_freq == expected_freq

        # Sarvam family defaults (provider sets these as fixed behavior)
        assert provider.qk_layernorm is True
        assert provider.add_qkv_bias is False
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"

    def test_provider_bridge_maps_moe_specific_gqa_fields(self, mock_pretrained_moe):
        bridge = SarvamMoEBridge()
        provider = bridge.provider_bridge(mock_pretrained_moe)

        assert provider.num_query_groups == mock_pretrained_moe.config.num_key_value_heads
        assert provider.kv_channels == mock_pretrained_moe.config.head_dim

    def test_provider_bridge_dtype_handling(self, mock_pretrained_moe):
        bridge = SarvamMoEBridge()
        provider = bridge.provider_bridge(mock_pretrained_moe)

        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16

    def test_provider_bridge_dtype_handling_multiple_precisions(self, sarvam_moe_config_dict_small):
        cfg = Mock()
        for k, v in sarvam_moe_config_dict_small.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = SarvamMoEBridge()

        cfg.torch_dtype = "bfloat16"
        result = bridge.provider_bridge(mock_pretrained)
        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

        cfg.torch_dtype = "float16"
        result = bridge.provider_bridge(mock_pretrained)
        assert result.fp16 is True
        assert result.bf16 is False
        assert result.params_dtype == torch.float16

    def test_provider_bridge_requires_torch_dtype_attribute(self, sarvam_moe_config_dict_small):
        """Match Qwen-style robustness tests: torch_dtype is required for dtype mapping."""
        cfg_dict = dict(sarvam_moe_config_dict_small)
        cfg_dict.pop("torch_dtype", None)
        cfg = SimpleNamespace(**cfg_dict)
        hf = SimpleNamespace(config=cfg)

        with pytest.raises(AssertionError, match="torch_dtype"):
            SarvamMoEBridge().provider_bridge(hf)

    @pytest.mark.parametrize("first_k_dense_replace", [0, 1, 4])
    def test_provider_bridge_moe_layer_freq_edge_cases(self, sarvam_moe_config_dict_small, first_k_dense_replace):
        cfg = Mock()
        for k, v in sarvam_moe_config_dict_small.items():
            setattr(cfg, k, v)
        cfg.first_k_dense_replace = first_k_dense_replace
        cfg.num_hidden_layers = 6

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = SarvamMoEBridge()
        provider = bridge.provider_bridge(mock_pretrained)

        expected = [0] * first_k_dense_replace + [1] * (cfg.num_hidden_layers - first_k_dense_replace)
        assert provider.moe_layer_freq == expected

    def test_provider_bridge_second_config_variant(self, sarvam_moe_config_dict_large):
        cfg = Mock()
        for k, v in sarvam_moe_config_dict_large.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = SarvamMoEBridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.num_layers == cfg.num_hidden_layers
        assert provider.hidden_size == cfg.hidden_size
        assert provider.num_attention_heads == cfg.num_attention_heads
        assert provider.ffn_hidden_size == cfg.intermediate_size
        assert provider.moe_ffn_hidden_size == cfg.moe_intermediate_size
        assert provider.num_moe_experts == cfg.num_experts
        assert provider.moe_router_topk == cfg.num_experts_per_tok
        assert provider.vocab_size == cfg.vocab_size
        assert provider.seq_length == cfg.max_position_embeddings
        assert provider.rotary_base == cfg.rope_theta
        assert provider.num_query_groups == cfg.num_key_value_heads
        assert provider.kv_channels == cfg.head_dim

    def test_mapping_registry_basic_lookups(self):
        bridge = SarvamMoEBridge()
        reg = bridge.mapping_registry()

        assert reg is not None
        assert len(reg) > 0

        # Direct mapping
        m = reg.megatron_to_hf_lookup("embedding.word_embeddings.weight")
        assert m is not None
        assert m.hf_param == "model.word_embeddings.weight"

        # Wildcard mapping resolves layer index
        m = reg.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_proj.weight")
        assert m is not None
        assert m.hf_param == "model.layers.0.attention.dense.weight"

        # QKV concatenation mapping
        m = reg.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_qkv.weight")
        assert m is not None
        assert m.hf_param == "model.layers.0.attention.query_key_value.weight"

    def test_mapping_registry_types(self):
        bridge = SarvamMoEBridge()
        registry = bridge.mapping_registry()

        assert registry is not None
        assert len(registry.mappings) > 0

        mapping_types = [type(m).__name__ for m in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "ConcatenatedQKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types

    def test_mapping_registry_contains_expected_auto_mapping_patterns(self):
        """Validate that key AutoMapping patterns exist (similar to Qwen's mapping coverage tests)."""
        bridge = SarvamMoEBridge()
        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        hf_params = [m.hf_param for m in auto_mappings]
        megatron_params = [m.megatron_param for m in auto_mappings]

        expected_pairs = [
            # Embedding / output / final norm
            ("embedding.word_embeddings.weight", "model.word_embeddings.weight"),
            ("output_layer.weight", "lm_head.weight"),
            ("decoder.final_layernorm.weight", "model.norm.weight"),
            # Attention QK norm
            (
                "decoder.layers.*.self_attention.q_layernorm.weight",
                "model.layers.*.attention.query_layernorm.weight",
            ),
            (
                "decoder.layers.*.self_attention.k_layernorm.weight",
                "model.layers.*.attention.key_layernorm.weight",
            ),
            # Attention output projection
            (
                "decoder.layers.*.self_attention.linear_proj.weight",
                "model.layers.*.attention.dense.weight",
            ),
            # Router mappings (including expert_bias)
            ("decoder.layers.*.mlp.router.weight", "model.layers.*.mlp.gate.weight"),
            (
                "decoder.layers.*.mlp.router.expert_bias",
                "model.layers.*.mlp.gate.expert_bias",
            ),
            # Dense MLP down-proj
            (
                "decoder.layers.*.mlp.linear_fc2.weight",
                "model.layers.*.mlp.down_proj.weight",
            ),
            # Expert + shared expert down-proj
            (
                "decoder.layers.*.mlp.experts.linear_fc2.weight*",
                "model.layers.*.mlp.experts.*.down_proj.weight",
            ),
            (
                "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                "model.layers.*.mlp.shared_experts.down_proj.weight",
            ),
        ]

        for megatron_param, hf_param in expected_pairs:
            assert megatron_param in megatron_params
            assert hf_param in hf_params

    def test_mapping_registry_post_attention_layernorm_maps_to_two_targets(self):
        """
        Sarvam MoE maps HF post_attention_layernorm to two possible Megatron params
        (dense vs MoE layer type). Validate both mappings exist.
        """
        bridge = SarvamMoEBridge()
        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        pairs = {(m.megatron_param, m.hf_param) for m in auto_mappings}

        assert (
            "decoder.layers.*.pre_mlp_layernorm.weight",
            "model.layers.*.post_attention_layernorm.weight",
        ) in pairs
        assert (
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight",
        ) in pairs

        # Reverse lookup returns the *first* matching mapping; ensure it's one of the two.
        rev = registry.hf_to_megatron_lookup("model.layers.0.post_attention_layernorm.weight")
        assert rev is not None
        assert rev.megatron_param in {
            "decoder.layers.0.pre_mlp_layernorm.weight",
            "decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
        }

    def test_mapping_registry_gated_mlp_mappings_structure(self):
        """Check GatedMLPMapping objects include expected megatron patterns for dense+experts+shared."""
        bridge = SarvamMoEBridge()
        registry = bridge.mapping_registry()

        gated = [m for m in registry.mappings if type(m).__name__ == "GatedMLPMapping"]
        assert len(gated) >= 3

        megatron_params = {m.megatron_param for m in gated}
        assert "decoder.layers.*.mlp.linear_fc1.weight" in megatron_params
        assert "decoder.layers.*.mlp.experts.linear_fc1.weight*" in megatron_params
        assert "decoder.layers.*.mlp.shared_experts.linear_fc1.weight" in megatron_params

    def test_mapping_registry_concatenated_qkv_mapping_shape(self):
        """Validate ConcatenatedQKVMapping uses a single HF qkv tensor name (not dict)."""
        bridge = SarvamMoEBridge()
        registry = bridge.mapping_registry()

        qkv = [m for m in registry.mappings if type(m).__name__ == "ConcatenatedQKVMapping"]
        assert len(qkv) == 1
        qkv = qkv[0]
        assert isinstance(qkv.hf_param, str)
        assert qkv.megatron_param == "decoder.layers.*.self_attention.linear_qkv.weight"

        # Wildcard resolution should work
        resolved = registry.megatron_to_hf_lookup("decoder.layers.12.self_attention.linear_qkv.weight")
        assert resolved is not None
        assert resolved.hf_param == "model.layers.12.attention.query_key_value.weight"

    def test_mapping_registry_parameter_mappings(self):
        bridge = SarvamMoEBridge()
        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        hf_params = [m.hf_param for m in auto_mappings]
        megatron_params = [m.megatron_param for m in auto_mappings]

        # Embedding / output
        assert "model.word_embeddings.weight" in hf_params
        assert "embedding.word_embeddings.weight" in megatron_params
        assert "lm_head.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        # Norm
        assert "model.norm.weight" in hf_params
        assert "decoder.final_layernorm.weight" in megatron_params

        # Router
        assert "model.layers.*.mlp.gate.weight" in hf_params
        assert "decoder.layers.*.mlp.router.weight" in megatron_params

    def test_mapping_registry_reverse_lookup(self):
        bridge = SarvamMoEBridge()
        registry = bridge.mapping_registry()

        m = registry.hf_to_megatron_lookup("model.layers.5.attention.dense.weight")
        assert m is not None
        assert m.megatron_param == "decoder.layers.5.self_attention.linear_proj.weight"

        m = registry.hf_to_megatron_lookup("model.layers.3.attention.query_key_value.weight")
        assert m is not None
        assert m.megatron_param == "decoder.layers.3.self_attention.linear_qkv.weight"

        # Router
        m = registry.hf_to_megatron_lookup("model.layers.2.mlp.gate.weight")
        assert m is not None
        assert m.megatron_param == "decoder.layers.2.mlp.router.weight"
