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
Unit tests for Sarvam MLA bridge.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.sarvam.sarvam_mla_bridge import SarvamMLABridge
from megatron.bridge.models.sarvam.sarvam_provider import SarvamMLAModelProvider


class TestSarvamMLABridge:
    @pytest.fixture
    def sarvam_mla_config_dict(self):
        return {
            "architectures": ["SarvamMLAForCausalLM"],
            "auto_map": {"AutoModelForCausalLM": "modeling_sarvam.SarvamMLAForCausalLM"},
            # Common Sarvam fields
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "moe_intermediate_size": 2048,
            "num_attention_heads": 64,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "vocab_size": 262144,
            "max_position_embeddings": 131072,
            "rope_theta": 8_000_000.0,
            # dtype
            "torch_dtype": torch.float16,
            # MLA fields (bridge specific)
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rope_scaling": {"factor": 40, "mscale": 1.0, "mscale_all_dim": 1.0},
        }

    @pytest.fixture
    def mock_pretrained_mla(self, sarvam_mla_config_dict):
        cfg = Mock()
        for k, v in sarvam_mla_config_dict.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)
        return m

    def test_registration(self):
        assert issubclass(SarvamMLABridge, MegatronModelBridge)

    def test_provider_bridge_maps_common_and_mla_fields(self, mock_pretrained_mla):
        bridge = SarvamMLABridge()
        provider = bridge.provider_bridge(mock_pretrained_mla)

        assert isinstance(provider, SarvamMLAModelProvider)

        # Common config
        assert provider.num_layers == mock_pretrained_mla.config.num_hidden_layers
        assert provider.hidden_size == mock_pretrained_mla.config.hidden_size
        assert provider.ffn_hidden_size == mock_pretrained_mla.config.intermediate_size
        assert provider.moe_ffn_hidden_size == mock_pretrained_mla.config.moe_intermediate_size
        assert provider.num_attention_heads == mock_pretrained_mla.config.num_attention_heads
        assert provider.vocab_size == mock_pretrained_mla.config.vocab_size
        assert provider.seq_length == mock_pretrained_mla.config.max_position_embeddings
        assert provider.rotary_base == mock_pretrained_mla.config.rope_theta
        assert provider.generation_config == mock_pretrained_mla.generation_config

        # MLA-specific
        assert provider.kv_channels == mock_pretrained_mla.config.hidden_size // mock_pretrained_mla.config.num_attention_heads
        assert provider.kv_lora_rank == mock_pretrained_mla.config.kv_lora_rank
        assert provider.qk_head_dim == mock_pretrained_mla.config.qk_nope_head_dim
        assert provider.qk_pos_emb_head_dim == mock_pretrained_mla.config.qk_rope_head_dim
        assert provider.v_head_dim == mock_pretrained_mla.config.v_head_dim

        assert provider.rotary_scaling_factor == mock_pretrained_mla.config.rope_scaling["factor"]
        assert provider.mscale == mock_pretrained_mla.config.rope_scaling["mscale"]
        assert provider.mscale_all_dim == mock_pretrained_mla.config.rope_scaling["mscale_all_dim"]

        # dtype mapping
        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

        # Sarvam family defaults
        assert provider.multi_latent_attention is True
        assert provider.qk_layernorm is True
        assert provider.add_qkv_bias is False
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"

    def test_provider_bridge_rope_scaling_defaults_when_missing(self, sarvam_mla_config_dict):
        cfg = Mock()
        for k, v in sarvam_mla_config_dict.items():
            if k == "rope_scaling":
                continue
            setattr(cfg, k, v)
        cfg.rope_scaling = None

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = None

        bridge = SarvamMLABridge()
        provider = bridge.provider_bridge(m)

        assert provider.rotary_scaling_factor == 1.0
        assert provider.mscale == 1.0
        assert provider.mscale_all_dim == 1.0

    def test_provider_bridge_rope_scaling_attribute_absent_defaults(self):
        """Cover the branch where rope_scaling attr is entirely absent (not just None)."""
        cfg = SimpleNamespace(
            architectures=["SarvamMLAForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_sarvam.SarvamMLAForCausalLM"},
            # Common Sarvam fields
            num_hidden_layers=2,
            hidden_size=256,
            intermediate_size=1024,
            moe_intermediate_size=128,
            num_attention_heads=8,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            first_k_dense_replace=0,
            vocab_size=32000,
            max_position_embeddings=2048,
            rope_theta=8_000_000.0,
            # dtype
            torch_dtype="bfloat16",
            # MLA fields
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            # NOTE: no rope_scaling attribute here on purpose
        )

        # No generation_config attribute either; get_common_config should default to None.
        hf = SimpleNamespace(config=cfg)

        bridge = SarvamMLABridge()
        provider = bridge.provider_bridge(hf)

        assert provider.rotary_scaling_factor == 1.0
        assert provider.mscale == 1.0
        assert provider.mscale_all_dim == 1.0
        assert provider.generation_config is None
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16

    def test_provider_bridge_requires_torch_dtype_attribute(self):
        """Match Qwen-style robustness tests: torch_dtype is required for dtype mapping."""
        cfg = SimpleNamespace(
            architectures=["SarvamMLAForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_sarvam.SarvamMLAForCausalLM"},
            num_hidden_layers=1,
            hidden_size=128,
            intermediate_size=512,
            moe_intermediate_size=64,
            num_attention_heads=4,
            num_experts=2,
            num_experts_per_tok=1,
            num_shared_experts=1,
            first_k_dense_replace=0,
            vocab_size=32000,
            max_position_embeddings=1024,
            rope_theta=8_000_000.0,
            # MLA fields
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
        )
        hf = SimpleNamespace(config=cfg)

        with pytest.raises(AssertionError, match="torch_dtype"):
            SarvamMLABridge().provider_bridge(hf)

    def test_provider_bridge_dtype_handling_multiple_precisions(self, sarvam_mla_config_dict):
        cfg = Mock()
        for k, v in sarvam_mla_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg
        mock_pretrained.generation_config = None

        bridge = SarvamMLABridge()

        cfg.torch_dtype = "bfloat16"
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16

        cfg.torch_dtype = "float16"
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    def test_provider_bridge_generation_config_passthrough(self, mock_pretrained_mla):
        bridge = SarvamMLABridge()
        provider = bridge.provider_bridge(mock_pretrained_mla)
        assert provider.generation_config == mock_pretrained_mla.generation_config

    def test_mapping_registry_basic_lookups(self):
        bridge = SarvamMLABridge()
        reg = bridge.mapping_registry()

        assert reg is not None
        assert len(reg) > 0

        m = reg.megatron_to_hf_lookup("embedding.word_embeddings.weight")
        assert m is not None
        assert m.hf_param == "model.embed_tokens.weight"

        m = reg.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_proj.weight")
        assert m is not None
        assert m.hf_param == "model.layers.0.self_attn.o_proj.weight"

    def test_mapping_registry_types(self):
        bridge = SarvamMLABridge()
        registry = bridge.mapping_registry()

        assert registry is not None
        assert len(registry.mappings) > 0

        mapping_types = [type(m).__name__ for m in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types

    def test_mapping_registry_contains_expected_auto_mapping_patterns(self):
        """Validate presence of key MLA-specific AutoMapping patterns (mirrors Qwen-style mapping coverage)."""
        bridge = SarvamMLABridge()
        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        hf_params = [m.hf_param for m in auto_mappings]
        megatron_params = [m.megatron_param for m in auto_mappings]

        expected_pairs = [
            # Embedding / output / final norm
            ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            ("output_layer.weight", "lm_head.weight"),
            ("decoder.final_layernorm.weight", "model.norm.weight"),
            # Layernorm + proj
            ("decoder.layers.*.input_layernorm.weight", "model.layers.*.input_layernorm.weight"),
            ("decoder.layers.*.self_attention.linear_proj.weight", "model.layers.*.self_attn.o_proj.weight"),
            # MLA q projection and kv projections/layernorm
            ("decoder.layers.*.self_attention.linear_q_proj.weight", "model.layers.*.self_attn.q_proj.weight"),
            (
                "decoder.layers.*.self_attention.linear_kv_down_proj.weight",
                "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            ),
            ("decoder.layers.*.self_attention.linear_kv_up_proj.weight", "model.layers.*.self_attn.kv_b_proj.weight"),
            (
                "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight",
                "model.layers.*.self_attn.kv_a_layernorm.weight",
            ),
            # Mcore local spec alias
            ("decoder.layers.*.self_attention.kv_layernorm.weight", "model.layers.*.self_attn.kv_a_layernorm.weight"),
            # Router mappings
            ("decoder.layers.*.mlp.router.weight", "model.layers.*.mlp.gate.weight"),
            ("decoder.layers.*.mlp.router.expert_bias", "model.layers.*.mlp.gate.e_score_correction_bias"),
            # Dense MLP down-proj
            ("decoder.layers.*.mlp.linear_fc2.weight", "model.layers.*.mlp.down_proj.weight"),
            # Expert + shared expert down-proj
            ("decoder.layers.*.mlp.experts.linear_fc2.weight*", "model.layers.*.mlp.experts.*.down_proj.weight"),
            ("decoder.layers.*.mlp.shared_experts.linear_fc2.weight", "model.layers.*.mlp.shared_experts.down_proj.weight"),
        ]

        for megatron_param, hf_param in expected_pairs:
            assert megatron_param in megatron_params
            assert hf_param in hf_params

    def test_mapping_registry_post_attention_layernorm_maps_to_two_targets(self):
        """
        Sarvam MLA maps HF post_attention_layernorm to two possible Megatron params
        (dense vs MoE layer type). Validate both mappings exist.
        """
        bridge = SarvamMLABridge()
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

        rev = registry.hf_to_megatron_lookup("model.layers.0.post_attention_layernorm.weight")
        assert rev is not None
        assert rev.megatron_param in {
            "decoder.layers.0.pre_mlp_layernorm.weight",
            "decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
        }

    def test_mapping_registry_gated_mlp_mappings_structure(self):
        bridge = SarvamMLABridge()
        registry = bridge.mapping_registry()

        gated = [m for m in registry.mappings if type(m).__name__ == "GatedMLPMapping"]
        assert len(gated) >= 3

        megatron_params = {m.megatron_param for m in gated}
        assert "decoder.layers.*.mlp.linear_fc1.weight" in megatron_params
        assert "decoder.layers.*.mlp.experts.linear_fc1.weight*" in megatron_params
        assert "decoder.layers.*.mlp.shared_experts.linear_fc1.weight" in megatron_params

    def test_mapping_registry_parameter_mappings(self):
        bridge = SarvamMLABridge()
        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        hf_params = [m.hf_param for m in auto_mappings]
        megatron_params = [m.megatron_param for m in auto_mappings]

        # Embedding / output
        assert "model.embed_tokens.weight" in hf_params
        assert "embedding.word_embeddings.weight" in megatron_params
        assert "lm_head.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        # Router bias mapping is MLA-specific in this bridge
        assert "model.layers.*.mlp.gate.e_score_correction_bias" in hf_params
        assert "decoder.layers.*.mlp.router.expert_bias" in megatron_params

        # KV layernorm mapping
        assert "model.layers.*.self_attn.kv_a_layernorm.weight" in hf_params

    def test_mapping_registry_reverse_lookup(self):
        bridge = SarvamMLABridge()
        registry = bridge.mapping_registry()

        m = registry.hf_to_megatron_lookup("model.layers.7.self_attn.o_proj.weight")
        assert m is not None
        assert m.megatron_param == "decoder.layers.7.self_attention.linear_proj.weight"

        m = registry.hf_to_megatron_lookup("model.layers.7.self_attn.q_proj.weight")
        assert m is not None
        assert m.megatron_param == "decoder.layers.7.self_attention.linear_q_proj.weight"

        m = registry.hf_to_megatron_lookup("model.layers.7.self_attn.kv_a_layernorm.weight")
        assert m is not None
        # There are two megatron aliases mapping to the same HF weight; accept either.
        assert m.megatron_param in {
            "decoder.layers.7.self_attention.linear_kv_up_proj.layer_norm_weight",
            "decoder.layers.7.self_attention.kv_layernorm.weight",
        }

        m = registry.hf_to_megatron_lookup("model.layers.2.mlp.gate.e_score_correction_bias")
        assert m is not None
        assert m.megatron_param == "decoder.layers.2.mlp.router.expert_bias"

