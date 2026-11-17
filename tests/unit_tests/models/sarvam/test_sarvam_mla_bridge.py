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

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.deepseek.attention import get_deepseek_decoder_block_spec
from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig
from megatron.bridge.models.sarvam.sarvam_mla_bridge import SarvamMLABridge
from megatron.bridge.models.sarvam.sarvam_provider import SarvamMLAModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


def _sarvam_mla_config(**overrides) -> SimpleNamespace:
    """A Sarvam MLA config with the attributes the remote ``SarvamMLAConfig`` sets, at sarvam-105b sizes."""
    values = dict(
        architectures=["SarvamMLAForCausalLM"],
        auto_map={"AutoModelForCausalLM": "modeling_sarvam_moe.SarvamMLAForCausalLM"},
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=16384,
        moe_intermediate_size=2048,
        num_attention_heads=64,
        num_experts=128,
        num_experts_per_tok=8,
        num_shared_experts=1,
        first_k_dense_replace=1,
        vocab_size=262144,
        max_position_embeddings=131072,
        rope_theta=10000.0,
        rope_scaling={
            "type": "deepseek_yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
        },
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_head_dim=192,
        # The remote config sets head_dim to kv_lora_rank + qk_rope_head_dim for vLLM.
        head_dim=576,
        rms_norm_eps=1e-6,
        initializer_range=0.006,
        attention_dropout=0.0,
        hidden_act="silu",
        use_qk_norm=True,
        routed_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        tie_word_embeddings=False,
        torch_dtype="float16",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _provider(hf_config) -> SarvamMLAModelProvider:
    with pytest.warns(FutureWarning, match="deprecated for builder-backed models"):
        return SarvamMLABridge().provider_bridge(SimpleNamespace(config=hf_config))


class TestSarvamMLABridge:
    def test_registration(self):
        assert issubclass(SarvamMLABridge, MegatronModelBridge)

    def test_conversion_uses_builder_config(self):
        assert SarvamMLABridge.MODEL_CONFIG_CLASS is BridgeGPTMLAModelConfig
        assert SarvamMLABridge.USE_MODEL_CONFIG_FOR_CONVERSION is True
        assert SarvamMLABridge.PROVIDER_CLASS is SarvamMLAModelProvider
        assert "provider_bridge" not in SarvamMLABridge.__dict__

    def test_model_config_maps_common_and_mla_fields(self):
        hf_config = _sarvam_mla_config()
        bridge = SarvamMLABridge()
        # The builder path must not route through the legacy provider.
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(hf_config)

        assert isinstance(model_config, BridgeGPTMLAModelConfig)
        assert isinstance(model_config.transformer, MLATransformerConfig)
        assert model_config.num_layers == 32
        assert model_config.hidden_size == 4096
        assert model_config.ffn_hidden_size == 16384
        assert model_config.moe_ffn_hidden_size == 2048
        assert model_config.moe_shared_expert_intermediate_size == 2048
        assert model_config.moe_layer_freq == [0] + [1] * 31
        assert model_config.num_moe_experts == 128
        assert model_config.moe_router_topk == 8
        assert model_config.vocab_size == 262144
        assert model_config.seq_length == 131072
        assert model_config.transformer.rotary_base == 10000.0
        # kv_channels follows the attention heads, not the remote config's vLLM head_dim.
        assert model_config.kv_channels == 64
        assert model_config.multi_latent_attention is True
        assert model_config.q_lora_rank is None
        assert model_config.kv_lora_rank == 512
        assert model_config.qk_head_dim == 128
        assert model_config.qk_pos_emb_head_dim == 64
        assert model_config.v_head_dim == 128
        assert model_config.qk_layernorm is True
        assert model_config.position_embedding_type == "rope"
        assert model_config.rope_type == "yarn"
        assert model_config.rotary_scaling_factor == 40
        assert model_config.original_max_position_embeddings == 4096
        assert model_config.mscale_all_dim == 1.0
        assert model_config.params_dtype == torch.float16
        assert model_config.transformer_layer_spec.func is get_deepseek_decoder_block_spec
        assert model_config.transformer_layer_spec.keywords["normalization"] == "RMSNorm"

    def test_yarn_settings_follow_the_checkpoint(self):
        rope_scaling = {
            "type": "deepseek_yarn",
            "factor": 8,
            "beta_fast": 16,
            "beta_slow": 2,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 2048,
        }
        hf_config = _sarvam_mla_config(rope_scaling=rope_scaling)

        for config in (SarvamMLABridge().hf_config_to_model_config(hf_config), _provider(hf_config)):
            assert config.rotary_scaling_factor == 8
            assert config.beta_fast == 16
            assert config.beta_slow == 2
            assert config.mscale == 0.707
            assert config.mscale_all_dim == 0.707
            assert config.original_max_position_embeddings == 2048

    def test_model_config_matches_provider(self):
        hf_config = _sarvam_mla_config()
        model_config = SarvamMLABridge().hf_config_to_model_config(hf_config)
        provider = _provider(hf_config)

        model_fields = {f.name for f in fields(model_config)} | {f.name for f in fields(model_config.transformer)}
        shared = ({f.name for f in fields(provider)} & model_fields) - {"transformer_layer_spec"}
        assert len(shared) > 400
        for name in sorted(shared):
            assert getattr(model_config, name) == getattr(provider, name), name
        for spec in (model_config.transformer_layer_spec, provider.transformer_layer_spec):
            assert spec.func is get_deepseek_decoder_block_spec
            assert spec.keywords == model_config.transformer_layer_spec.keywords

    def test_model_config_round_trip(self):
        model_config = SarvamMLABridge().hf_config_to_model_config(_sarvam_mla_config())

        restored = ModelConfig.from_dict(model_config.as_dict())

        assert isinstance(restored, BridgeGPTMLAModelConfig)
        assert restored.transformer_layer_spec.func is get_deepseek_decoder_block_spec
        assert restored.transformer.rotary_base == 10000.0
        assert restored.rotary_scaling_factor == 40
        assert restored.q_lora_rank is None

    def test_provider_bridge_maps_common_and_mla_fields(self):
        hf_config = _sarvam_mla_config()
        provider = _provider(hf_config)

        assert isinstance(provider, SarvamMLAModelProvider)

        # Common config
        assert provider.num_layers == hf_config.num_hidden_layers
        assert provider.hidden_size == hf_config.hidden_size
        assert provider.ffn_hidden_size == hf_config.intermediate_size
        assert provider.moe_ffn_hidden_size == hf_config.moe_intermediate_size
        assert provider.num_attention_heads == hf_config.num_attention_heads
        assert provider.vocab_size == hf_config.vocab_size
        assert provider.seq_length == hf_config.max_position_embeddings
        assert provider.rotary_base == hf_config.rope_theta

        # MLA-specific
        assert provider.kv_channels == hf_config.hidden_size // hf_config.num_attention_heads
        assert provider.kv_lora_rank == hf_config.kv_lora_rank
        assert provider.qk_head_dim == hf_config.qk_nope_head_dim
        assert provider.qk_pos_emb_head_dim == hf_config.qk_rope_head_dim
        assert provider.v_head_dim == hf_config.v_head_dim

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
        assert provider.transformer_layer_spec.func is get_deepseek_decoder_block_spec

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

        with pytest.warns(FutureWarning), pytest.raises(AssertionError, match="torch_dtype"):
            SarvamMLABridge().provider_bridge(hf)

    @pytest.mark.parametrize(
        ("torch_dtype", "expected"),
        [("bfloat16", torch.bfloat16), ("float16", torch.float16), ("float32", torch.float32)],
    )
    def test_dtype_handling_multiple_precisions(self, torch_dtype, expected):
        hf_config = _sarvam_mla_config(torch_dtype=torch_dtype)

        for config in (SarvamMLABridge().hf_config_to_model_config(hf_config), _provider(hf_config)):
            assert config.params_dtype == expected
            assert config.bf16 is (expected == torch.bfloat16)
            assert config.fp16 is (expected == torch.float16)

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
            (
                "decoder.layers.*.input_layernorm.weight",
                "model.layers.*.input_layernorm.weight",
            ),
            (
                "decoder.layers.*.self_attention.linear_proj.weight",
                "model.layers.*.self_attn.o_proj.weight",
            ),
            # MLA q projection and kv projections/layernorm
            (
                "decoder.layers.*.self_attention.linear_q_proj.weight",
                "model.layers.*.self_attn.q_proj.weight",
            ),
            (
                "decoder.layers.*.self_attention.linear_kv_down_proj.weight",
                "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            ),
            (
                "decoder.layers.*.self_attention.linear_kv_up_proj.weight",
                "model.layers.*.self_attn.kv_b_proj.weight",
            ),
            (
                "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight",
                "model.layers.*.self_attn.kv_a_layernorm.weight",
            ),
            # Mcore local spec alias
            (
                "decoder.layers.*.self_attention.kv_layernorm.weight",
                "model.layers.*.self_attn.kv_a_layernorm.weight",
            ),
            # Router mappings
            ("decoder.layers.*.mlp.router.weight", "model.layers.*.mlp.gate.weight"),
            (
                "decoder.layers.*.mlp.router.expert_bias",
                "model.layers.*.mlp.gate.e_score_correction_bias",
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
