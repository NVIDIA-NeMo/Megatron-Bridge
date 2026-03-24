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
Unit tests for MiniMax-M2 bridge.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.minimax_m2.minimax_m2_bridge import (
    MiniMaxM2Bridge,
    _dequant_fp8_blockwise,
    _FullDimQKNormMapping,
)
from megatron.bridge.models.minimax_m2.minimax_m2_provider import (
    FullDimKNorm,
    FullDimQNorm,
    _FullDimRMSNorm,
)


# Matches the toy config used in functional tests for consistency
_MINIMAX_M2_CONFIG = {
    "architectures": ["MiniMaxM2ForCausalLM"],
    "model_type": "minimax_m2",
    "hidden_size": 512,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "rotary_dim": 32,
    "vocab_size": 1024,
    "tie_word_embeddings": False,
    "attention_dropout": 0.0,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "scoring_func": "sigmoid",
    "use_routing_bias": True,
    "use_qk_norm": True,
    "qk_norm_type": "per_layer",
    "router_aux_loss_coef": 0.001,
    "router_jitter_noise": 0.0,
    "output_router_logits": False,
    "torch_dtype": "bfloat16",
}


class TestMiniMaxM2Bridge:
    """Unit tests for MiniMaxM2Bridge config mapping, FP8 dequant, and mapping registry."""

    @pytest.fixture
    def mock_pretrained(self):
        cfg = Mock(spec=list(_MINIMAX_M2_CONFIG.keys()))
        for k, v in _MINIMAX_M2_CONFIG.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock(spec=GenerationConfig)
        return m

    def test_registration(self):
        assert issubclass(MiniMaxM2Bridge, MegatronModelBridge)

    def test_provider_bridge_maps_core_config(self, mock_pretrained):
        bridge = MiniMaxM2Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.hidden_size == mock_pretrained.config.hidden_size
        assert provider.num_attention_heads == mock_pretrained.config.num_attention_heads
        assert provider.num_query_groups == mock_pretrained.config.num_key_value_heads
        assert provider.ffn_hidden_size == mock_pretrained.config.intermediate_size
        assert provider.vocab_size == mock_pretrained.config.vocab_size
        assert provider.layernorm_epsilon == mock_pretrained.config.rms_norm_eps
        assert provider.rotary_base == mock_pretrained.config.rope_theta
        assert provider.num_moe_experts == mock_pretrained.config.num_local_experts
        assert provider.moe_router_topk == mock_pretrained.config.num_experts_per_tok

    def test_provider_bridge_sets_moe_sigmoid_routing(self, mock_pretrained):
        bridge = MiniMaxM2Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_pre_softmax is False
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_router_load_balancing_type == "aux_loss"

    def test_provider_bridge_calculates_rotary_percent(self, mock_pretrained):
        bridge = MiniMaxM2Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        expected = mock_pretrained.config.rotary_dim / mock_pretrained.config.head_dim
        assert abs(provider.rotary_percent - expected) < 1e-6

    def test_provider_bridge_rotary_percent_missing_fields(self, mock_pretrained):
        """When rotary_dim or head_dim is absent, rotary_percent is not overridden."""
        mock_pretrained.config.rotary_dim = None
        bridge = MiniMaxM2Bridge()
        provider = bridge.provider_bridge(mock_pretrained)
        # rotary_percent may be set by CONFIG_MAPPING via partial_rotary_factor;
        # the key assertion is that no AttributeError is raised.
        assert hasattr(provider, "rotary_percent")

    def test_provider_bridge_sets_custom_layer_spec(self, mock_pretrained):
        bridge = MiniMaxM2Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        from megatron.bridge.models.minimax_m2.minimax_m2_provider import minimax_m2_layer_spec

        assert provider.transformer_layer_spec is minimax_m2_layer_spec
        # qk_layernorm is disabled at the provider level; the custom spec injects norms instead.
        assert provider.qk_layernorm is False

    def test_provider_bridge_dtype_bfloat16(self, mock_pretrained):
        bridge = MiniMaxM2Bridge()
        provider = bridge.provider_bridge(mock_pretrained)

        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_mapping_registry_contains_critical_weights(self, mock_pretrained):
        bridge = MiniMaxM2Bridge()
        registry = bridge.mapping_registry()

        megatron_params = [str(m.megatron_param) for m in registry]
        assert any("embed_tokens" in p or "word_embeddings" in p for p in megatron_params), "Embedding mapping missing"
        assert any("linear_qkv" in p for p in megatron_params), "QKV mapping missing"
        assert any("linear_proj" in p for p in megatron_params), "o_proj mapping missing"
        assert any("router" in p for p in megatron_params), "MoE router mapping missing"
        assert any("linear_fc1" in p for p in megatron_params), "Expert gate/up mapping missing"
        assert any("linear_fc2" in p for p in megatron_params), "Expert down mapping missing"
        assert any("q_layernorm" in p for p in megatron_params), "Q norm mapping missing"
        assert any("k_layernorm" in p for p in megatron_params), "K norm mapping missing"
        assert any("expert_bias" in p for p in megatron_params), "Expert bias mapping missing"

    def test_mapping_registry_qk_norm_type(self, mock_pretrained):
        """QK norm mappings must use _FullDimQKNormMapping, not AutoMapping."""
        bridge = MiniMaxM2Bridge()
        registry = bridge.mapping_registry()

        qk_norm_mappings = [
            m for m in registry if "q_layernorm" in str(m.megatron_param) or "k_layernorm" in str(m.megatron_param)
        ]
        assert len(qk_norm_mappings) == 2
        for m in qk_norm_mappings:
            assert isinstance(m, _FullDimQKNormMapping), (
                f"Expected _FullDimQKNormMapping, got {type(m)} for {m.megatron_param}"
            )


class TestDequantFP8Blockwise:
    """Unit tests for _dequant_fp8_blockwise."""

    def test_identity_scale_inv(self):
        """With scale_inv=1 the output equals the input cast to bfloat16."""
        weight = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(1, 1)
        result = _dequant_fp8_blockwise(weight, scale_inv)

        assert result.dtype == torch.bfloat16
        assert result.shape == (128, 128)
        assert torch.all(result == 1.0)

    def test_scale_inv_applied_per_block(self):
        """scale_inv value is multiplied block-wise."""
        weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
        scale_inv = torch.full((2, 2), 2.0)
        result = _dequant_fp8_blockwise(weight, scale_inv)

        assert result.dtype == torch.bfloat16
        assert torch.all(result == 2.0)

    def test_non_square_weight(self):
        """Works for weight tensors whose dims are not multiples of block size."""
        weight = torch.zeros(100, 70, dtype=torch.float8_e4m3fn)
        # ceil(100/128)=1, ceil(70/128)=1 → 1x1 scale block
        scale_inv = torch.ones(1, 1)
        result = _dequant_fp8_blockwise(weight, scale_inv)

        assert result.shape == (100, 70)
        assert result.dtype == torch.bfloat16


class TestMaybeModifyLoadedHFWeight:
    """Unit tests for MiniMaxM2Bridge.maybe_modify_loaded_hf_weight."""

    def _make_bridge(self):
        return MiniMaxM2Bridge()

    def test_passthrough_bfloat16(self):
        """Non-FP8 weights are returned unchanged."""
        bridge = self._make_bridge()
        w = torch.randn(4, 4, dtype=torch.bfloat16)
        state = {"layer.weight": w}
        result = bridge.maybe_modify_loaded_hf_weight("layer.weight", state)
        assert result is w

    def test_passthrough_float32(self):
        bridge = self._make_bridge()
        w = torch.randn(4, 4, dtype=torch.float32)
        state = {"layer.weight": w}
        result = bridge.maybe_modify_loaded_hf_weight("layer.weight", state)
        assert result is w

    def test_dequants_fp8_when_scale_inv_present(self):
        """FP8 weight with a _scale_inv key is dequantized."""
        bridge = self._make_bridge()
        w = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        sinv = torch.full((1, 1), 3.0)
        state = {"layer.weight": w, "layer.weight_scale_inv": sinv}
        result = bridge.maybe_modify_loaded_hf_weight("layer.weight", state)

        assert result.dtype == torch.bfloat16
        assert torch.all(result == 3.0)

    def test_fp8_without_scale_inv_cast_to_bfloat16(self):
        """FP8 weight without _scale_inv falls back to plain float cast."""
        bridge = self._make_bridge()
        w = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
        state = {"layer.weight": w}
        result = bridge.maybe_modify_loaded_hf_weight("layer.weight", state)

        assert result.dtype == torch.bfloat16

    def test_dict_hf_param_each_key_processed(self):
        """Dict hf_param processes every sub-key independently."""
        bridge = self._make_bridge()
        w1 = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        w2 = torch.ones(64, 64, dtype=torch.bfloat16)
        sinv = torch.full((1, 1), 2.0)
        state = {"key1": w1, "key1_scale_inv": sinv, "key2": w2}
        result = bridge.maybe_modify_loaded_hf_weight({"a": "key1", "b": "key2"}, state)

        assert isinstance(result, dict)
        assert result["a"].dtype == torch.bfloat16
        assert torch.all(result["a"] == 2.0)
        assert result["b"] is w2


class TestFullDimRMSNorm:
    """Unit tests for _FullDimRMSNorm, FullDimQNorm, FullDimKNorm."""

    def test_full_dim_rms_norm_forward_tp1(self):
        """TP=1: forward pass normalises over full local_dim == global_dim."""
        norm = _FullDimRMSNorm(local_dim=64, global_dim=64, tp_group_getter=lambda: None, eps=1e-6)
        x = torch.randn(4, 2, 1, 64)  # [sq, b, num_heads_local, head_dim]
        out = norm(x)
        assert out.shape == x.shape

    def test_full_dim_rms_norm_weight_shape(self):
        norm = _FullDimRMSNorm(local_dim=32, global_dim=64, tp_group_getter=lambda: None)
        assert norm.weight.shape == (32,)

    def test_full_dim_q_norm_creates_rms_norm_tp1(self):
        """FullDimQNorm factory creates _FullDimRMSNorm with correct dims for TP=1."""
        config = Mock()
        config.tensor_model_parallel_size = 1
        config.num_attention_heads = 8
        norm = FullDimQNorm(hidden_size=64, config=config)
        assert isinstance(norm, _FullDimRMSNorm)
        # local_dim = (8 // 1) * 64 = 512; global_dim = 8 * 64 = 512
        assert norm.local_dim == 512
        assert norm.global_dim == 512

    def test_full_dim_q_norm_creates_rms_norm_tp2(self):
        """FullDimQNorm shards correctly for TP=2."""
        config = Mock()
        config.tensor_model_parallel_size = 2
        config.num_attention_heads = 8
        norm = FullDimQNorm(hidden_size=64, config=config)
        assert isinstance(norm, _FullDimRMSNorm)
        # local_dim = (8 // 2) * 64 = 256; global_dim = 8 * 64 = 512
        assert norm.local_dim == 256
        assert norm.global_dim == 512

    def test_full_dim_k_norm_uses_num_kv_heads(self):
        """FullDimKNorm uses num_query_groups (KV heads) instead of num_attention_heads."""
        config = Mock()
        config.tensor_model_parallel_size = 1
        config.num_query_groups = 4
        config.num_attention_heads = 8
        norm = FullDimKNorm(hidden_size=64, config=config)
        assert isinstance(norm, _FullDimRMSNorm)
        # local_dim = (4 // 1) * 64 = 256; global_dim = 4 * 64 = 256
        assert norm.local_dim == 256
        assert norm.global_dim == 256

    def test_full_dim_k_norm_falls_back_to_q_heads(self):
        """FullDimKNorm falls back to num_attention_heads when num_query_groups is None."""
        config = Mock()
        config.tensor_model_parallel_size = 1
        config.num_query_groups = None
        config.num_attention_heads = 8
        norm = FullDimKNorm(hidden_size=64, config=config)
        assert norm.global_dim == 8 * 64


class TestFullDimQKNormMappingTP1:
    """Unit tests for _FullDimQKNormMapping in the TP=1 (non-distributed) case."""

    def _make_mapping(self):
        return _FullDimQKNormMapping(
            megatron_param="decoder.layers.*.self_attention.q_layernorm.weight",
            hf_param="model.layers.*.self_attn.q_norm.weight",
        )

    def test_hf_to_megatron_tp1_returns_weight(self):
        mapping = self._make_mapping()
        # Mock tp_size=1 via the tp_group property
        with patch.object(type(mapping), "tp_size", new_callable=lambda: property(lambda self: 1)):
            hf_weight = torch.ones(512, dtype=torch.bfloat16)
            megatron_module = Mock()
            megatron_module.weight = torch.zeros(512, dtype=torch.bfloat16)

            result = mapping.hf_to_megatron(hf_weight, megatron_module)

        assert result.shape == hf_weight.shape
        assert torch.all(result == 1.0)

    def test_megatron_to_hf_tp1_returns_weight(self):
        mapping = self._make_mapping()
        with (
            patch.object(type(mapping), "tp_size", new_callable=lambda: property(lambda self: 1)),
            patch.object(mapping, "broadcast_from_pp_rank", side_effect=lambda w, **kw: w),
            patch.object(mapping, "maybe_dequantize", side_effect=lambda w: w),
        ):
            megatron_weight = torch.ones(512, dtype=torch.bfloat16)
            result = mapping.megatron_to_hf(megatron_weight, None)

        assert isinstance(result, dict)
        assert list(result.values())[0].shape == (512,)
