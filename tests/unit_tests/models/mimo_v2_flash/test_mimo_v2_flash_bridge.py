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

"""Unit tests for MiMo-V2-Flash bridge."""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import (
    MiMoV2FlashBridge,
    MiMoV2FlashQKVMapping,
    _dequant_fp8_blockwise,
)
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import (
    MiMoV2FlashModelProvider,
    mimo_v2_flash_layer_spec,
)


_MIMO_V2_FLASH_CONFIG = {
    # HF metadata
    "architectures": ["MiMoV2FlashForCausalLM"],
    "model_type": "mimo_v2_flash",
    "transformers_version": "4.40.1",
    "use_cache": True,
    "torch_dtype": "bfloat16",
    # Core architecture
    "num_hidden_layers": 6,
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 192,
    "vocab_size": 152576,
    "max_position_embeddings": 262144,
    "rope_theta": 5000000,
    "rms_norm_eps": 1e-5,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "partial_rotary_factor": 0.334,
    "rope_scaling": None,
    "use_qk_norm": False,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    # MiMo-specific
    "layernorm_epsilon": 1e-5,
    "v_head_dim": 128,
    "hybrid_layer_pattern": [0, 1, 1, 1, 0, 1],
    "sliding_window_size": 128,
    "sliding_window": 128,
    "attention_chunk_size": 128,
    "swa_rope_theta": 10000,
    "swa_num_key_value_heads": 8,
    "swa_num_attention_heads": 64,
    "swa_head_dim": 192,
    "swa_v_head_dim": 128,
    "add_swa_attention_sink_bias": True,
    "add_full_attention_sink_bias": False,
    "attention_value_scale": 0.707,
    # MoE
    "moe_layer_freq": [0, 1, 1, 1, 1, 1],
    "n_routed_experts": 256,
    "moe_intermediate_size": 2048,
    "num_experts_per_tok": 8,
    "scoring_func": "sigmoid",
    "n_shared_experts": None,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": None,
}


def _make_mock_config():
    """Create a mock HF config. Uses Mock(spec=) to prevent auto-generated attributes."""
    config = Mock(spec=list(_MIMO_V2_FLASH_CONFIG.keys()))
    for k, v in _MIMO_V2_FLASH_CONFIG.items():
        setattr(config, k, v)
    return config


def _make_mock_pretrained(with_state=False):
    """Create a mock PreTrainedCausalLM with MiMo config."""
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = _make_mock_config()
    if not with_state:
        del pretrained.state
    return pretrained


class TestMiMoV2FlashBridgeProviderBridge:
    @pytest.fixture
    def bridge(self):
        return MiMoV2FlashBridge()

    @pytest.fixture
    def mock_pretrained(self):
        return _make_mock_pretrained()

    @pytest.fixture
    def provider(self, bridge, mock_pretrained):
        return bridge.provider_bridge(mock_pretrained)

    def test_provider_type(self, provider):
        assert isinstance(provider, MiMoV2FlashModelProvider)

    def test_dual_rope_bases(self, provider):
        assert provider.rotary_base == (10000, 5000000)

    def test_hybrid_attention_pattern(self, provider):
        assert isinstance(provider.hybrid_attention_pattern, list)
        assert len(provider.hybrid_attention_pattern) == 6
        assert provider.hybrid_attention_pattern[0] == 0

    def test_per_layer_kv_heads(self, provider):
        assert provider.full_attn_num_query_groups == 4
        assert provider.swa_num_query_groups == 8
        assert provider.num_query_groups == 4

    def test_v_head_dim(self, provider):
        assert provider.v_head_dim == 128

    def test_window_size(self, provider):
        assert provider.window_size == 128

    def test_layernorm_epsilon(self, provider):
        assert provider.layernorm_epsilon == 1e-5

    def test_attention_sink_bias(self, provider):
        assert provider.add_swa_attention_sink_bias is True
        assert provider.add_full_attention_sink_bias is False

    def test_attention_value_scale(self, provider):
        assert provider.attention_value_scale == 0.707

    def test_moe_noaux_tc(self, provider):
        assert provider.moe_router_load_balancing_type == "none"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_pre_softmax is True
        assert provider.moe_token_dispatcher_type == "alltoall"

    def test_moe_layer_freq(self, provider):
        assert isinstance(provider.moe_layer_freq, list)
        assert provider.moe_layer_freq[0] == 0
        assert all(f == 1 for f in provider.moe_layer_freq[1:])

    def test_custom_layer_spec(self, provider):
        assert provider.transformer_layer_spec is mimo_v2_flash_layer_spec


class TestMiMoV2FlashMTPDetection:
    @pytest.fixture
    def bridge(self):
        return MiMoV2FlashBridge()

    def test_mtp_detected_from_state_source(self, bridge):
        pretrained = _make_mock_pretrained(with_state=True)
        pretrained.state.source = Mock()
        pretrained.state.source.get_all_keys.return_value = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.mtp.layers.0.enorm.weight",
            "model.mtp.layers.0.eh_proj.weight",
            "model.mtp.layers.1.enorm.weight",
            "model.mtp.layers.2.enorm.weight",
        ]
        provider = bridge.provider_bridge(pretrained)
        assert provider.mtp_num_layers == 3

    def test_mtp_zero_when_no_state(self, bridge):
        pretrained = _make_mock_pretrained()  # with_state=False by default
        provider = bridge.provider_bridge(pretrained)
        assert provider.mtp_num_layers == 0

    def test_mtp_zero_when_no_mtp_keys(self, bridge):
        pretrained = _make_mock_pretrained(with_state=True)
        pretrained.state.source = Mock()
        pretrained.state.source.get_all_keys.return_value = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.mlp.gate.weight",
        ]
        provider = bridge.provider_bridge(pretrained)
        assert provider.mtp_num_layers == 0


class TestMiMoV2FlashMegatronToHfConfig:
    def test_roundtrip_hf_to_megatron_to_hf(self):
        """Test roundtrip: HF config → provider_bridge → megatron_to_hf_config preserves values."""
        pretrained = _make_mock_pretrained()
        original = pretrained.config

        bridge = MiMoV2FlashBridge()
        provider = bridge.provider_bridge(pretrained)
        result = MiMoV2FlashBridge.megatron_to_hf_config(provider)

        # Dual RoPE
        assert result["swa_rope_theta"] == original.swa_rope_theta
        assert result["rope_theta"] == original.rope_theta

        # Per-layer KV heads
        assert result["num_key_value_heads"] == original.num_key_value_heads
        assert result["swa_num_key_value_heads"] == original.swa_num_key_value_heads

        # Sliding window
        assert result["sliding_window_size"] == original.sliding_window_size
        assert result["sliding_window"] == original.sliding_window

        # Hybrid attention pattern
        assert result["hybrid_layer_pattern"] == list(original.hybrid_layer_pattern)

        # Asymmetric V head dim
        assert result["v_head_dim"] == original.v_head_dim

        # MoE
        assert result["moe_layer_freq"] == list(original.moe_layer_freq)

        # Attention sink bias
        assert result["add_swa_attention_sink_bias"] == original.add_swa_attention_sink_bias
        assert result["add_full_attention_sink_bias"] == original.add_full_attention_sink_bias

        # Attention value scale
        assert result["attention_value_scale"] == original.attention_value_scale

        # Layernorm epsilon
        assert result["layernorm_epsilon"] == original.layernorm_epsilon

        # norm_topk_prob (hardcoded, not on original config)
        assert result["norm_topk_prob"] is True


class TestMiMoV2FlashBridgeMappingRegistry:
    @pytest.fixture
    def registry(self):
        return MiMoV2FlashBridge().mapping_registry()

    @pytest.fixture
    def megatron_params(self, registry):
        return {m.megatron_param for m in registry.mappings}

    def test_has_all_weight_families(self, megatron_params):
        """Every weight family the model needs must have a mapping."""
        required = {
            # Embeddings
            "word_embeddings": "embedding.word_embeddings.weight",
            "output_layer": "output_layer.weight",
            # Decoder attention
            "decoder_qkv": "decoder.layers.*.self_attention.linear_qkv.weight",
            "decoder_o_proj": "decoder.layers.*.self_attention.linear_proj.weight",
            "decoder_fused_input_ln": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "decoder_sink_bias": "decoder.layers.*.self_attention.core_attention.softmax_offset",
            # Decoder layernorms
            "decoder_pre_mlp_ln": "decoder.layers.*.pre_mlp_layernorm.weight",
            "decoder_dense_mlp_ln": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "decoder_final_ln": "decoder.final_layernorm.weight",
            # Decoder dense MLP
            "decoder_dense_fc1": "decoder.layers.*.mlp.linear_fc1.weight",
            "decoder_dense_fc2": "decoder.layers.*.mlp.linear_fc2.weight",
            # Decoder MoE
            "decoder_expert_fc1": "decoder.layers.*.mlp.experts.linear_fc1",
            "decoder_expert_fc2": "decoder.layers.*.mlp.experts.linear_fc2",
            "decoder_router_weight": "decoder.layers.*.mlp.router.weight",
            "decoder_router_bias": "decoder.layers.*.mlp.router.expert_bias",
            # MTP wrapper
            "mtp_enorm": "mtp.layers.*.enorm.weight",
            "mtp_hnorm": "mtp.layers.*.hnorm.weight",
            "mtp_eh_proj": "mtp.layers.*.eh_proj.weight",
            "mtp_final_ln": "mtp.layers.*.final_layernorm.weight",
            # Decoder non-TE layernorm path
            "decoder_input_ln": "decoder.layers.*.input_layernorm.weight",
            # MTP transformer layer
            "mtp_mml_qkv": "mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
            "mtp_mml_o_proj": "mtp.layers.*.mtp_model_layer.self_attention.linear_proj.weight",
            "mtp_mml_sink_bias": "mtp.layers.*.mtp_model_layer.self_attention.core_attention.softmax_offset",
            "mtp_mml_fused_input_ln": "mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight",
            "mtp_mml_mlp_fc1": "mtp.layers.*.mtp_model_layer.mlp.linear_fc1.weight",
            "mtp_mml_mlp_fc2": "mtp.layers.*.mtp_model_layer.mlp.linear_fc2.weight",
            "mtp_mml_mlp_ln": "mtp.layers.*.mtp_model_layer.mlp.linear_fc1.layer_norm_weight",
            # MTP transformer layer
            "mtp_tl_qkv": "mtp.layers.*.transformer_layer.self_attention.linear_qkv.weight",
            "mtp_tl_o_proj": "mtp.layers.*.transformer_layer.self_attention.linear_proj.weight",
            "mtp_tl_sink_bias": "mtp.layers.*.transformer_layer.self_attention.core_attention.softmax_offset",
            "mtp_tl_fused_input_ln": "mtp.layers.*.transformer_layer.self_attention.linear_qkv.layer_norm_weight",
            "mtp_tl_mlp_fc1": "mtp.layers.*.transformer_layer.mlp.linear_fc1.weight",
            "mtp_tl_mlp_fc2": "mtp.layers.*.transformer_layer.mlp.linear_fc2.weight",
            "mtp_tl_mlp_ln": "mtp.layers.*.transformer_layer.mlp.linear_fc1.layer_norm_weight",
        }
        for name, pattern in required.items():
            assert any(pattern in p for p in megatron_params), f"Missing mapping for {name}: {pattern}"

    def test_qkv_uses_custom_mapping(self, registry):
        """Decoder and MTP QKV must use MiMoV2FlashQKVMapping for asymmetric V dim."""
        qkv_mappings = [m for m in registry.mappings if "linear_qkv.weight" in str(m.megatron_param)]
        assert len(qkv_mappings) > 0
        for m in qkv_mappings:
            assert isinstance(m, MiMoV2FlashQKVMapping), (
                f"Expected MiMoV2FlashQKVMapping for {m.megatron_param}, got {type(m).__name__}"
            )


class TestMiMoV2FlashQKVMapping:
    """Test asymmetric V head dim QKV merge/split via actual MiMoV2FlashQKVMapping (TP=1)."""

    NUM_HEADS = 64
    NUM_QG = 4
    QK_CH = 192
    V_CH = 128
    HIDDEN = 4096

    @pytest.fixture
    def config(self):
        cfg = Mock()
        cfg.num_attention_heads = self.NUM_HEADS
        cfg.num_query_groups = self.NUM_QG
        cfg.kv_channels = self.QK_CH
        cfg.v_head_dim = self.V_CH
        cfg.hidden_size = self.HIDDEN
        return cfg

    @pytest.fixture
    def mapping(self):
        return MiMoV2FlashQKVMapping(
            megatron_param="decoder.layers.0.self_attention.linear_qkv.weight",
            q="model.layers.0.self_attn.q_proj.weight",
            k="model.layers.0.self_attn.k_proj.weight",
            v="model.layers.0.self_attn.v_proj.weight",
        )

    @pytest.fixture
    def qkv(self):
        torch.manual_seed(0)
        q = torch.randn(self.NUM_HEADS * self.QK_CH, self.HIDDEN)
        k = torch.randn(self.NUM_QG * self.QK_CH, self.HIDDEN)
        v = torch.randn(self.NUM_QG * self.V_CH, self.HIDDEN)
        return q, k, v

    def _call_hf_to_megatron(self, mapping, config, qkv):
        """Call hf_to_megatron with TP=1 mocking."""
        q, k, v = qkv
        module = Mock()
        with (
            patch.object(type(mapping), "tp_rank", new_callable=lambda: property(lambda self: 0)),
            patch.object(mapping, "_get_config", return_value=config),
            patch.object(mapping._tp_mapping, "hf_to_megatron", side_effect=lambda m, mod: m),
        ):
            return mapping.hf_to_megatron({"q": q, "k": k, "v": v}, module)

    def _call_megatron_to_hf(self, mapping, config, packed):
        """Call megatron_to_hf with TP=1 mocking."""
        with (
            patch.object(mapping, "maybe_dequantize", side_effect=lambda w: w),
            patch.object(mapping, "broadcast_obj_from_pp_rank", side_effect=lambda obj, *args, **kw: obj),
            patch.object(mapping, "_get_config", return_value=config),
            patch.object(
                mapping._tp_mapping,
                "megatron_to_hf",
                side_effect=lambda w, mod: {mapping.hf_param["q"]: w} if w is not None else {},
            ),
        ):
            return mapping.megatron_to_hf(packed, Mock())

    def test_merge_shape(self, mapping, config, qkv):
        merged = self._call_hf_to_megatron(mapping, config, qkv)
        expected_rows = self.NUM_HEADS * self.QK_CH + self.NUM_QG * self.QK_CH + self.NUM_QG * self.V_CH
        assert merged.shape == (expected_rows, self.HIDDEN)

    def test_split_shapes(self, mapping, config, qkv):
        merged = self._call_hf_to_megatron(mapping, config, qkv)
        result = self._call_megatron_to_hf(mapping, config, merged)
        q_key = mapping.hf_param["q"]
        k_key = mapping.hf_param["k"]
        v_key = mapping.hf_param["v"]
        assert result[q_key].shape == (self.NUM_HEADS * self.QK_CH, self.HIDDEN)
        assert result[k_key].shape == (self.NUM_QG * self.QK_CH, self.HIDDEN)
        assert result[v_key].shape == (self.NUM_QG * self.V_CH, self.HIDDEN)

    def test_roundtrip(self, mapping, config, qkv):
        q, k, v = qkv
        merged = self._call_hf_to_megatron(mapping, config, qkv)
        result = self._call_megatron_to_hf(mapping, config, merged)
        assert torch.equal(q, result[mapping.hf_param["q"]])
        assert torch.equal(k, result[mapping.hf_param["k"]])
        assert torch.equal(v, result[mapping.hf_param["v"]])


class TestDequantFP8Blockwise:
    """Test FP8 block-wise dequantization."""

    def test_scale_inv_applied_per_block(self):
        """scale_inv value is multiplied block-wise, output is fp32."""
        weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
        scale_inv = torch.full((2, 2), 2.0)
        result = _dequant_fp8_blockwise(weight, scale_inv)
        assert result.dtype == torch.float32
        assert result.shape == (256, 256)
        assert torch.all(result == 2.0)

    def test_non_uniform_block_sizes(self):
        """MiMo full-attn k_proj uses different blocks."""
        weight = torch.ones(768, 4096, dtype=torch.float8_e4m3fn)
        scale_inv = torch.full((8, 32), 3.0)
        result = _dequant_fp8_blockwise(weight, scale_inv)
        assert result.shape == (768, 4096)
        assert torch.all(result == 3.0)


class TestMaybeModifyLoadedHFWeight:
    """Test FP8 dequantization hook during weight loading."""

    def _make_bridge(self):
        return MiMoV2FlashBridge()

    def test_passthrough_bfloat16(self):
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

    def test_dequants_fp8_with_scale_inv(self):
        bridge = self._make_bridge()
        w = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        sinv = torch.full((1, 1), 3.0)
        state = {"layer.weight": w, "layer.weight_scale_inv": sinv}
        result = bridge.maybe_modify_loaded_hf_weight("layer.weight", state)
        assert result.dtype == torch.float32
        assert torch.all(result == 3.0)
