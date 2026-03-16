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

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider


pytestmark = pytest.mark.unit


def _make_mock_text_config():
    """Create a mock text_config matching Kimi K2.5 VL's DeepSeek-v3-style backbone."""
    tc = Mock(spec=[])
    tc.num_hidden_layers = 4
    tc.hidden_size = 7168
    tc.intermediate_size = 18432
    tc.num_attention_heads = 64
    tc.num_key_value_heads = 64
    tc.vocab_size = 163840
    tc.max_position_embeddings = 262144
    tc.hidden_act = "silu"
    tc.rms_norm_eps = 1e-5
    tc.initializer_range = 0.006
    tc.tie_word_embeddings = False
    tc.bos_token_id = 163584
    tc.eos_token_id = 163585
    tc.torch_dtype = "bfloat16"
    # MLA
    tc.q_lora_rank = 1536
    tc.kv_lora_rank = 512
    tc.qk_nope_head_dim = 128
    tc.qk_rope_head_dim = 64
    tc.v_head_dim = 128
    tc.attention_bias = False
    tc.attention_dropout = 0.0
    # MoE
    tc.n_routed_experts = 8
    tc.n_shared_experts = 1
    tc.num_experts_per_tok = 2
    tc.moe_intermediate_size = 2048
    tc.moe_layer_freq = 1
    tc.first_k_dense_replace = 1
    tc.n_group = 1
    tc.topk_group = 1
    tc.scoring_func = "sigmoid"
    tc.routed_scaling_factor = 2.827
    tc.aux_loss_alpha = 0.001
    # RoPE
    tc.rope_theta = 50000.0
    tc.rope_scaling = {
        "type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    return tc


def _make_mock_vision_config():
    """Create a mock vision_config for Kimi K2.5's MoonViT3d."""
    vc = Mock(spec=[])
    vc.vt_hidden_size = 1152
    vc.vt_intermediate_size = 4304
    vc.vt_num_attention_heads = 16
    vc.vt_num_hidden_layers = 27
    vc.mm_hidden_size = 1152
    vc.text_hidden_size = 7168
    vc.patch_size = 14
    vc.mm_projector_type = "patchmerger"
    vc.merge_kernel_size = [2, 2]
    return vc


@pytest.fixture
def mock_hf_pretrained():
    pretrained = Mock(spec=PreTrainedVLM)
    hf_config = Mock()
    hf_config.text_config = _make_mock_text_config()
    hf_config.vision_config = _make_mock_vision_config()
    hf_config.tie_word_embeddings = False
    hf_config.media_placeholder_token_id = 163605
    hf_config.pad_token_id = 163839
    hf_config.ignore_index = -100
    pretrained.config = hf_config
    pretrained.generation_config = None
    pretrained._model_name_or_path = "moonshotai/Kimi-K2.5"
    return pretrained


@pytest.fixture
def bridge():
    return KimiK25VLBridge()


class TestKimiK25VLBridgeInit:
    def test_bridge_creates(self, bridge):
        assert isinstance(bridge, KimiK25VLBridge)

    def test_bridge_has_required_methods(self, bridge):
        assert callable(bridge.provider_bridge)
        assert callable(bridge.mapping_registry)
        assert callable(bridge.maybe_modify_loaded_hf_weight)
        assert callable(bridge.maybe_modify_converted_hf_weight)


class TestKimiK25VLBridgeProviderBridge:
    def test_returns_correct_provider_type(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert isinstance(provider, KimiK25VLModelProvider)

    def test_language_config_mapping(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.num_layers == 4
        assert provider.hidden_size == 7168
        assert provider.ffn_hidden_size == 18432
        assert provider.num_attention_heads == 64
        assert provider.vocab_size == 163840

    def test_mla_config_mapping(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.multi_latent_attention is True
        assert provider.q_lora_rank == 1536
        assert provider.kv_lora_rank == 512
        assert provider.qk_head_dim == 128
        assert provider.qk_pos_emb_head_dim == 64
        assert provider.v_head_dim == 128

    def test_moe_config_mapping(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.num_moe_experts == 8
        assert provider.moe_router_topk == 2
        assert provider.moe_ffn_hidden_size == 2048
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_aux_loss_coeff == 0.001

    def test_moe_layer_freq(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert isinstance(provider.moe_layer_freq, list)
        assert provider.moe_layer_freq == [0, 1, 1, 1]

    def test_vision_config_passthrough(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.vision_config is not None
        assert provider.vision_config.vt_hidden_size == 1152

    def test_token_ids(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.image_token_id == 163605
        assert provider.media_placeholder_token_id == 163605
        assert provider.pad_token_id == 163839
        assert provider.bos_token_id == 163584
        assert provider.eos_token_id == 163585
        assert provider.ignore_index == -100

    def test_tie_word_embeddings_from_top_level(self, bridge, mock_hf_pretrained):
        """tie_word_embeddings should come from top-level config, not text_config."""
        mock_hf_pretrained.config.tie_word_embeddings = False
        mock_hf_pretrained.config.text_config.tie_word_embeddings = True
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.share_embeddings_and_output_weights is False

    def test_rope_scaling(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.rotary_scaling_factor == 64.0
        assert provider.mscale == 1.0
        assert provider.mscale_all_dim == 1.0

    def test_hf_model_path(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.hf_model_path == "moonshotai/Kimi-K2.5"

    def test_scatter_embedding_disabled(self, bridge, mock_hf_pretrained):
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.scatter_embedding_sequence_parallel is False

    @patch.object(KimiK25VLBridge, "dtype_from_hf")
    def test_bf16_dtype(self, mock_dtype, bridge, mock_hf_pretrained):
        mock_dtype.return_value = torch.bfloat16
        provider = bridge.provider_bridge(mock_hf_pretrained)
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16

    def test_config_restore_on_error(self, bridge, mock_hf_pretrained):
        """Ensure hf_pretrained.config is restored even if get_common_configs fails."""
        original_config = mock_hf_pretrained.config
        with patch(
            "megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge.get_common_configs",
            side_effect=RuntimeError("test error"),
        ):
            with pytest.raises(RuntimeError, match="test error"):
                bridge.provider_bridge(mock_hf_pretrained)
        assert mock_hf_pretrained.config is original_config


class TestKimiK25VLBridgeMappingRegistry:
    def test_returns_correct_type(self, bridge):
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

    def test_has_language_model_prefix(self, bridge):
        """All language mappings should have language_model. prefix."""
        registry = bridge.mapping_registry()
        for mapping in registry.mappings:
            megatron_param = getattr(mapping, "megatron_param", "")
            if isinstance(mapping, ReplicatedMapping):
                continue
            assert megatron_param.startswith("language_model."), f"Language mapping missing prefix: {megatron_param}"

    def test_has_vision_mappings(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = [getattr(m, "megatron_param", "") for m in registry.mappings]
        assert any("vision_tower" in p for p in megatron_params)
        assert any("mm_projector" in p for p in megatron_params)

    def test_has_expert_bias_mapping(self, bridge):
        registry = bridge.mapping_registry()
        hf_params = []
        for m in registry.mappings:
            hf = getattr(m, "hf_param", None)
            if isinstance(hf, str):
                hf_params.append(hf)
            elif isinstance(hf, dict):
                hf_params.extend(hf.values())
        assert any("e_score_correction_bias" in p for p in hf_params)

    def test_hf_language_params_have_prefix(self, bridge):
        """HF params for language mappings should have language_model.model prefix."""
        registry = bridge.mapping_registry()
        for mapping in registry.mappings:
            if isinstance(mapping, ReplicatedMapping):
                continue
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, str):
                assert hf.startswith("language_model."), f"Missing prefix: {hf}"
            elif isinstance(hf, dict):
                for v in hf.values():
                    assert v.startswith("language_model."), f"Missing prefix: {v}"

    def test_vision_mappings_are_replicated(self, bridge):
        registry = bridge.mapping_registry()
        for mapping in registry.mappings:
            mp = getattr(mapping, "megatron_param", "")
            if "vision_tower" in mp or "mm_projector" in mp:
                assert isinstance(mapping, ReplicatedMapping)


class TestKimiK25VLBridgeDequantization:
    def test_maybe_modify_loaded_hf_weight_passthrough(self, bridge):
        """Non-quantized weight should pass through unchanged."""
        state_dict = {"language_model.model.norm.weight": torch.randn(128)}
        result = bridge.maybe_modify_loaded_hf_weight("language_model.model.norm.weight", state_dict)
        assert torch.equal(result, state_dict["language_model.model.norm.weight"])

    def test_maybe_modify_loaded_hf_weight_int4(self, bridge):
        """INT4 packed weight should be dequantized."""
        out_features, in_features = 64, 128
        packed = torch.randint(0, 2**31, (out_features, in_features // 8), dtype=torch.int32)
        scale = torch.randn(out_features, in_features // 32, dtype=torch.float16)
        shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        state_dict = {
            "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed": packed,
            "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale": scale,
            "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_shape": shape,
        }
        result = bridge.maybe_modify_loaded_hf_weight(
            "language_model.model.layers.1.mlp.experts.0.gate_proj.weight", state_dict
        )
        assert result.shape == (out_features, in_features)
        assert result.dtype == torch.bfloat16

    def test_maybe_modify_loaded_hf_weight_dict(self, bridge):
        """Dict of params (e.g., QKV) should also pass through."""
        state_dict = {
            "q": torch.randn(64, 128),
            "k": torch.randn(64, 128),
        }
        result = bridge.maybe_modify_loaded_hf_weight({"q_key": "q", "k_key": "k"}, state_dict)
        assert isinstance(result, dict)
        assert "q_key" in result
        assert "k_key" in result

    def test_is_quantized_expert_key(self, bridge):
        assert bridge._is_quantized_expert_key("model.layers.2.mlp.experts.0.gate_proj.weight")
        assert not bridge._is_quantized_expert_key("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert not bridge._is_quantized_expert_key("model.layers.2.mlp.shared_experts.gate_proj.weight")
        assert not bridge._is_quantized_expert_key("model.norm.weight")


class TestKimiK25VLProviderDefaults:
    def test_default_token_ids(self):
        provider = KimiK25VLModelProvider()
        assert provider.image_token_id == 163605
        assert provider.media_placeholder_token_id == 163605
        assert provider.pad_token_id == 163839
        assert provider.bos_token_id == 163584
        assert provider.eos_token_id == 163585

    def test_freeze_defaults(self):
        provider = KimiK25VLModelProvider()
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_scatter_embedding_disabled(self):
        provider = KimiK25VLModelProvider()
        assert provider.scatter_embedding_sequence_parallel is False

    def test_inherits_kimi_k2(self):
        from megatron.bridge.models.kimi.kimi_provider import KimiK2Provider

        provider = KimiK25VLModelProvider()
        assert isinstance(provider, KimiK2Provider)

    def test_has_provide_language_model(self):
        provider = KimiK25VLModelProvider()
        assert callable(provider.provide_language_model)
