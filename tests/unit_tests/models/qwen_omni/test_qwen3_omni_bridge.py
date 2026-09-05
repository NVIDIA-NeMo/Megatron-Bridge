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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, QKVMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen_omni import Qwen3OmniBridge, Qwen3OmniModelProvider


@pytest.fixture
def mock_text_config():
    text_config = Mock(spec=[])
    text_config.num_hidden_layers = 48
    text_config.hidden_size = 2048
    text_config.intermediate_size = 6144
    text_config.moe_intermediate_size = 768
    text_config.num_attention_heads = 32
    text_config.num_key_value_heads = 4
    text_config.head_dim = 128
    text_config.num_experts = 128
    text_config.num_experts_per_tok = 8
    text_config.initializer_range = 0.02
    text_config.rms_norm_eps = 1e-6
    text_config.vocab_size = 152064
    text_config.max_position_embeddings = 32768
    text_config.rope_theta = 1000000.0
    text_config.attention_bias = False
    text_config.rope_scaling = {
        "rope_type": "default",
        "interleaved": True,
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
    }
    return text_config


@pytest.fixture
def mock_thinker_config(mock_text_config):
    thinker = Mock(spec=[])
    thinker.text_config = mock_text_config
    thinker.torch_dtype = torch.float32
    thinker.image_token_id = 151655
    thinker.video_token_id = 151656
    thinker.audio_token_id = 151646
    thinker.vision_start_token_id = 151652
    thinker.vision_end_token_id = 151753
    thinker.audio_end_token_id = 151748
    vision_config = Mock(spec=[])
    vision_config.patch_size = 32
    vision_config.temporal_patch_size = 4
    vision_config.spatial_merge_size = 3
    thinker.vision_config = vision_config
    return thinker


@pytest.fixture
def mock_hf_config(mock_thinker_config):
    config = Mock()
    config.thinker_config = mock_thinker_config
    config.torch_dtype = torch.float32
    config.enable_audio_output = False
    config.talker_config = Mock()
    config.code2wav_config = Mock()
    config.tie_word_embeddings = True
    config.bos_token_id = 151743
    config.eos_token_id = 151745
    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = mock_hf_config
    return pretrained


class TestQwen3OmniBridge:
    def test_provider_bridge_basic_config(self, mock_hf_pretrained):
        bridge = Qwen3OmniBridge()
        provider = bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Qwen3OmniModelProvider)
        assert provider.num_layers == 48
        assert provider.hidden_size == 2048
        assert provider.ffn_hidden_size == 6144
        assert provider.moe_ffn_hidden_size == 768
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 4
        assert provider.kv_channels == 128
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.num_moe_experts == 128
        assert provider.moe_router_topk == 8
        assert provider.share_embeddings_and_output_weights is False
        assert provider.mrope_section == [24, 20, 20]
        assert provider.rotary_interleaved is False
        assert provider.language_max_sequence_length == 32768
        assert provider.patch_size == 32
        assert provider.temporal_patch_size == 4
        assert provider.spatial_merge_size == 3
        assert provider.image_token_id == 151655
        assert provider.video_token_id == 151656
        assert provider.audio_token_id == 151646
        assert provider.vision_start_token_id == 151652
        assert provider.vision_end_token_id == 151753
        assert provider.audio_start_token_id == 151647
        assert provider.audio_end_token_id == 151748
        assert provider.bos_token_id == 151743
        assert provider.eos_token_id == 151745

    @patch.object(Qwen3OmniBridge, "dtype_from_hf")
    def test_provider_bridge_dtype(self, mock_dtype_from_hf, mock_hf_pretrained):
        mock_dtype_from_hf.return_value = torch.bfloat16
        bridge = Qwen3OmniBridge()
        provider = bridge.provider_bridge(mock_hf_pretrained)

        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16

    def test_mapping_registry(self):
        bridge = Qwen3OmniBridge()
        registry = bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)
        mapping_names = []
        for mapping in registry.mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))

        assert any("thinker.language_model.embedding.word_embeddings.weight" in name for name in mapping_names)
        assert any(
            "thinker.language_model.decoder.layers.*.self_attention.linear_qkv.weight" in name
            for name in mapping_names
        )
        assert any("thinker.language_model.decoder.layers.*.mlp.router.weight" in name for name in mapping_names)

    def test_mapping_registry_resolves_parallel_parameter_names(self):
        bridge = Qwen3OmniBridge()
        registry = bridge.mapping_registry()

        expected = {
            "thinker.language_model.decoder.layers.0.self_attention.linear_qkv.weight": {
                "q": "thinker.model.layers.0.self_attn.q_proj.weight",
                "k": "thinker.model.layers.0.self_attn.k_proj.weight",
                "v": "thinker.model.layers.0.self_attn.v_proj.weight",
            },
            "thinker.language_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": (
                "thinker.model.layers.0.input_layernorm.weight"
            ),
            "thinker.language_model.decoder.layers.0.input_layernorm.weight": (
                "thinker.model.layers.0.input_layernorm.weight"
            ),
            "thinker.language_model.decoder.layers.0.pre_mlp_layernorm.weight": (
                "thinker.model.layers.0.post_attention_layernorm.weight"
            ),
            "thinker.language_model.decoder.layers.0.self_attention.q_layernorm.weight": (
                "thinker.model.layers.0.self_attn.q_norm.weight"
            ),
            "thinker.language_model.decoder.layers.0.self_attention.k_layernorm.weight": (
                "thinker.model.layers.0.self_attn.k_norm.weight"
            ),
            "thinker.language_model.decoder.layers.0.self_attention.linear_proj.weight": (
                "thinker.model.layers.0.self_attn.o_proj.weight"
            ),
            "thinker.language_model.decoder.layers.0.mlp.router.weight": "thinker.model.layers.0.mlp.gate.weight",
            "thinker.language_model.decoder.layers.0.mlp.experts.linear_fc1.weight7": {
                "gate": "thinker.model.layers.0.mlp.experts.7.gate_proj.weight",
                "up": "thinker.model.layers.0.mlp.experts.7.up_proj.weight",
            },
            "thinker.language_model.decoder.layers.0.mlp.experts.linear_fc2.weight7": (
                "thinker.model.layers.0.mlp.experts.7.down_proj.weight"
            ),
            "thinker.language_model.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.weight": {
                "gate": "thinker.model.layers.0.mlp.experts.7.gate_proj.weight",
                "up": "thinker.model.layers.0.mlp.experts.7.up_proj.weight",
            },
            "thinker.language_model.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2.weight": (
                "thinker.model.layers.0.mlp.experts.7.down_proj.weight"
            ),
        }

        for megatron_param, hf_param in expected.items():
            mapping = registry.megatron_to_hf_lookup(megatron_param)
            assert mapping is not None, megatron_param
            assert mapping.hf_param == hf_param

    def test_qkv_mapping_preserves_asymmetric_projection_slices(self):
        registry = Qwen3OmniBridge().mapping_registry()
        mapping = registry.megatron_to_hf_lookup(
            "thinker.language_model.decoder.layers.0.self_attention.linear_qkv.weight"
        )
        assert isinstance(mapping, QKVMapping)

        config = SimpleNamespace(num_attention_heads=4, num_query_groups=2, kv_channels=2, hidden_size=4)
        module = SimpleNamespace(config=config)
        q = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) + 10
        k = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4) + 100
        v = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4) + 200

        with patch.object(mapping._tp_mapping, "hf_to_megatron", side_effect=lambda weight, _module: weight):
            fused = mapping.hf_to_megatron({"q": q, "k": k, "v": v}, module)

        expected = torch.cat([q[:4], k[:2], v[:2], q[4:], k[2:], v[2:]], dim=0)
        torch.testing.assert_close(fused, expected)

        with (
            patch.object(mapping, "broadcast_obj_from_pp_rank", side_effect=lambda value, _key: value),
            patch.object(mapping._tp_mapping, "megatron_to_hf", return_value={mapping.megatron_param: fused}),
        ):
            restored = mapping.megatron_to_hf(fused, module)

        torch.testing.assert_close(restored[mapping.hf_param["q"]], q)
        torch.testing.assert_close(restored[mapping.hf_param["k"]], k)
        torch.testing.assert_close(restored[mapping.hf_param["v"]], v)

    def test_expert_mapping_preserves_asymmetric_gate_up_slices(self):
        registry = Qwen3OmniBridge().mapping_registry()
        mapping = registry.megatron_to_hf_lookup(
            "thinker.language_model.decoder.layers.0.mlp.experts.linear_fc1.weight7"
        )
        assert isinstance(mapping, GatedMLPMapping)

        gate = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 10
        up = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100
        fused = mapping.hf_to_megatron({"gate": gate, "up": up}, SimpleNamespace())
        torch.testing.assert_close(fused, torch.cat([gate, up], dim=0))

        with patch.object(mapping, "broadcast_from_pp_rank", side_effect=lambda value, cache_key=None: value):
            restored = mapping.megatron_to_hf(fused, SimpleNamespace())

        torch.testing.assert_close(restored[mapping.hf_param["gate"]], gate)
        torch.testing.assert_close(restored[mapping.hf_param["up"]], up)

    def test_expert_mapping_exports_gate_up_across_two_ep_ranks(self, monkeypatch):
        registry = Qwen3OmniBridge().mapping_registry()
        mapping = registry.megatron_to_hf_lookup(
            "thinker.language_model.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight"
        )
        assert isinstance(mapping, GatedMLPMapping)

        class _FakeGroup:
            def size(self):
                return 2

        mapping.ep_group = _FakeGroup()
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.param_mapping.get_pg_size",
            lambda group: 1 if group is None else group.size(),
        )
        monkeypatch.setattr(mapping, "broadcast_from_pp_rank", lambda value, cache_key=None: value)
        monkeypatch.setattr(mapping, "broadcast_obj_from_pp_rank", lambda value, cache_key=None: value)

        def fake_all_gather(outputs, local_weight, group):
            assert group is mapping.ep_group
            outputs[0].copy_(local_weight)
            outputs[1].copy_(local_weight + 1000)

        monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

        gate = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 10
        up = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 100
        module = SimpleNamespace(config=SimpleNamespace(num_moe_experts=4))
        restored = mapping.megatron_to_hf(torch.cat([gate, up], dim=0), module)

        gate_name = "thinker.model.layers.0.mlp.experts.{}.gate_proj.weight"
        up_name = "thinker.model.layers.0.mlp.experts.{}.up_proj.weight"
        torch.testing.assert_close(restored[gate_name.format(1)], gate)
        torch.testing.assert_close(restored[gate_name.format(3)], gate + 1000)
        torch.testing.assert_close(restored[up_name.format(1)], up)
        torch.testing.assert_close(restored[up_name.format(3)], up + 1000)

    def test_provider_bridge_warns_for_audio_output_stack(self, mock_hf_pretrained, caplog):
        mock_hf_pretrained.config.enable_audio_output = True
        caplog.set_level("WARNING")

        bridge = Qwen3OmniBridge()
        provider = bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Qwen3OmniModelProvider)
        assert "converting thinker-side weights only" in caplog.text
