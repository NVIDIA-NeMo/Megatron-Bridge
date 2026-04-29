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

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
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
    text_config.rope_scaling = {"mrope_section": [24, 20, 20]}
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
    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    pretrained = Mock(spec=PreTrainedVLM)
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
        assert provider.language_max_sequence_length == 32768
        assert provider.patch_size == 32
        assert provider.temporal_patch_size == 4
        assert provider.spatial_merge_size == 3

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

    def test_provider_bridge_warns_for_audio_output_stack(self, mock_hf_pretrained, caplog):
        mock_hf_pretrained.config.enable_audio_output = True
        caplog.set_level("WARNING")

        bridge = Qwen3OmniBridge()
        provider = bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Qwen3OmniModelProvider)
        assert "converting thinker-side weights only" in caplog.text
