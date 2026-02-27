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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_audio.qwen2_audio_bridge import Qwen2AudioBridge
from megatron.bridge.models.qwen_audio.qwen2_audio_provider import Qwen2AudioModelProvider


@pytest.fixture
def mock_text_config():
    """Create a mock text config for Qwen2-Audio."""
    text_config = Mock()
    text_config.num_hidden_layers = 32
    text_config.hidden_size = 4096
    text_config.intermediate_size = 11008
    text_config.num_attention_heads = 32
    text_config.num_key_value_heads = 32
    text_config.initializer_range = 0.02
    text_config.rms_norm_eps = 1e-6
    text_config.vocab_size = 151936
    text_config.max_position_embeddings = 4096
    text_config.rope_theta = 1000000.0
    text_config.tie_word_embeddings = False
    text_config.hidden_act = "silu"
    text_config.q_lora_rank = None
    text_config.kv_lora_rank = None
    text_config.qk_nope_head_dim = None
    text_config.qk_rope_head_dim = None
    text_config.v_head_dim = None
    text_config.num_nextn_predict_layers = None
    text_config.rope_scaling = None
    text_config.bos_token_id = 151643
    text_config.eos_token_id = 151645
    return text_config


@pytest.fixture
def mock_audio_config():
    """Create a mock audio encoder config for Qwen2-Audio."""
    audio_config = Mock()
    audio_config.d_model = 1280
    audio_config.encoder_layers = 32
    audio_config.encoder_attention_heads = 20
    audio_config.encoder_ffn_dim = 5120
    return audio_config


@pytest.fixture
def mock_hf_config(mock_text_config, mock_audio_config):
    """Create a mock HF config for Qwen2-Audio."""
    config = Mock()
    config.text_config = mock_text_config
    config.audio_config = mock_audio_config
    config.tie_word_embeddings = False
    config.audio_token_index = 151646
    config.bos_token_id = 151643
    config.eos_token_id = 151645
    config.pad_token_id = 151643
    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    return pretrained


@pytest.fixture
def qwen2_audio_bridge():
    """Create a Qwen2AudioBridge instance."""
    return Qwen2AudioBridge()


class TestQwen2AudioBridgeInitialization:
    """Test Qwen2AudioBridge initialization and basic functionality."""

    def test_bridge_initialization(self, qwen2_audio_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(qwen2_audio_bridge, Qwen2AudioBridge)

    def test_bridge_has_required_methods(self, qwen2_audio_bridge):
        """Test that bridge has required methods."""
        assert hasattr(qwen2_audio_bridge, "provider_bridge")
        assert callable(qwen2_audio_bridge.provider_bridge)

        assert hasattr(qwen2_audio_bridge, "mapping_registry")
        assert callable(qwen2_audio_bridge.mapping_registry)


class TestQwen2AudioBridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Qwen2AudioModelProvider)

        # Check basic transformer config
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 11008
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 32
        assert provider.init_method_std == 0.02
        assert provider.layernorm_epsilon == 1e-6
        assert provider.vocab_size == 151936
        assert provider.seq_length == 4096
        assert provider.rotary_base == 1000000.0
        assert provider.share_embeddings_and_output_weights is False

    def test_provider_bridge_audio_specific_config(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct audio-specific configuration."""
        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        # Check audio-specific token IDs
        assert provider.audio_token_id == 151646
        assert provider.bos_token_id == 151643
        assert provider.eos_token_id == 151645
        assert provider.pad_token_id == 151643

        # Check hf_config is propagated
        assert provider.hf_config is mock_hf_pretrained.config

        # Check Qwen2-specific settings
        assert provider.add_qkv_bias is True

    def test_provider_bridge_qwen2_settings(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge sets Qwen2-specific settings correctly."""
        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_qkv_bias is True
        assert provider.add_bias_linear is False
        assert provider.hidden_dropout == 0.0

    def test_provider_bridge_with_custom_token_ids(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom token IDs from config."""
        mock_hf_pretrained.config.audio_token_index = 200000
        mock_hf_pretrained.config.bos_token_id = 200001
        mock_hf_pretrained.config.eos_token_id = 200002
        mock_hf_pretrained.config.pad_token_id = 200003

        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.audio_token_id == 200000
        assert provider.bos_token_id == 200001
        assert provider.eos_token_id == 200002
        assert provider.pad_token_id == 200003

    def test_provider_bridge_with_tied_embeddings(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge with tied embeddings."""
        mock_hf_pretrained.config.text_config.tie_word_embeddings = True

        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.share_embeddings_and_output_weights is True

    @patch.object(Qwen2AudioBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_handling_fp16(self, mock_dtype_from_hf, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge handles fp16 dtype correctly."""
        mock_dtype_from_hf.return_value = torch.float16

        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    @patch.object(Qwen2AudioBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_handling_bf16(self, mock_dtype_from_hf, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge handles bfloat16 dtype correctly."""
        mock_dtype_from_hf.return_value = torch.bfloat16

        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    @patch.object(Qwen2AudioBridge, "make_vocab_size_divisible_by")
    def test_provider_bridge_vocab_size_divisibility(self, mock_divisible, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge handles vocab size divisibility."""
        mock_divisible.return_value = 128

        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        mock_divisible.assert_called_once_with(151936)
        assert provider.make_vocab_size_divisible_by == 128


class TestQwen2AudioBridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def _get_mapping_names(self, registry):
        """Helper to extract all mapping param names from a registry."""
        mapping_names = []
        for mapping in registry.mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)
        return mapping_names

    def test_mapping_registry_returns_correct_type(self, qwen2_audio_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = qwen2_audio_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_embeddings(self, qwen2_audio_bridge):
        """Test mapping_registry contains word embeddings mapping."""
        registry = qwen2_audio_bridge.mapping_registry()
        mapping_names = self._get_mapping_names(registry)

        has_embeddings = any("embed_tokens" in name or "word_embeddings" in name for name in mapping_names)
        assert has_embeddings, "Should contain embeddings mapping"

    def test_mapping_registry_contains_audio_tower(self, qwen2_audio_bridge):
        """Test mapping_registry contains audio_tower mapping."""
        registry = qwen2_audio_bridge.mapping_registry()
        mapping_names = self._get_mapping_names(registry)

        has_audio_tower = any("audio_tower" in name for name in mapping_names)
        assert has_audio_tower, "Should contain audio_tower mapping"

    def test_mapping_registry_contains_projector(self, qwen2_audio_bridge):
        """Test mapping_registry contains multi_modal_projector mapping."""
        registry = qwen2_audio_bridge.mapping_registry()
        mapping_names = self._get_mapping_names(registry)

        has_projector = any("multi_modal_projector" in name for name in mapping_names)
        assert has_projector, "Should contain multi_modal_projector mapping"

    def test_mapping_registry_contains_qkv(self, qwen2_audio_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = qwen2_audio_bridge.mapping_registry()
        mapping_names = self._get_mapping_names(registry)

        has_qkv = any("linear_qkv" in name for name in mapping_names)
        assert has_qkv, "Should contain QKV mappings"

    def test_mapping_registry_contains_mlp(self, qwen2_audio_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = qwen2_audio_bridge.mapping_registry()
        mapping_names = self._get_mapping_names(registry)

        has_mlp = any("mlp" in name for name in mapping_names)
        assert has_mlp, "Should contain MLP mappings"


class TestQwen2AudioBridgeEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_bridge_with_minimal_config(self, qwen2_audio_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedVLM)
        minimal_config = Mock()

        text_config = Mock()
        text_config.num_hidden_layers = 24
        text_config.hidden_size = 2048
        text_config.intermediate_size = 5504
        text_config.num_attention_heads = 16
        text_config.num_key_value_heads = 16
        text_config.initializer_range = 0.02
        text_config.rms_norm_eps = 1e-6
        text_config.vocab_size = 151936
        text_config.max_position_embeddings = 4096
        text_config.rope_theta = 1000000.0
        text_config.hidden_act = "silu"
        text_config.tie_word_embeddings = False
        text_config.q_lora_rank = None
        text_config.kv_lora_rank = None
        text_config.qk_nope_head_dim = None
        text_config.qk_rope_head_dim = None
        text_config.v_head_dim = None
        text_config.num_nextn_predict_layers = None
        text_config.rope_scaling = None

        minimal_config.text_config = text_config
        minimal_config.tie_word_embeddings = False
        minimal_config.audio_token_index = 151646
        minimal_config.bos_token_id = 151643
        minimal_config.eos_token_id = 151645
        minimal_config.pad_token_id = 151643
        minimal_pretrained.config = minimal_config

        provider = qwen2_audio_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, Qwen2AudioModelProvider)
        assert provider.num_layers == 24
        assert provider.hidden_size == 2048

    def test_provider_bridge_with_different_vocab_sizes(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge with different vocabulary sizes."""
        test_vocab_sizes = [32000, 151936, 152064]

        for vocab_size in test_vocab_sizes:
            mock_hf_pretrained.config.text_config.vocab_size = vocab_size
            provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.vocab_size == vocab_size

    def test_provider_bridge_with_different_sequence_lengths(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge with different sequence lengths."""
        test_seq_lengths = [2048, 4096, 8192, 32768]

        for seq_length in test_seq_lengths:
            mock_hf_pretrained.config.text_config.max_position_embeddings = seq_length
            provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.seq_length == seq_length


class TestQwen2AudioBridgeCompatibility:
    """Test compatibility with different HF model configurations."""

    def test_provider_bridge_with_group_query_attention(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge with group query attention."""
        mock_hf_pretrained.config.text_config.num_attention_heads = 32
        mock_hf_pretrained.config.text_config.num_key_value_heads = 8

        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8

    def test_provider_bridge_with_different_rope_theta(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge with different RoPE theta values."""
        test_rope_values = [10000.0, 500000.0, 1000000.0]

        for rope_theta in test_rope_values:
            mock_hf_pretrained.config.text_config.rope_theta = rope_theta
            provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.rotary_base == rope_theta

    def test_provider_bridge_with_missing_audio_token_index(self, qwen2_audio_bridge, mock_hf_pretrained):
        """Test provider_bridge with missing audio_token_index uses default."""
        delattr(mock_hf_pretrained.config, "audio_token_index")

        provider = qwen2_audio_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.audio_token_id == 151646
