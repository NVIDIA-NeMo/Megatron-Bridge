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
from transformers import GenerationConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.qwen25_omni_bridge import Qwen25OmniBridge
from megatron.bridge.models.qwen_vl.qwen_vl_provider import Qwen25VLModelProvider


@pytest.fixture
def mock_text_config():
    """Create a mock text_config for Qwen2.5-Omni."""
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
    text_config.bos_token_id = 151643
    text_config.eos_token_id = 151645
    return text_config


@pytest.fixture
def mock_thinker_config(mock_text_config):
    """Create a mock thinker_config for Qwen2.5-Omni."""
    thinker_config = Mock()
    thinker_config.text_config = mock_text_config
    thinker_config.vision_config = Qwen2_5_VLVisionConfig()
    thinker_config.image_token_id = 151655
    thinker_config.video_token_id = 151656
    thinker_config.vision_start_token_id = 151652
    thinker_config.vision_end_token_id = 151653
    thinker_config.vision_token_id = 151654
    return thinker_config


@pytest.fixture
def mock_hf_omni_config(mock_thinker_config):
    """Create a mock HF config for Qwen2.5-Omni with nested structure."""
    config = Mock()
    config.thinker_config = mock_thinker_config
    return config


@pytest.fixture
def mock_hf_omni_pretrained(mock_hf_omni_config):
    """Create a mock HF pretrained VLM for Qwen2.5-Omni."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_omni_config
    pretrained.generation_config = GenerationConfig()
    return pretrained


@pytest.fixture
def qwen25_omni_bridge():
    """Create a Qwen25OmniBridge instance."""
    return Qwen25OmniBridge()


class TestQwen25OmniBridgeInitialization:
    """Test Qwen25OmniBridge initialization and basic functionality."""

    def test_bridge_initialization(self, qwen25_omni_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(qwen25_omni_bridge, Qwen25OmniBridge)

    def test_bridge_has_required_methods(self, qwen25_omni_bridge):
        """Test that bridge has required methods."""
        assert hasattr(qwen25_omni_bridge, "provider_bridge")
        assert callable(qwen25_omni_bridge.provider_bridge)

        assert hasattr(qwen25_omni_bridge, "mapping_registry")
        assert callable(qwen25_omni_bridge.mapping_registry)


class TestQwen25OmniBridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert isinstance(provider, Qwen25VLModelProvider)

        # Check basic transformer config from text_config
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        assert provider.num_layers == text_config.num_hidden_layers
        assert provider.hidden_size == text_config.hidden_size
        assert provider.ffn_hidden_size == text_config.intermediate_size
        assert provider.num_attention_heads == text_config.num_attention_heads
        assert provider.num_query_groups == text_config.num_key_value_heads
        assert provider.init_method_std == text_config.initializer_range
        assert provider.layernorm_epsilon == text_config.rms_norm_eps
        assert provider.vocab_size == text_config.vocab_size
        assert provider.seq_length == text_config.max_position_embeddings
        assert provider.rotary_base == text_config.rope_theta
        assert provider.share_embeddings_and_output_weights == text_config.tie_word_embeddings

    def test_provider_bridge_vl_specific_config(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge creates correct VL-specific configuration."""
        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        thinker_config = mock_hf_omni_pretrained.config.thinker_config
        text_config = thinker_config.text_config

        # Check VL-specific token IDs from thinker_config
        assert provider.image_token_id == 151655
        assert provider.video_token_id == 151656
        assert provider.vision_start_token_id == 151652
        assert provider.vision_end_token_id == 151653
        assert provider.vision_token_id == 151654

        # Check token IDs from text_config
        assert provider.bos_token_id == text_config.bos_token_id
        assert provider.eos_token_id == text_config.eos_token_id

        # Check vision config
        assert isinstance(provider.vision_config, Qwen2_5_VLVisionConfig)
        assert provider.add_qkv_bias is True

    def test_provider_bridge_with_custom_token_ids(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge with custom token IDs."""
        thinker_config = mock_hf_omni_pretrained.config.thinker_config
        text_config = thinker_config.text_config

        # Modify token IDs
        text_config.bos_token_id = 100
        text_config.eos_token_id = 101
        thinker_config.vision_start_token_id = 102
        thinker_config.image_token_id = 103

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.vision_start_token_id == 102
        assert provider.image_token_id == 103

    def test_provider_bridge_with_missing_token_ids(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge with missing token IDs uses defaults."""
        thinker_config = mock_hf_omni_pretrained.config.thinker_config

        # Remove some token IDs and their fallback attributes
        delattr(thinker_config, "vision_start_token_id")
        delattr(thinker_config, "image_token_id")
        
        if hasattr(thinker_config, "image_token_index"):
            delattr(thinker_config, "image_token_index")

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)
        # Should use defaults
        assert provider.vision_start_token_id == 151652
        assert provider.image_token_id == 151655

    def test_provider_bridge_with_image_token_index_fallback(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge falls back to image_token_index if image_token_id is missing."""
        thinker_config = mock_hf_omni_pretrained.config.thinker_config

        # Remove image_token_id but set image_token_index
        delattr(thinker_config, "image_token_id")
        thinker_config.image_token_index = 999

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.image_token_id == 999

    def test_provider_bridge_with_video_token_index_fallback(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge falls back to video_token_index if video_token_id is missing."""
        thinker_config = mock_hf_omni_pretrained.config.thinker_config

        # Remove video_token_id but set video_token_index
        delattr(thinker_config, "video_token_id")
        thinker_config.video_token_index = 888

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.video_token_id == 888

    @patch.object(Qwen25OmniBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_handling(self, mock_dtype_from_hf, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge handles dtype correctly."""
        mock_dtype_from_hf.return_value = torch.float16

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    @patch.object(Qwen25OmniBridge, "dtype_from_hf")
    def test_provider_bridge_bfloat16_handling(self, mock_dtype_from_hf, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge handles bfloat16 correctly."""
        mock_dtype_from_hf.return_value = torch.bfloat16

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    @patch.object(Qwen25OmniBridge, "make_vocab_size_divisible_by")
    def test_provider_bridge_vocab_size_divisibility(self, mock_divisible, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge handles vocab size divisibility."""
        mock_divisible.return_value = 128

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        mock_divisible.assert_called_once_with(text_config.vocab_size)
        assert provider.make_vocab_size_divisible_by == 128

    def test_provider_bridge_with_tied_embeddings(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge with tied embeddings."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        text_config.tie_word_embeddings = True

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.share_embeddings_and_output_weights is True

    def test_provider_bridge_generation_config(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge includes generation config."""
        custom_gen_config = GenerationConfig(max_length=2048, temperature=0.8)
        mock_hf_omni_pretrained.generation_config = custom_gen_config

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.generation_config is custom_gen_config

    def test_provider_bridge_mrope_section_from_dict(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge extracts mRoPE section from dict rope_parameters."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        text_config.rope_parameters = {"mrope_section": [20, 30, 30], "rope_theta": 2000000.0}

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.mrope_section == [20, 30, 30]

    def test_provider_bridge_mrope_section_from_object(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge extracts mRoPE section from object rope_parameters."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        rope_params = Mock()
        rope_params.mrope_section = [24, 32, 32]
        text_config.rope_parameters = rope_params

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.mrope_section == [24, 32, 32]

    def test_provider_bridge_mrope_section_default(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge uses default mRoPE section when not provided."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        # Remove rope_parameters
        delattr(text_config, "rope_parameters")

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.mrope_section == [16, 24, 24]

    def test_provider_bridge_rope_theta_from_text_config(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge extracts rope_theta from text_config."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        text_config.rope_theta = 2000000.0

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.rotary_base == 2000000.0

    def test_provider_bridge_rope_theta_from_rope_parameters_dict(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge extracts rope_theta from rope_parameters dict."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        delattr(text_config, "rope_theta")
        text_config.rope_parameters = {"rope_theta": 3000000.0}

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.rotary_base == 3000000.0

    def test_provider_bridge_rope_theta_from_rope_parameters_object(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge extracts rope_theta from rope_parameters object."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        delattr(text_config, "rope_theta")
        rope_params = Mock()
        rope_params.rope_theta = 4000000.0
        text_config.rope_parameters = rope_params

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.rotary_base == 4000000.0

    def test_provider_bridge_rope_theta_default(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge uses default rope_theta when not provided."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        delattr(text_config, "rope_theta")
        delattr(text_config, "rope_parameters")

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.rotary_base == 1000000.0

    def test_provider_bridge_missing_thinker_config(self, qwen25_omni_bridge):
        """Test provider_bridge raises ValueError when thinker_config is missing."""
        pretrained = Mock(spec=PreTrainedVLM)
        config = Mock()
        delattr(config, "thinker_config")
        pretrained.config = config

        with pytest.raises(ValueError, match="thinker_config"):
            qwen25_omni_bridge.provider_bridge(pretrained)

    def test_provider_bridge_missing_text_config(self, qwen25_omni_bridge):
        """Test provider_bridge raises ValueError when text_config is missing."""
        pretrained = Mock(spec=PreTrainedVLM)
        config = Mock()
        thinker_config = Mock()
        delattr(thinker_config, "text_config")
        config.thinker_config = thinker_config
        pretrained.config = config

        with pytest.raises(ValueError, match="text_config"):
            qwen25_omni_bridge.provider_bridge(pretrained)


class TestQwen25OmniBridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_correct_type(self, qwen25_omni_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = qwen25_omni_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_required_mappings(self, qwen25_omni_bridge):
        """Test mapping_registry contains all required parameter mappings."""
        registry = qwen25_omni_bridge.mapping_registry()

        # Extract mappings - registry should contain mappings for common parameters
        mappings = registry.mappings
        assert len(mappings) > 0

        # Check that we have mappings for embeddings, output layer, layernorms
        mapping_names = []
        for mapping in mappings:
            # Collect Megatron param pattern
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            # Collect HF param pattern(s)
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain word embeddings mapping (thinker.model.embed_tokens)
        has_embeddings = any("embed_tokens" in name or "word_embeddings" in name for name in mapping_names)
        assert has_embeddings, "Should contain embeddings mapping"

        # Should contain output layer mapping (thinker.lm_head)
        has_output = any("lm_head" in name or "output_layer" in name for name in mapping_names)
        assert has_output, "Should contain output layer mapping"

    def test_mapping_registry_thinker_model_mappings(self, qwen25_omni_bridge):
        """Test mapping_registry contains thinker.model.* parameter mappings."""
        registry = qwen25_omni_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain thinker.model.* mappings
        has_thinker_model = any("thinker.model" in name for name in mapping_names)
        assert has_thinker_model, "Should contain thinker.model.* mappings"

    def test_mapping_registry_visual_params(self, qwen25_omni_bridge):
        """Test mapping_registry handles visual parameters correctly."""
        registry = qwen25_omni_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        has_visual = any("visual" in name or "thinker.visual" in name for name in mapping_names)
        assert has_visual, "Should contain visual parameter mappings"

    def test_mapping_registry_audio_tower_params(self, qwen25_omni_bridge):
        """Test mapping_registry handles audio tower parameters correctly."""
        registry = qwen25_omni_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        has_audio = any("audio" in name or "audio_tower" in name for name in mapping_names)
        assert has_audio, "Should contain audio_tower parameter mappings"

    def test_mapping_registry_talker_params(self, qwen25_omni_bridge):
        """Test mapping_registry handles talker parameters correctly."""
        registry = qwen25_omni_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        has_talker = any("talker" in name for name in mapping_names)
        assert has_talker, "Should contain talker parameter mappings"

    def test_mapping_registry_token2wav_params(self, qwen25_omni_bridge):
        """Test mapping_registry handles token2wav parameters correctly."""
        registry = qwen25_omni_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        has_token2wav = any("token2wav" in name for name in mapping_names)
        assert has_token2wav, "Should contain token2wav parameter mappings"

    def test_mapping_registry_qkv_mappings(self, qwen25_omni_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = qwen25_omni_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain QKV mappings
        has_qkv = any("linear_qkv" in name for name in mapping_names)
        assert has_qkv, "Should contain QKV mappings"

    def test_mapping_registry_mlp_mappings(self, qwen25_omni_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = qwen25_omni_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain MLP mappings
        has_mlp = any("mlp" in name for name in mapping_names)
        assert has_mlp, "Should contain MLP mappings"


class TestQwen25OmniBridgeEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_bridge_with_minimal_config(self, qwen25_omni_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedVLM)
        minimal_config = Mock()
        minimal_thinker_config = Mock()
        minimal_text_config = Mock()

        # Set only required fields in text_config
        minimal_text_config.num_hidden_layers = 24
        minimal_text_config.hidden_size = 2048
        minimal_text_config.intermediate_size = 5504
        minimal_text_config.num_attention_heads = 16
        minimal_text_config.num_key_value_heads = 16
        minimal_text_config.initializer_range = 0.02
        minimal_text_config.rms_norm_eps = 1e-6
        minimal_text_config.vocab_size = 151936
        minimal_text_config.max_position_embeddings = 4096

        # Set thinker_config
        minimal_thinker_config.text_config = minimal_text_config
        minimal_thinker_config.vision_config = Qwen2_5_VLVisionConfig()

        minimal_config.thinker_config = minimal_thinker_config
        minimal_pretrained.config = minimal_config
        minimal_pretrained.generation_config = GenerationConfig()

        provider = qwen25_omni_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, Qwen25VLModelProvider)
        assert provider.num_layers == 24
        assert provider.hidden_size == 2048

    def test_provider_bridge_with_different_vocab_sizes(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge with different vocabulary sizes."""
        test_vocab_sizes = [32000, 151936, 152064]
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config

        for vocab_size in test_vocab_sizes:
            text_config.vocab_size = vocab_size
            provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)
            assert provider.vocab_size == vocab_size

    def test_provider_bridge_with_different_sequence_lengths(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge with different sequence lengths."""
        test_seq_lengths = [2048, 4096, 8192, 32768]
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config

        for seq_length in test_seq_lengths:
            text_config.max_position_embeddings = seq_length
            provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)
            assert provider.seq_length == seq_length


class TestQwen25OmniBridgeCompatibility:
    """Test compatibility with different HF model configurations."""

    def test_provider_bridge_with_group_query_attention(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge with group query attention."""
        text_config = mock_hf_omni_pretrained.config.thinker_config.text_config
        text_config.num_attention_heads = 32
        text_config.num_key_value_heads = 8

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8

    def test_provider_bridge_vision_config_types(self, qwen25_omni_bridge, mock_hf_omni_pretrained):
        """Test provider_bridge with different vision config types."""
        thinker_config = mock_hf_omni_pretrained.config.thinker_config
        # Test with custom vision config
        custom_vision_config = Qwen2_5_VLVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24)
        thinker_config.vision_config = custom_vision_config

        provider = qwen25_omni_bridge.provider_bridge(mock_hf_omni_pretrained)

        assert provider.vision_config.hidden_size == 1024
        assert provider.vision_config.intermediate_size == 4096
        assert provider.vision_config.num_hidden_layers == 24